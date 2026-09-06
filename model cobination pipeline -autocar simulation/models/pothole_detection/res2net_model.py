"""
Res2Net Pothole Model Runner — V2 (Classification + BBox Regression).

Architecture: PotholeDetectorV2
  - Backbone: res2net50d.in1k (feature extractor, no classifier)
  - cls_head:  Linear(2048→256→1)  → sigmoid probability of pothole present
  - bbox_head: Linear(2048→256→4)  → [x1, y1, x2, y2] in 128px space

Checkpoint: pothole detection model/best_model_v2.pt

Inference logic:
  1. Slide a 2-scale overlapping grid across the lower road region.
  2. For each patch, run the model.
  3. Accept the detection ONLY if cls_prob >= confidence_threshold (default 0.50).
  4. Map the 128px bbox back to full-image pixel coordinates.
  5. Apply NMS to remove overlapping boxes.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from models.base import BasePerceptionModel


class Res2NetPotholeModel(BasePerceptionModel):
    """
    Res2Net V2 pothole detector: classification gate + bbox regression.
    Only patches classified as pothole (prob > threshold) produce detections.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="res2net_pothole_detector_v2")
        self.model_path = config.get("path", "")
        self.backbone_name = config.get("backbone", "res2net50d.in1k")
        self.input_size = tuple(config.get("input_size", [128, 128]))
        self.min_box_size = float(config.get("min_box_size", 12.0))
        self.max_detections = int(config.get("max_detections", 15))
        # Confidence threshold for classification gate
        self.cls_threshold = float(config.get("cls_threshold", 0.50))
        self.model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build PotholeDetectorV2 architecture and load the state_dict."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Pothole model checkpoint not found at: {self.model_path}"
            )

        try:
            import torch
            import torch.nn as nn
            import timm
        except ImportError as e:
            raise ImportError(
                "PyTorch / timm not installed. Run: pip install torch torchvision timm"
            ) from e

        class PotholeDetectorV2(nn.Module):
            def __init__(self, backbone_name: str):
                super().__init__()
                self.backbone = timm.create_model(
                    backbone_name, pretrained=False, num_classes=0, global_pool="avg"
                )
                feat_dim = self.backbone.num_features
                self.cls_head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                )
                self.bbox_head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 4),
                )

            def forward(self, x):
                feats = self.backbone(x)
                return self.cls_head(feats), self.bbox_head(feats)

        self.model = PotholeDetectorV2(self.backbone_name)
        device = torch.device(
            self.device if (torch.cuda.is_available() and self.device == "cuda") else "cpu"
        )
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Single-patch inference
    # ------------------------------------------------------------------

    def _predict_patch(
        self, patch_bgr: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns (x1, y1, x2, y2, cls_prob) in patch pixel space.
        cls_prob is the probability that this patch contains a pothole.
        """
        import torch

        ph, pw = patch_bgr.shape[:2]
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        tensor = (
            torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        )
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)

        with torch.inference_mode():
            cls_logit, bbox_pred = self.model(tensor)
            cls_prob = float(torch.sigmoid(cls_logit[0, 0]).cpu())
            coords = bbox_pred[0].cpu().numpy()

        # Scale bbox from 128px model space back to patch pixel space
        scale_x = float(pw) / float(self.input_size[0])
        scale_y = float(ph) / float(self.input_size[1])

        p_x1 = float(coords[0]) * scale_x
        p_y1 = float(coords[1]) * scale_y
        p_x2 = float(coords[2]) * scale_x
        p_y2 = float(coords[3]) * scale_y

        return p_x1, p_y1, p_x2, p_y2, cls_prob

    # ------------------------------------------------------------------
    # Full-image inference
    # ------------------------------------------------------------------

    def infer(self, frame: Any) -> Dict[str, Any]:
        """
        Slide grid across road region. Accept patch as pothole only if
        classification probability >= cls_threshold.
        """
        if not self.is_loaded:
            raise RuntimeError("Res2Net model is not loaded. Call load() first.")

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        # Only scan lower portion of image (road surface, not sky)
        road_top_y = int(orig_h * 0.40)
        road_h = orig_h - road_top_y

        candidates: List[Dict[str, Any]] = []

        if road_h >= 60 and orig_w >= 80:
            # Two-scale grid
            grid_configs = [
                (2, 3, 0.65),   # coarse: 2 rows × 3 cols, 65% overlap stride
                (3, 4, 0.70),   # medium: 3 rows × 4 cols, 70% overlap stride
            ]

            for num_rows, num_cols, overlap in grid_configs:
                tile_w = int(orig_w / (num_cols * (1.0 - overlap) + overlap))
                tile_h = int(road_h / (num_rows * (1.0 - overlap) + overlap))
                tile_w = max(64, tile_w)
                tile_h = max(48, tile_h)

                step_x = max(20, int(tile_w * (1.0 - overlap)))
                step_y = max(15, int(tile_h * (1.0 - overlap)))

                y = road_top_y
                while y + tile_h <= orig_h:
                    x = 0
                    while x + tile_w <= orig_w:
                        ty1, ty2 = y, min(orig_h, y + tile_h)
                        tx1, tx2 = x, min(orig_w, x + tile_w)

                        patch = frame[ty1:ty2, tx1:tx2]
                        if patch.shape[0] < 32 or patch.shape[1] < 32:
                            x += step_x
                            continue

                        px1, py1, px2, py2, cls_prob = self._predict_patch(patch)

                        # === CLASSIFICATION GATE: reject if model says "no pothole" ===
                        if cls_prob < self.cls_threshold:
                            x += step_x
                            continue

                        bw, bh = px2 - px1, py2 - py1
                        if bw < self.min_box_size or bh < self.min_box_size:
                            x += step_x
                            continue

                        # Reject if box fills entire patch (degenerate prediction)
                        patch_w = float(tx2 - tx1)
                        patch_h = float(ty2 - ty1)
                        if bw > patch_w * 0.92 and bh > patch_h * 0.92:
                            x += step_x
                            continue

                        abs_x1 = float(np.clip(px1 + tx1, 0, orig_w))
                        abs_y1 = float(np.clip(py1 + ty1, 0, orig_h))
                        abs_x2 = float(np.clip(px2 + tx1, 0, orig_w))
                        abs_y2 = float(np.clip(py2 + ty1, 0, orig_h))

                        candidates.append({
                            "bbox": (abs_x1, abs_y1, abs_x2, abs_y2),
                            "confidence": round(cls_prob, 3),
                        })

                        x += step_x
                    y += step_y

        # ------------------------------------------------------------------
        # NMS
        # ------------------------------------------------------------------
        def _box_iou(b1, b2) -> float:
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
            a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
            union = a1 + a2 - inter
            return inter / max(1e-6, union)

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        final_boxes: List[Dict[str, Any]] = []

        for cand in candidates:
            b = cand["bbox"]
            if b[2] - b[0] < self.min_box_size or b[3] - b[1] < self.min_box_size:
                continue

            suppressed = any(
                _box_iou(b, k["bbox"]) > 0.35 for k in final_boxes
            )
            if not suppressed:
                final_boxes.append(cand)
                if len(final_boxes) >= self.max_detections:
                    break

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "pothole_boxes": final_boxes,
            "orig_shape": (orig_h, orig_w),
            "inference_time_ms": self.last_inference_time_ms,
        }

    def close(self) -> None:
        self.model = None
        self.is_loaded = False


# ---------------------------------------------------------------------------
# Lightweight mock for testing without GPU/PyTorch
# ---------------------------------------------------------------------------

class MockPotholeModel(BasePerceptionModel):
    """Mock pothole model — returns zero detections (safe default)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {"backend": "mock"}, model_name="mock_pothole_detector")

    def load(self) -> None:
        self.is_loaded = True

    def infer(self, frame: Any) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("MockPotholeModel is not loaded.")

        t0 = time.perf_counter()
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pothole_boxes": [
                {
                    "bbox": (int(w * 0.45), int(h * 0.70), int(w * 0.55), int(h * 0.80)),
                    "confidence": 0.88,
                }
            ],
            "orig_shape": (h, w),
            "inference_time_ms": self.last_inference_time_ms,
        }

    def close(self) -> None:
        self.is_loaded = False
