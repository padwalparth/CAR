"""
Res2Net Pothole Model Runner.

Wraps the existing Res2Net-50d bounding-box regression model trained on pothole datasets.
Architecture: timm.create_model('res2net50d.in1k', num_classes=4)
Checkpoint: pothole detection model/best_model.pt
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.base import BasePerceptionModel


class Res2NetPotholeModel(BasePerceptionModel):
    """
    Runner for existing Res2Net pothole bbox regression model.
    Loads best_model.pt state_dict, preprocesses 128x128 RGB input,
    and returns a single predicted bounding box.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="res2net_pothole_detector")
        self.model_path = config.get("path", "")
        self.backbone_name = config.get("backbone", "res2net50d.in1k")
        self.input_size = tuple(config.get("input_size", [128, 128]))
        self.min_box_size = float(config.get("min_box_size", 5.0))
        self.model = None

    def load(self) -> None:
        """Construct the PyTorch architecture and load the state_dict."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Pothole model checkpoint not found at: {self.model_path}")

        try:
            import torch
            import torch.nn as nn
            import timm
        except ImportError as e:
            raise ImportError(
                "PyTorch / timm is not installed. Install them with 'pip install torch torchvision timm' "
                "or configure pothole_model.backend: mock for testing."
            ) from e

        # Define model matching pothole-detection.ipynb
        class PotholeModel(nn.Module):
            def __init__(self, backbone_name: str):
                super().__init__()
                self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=4)

            def forward(self, images):
                return self.backbone(images)

        self.model = PotholeModel(self.backbone_name)
        device = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.is_loaded = True

    def _predict_patch(self, patch_bgr: np.ndarray) -> Tuple[float, float, float, float]:
        """Predict 128x128 bounding box coordinates for a single image patch."""
        import torch
        ph, pw = patch_bgr.shape[:2]
        try:
            import cv2
            rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self.input_size)
        except ImportError:
            from PIL import Image
            img = Image.fromarray(patch_bgr).convert("RGB")
            resized = np.array(img.resize(self.input_size))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)

        with torch.inference_mode():
            out_bbox = self.model(tensor)
            coords = out_bbox[0].cpu().numpy()

        scale_x = float(pw) / float(self.input_size[0])
        scale_y = float(ph) / float(self.input_size[1])

        p_xmin = float(coords[0] * scale_x)
        p_ymin = float(coords[1] * scale_y)
        p_xmax = float(coords[2] * scale_x)
        p_ymax = float(coords[3] * scale_y)

        return p_xmin, p_ymin, p_xmax, p_ymax

    def infer(self, frame: Any) -> Dict[str, Any]:
        """
        Run multi-crop contextual inference on frame.
        Detects both macro-scale road surface defects and micro-scale localized potholes.
        """
        if not self.is_loaded:
            raise RuntimeError("Res2Net pothole model is not loaded. Call load() before infer().")

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        candidates: List[Dict[str, Any]] = []

        # 1. Full-frame macro prediction
        fx1, fy1, fx2, fy2 = self._predict_patch(frame)
        candidates.append({"bbox": (fx1, fy1, fx2, fy2), "confidence": 0.95})

        # 2. Multi-tile micro predictions across road surface
        if orig_h >= 100 and orig_w >= 100:
            tiles = [
                (0, int(orig_w * 0.55), int(orig_h * 0.35), orig_h),            # Left road lane
                (int(orig_w * 0.45), orig_w, int(orig_h * 0.35), orig_h),       # Right road lane
                (int(orig_w * 0.20), int(orig_w * 0.80), int(orig_h * 0.40), orig_h), # Center road corridor
                (0, orig_w, int(orig_h * 0.45), orig_h),                        # Lower road ROI
            ]

            for tx1, tx2, ty1, ty2 in tiles:
                patch = frame[ty1:ty2, tx1:tx2]
                if patch.shape[0] < 30 or patch.shape[1] < 30:
                    continue
                px1, py1, px2, py2 = self._predict_patch(patch)
                pw, ph = px2 - px1, py2 - py1

                # Filter out degenerate or edge-touching background predictions
                if pw < self.min_box_size or ph < self.min_box_size:
                    continue
                if pw > (tx2 - tx1) * 0.95 and ph > (ty2 - ty1) * 0.95:
                    continue

                abs_x1 = max(0.0, min(float(orig_w), px1 + tx1))
                abs_y1 = max(0.0, min(float(orig_h), py1 + ty1))
                abs_x2 = max(0.0, min(float(orig_w), px2 + tx1))
                abs_y2 = max(0.0, min(float(orig_h), py2 + ty1))

                candidates.append({"bbox": (abs_x1, abs_y1, abs_x2, abs_y2), "confidence": 0.90})

        # Non-Maximum Suppression (NMS) to merge overlapping duplicate predictions
        def _box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
            a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
            union = a1 + a2 - inter
            return inter / max(1e-6, union)

        # Sort candidates by confidence then area
        candidates = sorted(candidates, key=lambda c: (c["confidence"], (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1])), reverse=True)
        final_boxes: List[Dict[str, Any]] = []

        for cand in candidates:
            b = cand["bbox"]
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bw < self.min_box_size or bh < self.min_box_size:
                continue

            overlap = False
            for k in final_boxes:
                if _box_iou(b, k["bbox"]) > 0.40:
                    overlap = True
                    break
            if not overlap:
                final_boxes.append(cand)

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "pothole_boxes": final_boxes,
            "orig_shape": (orig_h, orig_w),
            "inference_time_ms": self.last_inference_time_ms,
        }

    def close(self) -> None:
        self.model = None
        self.is_loaded = False


class MockPotholeModel(BasePerceptionModel):
    """
    Lightweight mock pothole model for testing without GPU or PyTorch.
    Produces synthetic pothole detections situated on the lower road surface.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {"backend": "mock"}, model_name="mock_pothole_detector")

    def load(self) -> None:
        self.is_loaded = True

    def infer(self, frame: Any) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("MockPotholeModel is not loaded.")

        t0 = time.perf_counter()
        if hasattr(frame, "shape"):
            h, w = frame.shape[:2]
        else:
            h, w = (720, 1280)

        # Synthetic pothole box in bottom center of road
        boxes = [
            {
                "bbox": (w * 0.40, h * 0.72, w * 0.54, h * 0.84),
                "confidence": 0.92,
            }
        ]
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "pothole_boxes": boxes,
            "orig_shape": (h, w),
            "inference_time_ms": self.last_inference_time_ms,
        }
