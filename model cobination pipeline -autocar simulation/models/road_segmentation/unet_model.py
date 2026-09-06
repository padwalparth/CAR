"""
U-Net Road Segmentation Model Runner.

Supports:
1. Keras/TensorFlow backend (.h5 saved model) - uses pretrained weights directly
2. ONNX Runtime backend
3. MockRoadModel for testing and lightweight development
"""

import os
import time
from typing import Any, Dict, Optional
import numpy as np
import cv2

from models.base import BasePerceptionModel


class UNetRoadModel(BasePerceptionModel):
    """
    Runner for U-Net road segmentation using pretrained Keras .h5 model.
    Loads the model once at startup, preprocesses frames, runs inference,
    and returns a standardized pixel-level drivable road mask.

    The model outputs a per-pixel class probability map (shape H,W,num_classes).
    Class 0 = background, Class 1 = drivable road.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="unet_road_segmentation")
        self.model_path = config.get("path", "")
        self.input_size = tuple(config.get("input_size", [160, 160]))  # (H, W)
        self.road_threshold = float(config.get("threshold", 0.35))
        self.use_multiscale = bool(config.get("multiscale", True))
        self._backend = str(config.get("backend", "keras")).lower()
        self.model = None
        self._ort_session = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights. Auto-detects backend from file extension if needed."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Road segmentation model not found at: {self.model_path}")

        ext = os.path.splitext(self.model_path)[1].lower()

        if self._backend in ("pytorch", "torch") or ext in (".pt", ".pth"):
            self._load_pytorch()
        elif self._backend in ("keras", "tensorflow", "tf") or ext in (".h5", ".keras"):
            self._load_keras()
        elif self._backend in ("onnx",) or ext == ".onnx":
            self._load_onnx()
        else:
            # Try ONNX first, then PyTorch, then Keras
            try:
                self._load_onnx()
            except Exception:
                try:
                    self._load_pytorch()
                except Exception:
                    self._load_keras()

        self.is_loaded = True

    def _load_pytorch(self) -> None:
        """Load PyTorch .pt model."""
        try:
            import torch
            import torch.nn as nn

            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )
                def forward(self, x):
                    return self.conv(x)

            class UNetRoad(nn.Module):
                def __init__(self, in_channels=3, out_channels=1, base_ch=32):
                    super().__init__()
                    self.enc1 = DoubleConv(in_channels, base_ch)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base_ch, base_ch * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
                    self.pool3 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)
                    self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
                    self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
                    self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
                    self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
                    self.dec1 = DoubleConv(base_ch * 2, base_ch)
                    self.head = nn.Conv2d(base_ch, out_channels, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    e3 = self.enc3(self.pool2(e2))
                    b = self.bottleneck(self.pool3(e3))
                    d3 = self.up3(b)
                    d3 = self.dec3(torch.cat([d3, e3], dim=1))
                    d2 = self.up2(d3)
                    d2 = self.dec2(torch.cat([d2, e2], dim=1))
                    d1 = self.up1(d2)
                    d1 = self.dec1(torch.cat([d1, e1], dim=1))
                    return torch.sigmoid(self.head(d1))

            self.model = UNetRoad()
            state = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self._backend = "pytorch"
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch road segmentation model: {e}")

    def _load_keras(self) -> None:
        """Load Keras .h5 model."""
        try:
            import tensorflow as tf
            # Suppress TF info logs
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self._backend = "keras"
        except ImportError:
            raise ImportError(
                "TensorFlow is not installed. Install with: pip install tensorflow"
            )

    def _load_onnx(self) -> None:
        """Load ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort
            self._ort_session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )
            self._backend = "onnx"
        except ImportError:
            raise ImportError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise a BGR frame → float32 RGB array (H, W, 3) in [0, 1]."""
        if len(img_bgr.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_size[1], self.input_size[0]))
        return resized.astype(np.float32) / 255.0  # (H, W, 3)

    def _predict_raw(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run a single-scale inference pass.
        Returns 2-D probability road map of shape (H_model, W_model), values in [0, 1].
        """
        img_norm = self._preprocess(img_bgr)          # (160, 160, 3)
        batch_nhwc = np.expand_dims(img_norm, axis=0) # (1, 160, 160, 3)

        if self._backend == "pytorch":
            import torch
            with torch.no_grad():
                inp_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
                out_t = self.model(inp_t)
                pred = out_t.numpy()
        elif self._backend == "keras":
            pred = self.model.predict(batch_nhwc, verbose=0)  # (1, H, W, num_classes)
        elif self._backend == "onnx":
            inp_meta = self._ort_session.get_inputs()[0]
            inp_shape = inp_meta.shape
            if len(inp_shape) == 4 and inp_shape[1] == 3:
                # NCHW
                batch_nchw = np.transpose(batch_nhwc, (0, 3, 1, 2))
                pred = self._ort_session.run(None, {inp_meta.name: batch_nchw})[0]
            else:
                # NHWC
                pred = self._ort_session.run(None, {inp_meta.name: batch_nhwc})[0]
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

        pred_sq = np.squeeze(pred)

        if pred_sq.ndim == 2:
            prob_road = pred_sq.astype(np.float32)
            if prob_road.min() < 0.0 or prob_road.max() > 1.0:
                prob_road = 1.0 / (1.0 + np.exp(-prob_road))
        elif pred_sq.ndim == 3:
            # Check if channel-first (C, H, W)
            if pred_sq.shape[0] in (1, 2, 3, 4) and pred_sq.shape[0] < pred_sq.shape[1]:
                pred_sq = np.transpose(pred_sq, (1, 2, 0))

            num_classes = pred_sq.shape[-1]
            if num_classes == 1:
                prob_road = pred_sq[:, :, 0].astype(np.float32)
                if prob_road.min() < 0.0 or prob_road.max() > 1.0:
                    prob_road = 1.0 / (1.0 + np.exp(-prob_road))
            elif num_classes == 2:
                prob_road = pred_sq[:, :, 1].astype(np.float32)
            else:
                # Multi-class KITTI model: 1=lane, 2=main road, 3=mud road
                argmax_mask = np.argmax(pred_sq, axis=-1)
                road_prob = np.zeros(argmax_mask.shape, dtype=np.float32)
                for road_cls in [1, 2, 3]:
                    if road_cls < num_classes:
                        road_prob += pred_sq[:, :, road_cls]
                argmax_road = ((argmax_mask == 1) | (argmax_mask == 2) | (argmax_mask == 3)).astype(np.float32)
                prob_road = np.maximum(road_prob, argmax_road * 0.9)
        else:
            prob_road = pred_sq.astype(np.float32)

        return prob_road  # (H_model, W_model) values in [0, 1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, frame: Any) -> Dict[str, Any]:
        """
        Run inference on a raw BGR numpy frame.
        Uses optional multi-scale contextual inference, calibrated thresholding,
        morphological smoothing, and connected-component noise removal.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        # 1. Global-scale prediction
        prob_small = self._predict_raw(frame)  # (160, 160)
        prob_full = cv2.resize(prob_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 2. Road-ROI scale prediction (lower 70% of frame)
        if self.use_multiscale and orig_h >= 100:
            roi_y0 = int(orig_h * 0.30)
            prob_roi_small = self._predict_raw(frame[roi_y0:, :])
            prob_roi = cv2.resize(
                prob_roi_small,
                (orig_w, orig_h - roi_y0),
                interpolation=cv2.INTER_LINEAR
            )
            prob_map = prob_full.copy()
            prob_map[roi_y0:, :] = np.maximum(prob_map[roi_y0:, :], prob_roi * 0.95)
        else:
            prob_map = prob_full

        # 3. Threshold → binary mask
        binary_mask = (prob_map >= self.road_threshold).astype(np.uint8)

        # 4. Morphological closure
        k_size = max(5, int(min(orig_w, orig_h) * 0.015))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        smooth_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Connected-component filtering (remove tiny noise blobs)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            smooth_mask, connectivity=8
        )
        min_area = int(orig_w * orig_h * 0.005)  # 0.5 % of image
        clean_mask = np.zeros_like(smooth_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 1

        # Confidence = mean road probability over detected road pixels
        road_pixels = clean_mask > 0
        mean_conf = float(np.mean(prob_map[road_pixels])) if np.any(road_pixels) else 0.0

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "raw_mask": clean_mask,
            "orig_shape": (orig_h, orig_w),
            "model_shape": self.input_size,
            "road_class_ids": [1],
            "confidence": round(mean_conf, 4),
            "inference_time_ms": round(self.last_inference_time_ms, 2),
        }

    def close(self) -> None:
        self.model = None
        self._ort_session = None
        self.is_loaded = False


# ---------------------------------------------------------------------------


class MockRoadModel(BasePerceptionModel):
    """
    Lightweight mock road model for tests and non-GPU / no-TF environments.
    Generates a realistic trapezoid road mask without any ML inference.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {"backend": "mock"}, model_name="mock_road_segmentation")
        self.input_size = (160, 160)

    def load(self) -> None:
        self.is_loaded = True

    def infer(self, frame: Any) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("MockRoadModel is not loaded.")

        t0 = time.perf_counter()
        if hasattr(frame, "shape"):
            orig_h, orig_w = frame.shape[:2]
        else:
            orig_h, orig_w = (720, 1280)

        h, w = self.input_size
        mask = np.zeros((h, w), dtype=np.uint8)
        horizon = int(h * 0.45)
        for y in range(horizon, h):
            ratio = (y - horizon) / max(1, (h - horizon))
            half_width = int((0.15 + 0.35 * ratio) * w)
            cx = w // 2
            x_left = max(0, cx - half_width)
            x_right = min(w, cx + half_width)
            mask[y, x_left:x_right] = 1

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "raw_mask": mask,
            "orig_shape": (orig_h, orig_w),
            "model_shape": self.input_size,
            "road_class_ids": [1],
            "confidence": 1.0,
        }

    def close(self) -> None:
        self.is_loaded = False
