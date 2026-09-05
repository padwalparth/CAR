"""
U-Net Road Segmentation Model Runner.

Supports:
1. ONNX Runtime backend (road_seg_160_160.onnx)
2. Keras / TensorFlow backend (.h5 models)
3. MockRoadModel for testing and lightweight development
"""

import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from models.base import BasePerceptionModel


class UNetRoadModel(BasePerceptionModel):
    """
    Runner for existing U-Net road segmentation model.
    Loads once, preprocesses frames, runs inference, and returns raw mask.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="unet_road_segmentation")
        self.model_path = config.get("path", "")
        self.fallback_keras_path = config.get("fallback_keras_path", "")
        self.input_size = tuple(config.get("input_size", [160, 160]))
        self.road_class_ids = config.get("road_class_ids", [1, 2, 3])
        self.road_threshold = float(config.get("threshold", 0.18))
        self.use_multiscale = bool(config.get("multiscale", True))
        self.session = None
        self.model = None

    def load(self) -> None:
        """Load ONNX or Keras model depending on configured backend."""
        if self.backend == "onnx":
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX road model not found at: {self.model_path}")
            try:
                import onnxruntime as rt
            except ImportError as e:
                raise ImportError(
                    "onnxruntime is not installed. Install it with 'pip install onnxruntime' "
                    "or configure road_model.backend: mock for testing."
                ) from e

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            # Filter available providers
            available = rt.get_available_providers()
            active_providers = [p for p in providers if p in available]
            self.session = rt.InferenceSession(self.model_path, providers=active_providers)
            self.is_loaded = True

        elif self.backend == "keras":
            target_path = self.model_path if os.path.exists(self.model_path) else self.fallback_keras_path
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Keras road model not found at: {target_path}")
            try:
                from tensorflow import keras
            except ImportError as e:
                raise ImportError(
                    "tensorflow is not installed. Install it with 'pip install tensorflow' "
                    "or configure road_model.backend: mock for testing."
                ) from e
            self.model = keras.models.load_model(target_path)
            self.is_loaded = True

        else:
            raise ValueError(f"Unsupported road segmentation backend: '{self.backend}'. Use 'onnx', 'keras', or 'mock'.")

    def _predict_raw(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run single-patch prediction on (160, 160) input and return softmax probabilities."""
        try:
            import cv2
            resized = cv2.resize(img_bgr, self.input_size)
        except ImportError:
            from PIL import Image
            img = Image.fromarray(img_bgr)
            resized = np.array(img.resize(self.input_size))

        data = np.expand_dims(resized.astype("float32"), axis=0)

        if self.backend == "onnx":
            input_name = self.session.get_inputs()[0].name
            raw_pred = self.session.run(None, {input_name: data})
            val_pred = raw_pred[0]
        else:
            val_pred = self.model.predict(data, verbose=0)

        if len(val_pred.shape) == 4:
            return val_pred[0]
        return val_pred

    def infer(self, frame: Any) -> Dict[str, Any]:
        """
        Run inference on raw frame (numpy ndarray, BGR or RGB).
        Uses multi-scale contextual inference, calibrated probability thresholding,
        and morphological road-surface smoothing.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() before infer().")

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        import cv2

        # 1. Global image scale prediction
        pred_full = self._predict_raw(frame)  # shape: (160, 160, 3)
        # In the ONNX model, class index 2 is primary road surface, index 1 is lane markings
        prob_road_full = pred_full[:, :, 2]
        if pred_full.shape[-1] > 1:
            prob_road_full = np.maximum(prob_road_full, pred_full[:, :, 1])

        prob_full_up = cv2.resize(prob_road_full, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 2. Road ROI scale prediction (lower 65% of frame where road surface resides)
        if self.use_multiscale and orig_h >= 100:
            roi_y0 = int(orig_h * 0.32)
            roi_frame = frame[roi_y0:, :]
            pred_roi = self._predict_raw(roi_frame)
            prob_road_roi = pred_roi[:, :, 2]
            if pred_roi.shape[-1] > 1:
                prob_road_roi = np.maximum(prob_road_roi, pred_roi[:, :, 1])

            prob_roi_up = cv2.resize(prob_road_roi, (orig_w, orig_h - roi_y0), interpolation=cv2.INTER_LINEAR)

            # Contextual blending: reinforce road confidence in driving zone
            prob_map = prob_full_up.copy()
            prob_map[roi_y0:, :] = np.maximum(prob_map[roi_y0:, :], prob_roi_up)
        else:
            prob_map = prob_full_up

        # 3. Calibrated probability thresholding (avoids argmax clipping at lane boundaries)
        binary_mask = (prob_map >= self.road_threshold).astype(np.uint8)

        # 4. Morphological closure (bridges lane marks, textured asphalt, and shadows)
        k_size = max(5, int(min(orig_w, orig_h) * 0.015))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        smooth_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Connected component filtering to remove small disjoint false positives
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(smooth_mask, connectivity=8)
        min_component_area = int(orig_w * orig_h * 0.003)  # 0.3% of image area
        clean_mask = np.zeros_like(smooth_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                clean_mask[labels == i] = 1

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "raw_mask": clean_mask,
            "orig_shape": (orig_h, orig_w),
            "model_shape": self.input_size,
            "road_class_ids": self.road_class_ids,
            "confidence": float(np.mean(prob_map[clean_mask > 0])) if np.any(clean_mask > 0) else 0.0,
        }

    def close(self) -> None:
        self.session = None
        self.model = None
        self.is_loaded = False


class MockRoadModel(BasePerceptionModel):
    """
    Lightweight Mock road model runner for tests and non-GPU environments.
    Produces a realistic synthetic road mask (trapezoidal drivable road corridor).
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

        # Generate synthetic 160x160 road mask where bottom half has drivable road
        h, w = self.input_size
        mask = np.zeros((h, w), dtype=np.uint8)
        horizon = int(h * 0.45)
        for y in range(horizon, h):
            ratio = (y - horizon) / max(1, (h - horizon))
            half_width = int((0.15 + 0.35 * ratio) * w)
            cx = w // 2
            x_left = max(0, cx - half_width)
            x_right = min(w, cx + half_width)
            mask[y, x_left:x_right] = 2  # class 2 = main road

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "raw_mask": mask,
            "orig_shape": (orig_h, orig_w),
            "model_shape": self.input_size,
            "road_class_ids": [1, 2, 3],
            "confidence": 1.0,
        }
