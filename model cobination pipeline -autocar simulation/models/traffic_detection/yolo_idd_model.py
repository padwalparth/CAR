"""
YOLOv8 IDD Traffic Model Runner.

Runs the YOLOv8.3 model trained on the Indian Driving Dataset (IDD).
Supports Ultralytics PyTorch backend and lightweight MockTrafficModel.
"""

import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from models.base import BasePerceptionModel

# Exact 15 classes from YOLOv8.3.0 Train on IDD/idd_fixed.yaml
IDD_CLASSES: Dict[int, str] = {
    0: "animal",
    1: "autorickshaw",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "caravan",
    6: "motorcycle",
    7: "person",
    8: "rider",
    9: "traffic light",
    10: "traffic sign",
    11: "trailer",
    12: "train",
    13: "truck",
    14: "vehicle fallback",
}


class YOLOIDDTrafficModel(BasePerceptionModel):
    """
    Runner for YOLOv8 model trained on Indian Driving Dataset (IDD).
    Loads model once, executes batched or single-frame inference.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="yolo_idd_traffic_detector")
        self.model_path = config.get("path", "")
        self.fallback_path = config.get("fallback_path", "")
        self.confidence_threshold = float(config.get("confidence", 0.35))
        self.iou_threshold = float(config.get("iou", 0.70))
        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.model = None

    def load(self) -> None:
        """Load YOLO model checkpoint using Ultralytics."""
        target_path = self.model_path if os.path.exists(self.model_path) else self.fallback_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"YOLO IDD checkpoint not found at: {target_path}")

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Install it with 'pip install ultralytics' "
                "or configure traffic_model.backend: mock for testing."
            ) from e

        self.model = YOLO(target_path)
        self.is_loaded = True

    def infer(self, frame: Any) -> Dict[str, Any]:
        """
        Run YOLO detection on frame.
        Returns raw detections structure for TrafficAdapter.
        """
        if not self.is_loaded:
            raise RuntimeError("YOLO model is not loaded. Call load() before infer().")

        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=self.input_size[0],
        )
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = IDD_CLASSES.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                })

        return {
            "detections": detections,
            "inference_time_ms": self.last_inference_time_ms,
        }

    def close(self) -> None:
        self.model = None
        self.is_loaded = False


class MockTrafficModel(BasePerceptionModel):
    """
    Lightweight mock YOLO traffic model for testing without GPU or PyTorch.
    Produces realistic IDD detections (cars, autorickshaws, pedestrians, motorcycles).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {"backend": "mock"}, model_name="mock_traffic_detector")
        self.confidence_threshold = 0.35

    def load(self) -> None:
        self.is_loaded = True

    def infer(self, frame: Any) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("MockTrafficModel is not loaded.")

        t0 = time.perf_counter()
        if hasattr(frame, "shape"):
            h, w = frame.shape[:2]
        else:
            h, w = (720, 1280)

        # Synthetic detections representative of an Indian road scene
        detections = [
            {
                "class_id": 4,  # car
                "class_name": "car",
                "confidence": 0.89,
                "bbox": (w * 0.45, h * 0.55, w * 0.62, h * 0.78),
            },
            {
                "class_id": 1,  # autorickshaw
                "class_name": "autorickshaw",
                "confidence": 0.91,
                "bbox": (w * 0.20, h * 0.52, w * 0.38, h * 0.76),
            },
            {
                "class_id": 6,  # motorcycle
                "class_name": "motorcycle",
                "confidence": 0.84,
                "bbox": (w * 0.65, h * 0.60, w * 0.75, h * 0.82),
            },
            {
                "class_id": 7,  # person
                "class_name": "person",
                "confidence": 0.78,
                "bbox": (w * 0.12, h * 0.50, w * 0.18, h * 0.70),
            },
        ]
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "detections": detections,
            "inference_time_ms": self.last_inference_time_ms,
        }
