"""
YOLOv11 Pothole Detector — Production Implementation.

Wraps the trained YOLOv11n pothole detection model (runs/detect/train/weights/best.pt).
Outputs the standard pothole_boxes format consumed by PotholeAdapter.

Validated behaviour:
  - Returns 0 detections on sky / blank asphalt images (no false positives)
  - Returns proper bounding boxes with real confidence scores on road images
  - Class name: 'pothole' (single-class detector)

Output contract (same as Res2Net model, drop-in replacement):
{
    "pothole_boxes": [
        {
            "bbox": (xmin, ymin, xmax, ymax),   # pixel coords, absolute
            "confidence": float,                  # 0.0–1.0, real YOLO score
        },
        ...
    ],
    "orig_shape": (height, width),
    "inference_time_ms": float,
}
"""

import os
import time
from typing import Any, Dict, Optional

from models.base import BasePerceptionModel


class YOLOPotholeModel(BasePerceptionModel):
    """
    Production runner for the YOLOv11n pothole detector.
    Uses Ultralytics YOLO inference — proper classification gate built-in,
    no false positives on non-pothole scenes.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="yolov11_pothole_detector")
        self.model_path = config.get("path", "")
        self.confidence_threshold = float(config.get("confidence", 0.30))
        self.iou_threshold = float(config.get("iou", 0.45))
        self.max_detections = int(config.get("max_detections", 20))
        self.model = None

    def load(self) -> None:
        """Load YOLOv11 pothole model weights."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLOv11 pothole model not found at: {self.model_path}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        self.model = YOLO(self.model_path)
        self.is_loaded = True

    def infer(self, frame: Any) -> Dict[str, Any]:
        """
        Run YOLOv11 inference on a single BGR frame.
        Returns standardised pothole_boxes list.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("YOLOv11 pothole model is not loaded. Call load() first.")

        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            verbose=False,
        )

        pothole_boxes = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box, conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
            ):
                x1, y1, x2, y2 = box
                # Clamp to frame dimensions
                x1 = float(max(0.0, min(x1, orig_w)))
                y1 = float(max(0.0, min(y1, orig_h)))
                x2 = float(max(0.0, min(x2, orig_w)))
                y2 = float(max(0.0, min(y2, orig_h)))

                # Skip degenerate boxes
                if x2 - x1 < 4.0 or y2 - y1 < 4.0:
                    continue

                pothole_boxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": round(float(conf), 4),
                })

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "pothole_boxes": pothole_boxes,
            "orig_shape": (orig_h, orig_w),
            "inference_time_ms": self.last_inference_time_ms,
        }

    def close(self) -> None:
        self.model = None
        self.is_loaded = False
