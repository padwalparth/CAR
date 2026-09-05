"""
Future YOLO Pothole Detector Interface Stub.

This module provides the interface for a future YOLO-based pothole detector
(e.g., YOLOv8/YOLOv11 trained specifically on multi-pothole datasets).

CONTRACT SPECIFICATION:
When implemented, YOLOPotholeModel must output the exact same dictionary format
expected by PotholeAdapter:
{
    "pothole_boxes": [
        {
            "bbox": (xmin, ymin, xmax, ymax),  # Pixel coordinates
            "confidence": float,                 # [0.0, 1.0]
        },
        ...
    ],
    "orig_shape": (height, width),
    "inference_time_ms": float,
}

This design guarantees that replacing Res2Net with YOLO requires ZERO changes
to adapters, fusion, analyzers, or simulation code.
"""

from typing import Any, Dict
from models.base import BasePerceptionModel


class YOLOPotholeModel(BasePerceptionModel):
    """
    Stub runner for future YOLO-based pothole detector.
    Raises NotImplementedError until real YOLO pothole weights are configured.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, model_name="yolo_pothole_detector")
        self.model_path = config.get("path", "")

    def load(self) -> None:
        raise NotImplementedError(
            "YOLO Pothole Model backend is a future extension interface. "
            "To use pothole detection with current trained weights, "
            "set pothole_model.backend: res2net (or 'mock' for testing) in config/models.yaml."
        )

    def infer(self, frame: Any) -> Dict[str, Any]:
        raise NotImplementedError("YOLOPotholeModel is not implemented yet.")
