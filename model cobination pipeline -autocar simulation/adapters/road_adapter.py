"""
Road Model Adapter.

Converts raw U-Net model outputs into standardized RoadMask objects.
Decouples U-Net tensor shapes, argmax class IDs, and frameworks from
perception fusion and simulation.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from perception.schemas import RoadMask


class RoadAdapter:
    """
    Adapter converting raw segmentation output into a standardized RoadMask.
    """
    def __init__(self, road_class_ids: Optional[List[int]] = None):
        # Default road classes from KITTI: 1 = lane mark, 2 = main road, 3 = mud road
        self.road_class_ids = road_class_ids or [1, 2, 3]

    def convert(self, raw_output: Any) -> RoadMask:
        """
        Convert raw model output to RoadMask.
        Accepts:
          - Dict containing 'raw_mask', 'orig_shape', 'road_class_ids', 'confidence'
          - Raw 2D/3D numpy array
        """
        if raw_output is None:
            return RoadMask(mask=None, confidence=0.0, road_area_ratio=0.0)

        confidence = 1.0
        road_class_ids = self.road_class_ids

        if isinstance(raw_output, dict):
            raw_mask = raw_output.get("raw_mask")
            confidence = raw_output.get("confidence") if raw_output.get("confidence") is not None else 1.0
            road_class_ids = raw_output.get("road_class_ids", self.road_class_ids)
        else:
            raw_mask = raw_output

        if raw_mask is None:
            return RoadMask(mask=None, confidence=0.0, road_area_ratio=0.0)

        # Ensure numpy array
        arr = np.asarray(raw_mask)
        if len(arr.shape) > 2:
            arr = np.squeeze(arr)

        # Convert to binary drivable mask: 1 where pixel class in road_class_ids or arr > 0
        if arr.dtype == bool:
            binary_mask = arr.astype(np.uint8)
        else:
            binary_mask = np.isin(arr, road_class_ids).astype(np.uint8)

        total_pixels = binary_mask.size
        road_pixels = int(np.count_nonzero(binary_mask))
        road_area_ratio = float(road_pixels / max(1, total_pixels))

        return RoadMask(
            mask=binary_mask,
            confidence=float(confidence),
            road_area_ratio=float(round(road_area_ratio, 4)),
            drivable_regions=[],
        )
