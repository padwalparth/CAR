"""
Generic Postprocessing Utilities.

Applies model-agnostic bounding-box clipping and area filtering.
Model-specific NMS and argmax decoding remain inside model adapters.
"""

from typing import List
from perception.schemas import Detection, PotholeDetection
from perception.coordinate_utils import clip_bbox


def clip_detections_to_frame(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
) -> List[Detection]:
    """Ensure all detection bounding boxes stay within frame pixel limits."""
    clipped = []
    for d in detections:
        if not d.bbox.is_normalized:
            d.bbox = clip_bbox(d.bbox, 0.0, 0.0, float(frame_width), float(frame_height))
        clipped.append(d)
    return clipped


def filter_detections_by_min_area(
    detections: List[Detection],
    min_area_pixels: float = 25.0,
) -> List[Detection]:
    """Filter out degenerate or microscopic bounding boxes."""
    return [d for d in detections if d.bbox.area >= min_area_pixels]
