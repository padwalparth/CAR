"""
Spatial Association Utilities for Linking Detections to the Road Mask.

Provides configurable spatial reasoning to determine:
1. Whether potholes lie within the drivable road region.
2. Whether vehicles/pedestrians are on the road surface via bottom-center grounding.
"""

from typing import Any, Tuple
from perception.schemas import BoundingBox


def is_point_in_road(
    point: Tuple[float, float],
    road_mask: Any,
    img_width: int,
    img_height: int,
) -> bool:
    """
    Check if a point (x, y) falls inside the positive road mask.
    point can be in pixel coordinates or normalized coordinates.
    """
    if road_mask is None:
        return True

    px, py = point
    # If point is normalized [0, 1], scale to mask coordinates
    if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and (img_width > 1 or img_height > 1):
        # Could be normalized
        px = px * img_width
        py = py * img_height

    # Retrieve dimensions of road_mask
    if hasattr(road_mask, "shape"):
        mask_h, mask_w = road_mask.shape[:2]
    elif isinstance(road_mask, (list, tuple)) and len(road_mask) > 0:
        mask_h = len(road_mask)
        mask_w = len(road_mask[0])
    else:
        return True

    # Scale pixel point to road_mask resolution
    col = int(min(max(0, round((px / max(1.0, float(img_width))) * mask_w)), mask_w - 1))
    row = int(min(max(0, round((py / max(1.0, float(img_height))) * mask_h)), mask_h - 1))

    val = road_mask[row][col] if isinstance(road_mask, (list, tuple)) else road_mask[row, col]
    return bool(val > 0)


def calculate_bbox_road_overlap(
    bbox: BoundingBox,
    road_mask: Any,
    img_width: int,
    img_height: int,
    grid_size: Tuple[int, int] = (3, 3),
) -> float:
    """
    Compute estimated road-mask overlap percentage for a bounding box
    by sampling points across a regular grid within the box.
    Returns value in range [0.0, 1.0].
    """
    if road_mask is None:
        return 1.0

    # Ensure box in pixel coordinates
    pix_box = bbox.to_pixel(img_width, img_height) if bbox.is_normalized else bbox

    if pix_box.width <= 0 or pix_box.height <= 0:
        return 0.0

    nx, ny = grid_size
    positive_samples = 0
    total_samples = nx * ny

    for i in range(nx):
        x = pix_box.xmin + (i + 0.5) * (pix_box.width / nx)
        for j in range(ny):
            y = pix_box.ymin + (j + 0.5) * (pix_box.height / ny)
            if is_point_in_road((x, y), road_mask, img_width, img_height):
                positive_samples += 1

    return float(positive_samples / max(1, total_samples))


def associate_pothole_with_road(
    bbox: BoundingBox,
    road_mask: Any,
    img_width: int,
    img_height: int,
    min_overlap: float = 0.30,
    center_weight: float = 0.50,
    overlap_weight: float = 0.50,
) -> Tuple[bool, float]:
    """
    Determine whether a pothole is located inside the road surface.
    
    Formula:
      confidence = (center_weight * is_center_on_road) + (overlap_weight * bbox_overlap_ratio)
      associated = confidence >= min_overlap
      
    Returns:
      (is_associated, association_confidence)
    """
    if road_mask is None:
        return True, 1.0

    pix_box = bbox.to_pixel(img_width, img_height) if bbox.is_normalized else bbox
    cx, cy = pix_box.center

    center_on_road = 1.0 if is_point_in_road((cx, cy), road_mask, img_width, img_height) else 0.0
    overlap_ratio = calculate_bbox_road_overlap(pix_box, road_mask, img_width, img_height)

    # Normalize weights if sum != 1.0
    w_sum = max(1e-6, center_weight + overlap_weight)
    cw = center_weight / w_sum
    ow = overlap_weight / w_sum

    association_confidence = (cw * center_on_road) + (ow * overlap_ratio)
    is_associated = bool(association_confidence >= min_overlap)

    return is_associated, float(round(association_confidence, 4))


def associate_traffic_with_road(
    bbox: BoundingBox,
    road_mask: Any,
    img_width: int,
    img_height: int,
    bottom_center_weight: float = 0.70,
    overlap_weight: float = 0.30,
    min_association: float = 0.25,
) -> Tuple[bool, float]:
    """
    Determine whether a traffic object (vehicle/pedestrian) is on the road surface.
    Uses bottom-center grounding point (tyres/feet) as the primary indicator.
    
    Returns:
      (is_associated, association_confidence)
    """
    if road_mask is None:
        return True, 1.0

    pix_box = bbox.to_pixel(img_width, img_height) if bbox.is_normalized else bbox
    bc_x, bc_y = pix_box.bottom_center

    # For bottom-center, nudge slightly upwards (e.g. 2% of box height) to avoid sub-pixel border jitter
    eval_y = max(pix_box.ymin, bc_y - 0.02 * pix_box.height)
    bc_on_road = 1.0 if is_point_in_road((bc_x, eval_y), road_mask, img_width, img_height) else 0.0

    overlap_ratio = calculate_bbox_road_overlap(pix_box, road_mask, img_width, img_height)

    w_sum = max(1e-6, bottom_center_weight + overlap_weight)
    bc_w = bottom_center_weight / w_sum
    ow = overlap_weight / w_sum

    association_confidence = (bc_w * bc_on_road) + (ow * overlap_ratio)
    is_associated = bool(association_confidence >= min_association)

    return is_associated, float(round(association_confidence, 4))
