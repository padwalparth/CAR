"""
Coordinate System Utilities for Bounding Boxes and Normalization.

Rules:
- Pixel coordinates: (0, 0) is top-left, x increases right, y increases downward.
- Normalized coordinates: x in [0.0, 1.0], y in [0.0, 1.0].
- Never silently mix pixel and normalized coordinates.
"""

from typing import Optional, Tuple
from perception.schemas import BoundingBox


def pixel_to_normalized(
    x: float, y: float, img_width: int, img_height: int
) -> Tuple[float, float]:
    """Convert a single (x, y) point from pixel to normalized [0, 1] coordinates."""
    w = max(1.0, float(img_width))
    h = max(1.0, float(img_height))
    return (max(0.0, min(1.0, x / w)), max(0.0, min(1.0, y / h)))


def normalized_to_pixel(
    nx: float, ny: float, img_width: int, img_height: int
) -> Tuple[float, float]:
    """Convert a normalized [0, 1] point to pixel coordinates."""
    w = float(img_width)
    h = float(img_height)
    return (nx * w, ny * h)


def normalize_bbox(
    bbox: BoundingBox, img_width: int, img_height: int
) -> BoundingBox:
    """Return a normalized BoundingBox copy."""
    return bbox.to_normalized(img_width, img_height)


def denormalize_bbox(
    bbox: BoundingBox, img_width: int, img_height: int
) -> BoundingBox:
    """Return a pixel-coordinate BoundingBox copy."""
    return bbox.to_pixel(img_width, img_height)


def bbox_center(bbox: BoundingBox) -> Tuple[float, float]:
    """Return the (x, y) midpoint of the bounding box."""
    return bbox.center


def bbox_bottom_center(bbox: BoundingBox) -> Tuple[float, float]:
    """
    Return the (x, y) bottom-center of the bounding box.
    Crucial for determining whether a vehicle or object contacts the road surface.
    """
    return bbox.bottom_center


def bbox_area(bbox: BoundingBox) -> float:
    """Compute the surface area of the bounding box."""
    return bbox.area


def bbox_intersection(
    b1: BoundingBox, b2: BoundingBox
) -> Optional[BoundingBox]:
    """
    Compute the intersection bounding box between b1 and b2.
    Both boxes must share the same coordinate representation (normalized or pixel).
    Returns None if there is no spatial overlap.
    """
    if b1.is_normalized != b2.is_normalized:
        raise ValueError(
            f"Coordinate system mismatch: b1.is_normalized={b1.is_normalized}, "
            f"b2.is_normalized={b2.is_normalized}. Normalize both before computing intersection."
        )

    ixmin = max(b1.xmin, b2.xmin)
    iymin = max(b1.ymin, b2.ymin)
    ixmax = min(b1.xmax, b2.xmax)
    iymax = min(b1.ymax, b2.ymax)

    if ixmax <= ixmin or iymax <= iymin:
        return None

    return BoundingBox(
        xmin=ixmin,
        ymin=iymin,
        xmax=ixmax,
        ymax=iymax,
        is_normalized=b1.is_normalized,
    )


def bbox_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Returns float in range [0.0, 1.0].
    """
    if b1.is_normalized != b2.is_normalized:
        raise ValueError(
            f"Coordinate system mismatch in IoU calculation: "
            f"b1 is_normalized={b1.is_normalized}, b2 is_normalized={b2.is_normalized}"
        )

    inter = bbox_intersection(b1, b2)
    if inter is None:
        return 0.0

    inter_area = inter.area
    union_area = b1.area + b2.area - inter_area
    if union_area <= 1e-9:
        return 0.0

    return float(max(0.0, min(1.0, inter_area / union_area)))


def clip_bbox(
    bbox: BoundingBox,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> BoundingBox:
    """Clip bounding box to specific coordinate bounds."""
    c_xmin = max(min_x, min(max_x, bbox.xmin))
    c_ymin = max(min_y, min(max_y, bbox.ymin))
    c_xmax = max(min_x, min(max_x, bbox.xmax))
    c_ymax = max(min_y, min(max_y, bbox.ymax))

    # Ensure min <= max
    if c_xmax < c_xmin:
        c_xmax = c_xmin
    if c_ymax < c_ymin:
        c_ymax = c_ymin

    return BoundingBox(
        xmin=c_xmin,
        ymin=c_ymin,
        xmax=c_xmax,
        ymax=c_ymax,
        is_normalized=bbox.is_normalized,
    )
