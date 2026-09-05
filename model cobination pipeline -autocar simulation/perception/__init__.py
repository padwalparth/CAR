"""
Perception package for road scene perception and traffic simulation pipeline.
Provides standardized schemas, coordinate utilities, tracking, and spatial association.
"""

from perception.schemas import (
    BoundingBox,
    Detection,
    RoadMask,
    PotholeDetection,
    TrafficState,
    PotholesState,
    RoadCondition,
    RoadConditionCategory,
    PotholeSeverityCategory,
    CongestionLevel,
    ModelStatus,
    PerceptionStatus,
    WorldStateMetadata,
    WorldState,
    FrameData,
)
from perception.coordinate_utils import (
    pixel_to_normalized,
    normalized_to_pixel,
    normalize_bbox,
    denormalize_bbox,
    bbox_center,
    bbox_bottom_center,
    bbox_area,
    bbox_intersection,
    bbox_iou,
    clip_bbox,
)
from perception.association import (
    is_point_in_road,
    calculate_bbox_road_overlap,
    associate_pothole_with_road,
    associate_traffic_with_road,
)
from perception.tracker import SimpleTracker
from perception.fusion import PerceptionFusion
from perception.world_state import WorldStateHistory, validate_world_state

__all__ = [
    "BoundingBox",
    "Detection",
    "RoadMask",
    "PotholeDetection",
    "TrafficState",
    "PotholesState",
    "RoadCondition",
    "RoadConditionCategory",
    "PotholeSeverityCategory",
    "CongestionLevel",
    "ModelStatus",
    "PerceptionStatus",
    "WorldStateMetadata",
    "WorldState",
    "FrameData",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "normalize_bbox",
    "denormalize_bbox",
    "bbox_center",
    "bbox_bottom_center",
    "bbox_area",
    "bbox_intersection",
    "bbox_iou",
    "clip_bbox",
    "is_point_in_road",
    "calculate_bbox_road_overlap",
    "associate_pothole_with_road",
    "associate_traffic_with_road",
    "SimpleTracker",
    "PerceptionFusion",
    "WorldStateHistory",
    "validate_world_state",
]
