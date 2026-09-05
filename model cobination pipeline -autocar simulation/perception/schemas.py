"""
Standardized Data Contracts for Road-Scene Perception and Traffic Simulation.

This module defines the clean intermediate representation (WorldState) that decouples
computer-vision perception models (U-Net, YOLO, Res2Net/YOLO-Pothole) from
downstream traffic simulation (MockSimulator, CARLA, SUMO).
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class CongestionLevel(str, Enum):
    """Estimated traffic congestion level."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HEAVY = "HEAVY"
    GRIDLOCK = "GRIDLOCK"


class RoadConditionCategory(str, Enum):
    """Categorical road quality level."""
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


class PotholeSeverityCategory(str, Enum):
    """
    Estimated pothole severity category based on bounding-box size.
    Note: Camera bounding boxes alone do NOT provide physical depth.
    """
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class ModelStatus(str, Enum):
    """Health status of an individual perception model."""
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "error"
    SKIPPED = "skipped"
    DISABLED = "disabled"


@dataclass
class BoundingBox:
    """
    Bounding box in pixel or normalized coordinates.
    Pixel coordinate system: (0,0) is top-left, x increases right, y increases down.
    Normalized coordinate system: x in [0, 1], y in [0, 1].
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    is_normalized: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Bottom-center point (useful for road surface contact / vehicle grounding)."""
        return ((self.xmin + self.xmax) / 2.0, self.ymax)

    def to_normalized(self, img_width: int, img_height: int) -> BoundingBox:
        """Convert pixel coordinates to normalized [0, 1] coordinates."""
        if self.is_normalized:
            return BoundingBox(self.xmin, self.ymin, self.xmax, self.ymax, is_normalized=True)
        w = max(1.0, float(img_width))
        h = max(1.0, float(img_height))
        return BoundingBox(
            xmin=max(0.0, min(1.0, self.xmin / w)),
            ymin=max(0.0, min(1.0, self.ymin / h)),
            xmax=max(0.0, min(1.0, self.xmax / w)),
            ymax=max(0.0, min(1.0, self.ymax / h)),
            is_normalized=True,
        )

    def to_pixel(self, img_width: int, img_height: int) -> BoundingBox:
        """Convert normalized coordinates to pixel coordinates."""
        if not self.is_normalized:
            return BoundingBox(self.xmin, self.ymin, self.xmax, self.ymax, is_normalized=False)
        w = float(img_width)
        h = float(img_height)
        return BoundingBox(
            xmin=self.xmin * w,
            ymin=self.ymin * h,
            xmax=self.xmax * w,
            ymax=self.ymax * h,
            is_normalized=False,
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "xmin": round(self.xmin, 3),
            "ymin": round(self.ymin, 3),
            "xmax": round(self.xmax, 3),
            "ymax": round(self.ymax, 3),
            "is_normalized": self.is_normalized,
            "center": (round(self.center[0], 3), round(self.center[1], 3)),
            "bottom_center": (round(self.bottom_center[0], 3), round(self.bottom_center[1], 3)),
            "area": round(self.area, 3),
        }


@dataclass
class FrameData:
    """Raw ingestion frame metadata."""
    frame_id: int
    timestamp: float
    image_width: int
    image_height: int
    source: str = "camera"


@dataclass
class Detection:
    """
    Standardized perception representation for a detected object (vehicle, pedestrian, sign, etc.).
    """
    object_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None
    estimated_distance: Optional[float] = None  # Estimated simulation parameter (meters)
    road_association: float = 0.0  # Confidence (0.0 to 1.0) that object contacts the road
    lane_id: Optional[int] = None

    @property
    def center_x(self) -> float:
        return self.bbox.center[0]

    @property
    def center_y(self) -> float:
        return self.bbox.center[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_dict(),
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
            "track_id": self.track_id,
            "estimated_distance": round(self.estimated_distance, 2) if self.estimated_distance is not None else None,
            "road_association": round(self.road_association, 3),
            "lane_id": self.lane_id,
        }


@dataclass
class RoadMask:
    """
    Standardized road / drivable-area mask representation.
    mask: 2D numpy-like array or list of lists (H, W) where 1 indicates drivable road.
    """
    mask: Any  # ndarray or nested list
    confidence: float = 1.0
    road_area_ratio: float = 0.0  # Percentage of image that is drivable road [0.0 - 1.0]
    drivable_regions: List[Any] = field(default_factory=list)

    def to_dict(self, include_raw_mask: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "confidence": round(self.confidence, 4),
            "road_area_ratio": round(self.road_area_ratio, 4),
            "num_regions": len(self.drivable_regions),
        }
        if include_raw_mask and self.mask is not None:
            if hasattr(self.mask, "shape"):
                data["mask_shape"] = list(self.mask.shape)
            else:
                data["mask_shape"] = [len(self.mask), len(self.mask[0])] if len(self.mask) > 0 else [0, 0]
        return data


@dataclass
class PotholeDetection:
    """
    Standardized pothole detection representation.
    Identical interface whether generated by Res2Net or YOLO-Pothole.
    """
    pothole_id: int
    confidence: float
    bbox: BoundingBox
    area_pixels: float = 0.0
    relative_area: float = 0.0  # Area normalized by total image area
    estimated_severity: PotholeSeverityCategory = PotholeSeverityCategory.SMALL
    road_association: float = 0.0  # Confidence that pothole is inside road surface
    estimated_distance: Optional[float] = None  # Estimated simulation parameter (meters)

    @property
    def center_x(self) -> float:
        return self.bbox.center[0]

    @property
    def center_y(self) -> float:
        return self.bbox.center[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pothole_id": self.pothole_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_dict(),
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
            "area_pixels": round(self.area_pixels, 1),
            "relative_area": round(self.relative_area, 6),
            "estimated_severity": self.estimated_severity.value,
            "road_association": round(self.road_association, 3),
            "estimated_distance": round(self.estimated_distance, 2) if self.estimated_distance is not None else None,
        }


@dataclass
class TrafficState:
    """Aggregated traffic analytical state for simulation consumption."""
    vehicles: List[Detection] = field(default_factory=list)
    pedestrians: List[Detection] = field(default_factory=list)
    other_objects: List[Detection] = field(default_factory=list)
    vehicle_count: int = 0
    vehicle_counts_by_class: Dict[str, int] = field(default_factory=dict)
    pedestrian_count: int = 0
    total_objects: int = 0
    traffic_density: float = 0.0  # Estimated density metric [0.0 - 1.0]
    congestion_level: CongestionLevel = CongestionLevel.LOW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_count": self.vehicle_count,
            "vehicle_counts_by_class": self.vehicle_counts_by_class,
            "pedestrian_count": self.pedestrian_count,
            "total_objects": self.total_objects,
            "traffic_density": round(self.traffic_density, 4),
            "congestion_level": self.congestion_level.value,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "pedestrians": [p.to_dict() for p in self.pedestrians],
            "other_objects": [o.to_dict() for o in self.other_objects],
        }


@dataclass
class PotholesState:
    """Aggregated pothole analytical state for simulation consumption."""
    potholes: List[PotholeDetection] = field(default_factory=list)
    count: int = 0
    total_area_pixels: float = 0.0
    total_relative_area: float = 0.0
    severity_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_area_pixels": round(self.total_area_pixels, 1),
            "total_relative_area": round(self.total_relative_area, 6),
            "severity_counts": self.severity_counts,
            "potholes": [p.to_dict() for p in self.potholes],
        }


@dataclass
class RoadCondition:
    """
    Configurable road-condition assessment score and simulation impact parameters.
    """
    score: float = 100.0  # 0 to 100 (100 = flawless road, 0 = impassable)
    category: RoadConditionCategory = RoadConditionCategory.GOOD
    estimated_friction_factor: float = 1.0  # Estimated simulation coefficient (1.0 = normal asphalt)
    speed_restriction_factor: float = 1.0  # Recommended speed multiplier (1.0 = normal speed)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "category": self.category.value,
            "estimated_friction_factor": round(self.estimated_friction_factor, 3),
            "speed_restriction_factor": round(self.speed_restriction_factor, 3),
            "details": self.details,
        }


@dataclass
class PerceptionStatus:
    """
    Health / availability status of each perception model.
    Guarantees that a single model failure (e.g. pothole model exception) does not
    bring down the entire simulation pipeline.
    """
    road: str = ModelStatus.OK.value
    traffic: str = ModelStatus.OK.value
    pothole: str = ModelStatus.OK.value
    errors: Dict[str, str] = field(default_factory=dict)

    def is_all_ok(self) -> bool:
        return (
            self.road == ModelStatus.OK.value
            and self.traffic == ModelStatus.OK.value
            and self.pothole == ModelStatus.OK.value
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road": self.road,
            "traffic": self.traffic,
            "pothole": self.pothole,
            "errors": self.errors,
        }


@dataclass
class WorldStateMetadata:
    """Execution latency, framerate, and model version metadata."""
    processing_time_ms: float = 0.0
    road_inference_ms: float = 0.0
    traffic_inference_ms: float = 0.0
    pothole_inference_ms: float = 0.0
    fusion_ms: float = 0.0
    fps: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processing_time_ms": round(self.processing_time_ms, 2),
            "road_inference_ms": round(self.road_inference_ms, 2),
            "traffic_inference_ms": round(self.traffic_inference_ms, 2),
            "pothole_inference_ms": round(self.pothole_inference_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "fps": round(self.fps, 1),
            "model_versions": self.model_versions,
        }


@dataclass
class WorldState:
    """
    The Central Intermediate Representation (Data Contract).
    
    The simulator consumes ONLY this object.
    Models produce standardized outputs which PerceptionFusion aggregates into WorldState.
    No model-specific tensors, class IDs, or frameworks are exposed to the simulator.
    """
    frame_id: int
    timestamp: float
    image_width: int
    image_height: int
    road: RoadMask
    road_condition: RoadCondition
    traffic: TrafficState
    potholes: PotholesState
    perception_status: PerceptionStatus = field(default_factory=PerceptionStatus)
    metadata: WorldStateMetadata = field(default_factory=WorldStateMetadata)

    def to_dict(self, include_raw_mask: bool = False) -> Dict[str, Any]:
        """Serialize WorldState to a pure JSON-serializable Python dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "environment": {
                "image_width": self.image_width,
                "image_height": self.image_height,
            },
            "road": {
                **self.road.to_dict(include_raw_mask=include_raw_mask),
                "condition": self.road_condition.to_dict(),
            },
            "traffic": self.traffic.to_dict(),
            "potholes": self.potholes.to_dict(),
            "perception_status": self.perception_status.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Export WorldState to formatted JSON string."""
        return json.dumps(self.to_dict(include_raw_mask=False), indent=indent)
