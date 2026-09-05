"""
Traffic Analyzer.

Consumes standardized List[Detection] and generates TrafficState.
Calculates vehicle counts, pedestrian counts, per-class counts, spatial density,
road occupancy, and estimated congestion levels.

Decoupled from ML frameworks and simulator APIs.
"""

from typing import Any, Dict, List, Optional, Set
from perception.schemas import Detection, RoadMask, TrafficState, CongestionLevel
from perception.association import associate_traffic_with_road

# Standard groupings based on IDD classes
DEFAULT_VEHICLE_CLASSES: Set[str] = {
    "autorickshaw",
    "bicycle",
    "bus",
    "car",
    "caravan",
    "motorcycle",
    "trailer",
    "train",
    "truck",
    "vehicle fallback",
}

DEFAULT_PEDESTRIAN_CLASSES: Set[str] = {
    "person",
    "rider",
    "animal",
}


class TrafficAnalyzer:
    """
    Pure analytical engine for road traffic perception.
    Calculates traffic density and congestion levels deterministically from detections.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vehicle_classes: Optional[Set[str]] = None,
        pedestrian_classes: Optional[Set[str]] = None,
    ):
        self.config = config or {}
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES
        self.pedestrian_classes = pedestrian_classes or DEFAULT_PEDESTRIAN_CLASSES

        # Extract thresholds from config with verified defaults
        traffic_cfg = self.config.get("traffic_analysis", {})
        c_thresh = traffic_cfg.get("congestion_thresholds", {})
        self.moderate_thresh = float(c_thresh.get("moderate_density", 0.25))
        self.heavy_thresh = float(c_thresh.get("heavy_density", 0.60))
        self.gridlock_thresh = float(c_thresh.get("gridlock_density", 0.85))
        self.scene_capacity = float(traffic_cfg.get("scene_capacity_vehicles", 12.0))

        # Road association parameters
        assoc_cfg = self.config.get("road_association", {})
        self.min_association = float(assoc_cfg.get("traffic_min_association", 0.25))

    def _determine_congestion(self, density: float) -> CongestionLevel:
        """Map normalized traffic density to a discrete CongestionLevel."""
        if density < self.moderate_thresh:
            return CongestionLevel.LOW
        elif density < self.heavy_thresh:
            return CongestionLevel.MODERATE
        elif density < self.gridlock_thresh:
            return CongestionLevel.HEAVY
        else:
            return CongestionLevel.GRIDLOCK

    def analyze(
        self,
        detections: List[Detection],
        img_width: int,
        img_height: int,
        road_mask: Optional[RoadMask] = None,
    ) -> TrafficState:
        """
        Derive TrafficState from detections and optional road mask.
        
        Args:
          detections: List of standardized Detection objects (may include persistent track_ids)
          img_width: Frame pixel width
          img_height: Frame pixel height
          road_mask: Optional RoadMask for spatial road-grounding association
          
        Returns:
          TrafficState with vehicle/pedestrian breakdowns, density, and congestion.
        """
        if not detections:
            return TrafficState(
                vehicles=[],
                pedestrians=[],
                other_objects=[],
                vehicle_count=0,
                vehicle_counts_by_class={},
                pedestrian_count=0,
                total_objects=0,
                traffic_density=0.0,
                congestion_level=CongestionLevel.LOW,
            )

        vehicles: List[Detection] = []
        pedestrians: List[Detection] = []
        other_objects: List[Detection] = []
        vehicle_counts_by_class: Dict[str, int] = {}

        total_frame_area = max(1.0, float(img_width * img_height))
        vehicle_box_area_sum = 0.0
        road_associated_vehicles_count = 0

        for det in detections:
            # Associate with road mask if not already associated and road_mask is available
            if road_mask is not None and road_mask.mask is not None and det.road_association == 0.0:
                is_assoc, conf = associate_traffic_with_road(
                    det.bbox,
                    road_mask.mask,
                    img_width,
                    img_height,
                    min_association=self.min_association,
                )
                det.road_association = conf

            cname = det.class_name.lower()
            if cname in self.vehicle_classes:
                vehicles.append(det)
                vehicle_counts_by_class[cname] = vehicle_counts_by_class.get(cname, 0) + 1
                vehicle_box_area_sum += det.bbox.area
                if det.road_association >= self.min_association:
                    road_associated_vehicles_count += 1
            elif cname in self.pedestrian_classes:
                pedestrians.append(det)
            else:
                other_objects.append(det)

        # Traffic Density Computation
        # Note: Traffic density is a normalized camera-view spatial occupancy metric in range [0.0, 1.0].
        # It does NOT represent highway vehicle flow rate or speed.
        v_count = len(vehicles)
        if road_mask is not None and road_mask.road_area_ratio > 0.05:
            # Count density relative to drivable corridor capacity
            effective_capacity = max(1.0, self.scene_capacity * road_mask.road_area_ratio)
            count_density = road_associated_vehicles_count / effective_capacity
            # Area occupancy of vehicles on the road
            road_area_pixels = road_mask.road_area_ratio * total_frame_area
            area_occupancy = vehicle_box_area_sum / max(1.0, road_area_pixels)
            # Blended density: 60% count factor + 40% area occupancy factor
            raw_density = 0.60 * count_density + 0.40 * area_occupancy
        else:
            raw_density = v_count / max(1.0, self.scene_capacity)

        traffic_density = float(max(0.0, min(1.0, round(raw_density, 4))))
        congestion_level = self._determine_congestion(traffic_density)

        return TrafficState(
            vehicles=vehicles,
            pedestrians=pedestrians,
            other_objects=other_objects,
            vehicle_count=v_count,
            vehicle_counts_by_class=vehicle_counts_by_class,
            pedestrian_count=len(pedestrians),
            total_objects=len(detections),
            traffic_density=traffic_density,
            congestion_level=congestion_level,
        )
