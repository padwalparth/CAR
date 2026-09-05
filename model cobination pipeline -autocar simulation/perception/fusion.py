"""
Perception Fusion Engine.

Fuses standardized perception outputs (RoadMask, List[Detection], List[PotholeDetection])
into a unified WorldState representation.

Responsibilities:
1. Validate inputs and coordinate representations.
2. Ground traffic objects on road surface using perception.association.
3. Determine road-mask containment for potholes using perception.association.
4. Execute SimpleTracker to maintain persistent track IDs.
5. Invoke analytical engines (TrafficAnalyzer, PotholeAnalyzer, RoadConditionAnalyzer).
6. Aggregate inference and fusion latencies into WorldStateMetadata.
7. Return an immutable WorldState snapshot.
"""

from typing import Any, Dict, List, Optional
import time

from perception.schemas import (
    Detection,
    FrameData,
    ModelStatus,
    PerceptionStatus,
    PotholeDetection,
    RoadMask,
    WorldState,
    WorldStateMetadata,
)
from perception.coordinate_utils import clip_bbox
from perception.association import (
    associate_traffic_with_road,
    associate_pothole_with_road,
)
from perception.tracker import SimpleTracker
from analysis.traffic_analysis import TrafficAnalyzer
from analysis.pothole_analysis import PotholeAnalyzer
from analysis.road_analysis import RoadConditionAnalyzer


class PerceptionFusion:
    """
    Central Perception Fusion Engine.
    Combines outputs from independent CV models into a unified WorldState.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tracker: Optional[SimpleTracker] = None,
        traffic_analyzer: Optional[TrafficAnalyzer] = None,
        pothole_analyzer: Optional[PotholeAnalyzer] = None,
        road_analyzer: Optional[RoadConditionAnalyzer] = None,
    ):
        self.config = config or {}

        # Tracking configuration
        tracking_cfg = self.config.get("tracking", {})
        self.tracking_enabled = bool(tracking_cfg.get("enabled", True))
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = SimpleTracker(
                min_iou_threshold=float(tracking_cfg.get("min_iou", 0.25)),
                max_disappeared=int(tracking_cfg.get("max_disappeared_frames", 5)),
                max_centroid_distance_norm=float(tracking_cfg.get("max_centroid_distance_norm", 0.15)),
            )

        # Analyzers
        self.traffic_analyzer = traffic_analyzer or TrafficAnalyzer(self.config)
        self.pothole_analyzer = pothole_analyzer or PotholeAnalyzer(self.config)
        self.road_analyzer = road_analyzer or RoadConditionAnalyzer(self.config)

        # Preprocessing / coordinate config
        pre_cfg = self.config.get("preprocessing", {})
        self.clip_to_frame = bool(pre_cfg.get("clip_to_frame", True))

        # Road association thresholds
        assoc_cfg = self.config.get("road_association", {})
        self.pothole_min_overlap = float(assoc_cfg.get("min_overlap", 0.30))
        self.traffic_min_assoc = float(assoc_cfg.get("traffic_min_association", 0.25))

    def reset_tracker(self) -> None:
        """Reset object tracker state between video sequences."""
        self.tracker.reset()

    def fuse(
        self,
        frame_data: FrameData,
        road_mask: Optional[RoadMask] = None,
        traffic_detections: Optional[List[Detection]] = None,
        pothole_detections: Optional[List[PotholeDetection]] = None,
        perception_status: Optional[PerceptionStatus] = None,
        inference_latencies: Optional[Dict[str, float]] = None,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> WorldState:
        """
        Execute perception fusion and return WorldState snapshot.
        """
        t_start = time.perf_counter()

        img_w = max(1, frame_data.image_width)
        img_h = max(1, frame_data.image_height)

        # Safe defaults
        traffic_list: List[Detection] = traffic_detections if traffic_detections is not None else []
        pothole_list: List[PotholeDetection] = pothole_detections if pothole_detections is not None else []
        road: RoadMask = road_mask if road_mask is not None else RoadMask(mask=None, confidence=0.0, road_area_ratio=0.0)
        status: PerceptionStatus = perception_status if perception_status is not None else PerceptionStatus()
        latencies: Dict[str, float] = inference_latencies or {}

        # 1. Coordinate boundary clipping if enabled
        if self.clip_to_frame:
            for d in traffic_list:
                if not d.bbox.is_normalized:
                    d.bbox = clip_bbox(d.bbox, 0.0, 0.0, float(img_w), float(img_h))
            for p in pothole_list:
                if not p.bbox.is_normalized:
                    p.bbox = clip_bbox(p.bbox, 0.0, 0.0, float(img_w), float(img_h))

        # 2. Road Association (Potholes & Traffic)
        if road.mask is not None:
            for p in pothole_list:
                if p.road_association == 0.0:
                    is_assoc, conf = associate_pothole_with_road(
                        p.bbox,
                        road.mask,
                        img_w,
                        img_h,
                        min_overlap=self.pothole_min_overlap,
                    )
                    p.road_association = conf

            for d in traffic_list:
                if d.road_association == 0.0:
                    is_assoc, conf = associate_traffic_with_road(
                        d.bbox,
                        road.mask,
                        img_w,
                        img_h,
                        min_association=self.traffic_min_assoc,
                    )
                    d.road_association = conf

        # 3. Object Tracking
        if self.tracking_enabled and traffic_list:
            try:
                traffic_list = self.tracker.update(traffic_list, img_w, img_h)
            except Exception as e:
                status.errors["tracker"] = f"Tracker exception: {str(e)}"

        # 4. Analytical Processing
        traffic_state = self.traffic_analyzer.analyze(traffic_list, img_w, img_h, road)
        potholes_state, _ = self.pothole_analyzer.analyze(pothole_list, img_w, img_h, road)
        road_condition = self.road_analyzer.analyze(road, pothole_list, traffic_state)

        # 5. Latency & Metadata aggregation
        fusion_ms = (time.perf_counter() - t_start) * 1000.0
        road_inf_ms = float(latencies.get("road_ms", 0.0))
        traffic_inf_ms = float(latencies.get("traffic_ms", 0.0))
        pothole_inf_ms = float(latencies.get("pothole_ms", 0.0))
        total_proc_ms = road_inf_ms + traffic_inf_ms + pothole_inf_ms + fusion_ms
        fps = (1000.0 / total_proc_ms) if total_proc_ms > 0.0 else 0.0

        metadata = WorldStateMetadata(
            processing_time_ms=total_proc_ms,
            road_inference_ms=road_inf_ms,
            traffic_inference_ms=traffic_inf_ms,
            pothole_inference_ms=pothole_inf_ms,
            fusion_ms=fusion_ms,
            fps=fps,
            model_versions=model_versions or {},
        )

        return WorldState(
            frame_id=frame_data.frame_id,
            timestamp=frame_data.timestamp,
            image_width=img_w,
            image_height=img_h,
            road=road,
            road_condition=road_condition,
            traffic=traffic_state,
            potholes=potholes_state,
            perception_status=status,
            metadata=metadata,
        )
