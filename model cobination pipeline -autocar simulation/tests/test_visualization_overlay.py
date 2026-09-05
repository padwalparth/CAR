"""
Unit tests for OverlayRenderer.
"""

import unittest
import numpy as np
from perception.schemas import (
    BoundingBox,
    Detection,
    PotholeDetection,
    PotholeSeverityCategory,
    RoadMask,
    RoadCondition,
    RoadConditionCategory,
    TrafficState,
    PotholesState,
    PerceptionStatus,
    WorldStateMetadata,
    WorldState,
    CongestionLevel,
)
from visualization.overlay import OverlayRenderer


class TestVisualizationOverlay(unittest.TestCase):
    def setUp(self):
        self.renderer = OverlayRenderer()
        self.h, self.w = 720, 1280
        self.blank_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def _create_world_state(self, has_road=True, has_traffic=True, has_potholes=True, is_ok=True):
        road_mask = RoadMask(mask=np.ones((160, 160), dtype=np.uint8), confidence=0.9, road_area_ratio=0.5) if has_road else RoadMask(mask=None)
        traffic = TrafficState(
            vehicles=[Detection(1, "car", 4, 0.88, BoundingBox(100, 200, 300, 400), track_id=5, road_association=0.9)],
            pedestrians=[Detection(2, "person", 7, 0.80, BoundingBox(50, 200, 90, 300), track_id=6)],
            total_objects=2,
            vehicle_count=1,
            pedestrian_count=1,
            traffic_density=0.15,
            congestion_level=CongestionLevel.LOW,
        ) if has_traffic else TrafficState()
        potholes = PotholesState(
            potholes=[PotholeDetection(1, 0.92, BoundingBox(400, 500, 550, 600), 15000.0, 0.016, PotholeSeverityCategory.MEDIUM)],
            count=1,
        ) if has_potholes else PotholesState()
        status = PerceptionStatus() if is_ok else PerceptionStatus(road="error", errors={"road": "Keras segmentation failed"})

        return WorldState(
            frame_id=1,
            timestamp=12.5,
            image_width=self.w,
            image_height=self.h,
            road=road_mask,
            road_condition=RoadCondition(score=85.0, category=RoadConditionCategory.GOOD),
            traffic=traffic,
            potholes=potholes,
            perception_status=status,
            metadata=WorldStateMetadata(fps=28.5, processing_time_ms=35.1),
        )

    def test_overlay_output_dimensions_unchanged(self):
        ws = self._create_world_state()
        rendered = self.renderer.render(self.blank_frame, ws)
        self.assertEqual(rendered.shape, (self.h, self.w, 3))
        self.assertEqual(rendered.dtype, np.uint8)

    def test_overlay_with_blank_world_state(self):
        ws = self._create_world_state(has_road=False, has_traffic=False, has_potholes=False)
        rendered = self.renderer.render(self.blank_frame, ws)
        self.assertEqual(rendered.shape, (self.h, self.w, 3))

    def test_overlay_with_perception_error(self):
        ws = self._create_world_state(is_ok=False)
        rendered = self.renderer.render(self.blank_frame, ws)
        self.assertEqual(rendered.shape, (self.h, self.w, 3))

    def test_overlay_does_not_modify_world_state(self):
        ws = self._create_world_state()
        score_before = ws.road_condition.score
        v_count_before = ws.traffic.vehicle_count

        _ = self.renderer.render(self.blank_frame, ws)

        # Invariant: WorldState is read-only
        self.assertEqual(ws.road_condition.score, score_before)
        self.assertEqual(ws.traffic.vehicle_count, v_count_before)


if __name__ == "__main__":
    unittest.main()
