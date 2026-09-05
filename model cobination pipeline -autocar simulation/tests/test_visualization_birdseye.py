"""
Unit tests for BirdseyeRenderer.
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
    TrafficState,
    PotholesState,
    PerceptionStatus,
    WorldStateMetadata,
    WorldState,
)
from visualization.birdseye import BirdseyeRenderer


class TestVisualizationBirdseye(unittest.TestCase):
    def setUp(self):
        self.renderer = BirdseyeRenderer(output_width=400, output_height=600)

    def _create_mock_state(self, has_entities=True):
        traffic = TrafficState(
            vehicles=[Detection(1, "car", 4, 0.9, BoundingBox(400, 400, 600, 600), track_id=12)],
            pedestrians=[Detection(2, "person", 7, 0.8, BoundingBox(200, 300, 250, 400), track_id=15)],
            total_objects=2,
            vehicle_count=1,
            pedestrian_count=1,
        ) if has_entities else TrafficState()

        potholes = PotholesState(
            potholes=[PotholeDetection(1, 0.9, BoundingBox(450, 500, 550, 580), 8000.0, 0.01, PotholeSeverityCategory.SMALL)],
            count=1,
        ) if has_entities else PotholesState()

        return WorldState(
            frame_id=10,
            timestamp=25.0,
            image_width=1280,
            image_height=720,
            road=RoadMask(mask=None, road_area_ratio=0.45),
            road_condition=RoadCondition(score=88.0),
            traffic=traffic,
            potholes=potholes,
            perception_status=PerceptionStatus(),
            metadata=WorldStateMetadata(),
        )

    def test_birdseye_output_dimensions(self):
        ws = self._create_mock_state(has_entities=True)
        img = self.renderer.render(ws)
        self.assertEqual(img.shape, (600, 400, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_birdseye_with_empty_state(self):
        ws = self._create_mock_state(has_entities=False)
        img = self.renderer.render(ws)
        self.assertEqual(img.shape, (600, 400, 3))

    def test_birdseye_does_not_mutate_world_state(self):
        ws = self._create_mock_state()
        before_hash = hash((ws.frame_id, ws.road_condition.score, len(ws.traffic.vehicles)))
        _ = self.renderer.render(ws)
        after_hash = hash((ws.frame_id, ws.road_condition.score, len(ws.traffic.vehicles)))
        self.assertEqual(before_hash, after_hash)


if __name__ == "__main__":
    unittest.main()
