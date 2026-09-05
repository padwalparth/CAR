"""
Unit tests for PerceptionFusion.
"""

import unittest
from perception.schemas import (
    BoundingBox,
    Detection,
    FrameData,
    ModelStatus,
    PerceptionStatus,
    PotholeDetection,
    RoadMask,
)
from perception.fusion import PerceptionFusion
from config.loader import load_pipeline_config


class TestFusion(unittest.TestCase):
    def setUp(self):
        self.config = load_pipeline_config("config/pipeline.yaml")
        self.fusion = PerceptionFusion(self.config)
        self.frame_data = FrameData(
            frame_id=1,
            timestamp=100.0,
            image_width=1000,
            image_height=1000,
            source="test_source",
        )

    def test_fusion_with_empty_inputs(self):
        ws = self.fusion.fuse(self.frame_data)
        self.assertEqual(ws.frame_id, 1)
        self.assertEqual(ws.traffic.vehicle_count, 0)
        self.assertEqual(ws.potholes.count, 0)
        self.assertEqual(ws.road_condition.score, 100.0)
        self.assertTrue(ws.perception_status.is_all_ok())
        self.assertGreater(ws.metadata.processing_time_ms, 0.0)

    def test_fusion_with_full_perception(self):
        road_mask = RoadMask(mask=[[1]], confidence=0.95, road_area_ratio=0.50)
        traffic = [
            Detection(1, "car", 4, 0.9, BoundingBox(100, 100, 200, 200)),
            Detection(2, "person", 7, 0.85, BoundingBox(300, 300, 320, 350)),
        ]
        potholes = [
            PotholeDetection(1, 0.9, BoundingBox(400, 400, 500, 500), 10000.0, 0.01),
        ]
        latencies = {"road_ms": 10.0, "traffic_ms": 15.0, "pothole_ms": 5.0}

        ws = self.fusion.fuse(
            frame_data=self.frame_data,
            road_mask=road_mask,
            traffic_detections=traffic,
            pothole_detections=potholes,
            inference_latencies=latencies,
        )

        self.assertEqual(ws.traffic.vehicle_count, 1)
        self.assertEqual(ws.traffic.pedestrian_count, 1)
        self.assertEqual(ws.potholes.count, 1)
        self.assertAlmostEqual(ws.road_condition.score, 92.5)
        self.assertEqual(ws.metadata.road_inference_ms, 10.0)
        self.assertEqual(ws.metadata.traffic_inference_ms, 15.0)
        self.assertEqual(ws.metadata.pothole_inference_ms, 5.0)
        self.assertGreater(ws.metadata.fusion_ms, 0.0)

        # Persistent tracking should have assigned track IDs
        self.assertIsNotNone(ws.traffic.vehicles[0].track_id)

    def test_model_failure_status_preservation(self):
        status = PerceptionStatus(
            road="error",
            traffic="ok",
            pothole="ok",
            errors={"road": "Keras segmentation model out of memory"},
        )
        ws = self.fusion.fuse(self.frame_data, perception_status=status)
        self.assertEqual(ws.perception_status.road, "error")
        self.assertIn("segmentation", ws.perception_status.errors["road"])
        self.assertFalse(ws.perception_status.is_all_ok())


if __name__ == "__main__":
    unittest.main()
