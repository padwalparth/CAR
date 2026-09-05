"""
Unit tests for FrameProcessor.
"""

import unittest
import numpy as np
from models.road_segmentation.unet_model import MockRoadModel
from models.traffic_detection.yolo_idd_model import MockTrafficModel
from models.pothole_detection.res2net_model import MockPotholeModel
from processing.frame_processor import FrameProcessor
from config.loader import load_pipeline_config


class FaultyModel(MockRoadModel):
    """Model that deliberately fails on inference to test failure isolation."""
    def infer(self, frame):
        raise RuntimeError("Simulated model inference crash")


class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.config = load_pipeline_config("config/pipeline.yaml")
        self.processor = FrameProcessor(
            road_model=MockRoadModel(),
            traffic_model=MockTrafficModel(),
            pothole_model=MockPotholeModel(),
            config=self.config,
        )

    def test_uninitialized_processor_raises_error(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            self.processor.process_frame(frame)

    def test_end_to_end_single_frame(self):
        self.processor.initialize()
        self.assertTrue(self.processor.is_initialized)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ws = self.processor.process_frame(frame, frame_id=101)

        self.assertEqual(ws.frame_id, 101)
        self.assertTrue(ws.perception_status.is_all_ok())
        self.assertGreater(ws.traffic.vehicle_count, 0)
        self.assertGreater(ws.potholes.count, 0)
        self.assertIsNotNone(ws.road.mask)
        self.assertGreater(ws.metadata.processing_time_ms, 0.0)

        self.processor.close()
        self.assertFalse(self.processor.is_initialized)

    def test_failure_isolation_does_not_crash_pipeline(self):
        """When the road model crashes, traffic and pothole processing must still succeed."""
        faulty_proc = FrameProcessor(
            road_model=FaultyModel(),
            traffic_model=MockTrafficModel(),
            pothole_model=MockPotholeModel(),
            config=self.config,
        )
        faulty_proc.initialize()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ws = faulty_proc.process_frame(frame)

        # Pipeline did not crash and returned WorldState!
        self.assertEqual(ws.perception_status.road, "error")
        self.assertEqual(ws.perception_status.traffic, "ok")
        self.assertEqual(ws.perception_status.pothole, "ok")
        self.assertIn("Simulated model inference crash", ws.perception_status.errors["road"])
        self.assertGreater(ws.traffic.vehicle_count, 0)

        faulty_proc.close()


if __name__ == "__main__":
    unittest.main()
