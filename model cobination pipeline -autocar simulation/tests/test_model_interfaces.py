"""
Unit tests for base perception model interface lifecycle.
"""

import unittest
import numpy as np
from models.road_segmentation.unet_model import MockRoadModel
from models.traffic_detection.yolo_idd_model import MockTrafficModel
from models.pothole_detection.res2net_model import MockPotholeModel


class TestModelInterfaces(unittest.TestCase):
    def test_unloaded_model_raises_runtime_error(self):
        road_model = MockRoadModel()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            road_model.infer(frame)

        traffic_model = MockTrafficModel()
        with self.assertRaises(RuntimeError):
            traffic_model.infer(frame)

        pothole_model = MockPotholeModel()
        with self.assertRaises(RuntimeError):
            pothole_model.infer(frame)

    def test_model_lifecycle(self):
        model = MockRoadModel()
        self.assertFalse(model.is_loaded)

        model.load()
        self.assertTrue(model.is_loaded)

        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        res = model.infer(frame)
        self.assertIn("raw_mask", res)
        self.assertGreater(model.last_inference_time_ms, 0.0)

        info = model.get_info()
        self.assertEqual(info["model_name"], "mock_road_segmentation")
        self.assertEqual(info["backend"], "mock")
        self.assertTrue(info["is_loaded"])

        model.close()
        self.assertFalse(model.is_loaded)


if __name__ == "__main__":
    unittest.main()
