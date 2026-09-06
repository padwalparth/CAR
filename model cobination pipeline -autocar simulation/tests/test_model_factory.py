"""
Unit tests for model factory and backend selection.
"""

import unittest
from models.factory import create_road_model, create_traffic_model, create_pothole_model
from models.road_segmentation.unet_model import UNetRoadModel, MockRoadModel
from models.traffic_detection.yolo_idd_model import YOLOIDDTrafficModel, MockTrafficModel
from models.pothole_detection.res2net_model import Res2NetPotholeModel, MockPotholeModel
from models.pothole_detection.yolo_pothole_model import YOLOPotholeModel


class TestModelFactory(unittest.TestCase):
    def test_create_mock_models(self):
        road_m = create_road_model({"backend": "mock"})
        self.assertIsInstance(road_m, MockRoadModel)

        traffic_m = create_traffic_model({"backend": "mock"})
        self.assertIsInstance(traffic_m, MockTrafficModel)

        pothole_m = create_pothole_model({"backend": "mock"})
        self.assertIsInstance(pothole_m, MockPotholeModel)

    def test_create_real_models_instantiation(self):
        road_m = create_road_model({"backend": "onnx", "path": "some/path.onnx"})
        self.assertIsInstance(road_m, UNetRoadModel)

        traffic_m = create_traffic_model({"backend": "ultralytics", "path": "some/best.pt"})
        self.assertIsInstance(traffic_m, YOLOIDDTrafficModel)

        pothole_m = create_pothole_model({"backend": "res2net", "path": "some/pothole.pt"})
        self.assertIsInstance(pothole_m, Res2NetPotholeModel)

        yolo_pothole = create_pothole_model({"backend": "yolo", "path": "some/yolo_pothole.pt"})
        self.assertIsInstance(yolo_pothole, YOLOPotholeModel)

    def test_invalid_backend_raises_error(self):
        with self.assertRaises(ValueError):
            create_road_model({"backend": "tensorflow_invalid"})

        with self.assertRaises(ValueError):
            create_traffic_model({"backend": "faster_rcnn"})

        with self.assertRaises(ValueError):
            create_pothole_model({"backend": "mask_rcnn"})

    def test_missing_checkpoint_raises_file_not_found(self):
        """Verify that missing checkpoint raises FileNotFoundError and does NOT silently fall back to mock."""
        road_m = create_road_model({"backend": "onnx", "path": "non_existent_weights.onnx"})
        with self.assertRaises(FileNotFoundError):
            road_m.load()

        traffic_m = create_traffic_model({"backend": "ultralytics", "path": "non_existent_yolo.pt"})
        with self.assertRaises(FileNotFoundError):
            traffic_m.load()

        pothole_m = create_pothole_model({"backend": "res2net", "path": "non_existent_pothole.pt"})
        with self.assertRaises(FileNotFoundError):
            pothole_m.load()

    def test_yolo_pothole_model_creation(self):
        yolo_p = create_pothole_model({"backend": "yolo", "path": "some/path.pt"})
        self.assertIsInstance(yolo_p, YOLOPotholeModel)
        self.assertEqual(yolo_p.backend, "yolo")


if __name__ == "__main__":
    unittest.main()
