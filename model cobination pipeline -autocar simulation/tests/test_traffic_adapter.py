"""
Unit tests for TrafficAdapter.
"""

import unittest
from adapters.traffic_adapter import TrafficAdapter
from perception.schemas import Detection, BoundingBox
from models.traffic_detection.yolo_idd_model import IDD_CLASSES


class TestTrafficAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = TrafficAdapter(min_confidence=0.40)

    def test_convert_raw_yolo_detections(self):
        raw_output = {
            "detections": [
                {
                    "class_id": 4,
                    "class_name": "car",
                    "confidence": 0.88,
                    "bbox": (100.0, 200.0, 300.0, 400.0),
                },
                {
                    "class_id": 1,
                    "class_name": "autorickshaw",
                    "confidence": 0.75,
                    "bbox": (50.0, 150.0, 150.0, 250.0),
                },
                {
                    "class_id": 7,
                    "class_name": "person",
                    "confidence": 0.20,  # Below min_confidence (0.40), must be filtered out
                    "bbox": (10.0, 20.0, 30.0, 60.0),
                },
            ]
        }

        detections = self.adapter.convert(raw_output)
        self.assertEqual(len(detections), 2)

        car = detections[0]
        self.assertIsInstance(car, Detection)
        self.assertEqual(car.object_id, 1)
        self.assertEqual(car.class_name, "car")
        self.assertEqual(car.class_id, 4)
        self.assertAlmostEqual(car.confidence, 0.88)
        self.assertEqual(car.bbox.as_tuple(), (100.0, 200.0, 300.0, 400.0))
        self.assertEqual(car.center_x, 200.0)
        self.assertEqual(car.center_y, 300.0)
        self.assertIsNone(car.track_id)
        self.assertIsNone(car.estimated_distance)

        auto = detections[1]
        self.assertEqual(auto.class_name, "autorickshaw")

    def test_idd_class_mapping_integrity(self):
        """Ensure all 15 IDD classes from idd_fixed.yaml are properly indexed."""
        self.assertEqual(len(IDD_CLASSES), 15)
        self.assertEqual(IDD_CLASSES[0], "animal")
        self.assertEqual(IDD_CLASSES[1], "autorickshaw")
        self.assertEqual(IDD_CLASSES[4], "car")
        self.assertEqual(IDD_CLASSES[7], "person")
        self.assertEqual(IDD_CLASSES[13], "truck")
        self.assertEqual(IDD_CLASSES[14], "vehicle fallback")

    def test_empty_or_none_output(self):
        self.assertEqual(self.adapter.convert(None), [])
        self.assertEqual(self.adapter.convert({}), [])
        self.assertEqual(self.adapter.convert([]), [])


if __name__ == "__main__":
    unittest.main()
