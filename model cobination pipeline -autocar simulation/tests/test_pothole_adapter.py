"""
Unit tests for PotholeAdapter.
"""

import unittest
from adapters.pothole_adapter import PotholeAdapter
from perception.schemas import PotholeDetection, PotholeSeverityCategory


class TestPotholeAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = PotholeAdapter(
            min_confidence=0.40,
            small_max_relative_area=0.015,
            medium_max_relative_area=0.045,
        )

    def test_single_pothole_res2net_style(self):
        # Frame 1000 x 1000 (area 1,000,000)
        # Pothole 100x100 (area 10,000 => relative area 0.01 => SMALL)
        raw_output = {
            "pothole_boxes": [
                {
                    "bbox": (400.0, 600.0, 500.0, 700.0),
                    "confidence": 0.95,
                }
            ],
            "orig_shape": (1000, 1000),
        }

        potholes = self.adapter.convert(raw_output)
        self.assertEqual(len(potholes), 1)

        p = potholes[0]
        self.assertIsInstance(p, PotholeDetection)
        self.assertEqual(p.pothole_id, 1)
        self.assertAlmostEqual(p.confidence, 0.95)
        self.assertEqual(p.bbox.as_tuple(), (400.0, 600.0, 500.0, 700.0))
        self.assertAlmostEqual(p.area_pixels, 10000.0)
        self.assertAlmostEqual(p.relative_area, 0.01)
        self.assertEqual(p.estimated_severity, PotholeSeverityCategory.SMALL)
        self.assertIsNone(p.estimated_distance)

    def test_severity_threshold_categorization(self):
        # Frame 1000 x 1000
        # Medium pothole: area 30,000 => rel area 0.03
        raw_medium = {
            "pothole_boxes": [{"bbox": (0, 0, 300, 100), "confidence": 1.0}],
            "orig_shape": (1000, 1000),
        }
        res_med = self.adapter.convert(raw_medium)
        self.assertEqual(res_med[0].estimated_severity, PotholeSeverityCategory.MEDIUM)

        # Large pothole: area 60,000 => rel area 0.06
        raw_large = {
            "pothole_boxes": [{"bbox": (0, 0, 300, 200), "confidence": 1.0}],
            "orig_shape": (1000, 1000),
        }
        res_large = self.adapter.convert(raw_large)
        self.assertEqual(res_large[0].estimated_severity, PotholeSeverityCategory.LARGE)

    def test_multi_pothole_future_yolo_compatibility(self):
        """Verify that PotholeAdapter handles multiple pothole outputs seamlessly."""
        raw_multi = {
            "pothole_boxes": [
                {"bbox": (100, 100, 150, 150), "confidence": 0.9},
                {"bbox": (300, 400, 400, 500), "confidence": 0.85},
                {"bbox": (500, 600, 550, 650), "confidence": 0.20},  # Filtered by min_confidence
            ],
            "orig_shape": (1000, 1000),
        }
        res = self.adapter.convert(raw_multi)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].pothole_id, 1)
        self.assertEqual(res[1].pothole_id, 2)


if __name__ == "__main__":
    unittest.main()
