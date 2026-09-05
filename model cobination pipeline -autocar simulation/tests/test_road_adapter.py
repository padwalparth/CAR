"""
Unit tests for RoadAdapter.
"""

import unittest
import numpy as np
from adapters.road_adapter import RoadAdapter
from perception.schemas import RoadMask


class TestRoadAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = RoadAdapter(road_class_ids=[1, 2, 3])

    def test_convert_raw_mask_array(self):
        # 10x10 mask where 40 pixels are class 2 (road) and 60 pixels are class 0 (background)
        raw_mask = np.zeros((10, 10), dtype=np.uint8)
        raw_mask[6:10, :] = 2

        road_mask = self.adapter.convert(raw_mask)
        self.assertIsInstance(road_mask, RoadMask)
        self.assertAlmostEqual(road_mask.road_area_ratio, 0.40)
        self.assertEqual(road_mask.confidence, 1.0)
        self.assertEqual(road_mask.mask.shape, (10, 10))
        self.assertEqual(road_mask.mask[0, 0], 0)
        self.assertEqual(road_mask.mask[8, 5], 1)

    def test_convert_dict_output(self):
        raw_mask = np.zeros((20, 20), dtype=np.uint8)
        raw_mask[10:, :] = 1  # class 1 = lane marking, part of drivable road
        raw_output = {
            "raw_mask": raw_mask,
            "orig_shape": (720, 1280),
            "confidence": 0.95,
            "road_class_ids": [1, 2],
        }
        road_mask = self.adapter.convert(raw_output)
        self.assertAlmostEqual(road_mask.road_area_ratio, 0.50)
        self.assertEqual(road_mask.confidence, 0.95)

    def test_convert_none_or_empty(self):
        road_mask = self.adapter.convert(None)
        self.assertIsNone(road_mask.mask)
        self.assertEqual(road_mask.road_area_ratio, 0.0)
        self.assertEqual(road_mask.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
