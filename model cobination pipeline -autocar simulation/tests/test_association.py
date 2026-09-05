"""
Unit tests for spatial association between detections and road mask.
"""

import unittest
from perception.schemas import BoundingBox
from perception.association import (
    is_point_in_road,
    calculate_bbox_road_overlap,
    associate_pothole_with_road,
    associate_traffic_with_road,
)


class TestAssociation(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 10x10 road mask
        # Top 4 rows (0..3) = Non-road (0)
        # Bottom 6 rows (4..9) = Road (1)
        self.road_mask = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        self.img_w = 100
        self.img_h = 100

    def test_point_in_road(self):
        # Point in sky / top (y=20) -> should be False
        self.assertFalse(is_point_in_road((50, 20), self.road_mask, self.img_w, self.img_h))
        # Point on road (y=70) -> should be True
        self.assertTrue(is_point_in_road((50, 70), self.road_mask, self.img_w, self.img_h))

    def test_pothole_inside_road(self):
        # Pothole firmly on the road surface (y from 60 to 80)
        pothole_box = BoundingBox(xmin=30, ymin=60, xmax=70, ymax=80)
        is_assoc, conf = associate_pothole_with_road(
            pothole_box, self.road_mask, self.img_w, self.img_h, min_overlap=0.3
        )
        self.assertTrue(is_assoc)
        self.assertGreaterEqual(conf, 0.9)

    def test_pothole_outside_road(self):
        # Pothole off-road in the upper non-road area (y from 10 to 30)
        offroad_box = BoundingBox(xmin=30, ymin=10, xmax=70, ymax=30)
        is_assoc, conf = associate_pothole_with_road(
            offroad_box, self.road_mask, self.img_w, self.img_h, min_overlap=0.3
        )
        self.assertFalse(is_assoc)
        self.assertAlmostEqual(conf, 0.0)

    def test_traffic_bottom_center_association(self):
        # Car spanning from non-road horizon (y=30) down to road contact (y=65)
        # Bottom-center is at y=65 (which is inside road: row 6)
        car_box = BoundingBox(xmin=30, ymin=30, xmax=70, ymax=65)
        is_assoc, conf = associate_traffic_with_road(
            car_box, self.road_mask, self.img_w, self.img_h, min_association=0.25
        )
        self.assertTrue(is_assoc)
        self.assertGreaterEqual(conf, 0.5)

        # Overhead sign in upper sky (y=5 to 25)
        sign_box = BoundingBox(xmin=30, ymin=5, xmax=70, ymax=25)
        is_assoc_sign, conf_sign = associate_traffic_with_road(
            sign_box, self.road_mask, self.img_w, self.img_h, min_association=0.25
        )
        self.assertFalse(is_assoc_sign)
        self.assertEqual(conf_sign, 0.0)


if __name__ == "__main__":
    unittest.main()
