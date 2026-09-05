"""
Unit tests for coordinate transformation and bounding box mathematics.
"""

import unittest
from perception.schemas import BoundingBox
from perception.coordinate_utils import (
    pixel_to_normalized,
    normalized_to_pixel,
    normalize_bbox,
    denormalize_bbox,
    bbox_center,
    bbox_bottom_center,
    bbox_area,
    bbox_intersection,
    bbox_iou,
    clip_bbox,
)


class TestCoordinateUtils(unittest.TestCase):
    def test_point_normalization(self):
        nx, ny = pixel_to_normalized(640, 360, 1280, 720)
        self.assertAlmostEqual(nx, 0.5)
        self.assertAlmostEqual(ny, 0.5)

        px, py = normalized_to_pixel(0.5, 0.5, 1280, 720)
        self.assertAlmostEqual(px, 640.0)
        self.assertAlmostEqual(py, 360.0)

    def test_bbox_area_and_center(self):
        box = BoundingBox(xmin=10, ymin=20, xmax=30, ymax=60)
        self.assertAlmostEqual(bbox_area(box), 800.0)
        self.assertEqual(bbox_center(box), (20.0, 40.0))
        self.assertEqual(bbox_bottom_center(box), (20.0, 60.0))

    def test_bbox_intersection(self):
        b1 = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        b2 = BoundingBox(xmin=5, ymin=5, xmax=15, ymax=15)
        inter = bbox_intersection(b1, b2)
        self.assertIsNotNone(inter)
        self.assertEqual(inter.as_tuple(), (5, 5, 10, 10))
        self.assertAlmostEqual(inter.area, 25.0)

        # Disjoint boxes
        b3 = BoundingBox(xmin=20, ymin=20, xmax=30, ymax=30)
        self.assertIsNone(bbox_intersection(b1, b3))

    def test_coordinate_system_mismatch_guard(self):
        """IoU and intersection must reject mixed pixel and normalized boxes."""
        b_pix = BoundingBox(xmin=0, ymin=0, xmax=100, ymax=100, is_normalized=False)
        b_norm = BoundingBox(xmin=0, ymin=0, xmax=0.5, ymax=0.5, is_normalized=True)
        with self.assertRaises(ValueError):
            bbox_intersection(b_pix, b_norm)
        with self.assertRaises(ValueError):
            bbox_iou(b_pix, b_norm)

    def test_bbox_iou(self):
        # Identical boxes => IoU = 1.0
        b1 = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        b2 = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        self.assertAlmostEqual(bbox_iou(b1, b2), 1.0)

        # 50% overlap scenario:
        # b1: 0..10 x 0..10 (area 100)
        # b3: 0..10 x 5..15 (area 100)
        # inter: 0..10 x 5..10 (area 50)
        # union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3
        b3 = BoundingBox(xmin=0, ymin=5, xmax=10, ymax=15)
        self.assertAlmostEqual(bbox_iou(b1, b3), 50.0 / 150.0)

        # Disjoint boxes => IoU = 0.0
        b4 = BoundingBox(xmin=20, ymin=20, xmax=30, ymax=30)
        self.assertEqual(bbox_iou(b1, b4), 0.0)

    def test_clip_bbox(self):
        box = BoundingBox(xmin=-10, ymin=-5, xmax=150, ymax=210)
        clipped = clip_bbox(box, min_x=0, min_y=0, max_x=100, max_y=100)
        self.assertEqual(clipped.as_tuple(), (0, 0, 100, 100))


if __name__ == "__main__":
    unittest.main()
