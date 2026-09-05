"""
Unit tests for multi-object tracking.
"""

import unittest
from perception.schemas import Detection, BoundingBox
from perception.tracker import SimpleTracker


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = SimpleTracker(min_iou_threshold=0.2, max_disappeared=3)

    def test_persistent_track_ids(self):
        # Frame 1: Vehicle 1 at [100, 100, 200, 200]
        det1 = Detection(
            object_id=1,
            class_name="car",
            class_id=4,
            confidence=0.9,
            bbox=BoundingBox(100, 100, 200, 200),
        )
        out1 = self.tracker.update([det1], 1000, 1000)
        self.assertEqual(len(out1), 1)
        initial_track_id = out1[0].track_id
        self.assertIsNotNone(initial_track_id)

        # Frame 2: Vehicle 1 moved slightly to [105, 102, 205, 202] (High IoU)
        det2 = Detection(
            object_id=2,
            class_name="car",
            class_id=4,
            confidence=0.92,
            bbox=BoundingBox(105, 102, 205, 202),
        )
        out2 = self.tracker.update([det2], 1000, 1000)
        self.assertEqual(len(out2), 1)
        # Track ID must remain constant across consecutive frames!
        self.assertEqual(out2[0].track_id, initial_track_id)

    def test_new_object_gets_distinct_track_id(self):
        # Frame 1: car A
        det_a = Detection(
            object_id=1,
            class_name="car",
            class_id=4,
            confidence=0.9,
            bbox=BoundingBox(50, 50, 100, 100),
        )
        out1 = self.tracker.update([det_a], 1000, 1000)
        tid_a = out1[0].track_id

        # Frame 2: car A + new pedestrian B at [700, 700, 750, 800]
        det_a2 = Detection(
            object_id=2,
            class_name="car",
            class_id=4,
            confidence=0.9,
            bbox=BoundingBox(52, 51, 102, 101),
        )
        det_b = Detection(
            object_id=3,
            class_name="person",
            class_id=7,
            confidence=0.85,
            bbox=BoundingBox(700, 700, 750, 800),
        )
        out2 = self.tracker.update([det_a2, det_b], 1000, 1000)
        self.assertEqual(len(out2), 2)

        # Check track IDs
        matched_a = next(d for d in out2 if d.class_name == "car")
        new_b = next(d for d in out2 if d.class_name == "person")
        self.assertEqual(matched_a.track_id, tid_a)
        self.assertNotEqual(new_b.track_id, tid_a)

    def test_track_expiration_after_max_disappeared(self):
        # Frame 1: object appears
        det = Detection(
            object_id=1,
            class_name="truck",
            class_id=13,
            confidence=0.8,
            bbox=BoundingBox(200, 200, 400, 400),
        )
        self.tracker.update([det], 1000, 1000)
        self.assertEqual(len(self.tracker.tracks), 1)

        # Object disappears for 4 frames (max_disappeared is 3)
        for _ in range(4):
            self.tracker.update([], 1000, 1000)

        self.assertEqual(len(self.tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
