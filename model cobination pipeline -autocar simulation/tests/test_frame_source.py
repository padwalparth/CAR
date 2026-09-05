"""
Unit tests for FrameSource implementations.
"""

import unittest
from processing.frame_source import SyntheticSource, ImageSource


class TestFrameSource(unittest.TestCase):
    def test_synthetic_source_lifecycle(self):
        source = SyntheticSource(max_frames=3, width=640, height=360, fps=15.0)
        source.open()
        self.assertTrue(source.is_opened)

        frames_read = 0
        while source.has_next():
            success, frame, meta = source.read()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (360, 640, 3))
            self.assertEqual(meta.frame_id, frames_read + 1)
            frames_read += 1

        self.assertEqual(frames_read, 3)
        self.assertFalse(source.has_next())
        source.close()
        self.assertFalse(source.is_opened)

    def test_synthetic_source_context_manager(self):
        with SyntheticSource(max_frames=2) as src:
            count = sum(1 for _ in src)
            self.assertEqual(count, 2)

    def test_image_source_missing_file_raises_error(self):
        src = ImageSource("non_existent_image_path.jpg")
        with self.assertRaises(FileNotFoundError):
            src.open()


if __name__ == "__main__":
    unittest.main()
