"""
Unit Tests for PipelineRunner.

Tests configuration integration, source construction, pipeline execution,
output saving, and error handling.
"""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from processing.pipeline_runner import (
    build_frame_source,
    format_frame_result,
    run_pipeline,
)
from processing.frame_source import CameraSource, ImageSource, SyntheticSource
from perception.schemas import (
    ModelStatus,
    PerceptionStatus,
    PotholesState,
    RoadCondition,
    RoadMask,
    TrafficState,
    WorldState,
    WorldStateMetadata,
)


class TestPipelineRunner(unittest.TestCase):
    """Test suite for pipeline_runner logic."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sih_pipeline_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_build_frame_source_mock(self):
        src = build_frame_source("mock", max_frames=5)
        self.assertIsInstance(src, SyntheticSource)
        self.assertEqual(src.max_frames, 5)

    def test_build_frame_source_camera(self):
        src = build_frame_source("camera:0")
        self.assertIsInstance(src, CameraSource)
        self.assertEqual(src.camera_index, 0)

    def test_build_frame_source_image(self):
        img_path = os.path.join(self.test_dir, "sample.jpg")
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(img_path)

        src = build_frame_source(img_path)
        self.assertIsInstance(src, ImageSource)

    def test_build_frame_source_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            build_frame_source(os.path.join(self.test_dir, "nonexistent.jpg"))

    def test_run_pipeline_mock_mode(self):
        stats = run_pipeline(
            source="mock",
            config_dir="config",
            mock_models=True,
            visualize=True,
            simulate=True,
            max_frames=4,
            display=False,
        )
        self.assertEqual(stats.frames_processed, 4)
        self.assertEqual(stats.successful_frames, 4)
        self.assertEqual(stats.failed_frames, 0)
        self.assertGreater(stats.avg_fps, 0.0)

    def test_run_pipeline_image_mode(self):
        img_path = os.path.join(self.test_dir, "test_road.png")
        img = Image.fromarray(np.zeros((120, 160, 3), dtype=np.uint8))
        img.save(img_path)

        stats = run_pipeline(
            source=img_path,
            config_dir="config",
            mock_models=True,
            visualize=False,
            simulate=False,
            max_frames=1,
            display=False,
        )
        self.assertEqual(stats.frames_processed, 1)
        self.assertEqual(stats.successful_frames, 1)

    def test_run_pipeline_output_saving(self):
        out_dir = os.path.join(self.test_dir, "output_frames")
        stats = run_pipeline(
            source="mock",
            config_dir="config",
            mock_models=True,
            visualize=True,
            simulate=False,
            save_output=out_dir,
            save_json=out_dir,
            max_frames=2,
            display=False,
        )
        self.assertEqual(stats.frames_processed, 2)

        # Verify generated files
        frame1 = os.path.join(out_dir, "frame_000001.jpg")
        frame2 = os.path.join(out_dir, "frame_000002.jpg")
        json1 = os.path.join(out_dir, "world_state_000001.json")
        json2 = os.path.join(out_dir, "world_state_000002.json")
        summary = os.path.join(out_dir, "run_summary.json")

        self.assertTrue(os.path.exists(frame1))
        self.assertTrue(os.path.exists(frame2))
        self.assertTrue(os.path.exists(json1))
        self.assertTrue(os.path.exists(json2))
        self.assertTrue(os.path.exists(summary))

        # Verify summary content
        with open(summary, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        self.assertEqual(summary_data["frames_processed"], 2)
        self.assertEqual(summary_data["successful_frames"], 2)

    def test_run_pipeline_real_model_mode_missing_checkpoint_raises_error(self):
        """Ensure missing checkpoint in real model mode raises error without falling back to mock."""
        temp_dir = os.path.join(self.test_dir, "bad_cfg_dir")
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "models.yaml"), "w", encoding="utf-8") as f:
            f.write(
                "road_model:\n  backend: onnx\n  path: nonexistent_road.onnx\n"
                "traffic_model:\n  backend: ultralytics\n  path: nonexistent_yolo.pt\n"
                "pothole_model:\n  backend: res2net\n  path: nonexistent_pothole.pt\n"
            )
        shutil.copy("config/pipeline.yaml", os.path.join(temp_dir, "pipeline.yaml"))
        shutil.copy("config/simulation.yaml", os.path.join(temp_dir, "simulation.yaml"))

        with self.assertRaises((ImportError, FileNotFoundError)):
            run_pipeline(
                source="mock",
                config_dir=temp_dir,
                mock_models=False,  # Enforce real models
                display=False,
                max_frames=1,
            )

    def test_format_frame_result(self):
        ws = WorldState(
            frame_id=1,
            timestamp=100.0,
            image_width=1280,
            image_height=720,
            road=RoadMask(mask=None, confidence=0.95, road_area_ratio=0.6),
            road_condition=RoadCondition(score=92.4),
            traffic=TrafficState(total_objects=8, vehicle_count=6, pedestrian_count=2),
            potholes=PotholesState(count=2, severity_counts={"LARGE": 1, "SMALL": 1}),
            perception_status=PerceptionStatus(),
            metadata=WorldStateMetadata(processing_time_ms=84.3, fps=11.9),
        )
        result_str = format_frame_result(ws)
        self.assertIn("FRAME RESULT", result_str)
        self.assertIn("Frame ID: 1", result_str)
        self.assertIn("Resolution: 1280x720", result_str)
        self.assertIn("Score: 92.4", result_str)
        self.assertIn("Objects: 8", result_str)
        self.assertIn("Vehicles: 6", result_str)
        self.assertIn("Pedestrians: 2", result_str)
        self.assertIn("Total: 84.3 ms", result_str)
        self.assertIn("FPS: 11.9", result_str)


if __name__ == "__main__":
    unittest.main()
