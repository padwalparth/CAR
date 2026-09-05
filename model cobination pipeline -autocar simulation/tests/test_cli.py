"""
Unit and Integration Tests for Pipeline CLI.

Tests command-line parsing, help text, subprocess smoke execution,
headless guards, and error handling.
"""

import os
import subprocess
import sys
import unittest

from scripts.run_pipeline import build_arg_parser


class TestPipelineCLI(unittest.TestCase):
    """Test suite for CLI argument parser and execution."""

    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_arg_parser_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.source, "mock")
        self.assertEqual(args.config, "config")
        self.assertFalse(args.mock_models)
        self.assertFalse(args.visualize)
        self.assertFalse(args.simulate)
        self.assertIsNone(args.save_output)
        self.assertIsNone(args.save_json)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.display)
        self.assertFalse(args.no_display)
        self.assertFalse(args.verbose)

    def test_arg_parser_custom_values(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--source", "road.jpg",
            "--mock-models",
            "--visualize",
            "--simulate",
            "--max-frames", "20",
            "--save-output", "out_dir",
            "--save-json", "json_dir",
            "--verbose",
        ])
        self.assertEqual(args.source, "road.jpg")
        self.assertTrue(args.mock_models)
        self.assertTrue(args.visualize)
        self.assertTrue(args.simulate)
        self.assertEqual(args.max_frames, 20)
        self.assertEqual(args.save_output, "out_dir")
        self.assertEqual(args.save_json, "json_dir")
        self.assertTrue(args.verbose)

    def test_cli_help_flag(self):
        cmd = [sys.executable, "scripts/run_pipeline.py", "--help"]
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: run_pipeline", result.stdout)
        self.assertIn("--source", result.stdout)
        self.assertIn("--mock-models", result.stdout)
        self.assertIn("--simulate", result.stdout)
        self.assertIn("--visualize", result.stdout)

    def test_cli_mock_smoke_test_subprocess(self):
        """Verify the primary end-to-end smoke test command runs cleanly via subprocess."""
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            "--source", "mock",
            "--mock-models",
            "--simulate",
            "--max-frames", "3",
            "--no-display",
        ]
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, f"CLI execution failed with: {result.stderr}")
        self.assertIn("RUN SUMMARY", result.stdout)
        self.assertIn("Frames processed:       3", result.stdout)
        self.assertIn("Simulation:             enabled", result.stdout)

    def test_cli_invalid_source_path(self):
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            "--source", "nonexistent_road_file.mp4",
            "--mock-models",
            "--no-display",
        ]
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("File Not Found", result.stderr)

    def test_cli_real_mode_missing_checkpoint_actionable_error(self):
        """Without --mock-models, missing checkpoint files must fail with exit code 1."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "road_model:\n  backend: onnx\n  path: nonexistent_road.onnx\n"
                "traffic_model:\n  backend: ultralytics\n  path: nonexistent_yolo.pt\n"
                "pothole_model:\n  backend: res2net\n  path: nonexistent_pothole.pt\n"
            )
            bad_cfg_path = f.name

        try:
            cmd = [
                sys.executable,
                "scripts/run_pipeline.py",
                "--source", "mock",
                "--config", bad_cfg_path,
                "--max-frames", "1",
                "--no-display",
            ]
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue("ERROR" in result.stderr or "File Not Found" in result.stderr)
        finally:
            if os.path.exists(bad_cfg_path):
                os.remove(bad_cfg_path)


if __name__ == "__main__":
    unittest.main()
