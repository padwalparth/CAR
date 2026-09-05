"""
Unit tests for configuration loading and validation.
"""

import os
import unittest
from config.loader import (
    load_models_config,
    load_pipeline_config,
    load_simulation_config,
    validate_models_config,
    validate_pipeline_config,
)


class TestConfig(unittest.TestCase):
    def test_load_valid_configs(self):
        models_cfg = load_models_config("config/models.yaml")
        self.assertIn("road_model", models_cfg)
        self.assertIn("traffic_model", models_cfg)
        self.assertIn("pothole_model", models_cfg)

        pipeline_cfg = load_pipeline_config("config/pipeline.yaml")
        self.assertIn("confidence_thresholds", pipeline_cfg)
        self.assertIn("road_association", pipeline_cfg)
        self.assertIn("pothole_severity", pipeline_cfg)

        sim_cfg = load_simulation_config("config/simulation.yaml")
        self.assertIn("mock_simulator", sim_cfg)
        self.assertIn("scenario_mapping", sim_cfg)

    def test_missing_config_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_models_config("config/non_existent_file.yaml")

    def test_invalid_models_config_raises_value_error(self):
        # Missing required section
        bad_cfg = {"road_model": {"backend": "onnx"}}
        with self.assertRaises(ValueError) as ctx:
            validate_models_config(bad_cfg)
        self.assertIn("traffic_model", str(ctx.exception))

        # Missing backend field
        bad_cfg_2 = {
            "road_model": {},
            "traffic_model": {"backend": "ultralytics"},
            "pothole_model": {"backend": "res2net"},
        }
        with self.assertRaises(ValueError) as ctx2:
            validate_models_config(bad_cfg_2)
        self.assertIn("backend", str(ctx2.exception))

    def test_invalid_pipeline_config_raises_value_error(self):
        bad_pipe = {"confidence_thresholds": {}}
        with self.assertRaises(ValueError) as ctx:
            validate_pipeline_config(bad_pipe)
        self.assertIn("road_association", str(ctx.exception))

    def test_invalid_parameter_ranges(self):
        # Negative confidence
        bad_conf = {
            "confidence_thresholds": {"traffic": -0.1},
            "road_association": {},
            "tracking": {},
            "pothole_severity": {},
            "road_condition": {},
        }
        with self.assertRaises(ValueError):
            validate_pipeline_config(bad_conf)

        # Inverted pothole severity thresholds (small > medium)
        bad_sev = {
            "confidence_thresholds": {"traffic": 0.5},
            "road_association": {},
            "tracking": {},
            "pothole_severity": {"small_max_relative_area": 0.05, "medium_max_relative_area": 0.02},
            "road_condition": {},
        }
        with self.assertRaises(ValueError):
            validate_pipeline_config(bad_sev)

        # Inverted road condition score thresholds (poor > good)
        bad_rc = {
            "confidence_thresholds": {"traffic": 0.5},
            "road_association": {},
            "tracking": {},
            "pothole_severity": {"small_max_relative_area": 0.01, "medium_max_relative_area": 0.04},
            "road_condition": {"thresholds": {"good_min_score": 30.0, "moderate_min_score": 50.0, "poor_min_score": 80.0}},
        }
        with self.assertRaises(ValueError):
            validate_pipeline_config(bad_rc)


if __name__ == "__main__":
    unittest.main()
