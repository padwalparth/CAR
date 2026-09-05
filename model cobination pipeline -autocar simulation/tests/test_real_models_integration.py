"""
Real Model Integration Tests (Conditional / Environment-Aware).

Validates the actual neural network checkpoints and adapters when dependencies
are installed, and cleanly skips with informative reasons when running in a
minimal / CI environment.

Separate from the mandatory 100-test core mock regression suite.
"""

import os
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def have_ml_packages() -> bool:
    try:
        import onnxruntime
        import torch
        import timm
        import ultralytics
        return True
    except ImportError:
        return False


class TestRealModelsIntegration(unittest.TestCase):
    """Integration test suite executing real neural network inference."""

    @classmethod
    def setUpClass(cls):
        cls.road_onnx_path = os.path.join(
            REPO_ROOT, "Road-segmentation-UNET-model-main", "models", "onnx_models", "road_seg_160_160.onnx"
        )
        cls.yolo_pt_path = os.path.join(
            REPO_ROOT, "YOLOv8.3.0 Train on IDD", "runs", "idd_yolov8_training", "weights", "best.pt"
        )
        cls.pothole_pt_path = os.path.join(
            REPO_ROOT, "pothole detection model", "best_model.pt"
        )

    def test_real_road_segmentation_contract(self):
        """Verify real U-Net ONNX model produces expected mask and 3-class contract."""
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("onnxruntime is not installed.")

        if not os.path.exists(self.road_onnx_path):
            self.skipTest(f"Road ONNX model not found at: {self.road_onnx_path}")

        from models.road_segmentation.unet_model import UNetRoadModel
        from adapters.road_adapter import RoadAdapter

        model = UNetRoadModel({
            "backend": "onnx",
            "path": self.road_onnx_path,
            "input_size": [160, 160],
            "road_class_ids": [1, 2],
        })
        model.load()

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        raw_output = model.infer(dummy_frame)
        mask = raw_output["raw_mask"]

        self.assertTrue(mask.shape in [(160, 160), (720, 1280)], f"Unexpected mask shape: {mask.shape}")
        unique_classes = set(np.unique(mask).tolist())
        self.assertTrue(unique_classes.issubset({0, 1, 2}), f"Unexpected classes: {unique_classes}")

        # Test Adapter conversion
        adapter = RoadAdapter()
        road_mask = adapter.convert(raw_output)
        self.assertIsNotNone(road_mask.mask)
        self.assertGreaterEqual(road_mask.road_area_ratio, 0.0)
        self.assertLessEqual(road_mask.road_area_ratio, 1.0)
        model.close()

    def test_real_yolo_idd_traffic_contract(self):
        """Verify real YOLOv8 IDD model produces valid pixel bboxes and adapter normalizes them."""
        try:
            import ultralytics
            import torch
        except ImportError:
            self.skipTest("ultralytics or torch is not installed.")

        if not os.path.exists(self.yolo_pt_path):
            self.skipTest(f"YOLO IDD checkpoint not found at: {self.yolo_pt_path}")

        from models.traffic_detection.yolo_idd_model import YOLOIDDTrafficModel
        from adapters.traffic_adapter import TrafficAdapter

        model = YOLOIDDTrafficModel({
            "backend": "ultralytics",
            "path": self.yolo_pt_path,
            "confidence": 0.25,
            "iou": 0.70,
        })
        model.load()

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        raw_output = model.infer(dummy_frame)
        self.assertIn("detections", raw_output)

        # Raw Ultralytics detections have pixel coordinates
        adapter = TrafficAdapter()
        detections = adapter.convert(raw_output)
        self.assertIsInstance(detections, list)

        for det in detections:
            self.assertFalse(det.bbox.is_normalized)
            self.assertLessEqual(det.bbox.xmin, det.bbox.xmax)
            self.assertLessEqual(det.bbox.ymin, det.bbox.ymax)
            norm_b = det.bbox.to_normalized(1280, 720)
            self.assertTrue(0.0 <= norm_b.xmin <= norm_b.xmax <= 1.0)
            self.assertTrue(0.0 <= norm_b.ymin <= norm_b.ymax <= 1.0)
            self.assertTrue(norm_b.is_normalized)
            self.assertIn(det.class_id, range(15))
        model.close()

    def test_real_res2net_pothole_contract(self):
        """Verify real Res2Net model predicts valid coordinates and documents estimated confidence."""
        try:
            import torch
            import timm
        except ImportError:
            self.skipTest("torch or timm is not installed.")

        if not os.path.exists(self.pothole_pt_path):
            self.skipTest(f"Res2Net checkpoint not found at: {self.pothole_pt_path}")

        from models.pothole_detection.res2net_model import Res2NetPotholeModel
        from adapters.pothole_adapter import PotholeAdapter

        model = Res2NetPotholeModel({
            "backend": "res2net",
            "path": self.pothole_pt_path,
            "input_size": [128, 128],
            "min_box_size": 1.0,
        })
        model.load()

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        raw_output = model.infer(dummy_frame)
        self.assertIn("pothole_boxes", raw_output)

        boxes = raw_output["pothole_boxes"]
        if boxes:
            box = boxes[0]
            xmin, ymin, xmax, ymax = box["bbox"]
            self.assertFalse(np.isnan(xmin) or np.isnan(ymin))
            self.assertLess(xmin, xmax)
            self.assertLess(ymin, ymax)
            # Verify documented estimated confidence
            self.assertEqual(box["confidence"], 1.0)

        adapter = PotholeAdapter()
        potholes = adapter.convert(raw_output)
        self.assertIsInstance(potholes, list)
        model.close()

    def test_real_integrated_pipeline_execution(self):
        """Verify integrated pipeline execution with all 3 real models without mock fallback."""
        if not have_ml_packages():
            self.skipTest("Full ML stack not installed.")

        if not (os.path.exists(self.road_onnx_path) and os.path.exists(self.yolo_pt_path) and os.path.exists(self.pothole_pt_path)):
            self.skipTest("One or more real model checkpoints are missing.")

        from config.loader import load_models_config, load_pipeline_config, load_simulation_config
        from models.factory import create_road_model, create_traffic_model, create_pothole_model
        from processing.frame_processor import FrameProcessor
        from simulation.mock_simulator import MockSimulator
        from simulation.simulation_adapter import SimulationAdapter

        models_cfg = load_models_config(os.path.join(REPO_ROOT, "config", "models.yaml"))
        pipeline_cfg = load_pipeline_config(os.path.join(REPO_ROOT, "config", "pipeline.yaml"))
        sim_cfg = load_simulation_config(os.path.join(REPO_ROOT, "config", "simulation.yaml"))

        road_m = create_road_model(models_cfg["road_model"])
        traffic_m = create_traffic_model(models_cfg["traffic_model"])
        pothole_m = create_pothole_model(models_cfg["pothole_model"])

        processor = FrameProcessor(
            road_model=road_m,
            traffic_model=traffic_m,
            pothole_model=pothole_m,
            config=pipeline_cfg,
        )
        processor.initialize(warmup=False)

        simulator = MockSimulator(sim_cfg)
        sim_adapter = SimulationAdapter(simulator=simulator, config=pipeline_cfg)
        sim_adapter.initialize()

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        world_state = processor.process_frame(dummy_frame, frame_id=1, source="test_real")

        self.assertTrue(world_state.perception_status.is_all_ok())
        self.assertNotEqual(world_state.perception_status.road, "FAILED")
        self.assertNotEqual(world_state.perception_status.traffic, "FAILED")
        self.assertNotEqual(world_state.perception_status.pothole, "FAILED")

        scenario = sim_adapter.update(world_state)
        sim_adapter.step(0.033)
        sim_state = sim_adapter.get_state()
        self.assertTrue(sim_state["has_scenario"])

        processor.close()
        sim_adapter.close()


if __name__ == "__main__":
    unittest.main()
