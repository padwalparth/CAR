"""
Unit and integration tests for TelemetryDashboard.
"""

import unittest
import numpy as np
from perception.schemas import (
    BoundingBox,
    Detection,
    PotholeDetection,
    PotholeSeverityCategory,
    RoadMask,
    RoadCondition,
    RoadConditionCategory,
    TrafficState,
    PotholesState,
    PerceptionStatus,
    WorldStateMetadata,
    WorldState,
    CongestionLevel,
)
from visualization.dashboard import TelemetryDashboard
from visualization.overlay import OverlayRenderer
from visualization.birdseye import BirdseyeRenderer
from processing.frame_source import SyntheticSource
from processing.frame_processor import FrameProcessor
from models.road_segmentation.unet_model import MockRoadModel
from models.traffic_detection.yolo_idd_model import MockTrafficModel
from models.pothole_detection.res2net_model import MockPotholeModel
from config.loader import load_pipeline_config


class TestVisualizationDashboard(unittest.TestCase):
    def setUp(self):
        self.dashboard = TelemetryDashboard()

    def _create_mock_state(self, has_errors=False) -> WorldState:
        status = PerceptionStatus() if not has_errors else PerceptionStatus(
            pothole="error",
            errors={"pothole": "Checkpoint loading failed"},
        )
        return WorldState(
            frame_id=105,
            timestamp=42.125,
            image_width=1280,
            image_height=720,
            road=RoadMask(mask=None, road_area_ratio=0.48),
            road_condition=RoadCondition(
                score=76.5,
                category=RoadConditionCategory.MODERATE,
                estimated_friction_factor=0.85,
                speed_restriction_factor=0.75,
            ),
            traffic=TrafficState(
                vehicles=[Detection(1, "car", 4, 0.9, BoundingBox(100, 100, 200, 200))],
                total_objects=1,
                vehicle_count=1,
                pedestrian_count=0,
                vehicle_counts_by_class={"car": 1},
                traffic_density=0.12,
                congestion_level=CongestionLevel.LOW,
            ),
            potholes=PotholesState(
                potholes=[PotholeDetection(1, 0.9, BoundingBox(300, 400, 400, 480), 8000.0, 0.0087, PotholeSeverityCategory.SMALL)],
                count=1,
                total_relative_area=0.0087,
                severity_counts={"SMALL": 1},
            ),
            perception_status=status,
            metadata=WorldStateMetadata(
                fps=24.2,
                processing_time_ms=41.3,
                road_inference_ms=12.0,
                traffic_inference_ms=20.0,
                pothole_inference_ms=6.0,
                fusion_ms=3.3,
            ),
        )

    def test_dashboard_render_text(self):
        ws = self._create_mock_state()
        text = self.dashboard.render_text(ws)
        self.assertIn("105", text)
        self.assertIn("MODERATE", text)
        self.assertIn("76.5", text)
        self.assertIn("LOW", text)
        self.assertIn("car: 1", text)
        self.assertIn("SMALL: 1", text)

    def test_dashboard_render_text_with_errors(self):
        ws = self._create_mock_state(has_errors=True)
        text = self.dashboard.render_text(ws)
        self.assertIn("Checkpoint loading failed", text)

    def test_dashboard_render_image(self):
        ws = self._create_mock_state()
        img = self.dashboard.render_image(ws, width=640, height=480)
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_full_visualization_pipeline_integration(self):
        """Verify that SyntheticSource -> FrameProcessor -> WorldState -> All 3 Renderers works cleanly."""
        config = load_pipeline_config("config/pipeline.yaml")
        processor = FrameProcessor(
            road_model=MockRoadModel(),
            traffic_model=MockTrafficModel(),
            pothole_model=MockPotholeModel(),
            config=config,
        )
        processor.initialize()

        overlay = OverlayRenderer()
        birdseye = BirdseyeRenderer()
        dashboard = TelemetryDashboard()

        with SyntheticSource(max_frames=1) as src:
            for frame, meta in src:
                ws = processor.process_frame(frame, meta.frame_id, meta.timestamp, meta.source)

                # Render overlay
                ann_frame = overlay.render(frame, ws)
                self.assertEqual(ann_frame.shape, frame.shape)

                # Render birdseye
                bev_frame = birdseye.render(ws)
                self.assertEqual(bev_frame.shape, (600, 400, 3))

                # Render dashboard
                dash_text = dashboard.render_text(ws)
                dash_img = dashboard.render_image(ws)
                self.assertGreater(len(dash_text), 100)
                self.assertEqual(dash_img.shape, (480, 640, 3))

        processor.close()


if __name__ == "__main__":
    unittest.main()
