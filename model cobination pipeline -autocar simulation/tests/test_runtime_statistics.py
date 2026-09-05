"""
Unit Tests for RuntimeStatistics.

Tests frame counting, latency math, error tracking, and serialization.
"""

import time
import unittest

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
from processing.statistics import RuntimeStatistics


class TestRuntimeStatistics(unittest.TestCase):
    """Test suite for RuntimeStatistics tracker."""

    def _make_dummy_world_state(
        self,
        frame_id: int = 1,
        road_status: str = ModelStatus.OK.value,
        traffic_status: str = ModelStatus.OK.value,
        pothole_status: str = ModelStatus.OK.value,
        pothole_count: int = 2,
        traffic_count: int = 5,
    ) -> WorldState:
        return WorldState(
            frame_id=frame_id,
            timestamp=time.time(),
            image_width=1280,
            image_height=720,
            road=RoadMask(mask=None, confidence=0.9, road_area_ratio=0.5),
            road_condition=RoadCondition(score=85.0),
            traffic=TrafficState(total_objects=traffic_count, vehicle_count=4, pedestrian_count=1),
            potholes=PotholesState(count=pothole_count),
            perception_status=PerceptionStatus(
                road=road_status,
                traffic=traffic_status,
                pothole=pothole_status,
            ),
            metadata=WorldStateMetadata(processing_time_ms=50.0, fps=20.0),
        )

    def test_initial_state(self):
        stats = RuntimeStatistics()
        self.assertEqual(stats.frames_processed, 0)
        self.assertEqual(stats.successful_frames, 0)
        self.assertEqual(stats.failed_frames, 0)
        self.assertEqual(stats.avg_latency_ms, 0.0)
        self.assertEqual(stats.min_latency_ms, 0.0)
        self.assertEqual(stats.max_latency_ms, 0.0)
        self.assertEqual(stats.avg_fps, 0.0)

    def test_record_successful_frames_latency_math(self):
        stats = RuntimeStatistics()
        stats.start()

        ws1 = self._make_dummy_world_state(frame_id=1, pothole_count=1, traffic_count=3)
        ws2 = self._make_dummy_world_state(frame_id=2, pothole_count=2, traffic_count=4)

        stats.record_frame(ws1, latency_ms=40.0)
        stats.record_frame(ws2, latency_ms=60.0)
        stats.stop()

        self.assertEqual(stats.frames_processed, 2)
        self.assertEqual(stats.successful_frames, 2)
        self.assertEqual(stats.failed_frames, 0)
        self.assertAlmostEqual(stats.avg_latency_ms, 50.0)
        self.assertAlmostEqual(stats.min_latency_ms, 40.0)
        self.assertAlmostEqual(stats.max_latency_ms, 60.0)
        self.assertAlmostEqual(stats.avg_fps, 20.0)  # 1000 / 50.0 = 20.0
        self.assertEqual(stats.total_potholes_detected, 3)
        self.assertEqual(stats.total_traffic_objects_detected, 7)

    def test_record_isolated_model_failures(self):
        stats = RuntimeStatistics()
        stats.start()

        ws_road_fail = self._make_dummy_world_state(road_status=ModelStatus.FAILED.value)
        ws_traffic_fail = self._make_dummy_world_state(traffic_status=ModelStatus.FAILED.value)
        ws_pothole_fail = self._make_dummy_world_state(pothole_status=ModelStatus.FAILED.value)

        stats.record_frame(ws_road_fail, latency_ms=30.0)
        stats.record_frame(ws_traffic_fail, latency_ms=30.0)
        stats.record_frame(ws_pothole_fail, latency_ms=30.0)

        self.assertEqual(stats.frames_processed, 3)
        self.assertEqual(stats.successful_frames, 0)
        self.assertEqual(stats.failed_frames, 3)
        self.assertEqual(stats.road_failures, 1)
        self.assertEqual(stats.traffic_failures, 1)
        self.assertEqual(stats.pothole_failures, 1)

    def test_to_dict_and_format_summary(self):
        stats = RuntimeStatistics()
        stats.start()
        ws = self._make_dummy_world_state()
        stats.record_frame(ws, latency_ms=25.0)
        stats.stop()

        data = stats.to_dict(sim_enabled=True, vis_enabled=True)
        self.assertEqual(data["frames_processed"], 1)
        self.assertEqual(data["successful_frames"], 1)
        self.assertEqual(data["average_latency_ms"], 25.0)
        self.assertTrue(data["simulation_enabled"])
        self.assertTrue(data["visualization_enabled"])

        summary_text = stats.format_summary(sim_enabled=True, vis_enabled=True)
        self.assertIn("RUN SUMMARY", summary_text)
        self.assertIn("Frames processed:       1", summary_text)
        self.assertIn("Simulation:             enabled", summary_text)
        self.assertIn("Visualization:          enabled", summary_text)


if __name__ == "__main__":
    unittest.main()
