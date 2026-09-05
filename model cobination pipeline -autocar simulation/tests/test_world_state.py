"""
Unit tests for WorldState validation and WorldStateHistory.
"""

import unittest
from perception.schemas import (
    RoadMask,
    RoadCondition,
    RoadConditionCategory,
    TrafficState,
    PotholesState,
    PerceptionStatus,
    WorldStateMetadata,
    WorldState,
)
from perception.world_state import WorldStateHistory, validate_world_state


class TestWorldState(unittest.TestCase):
    def _create_mock_state(self, frame_id: int, fps: float = 25.0) -> WorldState:
        return WorldState(
            frame_id=frame_id,
            timestamp=1000.0 + frame_id * 0.04,
            image_width=1280,
            image_height=720,
            road=RoadMask(mask=None, confidence=1.0, road_area_ratio=0.5),
            road_condition=RoadCondition(score=90.0, category=RoadConditionCategory.GOOD),
            traffic=TrafficState(),
            potholes=PotholesState(),
            perception_status=PerceptionStatus(),
            metadata=WorldStateMetadata(fps=fps, processing_time_ms=40.0),
        )

    def test_world_state_validator(self):
        state = self._create_mock_state(1)
        self.assertTrue(validate_world_state(state))

        # Invalid frame ID
        state.frame_id = -5
        with self.assertRaises(ValueError):
            validate_world_state(state)

    def test_world_state_history_rolling_buffer(self):
        history = WorldStateHistory(max_length=5)
        for i in range(10):
            history.add(self._create_mock_state(i + 1, fps=20.0 + i))

        self.assertEqual(len(history), 5)
        self.assertEqual(history.latest.frame_id, 10)
        recent = history.get_recent_states(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1].frame_id, 10)
        self.assertGreater(history.get_average_fps(), 0.0)


if __name__ == "__main__":
    unittest.main()
