"""
Unit tests for SimulationAdapter.
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
from simulation.mock_simulator import MockSimulator
from simulation.simulation_adapter import SimulationAdapter


class TestSimulationAdapter(unittest.TestCase):
    def setUp(self):
        self.simulator = MockSimulator()
        self.adapter = SimulationAdapter(self.simulator)
        self.adapter.initialize()

    def test_uninitialized_adapter_raises_error(self):
        uninit_adapter = SimulationAdapter(MockSimulator())
        ws = WorldState(
            frame_id=1, timestamp=0.0, image_width=100, image_height=100,
            road=RoadMask(mask=None), road_condition=RoadCondition(),
            traffic=TrafficState(), potholes=PotholesState(),
        )
        with self.assertRaises(RuntimeError):
            uninit_adapter.update(ws)

    def test_adapter_update_step_get_state(self):
        ws = WorldState(
            frame_id=5,
            timestamp=100.2,
            image_width=1280,
            image_height=720,
            road=RoadMask(mask=None, road_area_ratio=0.4),
            road_condition=RoadCondition(score=75.0, category=RoadConditionCategory.MODERATE),
            traffic=TrafficState(total_objects=3),
            potholes=PotholesState(count=2),
            perception_status=PerceptionStatus(),
            metadata=WorldStateMetadata(),
        )

        scenario = self.adapter.update(ws)
        self.assertEqual(scenario.frame_id, 5)

        self.adapter.step(0.04)
        sim_state = self.adapter.get_state()

        self.assertTrue(sim_state["has_scenario"])
        self.assertEqual(sim_state["current_frame_id"], 5)
        self.assertEqual(sim_state["road"]["category"], "MODERATE")

        self.adapter.close()
        self.assertFalse(self.adapter.is_initialized)


if __name__ == "__main__":
    unittest.main()
