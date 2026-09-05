"""
Unit tests for MockSimulator and simulator extension stubs.
"""

import unittest
from simulation.mock_simulator import MockSimulator
from simulation.scenario_builder import SimulationScenario, SimulationAgent, SimulationHazard
from simulation.carla_adapter import CarlaAdapter
from simulation.sumo_adapter import SumoAdapter


class TestMockSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = MockSimulator()
        self.sim.initialize()

    def test_mock_simulator_lifecycle(self):
        self.assertTrue(self.sim.is_initialized)

        scenario = SimulationScenario(
            frame_id=1,
            timestamp=10.0,
            road_condition_category="GOOD",
            road_condition_score=95.0,
            estimated_friction_factor=1.0,
            speed_restriction_factor=1.0,
            agents=[SimulationAgent(1, "car", 101, (10, 10, 50, 50), (0.2, 0.5), 0.9)],
            hazards=[SimulationHazard(1, "pothole", "SMALL", (20, 20, 30, 30), (0.3, 0.7), 0.001, 1.0)],
        )

        self.sim.update(scenario)
        self.sim.step(0.033)

        state = self.sim.get_state()
        self.assertTrue(state["has_scenario"])
        self.assertEqual(state["current_frame_id"], 1)
        self.assertEqual(state["traffic"]["agent_count"], 1)
        self.assertEqual(state["hazards"]["hazard_count"], 1)
        self.assertAlmostEqual(state["sim_time"], 0.033)

        self.sim.close()
        self.assertFalse(self.sim.is_initialized)

    def test_carla_adapter_missing_dependency_behavior(self):
        carla_sim = CarlaAdapter()
        with self.assertRaises(ImportError) as ctx:
            carla_sim.initialize()
        self.assertIn("CARLA Python API", str(ctx.exception))

    def test_sumo_adapter_missing_dependency_behavior(self):
        sumo_sim = SumoAdapter()
        with self.assertRaises(ImportError) as ctx:
            sumo_sim.initialize()
        self.assertIn("SUMO TraCI", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
