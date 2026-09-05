"""
Unit tests for ScenarioBuilder mapping.
"""

import unittest
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
from simulation.scenario_builder import ScenarioBuilder, SimulationScenario
from config.loader import load_simulation_config


class TestScenarioBuilder(unittest.TestCase):
    def setUp(self):
        self.config = load_simulation_config("config/simulation.yaml")
        self.builder = ScenarioBuilder(self.config)

    def test_build_scenario_from_world_state(self):
        ws = WorldState(
            frame_id=42,
            timestamp=1234.56,
            image_width=1000,
            image_height=1000,
            road=RoadMask(mask=None, confidence=0.9, road_area_ratio=0.4),
            road_condition=RoadCondition(
                score=65.0,
                category=RoadConditionCategory.MODERATE,
                estimated_friction_factor=0.85,
                speed_restriction_factor=0.75,
            ),
            traffic=TrafficState(
                vehicles=[Detection(1, "car", 4, 0.9, BoundingBox(100, 500, 200, 600), track_id=7)],
                pedestrians=[Detection(2, "person", 7, 0.8, BoundingBox(300, 400, 350, 500), track_id=8)],
                total_objects=2,
                traffic_density=0.3,
                congestion_level=CongestionLevel.MODERATE,
            ),
            potholes=PotholesState(
                potholes=[PotholeDetection(1, 0.95, BoundingBox(400, 700, 500, 750), 5000.0, 0.005, PotholeSeverityCategory.SMALL)],
                count=1,
            ),
            perception_status=PerceptionStatus(),
            metadata=WorldStateMetadata(),
        )

        scenario = self.builder.build_scenario(ws)
        self.assertIsInstance(scenario, SimulationScenario)
        self.assertEqual(scenario.frame_id, 42)
        self.assertEqual(scenario.road_condition_category, "MODERATE")
        self.assertEqual(scenario.speed_restriction_factor, 0.75)
        self.assertEqual(scenario.estimated_friction_factor, 0.85)
        self.assertEqual(len(scenario.agents), 2)
        self.assertEqual(scenario.agents[0].track_id, 7)
        self.assertEqual(len(scenario.hazards), 1)
        self.assertEqual(scenario.hazards[0].severity, "SMALL")

        # Verify dictionary serialization
        s_dict = scenario.to_dict()
        self.assertEqual(s_dict["frame_id"], 42)
        self.assertEqual(s_dict["road_condition"]["category"], "MODERATE")


if __name__ == "__main__":
    unittest.main()
