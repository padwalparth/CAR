"""
End-to-End Pipeline Integration Test.

Validates the full perception-to-simulation workflow:
SyntheticSource -> FrameProcessor -> MockModels -> Adapters -> PerceptionFusion -> WorldState -> SimulationAdapter -> MockSimulator.

Runs completely without GPU or ML framework dependencies.
"""

import unittest
from processing.frame_source import SyntheticSource
from processing.frame_processor import FrameProcessor
from models.road_segmentation.unet_model import MockRoadModel
from models.traffic_detection.yolo_idd_model import MockTrafficModel
from models.pothole_detection.res2net_model import MockPotholeModel
from adapters.road_adapter import RoadAdapter
from adapters.traffic_adapter import TrafficAdapter
from adapters.pothole_adapter import PotholeAdapter
from perception.fusion import PerceptionFusion
from perception.world_state import WorldStateHistory, validate_world_state
from simulation.mock_simulator import MockSimulator
from simulation.simulation_adapter import SimulationAdapter
from config.loader import load_pipeline_config, load_simulation_config


class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline_stream(self):
        pipeline_cfg = load_pipeline_config("config/pipeline.yaml")
        sim_cfg = load_simulation_config("config/simulation.yaml")

        # 1. Setup Frame Source
        source = SyntheticSource(max_frames=5, width=1280, height=720, fps=25.0)

        # 2. Setup FrameProcessor with isolated models
        processor = FrameProcessor(
            road_model=MockRoadModel(),
            traffic_model=MockTrafficModel(),
            pothole_model=MockPotholeModel(),
            road_adapter=RoadAdapter(),
            traffic_adapter=TrafficAdapter(),
            pothole_adapter=PotholeAdapter(),
            fusion_engine=PerceptionFusion(pipeline_cfg),
            config=pipeline_cfg,
        )

        # 3. Setup Simulation
        simulator = MockSimulator()
        simulation_adapter = SimulationAdapter(simulator, config=sim_cfg)

        # 4. History Tracker
        history = WorldStateHistory(max_length=10)

        # Initialize all layers
        source.open()
        processor.initialize()
        simulation_adapter.initialize()

        frames_processed = 0
        while source.has_next():
            success, frame, meta = source.read()
            self.assertTrue(success)

            # Perception Step
            world_state = processor.process_frame(
                frame=frame,
                frame_id=meta.frame_id,
                timestamp=meta.timestamp,
                source=meta.source,
            )

            # Validate WorldState contract
            self.assertTrue(validate_world_state(world_state))
            history.add(world_state)

            # Simulation Step
            scenario = simulation_adapter.update(world_state)
            simulation_adapter.step(0.04)

            # Assertions
            self.assertEqual(world_state.frame_id, frames_processed + 1)
            self.assertTrue(world_state.perception_status.is_all_ok())
            self.assertGreater(world_state.traffic.vehicle_count, 0)
            self.assertGreater(world_state.potholes.count, 0)
            self.assertEqual(scenario.frame_id, world_state.frame_id)

            frames_processed += 1

        self.assertEqual(frames_processed, 5)
        self.assertEqual(len(history), 5)

        # Verify simulator state after running
        sim_state = simulation_adapter.get_state()
        self.assertTrue(sim_state["has_scenario"])
        self.assertEqual(sim_state["current_frame_id"], 5)
        self.assertEqual(sim_state["step_count"], 5)
        self.assertGreater(len(sim_state["recent_telemetry"]), 0)

        # Cleanup
        processor.close()
        simulation_adapter.close()
        source.close()


if __name__ == "__main__":
    unittest.main()
