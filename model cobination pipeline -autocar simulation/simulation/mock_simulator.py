"""
Mock Simulator.

Lightweight in-memory simulator implementing SimulatorInterface.
Receives SimulationScenario objects and maintains telemetry, agent lists,
hazard maps, and road friction/speed constraints without requiring CARLA or SUMO.
"""

from typing import Any, Dict, List, Optional
from simulation.simulator_interface import SimulatorInterface
from simulation.scenario_builder import SimulationScenario


class MockSimulator(SimulatorInterface):
    """
    Simulation backend mock for end-to-end testing and development.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self.step_count = 0
        self.sim_time = 0.0
        self.last_scenario: Optional[SimulationScenario] = None
        self.telemetry_history: List[Dict[str, Any]] = []

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize mock simulation environment."""
        if config:
            self.config.update(config)
        self.is_initialized = True
        self.reset()

    def reset(self) -> None:
        """Reset internal simulator state."""
        self.step_count = 0
        self.sim_time = 0.0
        self.last_scenario = None
        self.telemetry_history.clear()

    def update(self, scenario: Any) -> None:
        """
        Receive updated scenario from perception.
        Accepts SimulationScenario or dictionary.
        """
        if not self.is_initialized:
            raise RuntimeError("MockSimulator is not initialized. Call initialize() before update().")

        if isinstance(scenario, SimulationScenario):
            self.last_scenario = scenario
        else:
            raise TypeError(f"MockSimulator expected SimulationScenario, got {type(scenario)}")

    def step(self, delta_time: float = 0.033) -> None:
        """Advance the simulated world by delta_time seconds."""
        if not self.is_initialized:
            raise RuntimeError("MockSimulator is not initialized.")

        self.step_count += 1
        self.sim_time += delta_time

        # Snapshot current step telemetry
        if self.last_scenario is not None:
            telemetry = {
                "step": self.step_count,
                "sim_time": round(self.sim_time, 3),
                "frame_id": self.last_scenario.frame_id,
                "road_condition": self.last_scenario.road_condition_category,
                "road_score": self.last_scenario.road_condition_score,
                "friction_factor": self.last_scenario.estimated_friction_factor,
                "speed_limit_factor": self.last_scenario.speed_restriction_factor,
                "agent_count": len(self.last_scenario.agents),
                "hazard_count": len(self.last_scenario.hazards),
                "traffic_intensity": self.last_scenario.traffic_intensity,
            }
            self.telemetry_history.append(telemetry)
            # Limit history length
            if len(self.telemetry_history) > 100:
                self.telemetry_history.pop(0)

    def get_state(self) -> Dict[str, Any]:
        """Expose current simulator state and active entities."""
        if self.last_scenario is None:
            return {
                "is_initialized": self.is_initialized,
                "step_count": self.step_count,
                "sim_time": round(self.sim_time, 3),
                "has_scenario": False,
            }

        return {
            "is_initialized": self.is_initialized,
            "step_count": self.step_count,
            "sim_time": round(self.sim_time, 3),
            "has_scenario": True,
            "current_frame_id": self.last_scenario.frame_id,
            "road": {
                "category": self.last_scenario.road_condition_category,
                "score": self.last_scenario.road_condition_score,
                "friction_factor": self.last_scenario.estimated_friction_factor,
                "speed_limit_factor": self.last_scenario.speed_restriction_factor,
            },
            "traffic": {
                "intensity": self.last_scenario.traffic_intensity,
                "density": self.last_scenario.traffic_density,
                "agent_count": len(self.last_scenario.agents),
                "agents": [a.to_dict() for a in self.last_scenario.agents],
            },
            "hazards": {
                "hazard_count": len(self.last_scenario.hazards),
                "hazards": [h.to_dict() for h in self.last_scenario.hazards],
            },
            "recent_telemetry": self.telemetry_history[-5:] if self.telemetry_history else [],
        }

    def close(self) -> None:
        """Close simulator."""
        self.is_initialized = False
