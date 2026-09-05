"""
Simulation Adapter.

The bridge between Perception and Simulation.
Consumes WorldState, converts it to a SimulationScenario via ScenarioBuilder,
and pushes it to any SimulatorInterface (MockSimulator, CarlaAdapter, SumoAdapter).
"""

from typing import Any, Dict, Optional
from perception.schemas import WorldState
from simulation.simulator_interface import SimulatorInterface
from simulation.scenario_builder import ScenarioBuilder, SimulationScenario


class SimulationAdapter:
    """
    Adapter translating perception WorldState to simulator-specific commands.
    Decouples simulators completely from computer-vision models.
    """
    def __init__(
        self,
        simulator: SimulatorInterface,
        scenario_builder: Optional[ScenarioBuilder] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.simulator = simulator
        self.config = config or {}
        self.scenario_builder = scenario_builder or ScenarioBuilder(self.config)
        self.is_initialized = False

    def initialize(self) -> None:
        """Initialize both the adapter and the underlying simulator."""
        self.simulator.initialize(self.config)
        self.is_initialized = True

    def reset(self) -> None:
        """Reset simulator environment."""
        self.simulator.reset()

    def update(self, world_state: WorldState) -> SimulationScenario:
        """
        Convert WorldState snapshot into SimulationScenario and update simulator.
        Returns the constructed SimulationScenario.
        """
        if not self.is_initialized:
            raise RuntimeError("SimulationAdapter is not initialized. Call initialize() before update().")

        scenario = self.scenario_builder.build_scenario(world_state)
        self.simulator.update(scenario)
        return scenario

    def step(self, delta_time: float = 0.033) -> None:
        """Advance simulator clock."""
        self.simulator.step(delta_time)

    def get_state(self) -> Dict[str, Any]:
        """Query simulator current state."""
        return self.simulator.get_state()

    def close(self) -> None:
        """Shut down simulation."""
        self.simulator.close()
        self.is_initialized = False
