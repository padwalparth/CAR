"""
Simulator Interface.

Defines the simulator-agnostic contract for all traffic and vehicle simulation engines
(MockSimulator, CarlaAdapter, SumoAdapter).

The simulator must consume ONLY simulation scenario representations derived from WorldState.
It must never know about computer-vision models, PyTorch/TensorFlow tensors, or model checkpoints.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class SimulatorInterface(ABC):
    """
    Abstract interface for all simulation backends.
    """
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize simulation environment, connections, or assets."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state to initial scenario."""
        pass

    @abstractmethod
    def update(self, scenario: Any) -> None:
        """
        Ingest the latest SimulationScenario.
        Updates agent targets, road friction, speed restrictions, and road hazards.
        """
        pass

    @abstractmethod
    def step(self, delta_time: float = 0.033) -> None:
        """Advance the simulation clock by delta_time seconds."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return the current simulator telemetry and agent state."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Gracefully shut down simulator connection and cleanup resources."""
        pass
