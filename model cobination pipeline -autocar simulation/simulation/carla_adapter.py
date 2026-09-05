"""
CARLA Simulator Integration Adapter (Future Extension Stub).

Implements SimulatorInterface for CARLA autonomous driving simulator.
Uses lazy imports to prevent runtime crashes if carla is not installed.
"""

from typing import Any, Dict, Optional
from simulation.simulator_interface import SimulatorInterface


class CarlaAdapter(SimulatorInterface):
    """
    Adapter bridging SimulationScenario to CARLA world actors.
    Requires the official CARLA Python API (carla).
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.is_connected = False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Connect to CARLA server.
        Raises ImportError with clear instructions if carla Python API is missing.
        """
        if config:
            carla_cfg = config.get("carla", {})
            self.host = str(carla_cfg.get("host", self.host))
            self.port = int(carla_cfg.get("port", self.port))

        try:
            import carla  # type: ignore
        except ImportError as e:
            raise ImportError(
                "CARLA Python API ('carla') is not installed. "
                "To connect to CARLA, install the appropriate carla wheel matching your CARLA server version "
                "(e.g., CARLA 0.9.13 / 0.9.15). For local development, use MockSimulator."
            ) from e

        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.is_connected = True
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to CARLA server at {self.host}:{self.port}: {str(e)}"
            ) from e

    def reset(self) -> None:
        if not self.is_connected:
            raise RuntimeError("CarlaAdapter is not connected.")

    def update(self, scenario: Any) -> None:
        if not self.is_connected:
            raise RuntimeError("CarlaAdapter is not connected. Call initialize() first.")
        # Future mapping: spawn or update CARLA vehicle actors based on scenario.agents

    def step(self, delta_time: float = 0.033) -> None:
        if not self.is_connected:
            raise RuntimeError("CarlaAdapter is not connected.")
        if self.world is not None:
            self.world.tick()

    def get_state(self) -> Dict[str, Any]:
        return {
            "simulator": "CARLA",
            "is_connected": self.is_connected,
            "host": self.host,
            "port": self.port,
        }

    def close(self) -> None:
        self.is_connected = False
        self.world = None
        self.client = None
