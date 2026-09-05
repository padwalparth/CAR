"""
SUMO Simulator Integration Adapter (Future Extension Stub).

Implements SimulatorInterface for Eclipse SUMO traffic simulator.
Uses lazy imports to prevent runtime crashes if traci / sumolib is not installed.
"""

from typing import Any, Dict, Optional
from simulation.simulator_interface import SimulatorInterface


class SumoAdapter(SimulatorInterface):
    """
    Adapter bridging SimulationScenario to SUMO via TraCI.
    Requires Eclipse SUMO and the 'traci' Python package.
    """
    def __init__(self, sumo_binary: str = "sumo", config_file: str = "simulation/sumo.sumocfg", port: int = 8813):
        self.sumo_binary = sumo_binary
        self.config_file = config_file
        self.port = port
        self.is_connected = False
        self.traci = None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Connect to SUMO TraCI instance.
        Raises ImportError with clear instructions if traci is missing.
        """
        if config:
            sumo_cfg = config.get("sumo", {})
            self.sumo_binary = str(sumo_cfg.get("binary", self.sumo_binary))
            self.config_file = str(sumo_cfg.get("config_file", self.config_file))
            self.port = int(sumo_cfg.get("port", self.port))

        try:
            import traci  # type: ignore
            self.traci = traci
        except ImportError as e:
            raise ImportError(
                "SUMO TraCI library ('traci') is not installed. "
                "Install Eclipse SUMO and set SUMO_HOME environment variable, "
                "or install with 'pip install traci'. For local testing, use MockSimulator."
            ) from e

        # In real usage: traci.start([self.sumo_binary, "-c", self.config_file, "--port", str(self.port)])
        self.is_connected = True

    def reset(self) -> None:
        if not self.is_connected:
            raise RuntimeError("SumoAdapter is not connected.")

    def update(self, scenario: Any) -> None:
        if not self.is_connected:
            raise RuntimeError("SumoAdapter is not connected. Call initialize() first.")
        # Future mapping: adjust SUMO traffic flows and speed limits based on scenario

    def step(self, delta_time: float = 0.033) -> None:
        if not self.is_connected:
            raise RuntimeError("SumoAdapter is not connected.")
        if self.traci is not None:
            self.traci.simulationStep()

    def get_state(self) -> Dict[str, Any]:
        return {
            "simulator": "SUMO",
            "is_connected": self.is_connected,
            "port": self.port,
        }

    def close(self) -> None:
        if self.is_connected and self.traci is not None:
            try:
                self.traci.close()
            except Exception:
                pass
        self.is_connected = False
