"""
World State Helpers and Temporal History Tracker.

Keeps individual WorldState snapshots immutable, while providing a clean
temporal history buffer for simulation monitoring, rolling telemetry,
and state transition tracking.
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional
from perception.schemas import WorldState


class WorldStateHistory:
    """
    Rolling temporal buffer of WorldState snapshots.
    Separate from the immutable WorldState instances to avoid hidden mutable state.
    """
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self._history: Deque[WorldState] = deque(maxlen=max_length)

    def add(self, state: WorldState) -> None:
        """Append a new snapshot to history."""
        self._history.append(state)

    def clear(self) -> None:
        """Clear all historical snapshots."""
        self._history.clear()

    @property
    def latest(self) -> Optional[WorldState]:
        """Return the most recent WorldState snapshot or None if empty."""
        return self._history[-1] if self._history else None

    def __len__(self) -> int:
        return len(self._history)

    def get_average_fps(self) -> float:
        """Compute rolling average FPS across buffered snapshots."""
        if not self._history:
            return 0.0
        fps_values = [s.metadata.fps for s in self._history if s.metadata.fps > 0]
        return float(sum(fps_values) / len(fps_values)) if fps_values else 0.0

    def get_average_processing_time_ms(self) -> float:
        """Compute rolling average frame processing time in ms."""
        if not self._history:
            return 0.0
        proc_times = [s.metadata.processing_time_ms for s in self._history]
        return float(sum(proc_times) / len(proc_times)) if proc_times else 0.0

    def get_recent_states(self, count: int = 10) -> List[WorldState]:
        """Return the most recent N states as a list."""
        c = min(count, len(self._history))
        return list(self._history)[-c:]


def validate_world_state(state: WorldState) -> bool:
    """
    Verify the integrity of a WorldState instance.
    Ensures all sub-contracts are well-formed.
    """
    if state.frame_id < 0:
        raise ValueError(f"Invalid frame_id: {state.frame_id}")
    if state.image_width <= 0 or state.image_height <= 0:
        raise ValueError(f"Invalid image dimensions: {state.image_width}x{state.image_height}")
    if state.road_condition.score < 0.0 or state.road_condition.score > 100.0:
        raise ValueError(f"Road condition score out of bounds: {state.road_condition.score}")
    if state.traffic.traffic_density < 0.0 or state.traffic.traffic_density > 1.0:
        raise ValueError(f"Traffic density out of bounds: {state.traffic.traffic_density}")
    return True
