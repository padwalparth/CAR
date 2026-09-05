"""
Runtime Statistics Tracker.

Lightweight tracker for per-run performance metrics, frame latencies,
model failure tallies, and perception aggregates.
Maintains complete separation from WorldState.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from perception.schemas import ModelStatus, WorldState


@dataclass
class RuntimeStatistics:
    """
    Tracks runtime statistics across a pipeline execution session.
    """
    frames_processed: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    road_failures: int = 0
    traffic_failures: int = 0
    pothole_failures: int = 0
    total_potholes_detected: int = 0
    total_traffic_objects_detected: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def start(self) -> None:
        """Mark start time."""
        self.start_time = time.time()
        self.end_time = None

    def stop(self) -> None:
        """Mark completion time."""
        self.end_time = time.time()

    def record_frame(self, world_state: WorldState, latency_ms: float) -> None:
        """
        Record a processed frame and inspect perception status.
        """
        self.frames_processed += 1
        self.latencies_ms.append(max(0.0, float(latency_ms)))

        status = world_state.perception_status
        if status.road != ModelStatus.OK.value:
            self.road_failures += 1
        if status.traffic != ModelStatus.OK.value:
            self.traffic_failures += 1
        if status.pothole != ModelStatus.OK.value:
            self.pothole_failures += 1

        if world_state.perception_status.is_all_ok():
            self.successful_frames += 1
        else:
            self.failed_frames += 1

        self.total_potholes_detected += world_state.potholes.count
        self.total_traffic_objects_detected += world_state.traffic.total_objects

    def record_frame_failure(self) -> None:
        """Record an unhandled frame failure."""
        self.frames_processed += 1
        self.failed_frames += 1

    @property
    def total_elapsed_seconds(self) -> float:
        """Total wall-clock runtime in seconds."""
        t_end = self.end_time if self.end_time is not None else time.time()
        return max(0.0001, t_end - self.start_time)

    @property
    def avg_latency_ms(self) -> float:
        """Average processing latency per frame in ms."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def min_latency_ms(self) -> float:
        """Minimum processing latency in ms."""
        if not self.latencies_ms:
            return 0.0
        return min(self.latencies_ms)

    @property
    def max_latency_ms(self) -> float:
        """Maximum processing latency in ms."""
        if not self.latencies_ms:
            return 0.0
        return max(self.latencies_ms)

    @property
    def avg_fps(self) -> float:
        """Effective processing frames per second."""
        if self.frames_processed == 0:
            return 0.0
        if self.avg_latency_ms > 0:
            return 1000.0 / self.avg_latency_ms
        return self.frames_processed / self.total_elapsed_seconds

    def to_dict(self, sim_enabled: bool = False, vis_enabled: bool = False) -> Dict[str, Any]:
        """Convert statistics to structured dictionary."""
        return {
            "frames_processed": self.frames_processed,
            "successful_frames": self.successful_frames,
            "failed_frames": self.failed_frames,
            "elapsed_seconds": round(self.total_elapsed_seconds, 3),
            "average_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "average_fps": round(self.avg_fps, 2),
            "failures": {
                "road": self.road_failures,
                "traffic": self.traffic_failures,
                "pothole": self.pothole_failures,
            },
            "scene_aggregates": {
                "total_potholes_detected": self.total_potholes_detected,
                "total_traffic_objects_detected": self.total_traffic_objects_detected,
            },
            "simulation_enabled": sim_enabled,
            "visualization_enabled": vis_enabled,
        }

    def format_summary(self, sim_enabled: bool = False, vis_enabled: bool = False) -> str:
        """Format human-readable run summary text."""
        sim_str = "enabled" if sim_enabled else "disabled"
        vis_str = "enabled" if vis_enabled else "disabled"

        return f"""
========================================
RUN SUMMARY
========================================
Frames processed:       {self.frames_processed}
Successful frames:      {self.successful_frames}
Failed frames:          {self.failed_frames}
Average FPS:            {self.avg_fps:.1f}
Average latency:        {self.avg_latency_ms:.1f} ms
Min latency:            {self.min_latency_ms:.1f} ms
Max latency:            {self.max_latency_ms:.1f} ms
Road failures:          {self.road_failures}
Traffic failures:       {self.traffic_failures}
Pothole failures:       {self.pothole_failures}
Total potholes:         {self.total_potholes_detected}
Total traffic objects:  {self.total_traffic_objects_detected}
Simulation:             {sim_str}
Visualization:          {vis_str}
========================================
""".strip()
