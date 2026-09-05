"""
Telemetry Dashboard.

Lightweight read-only telemetry dashboard for WorldState inspection.
Supports:
1. Formatted ASCII / text console dashboard (render_text)
2. Graphic telemetry panel image (render_image)

Does NOT recompute metrics or modify state.
"""

from typing import Any, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw

from perception.schemas import WorldState


class TelemetryDashboard:
    """
    Consumes WorldState and generates human-readable telemetry reports.
    """
    def __init__(self, title: str = "SIH 2026 ROAD PERCEPTION DASHBOARD"):
        self.title = title

    def render_text(self, world_state: WorldState) -> str:
        """
        Generate structured text telemetry dashboard.
        
        Args:
          world_state: WorldState perception snapshot (Read-Only)
          
        Returns:
          Formatted multi-line string.
        """
        meta = world_state.metadata
        road = world_state.road
        cond = world_state.road_condition
        traffic = world_state.traffic
        potholes = world_state.potholes
        status = world_state.perception_status

        fps_str = f"{meta.fps:.1f}" if meta.fps > 0 else "N/A"
        total_time_str = f"{meta.processing_time_ms:.1f} ms" if meta.processing_time_ms > 0 else "N/A"

        road_cat = cond.category.value if hasattr(cond.category, "value") else str(cond.category)
        cong_level = traffic.congestion_level.value if hasattr(traffic.congestion_level, "value") else str(traffic.congestion_level)

        class_breakdown = ", ".join(f"{k}: {v}" for k, v in traffic.vehicle_counts_by_class.items()) if traffic.vehicle_counts_by_class else "None"
        sev_breakdown = ", ".join(f"{k}: {v}" for k, v in potholes.severity_counts.items()) if potholes.severity_counts else "None"

        error_lines = ""
        if status.errors:
            error_lines = "\n  ACTIVE ERRORS:\n" + "\n".join(f"    - [{k}]: {v}" for k, v in status.errors.items())

        text = f"""
================================================================================
  {self.title}
================================================================================
[FRAME & PERFORMANCE]
  Frame ID:           {world_state.frame_id}
  Timestamp:          {world_state.timestamp:.3f} s
  FPS:                {fps_str}
  Total Latency:      {total_time_str}
  Latency Breakdown:  Road: {meta.road_inference_ms:.1f}ms | Traffic: {meta.traffic_inference_ms:.1f}ms | Pothole: {meta.pothole_inference_ms:.1f}ms | Fusion: {meta.fusion_ms:.1f}ms

[ROAD HEALTH]
  Condition Category: {road_cat}
  Condition Score:    {cond.score:.1f} / 100
  Drivable Area Ratio:{road.road_area_ratio * 100:.1f}%
  Sim. Friction Factor:{cond.estimated_friction_factor:.2f} (Estimated simulation parameter)
  Sim. Speed Limit:   {cond.speed_restriction_factor * 100:.0f}% (Estimated simulation parameter)

[TRAFFIC STATE]
  Total Objects:      {traffic.total_objects}
  Vehicles:           {traffic.vehicle_count}
  Pedestrians:        {traffic.pedestrian_count}
  Traffic Density:    {traffic.traffic_density:.3f} (Spatial occupancy [0.0 - 1.0])
  Congestion Level:   {cong_level}
  Class Breakdown:    {class_breakdown}

[POTHOLE HAZARDS]
  Total Potholes:     {potholes.count}
  Total Rel. Area:    {potholes.total_relative_area * 100:.3f}% of scene
  Severity Breakdown: {sev_breakdown}

[SYSTEM HEALTH]
  Perception Status:  Road: {status.road} | Traffic: {status.traffic} | Pothole: {status.pothole}{error_lines}
================================================================================
"""
        return text.strip()

    def render_image(self, world_state: WorldState, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Generate graphic telemetry dashboard image as BGR numpy ndarray.
        """
        img = Image.new("RGB", (width, height), (20, 24, 28))
        draw = ImageDraw.Draw(img)

        # Header banner
        draw.rectangle([0, 0, width, 40], fill=(12, 15, 18))
        draw.text((12, 12), self.title, fill=(255, 255, 255))

        # Status badge
        is_ok = world_state.perception_status.is_all_ok()
        badge_text = "STATUS: OK" if is_ok else "STATUS: DEGRADED"
        badge_color = (0, 255, 0) if is_ok else (255, 60, 60)
        draw.text((width - 160, 12), badge_text, fill=badge_color)

        # Content lines
        meta = world_state.metadata
        cond = world_state.road_condition
        traffic = world_state.traffic
        potholes = world_state.potholes

        road_cat = cond.category.value if hasattr(cond.category, "value") else str(cond.category)
        cong_level = traffic.congestion_level.value if hasattr(traffic.congestion_level, "value") else str(traffic.congestion_level)

        lines = [
            ("[FRAME METRICS]", (100, 200, 255)),
            (f"Frame ID: {world_state.frame_id}   |   Time: {world_state.timestamp:.2f}s   |   FPS: {meta.fps:.1f}", (240, 240, 240)),
            (f"Processing Time: {meta.processing_time_ms:.1f} ms  (Fusion: {meta.fusion_ms:.1f} ms)", (200, 200, 200)),
            ("", (0, 0, 0)),
            ("[ROAD CONDITION]", (100, 200, 255)),
            (f"Quality: {road_cat} ({cond.score:.1f} / 100)", (0, 255, 120) if road_cat == "GOOD" else (255, 200, 0)),
            (f"Friction Factor: {cond.estimated_friction_factor:.2f}   |   Speed Factor: {cond.speed_restriction_factor:.2f}", (200, 200, 200)),
            ("", (0, 0, 0)),
            ("[TRAFFIC STATE]", (100, 200, 255)),
            (f"Vehicles: {traffic.vehicle_count}   |   Pedestrians: {traffic.pedestrian_count}   |   Total: {traffic.total_objects}", (240, 240, 240)),
            (f"Congestion: {cong_level}   |   Spatial Density: {traffic.traffic_density:.3f}", (240, 240, 240)),
            ("", (0, 0, 0)),
            ("[POTHOLE HAZARDS]", (100, 200, 255)),
            (f"Pothole Count: {potholes.count}   |   Total Coverage: {potholes.total_relative_area * 100:.3f}%", (240, 240, 240)),
            (f"Severities: Small: {potholes.severity_counts.get('SMALL', 0)}, Med: {potholes.severity_counts.get('MEDIUM', 0)}, Large: {potholes.severity_counts.get('LARGE', 0)}", (200, 200, 200)),
        ]

        y = 52
        for text, color in lines:
            if text:
                draw.text((16, y), text, fill=color)
                y += 22
            else:
                y += 10

        # Convert PIL RGB -> NumPy BGR
        res_rgb = np.array(img)
        return res_rgb[:, :, ::-1].copy()
