"""
Overlay Renderer.

Renders read-only perception annotations over camera/video frames:
1. Translucent drivable road mask overlay.
2. Traffic object bounding boxes with IDD class labels, confidence, track IDs, and road association.
3. Pothole bounding boxes color-coded by severity category.
4. Top HUD telemetry bar with road condition score, congestion, FPS, latency, and system health status.

Does NOT modify WorldState.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from perception.schemas import (
    PotholeSeverityCategory,
    RoadConditionCategory,
    WorldState,
)

# Standard BGR color constants
COLOR_ROAD_OVERLAY = (0, 200, 100)      # Translucent Emerald Green
COLOR_VEHICLE_ON_ROAD = (0, 255, 0)     # Bright Green
COLOR_VEHICLE_OFF_ROAD = (0, 165, 255)  # Orange
COLOR_PEDESTRIAN = (255, 0, 255)        # Magenta
COLOR_POTHOLE_SMALL = (0, 255, 255)     # Yellow
COLOR_POTHOLE_MEDIUM = (0, 140, 255)    # Deep Orange
COLOR_POTHOLE_LARGE = (0, 0, 255)       # Red
COLOR_HUD_BG = (20, 20, 20)             # Near Black
COLOR_TEXT_WHITE = (255, 255, 255)      # White
COLOR_STATUS_OK = (0, 255, 0)           # Green
COLOR_STATUS_WARN = (0, 140, 255)       # Orange
COLOR_STATUS_ERR = (0, 0, 255)          # Red


class OverlayRenderer:
    """
    Renders visual overlays onto camera frames using WorldState.
    Operates in headless mode using Pillow and NumPy without requiring an active display.
    """
    def __init__(
        self,
        road_alpha: float = 0.30,
        box_thickness: int = 2,
        show_hud: bool = True,
        show_road_mask: bool = True,
        show_traffic: bool = True,
        show_potholes: bool = True,
    ):
        self.road_alpha = road_alpha
        self.box_thickness = box_thickness
        self.show_hud = show_hud
        self.show_road_mask = show_road_mask
        self.show_traffic = show_traffic
        self.show_potholes = show_potholes

    def render(self, frame: np.ndarray, world_state: WorldState) -> np.ndarray:
        """
        Draw perception annotations on a copy of the frame.
        
        Args:
          frame: BGR numpy ndarray (H, W, 3)
          world_state: WorldState perception snapshot (Read-Only)
          
        Returns:
          Annotated BGR numpy ndarray of identical shape.
        """
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return frame

        h, w = frame.shape[:2]
        output_frame = frame.copy()

        # 1. Render Road Mask Overlay
        if self.show_road_mask and world_state.road is not None and world_state.road.mask is not None:
            output_frame = self._render_road_mask(output_frame, world_state.road.mask, w, h)

        # Convert to PIL Image for high-quality text and box rendering
        # Convert BGR -> RGB for PIL
        rgb_img = Image.fromarray(output_frame[:, :, ::-1])
        draw = ImageDraw.Draw(rgb_img, "RGBA")

        # 2. Render Traffic Objects
        if self.show_traffic and world_state.traffic is not None:
            all_traffic = world_state.traffic.vehicles + world_state.traffic.pedestrians + world_state.traffic.other_objects
            for det in all_traffic:
                self._draw_detection(draw, det, w, h)

        # 3. Render Potholes
        if self.show_potholes and world_state.potholes is not None:
            for pot in world_state.potholes.potholes:
                self._draw_pothole(draw, pot, w, h)

        # 4. Render HUD Telemetry
        if self.show_hud:
            self._draw_hud(draw, world_state, w, h)

        # Convert back from PIL RGB -> NumPy BGR
        res_rgb = np.array(rgb_img)
        return res_rgb[:, :, ::-1].copy()

    def _render_road_mask(self, frame: np.ndarray, mask: Any, target_w: int, target_h: int) -> np.ndarray:
        """Blend translucent color over road mask pixels."""
        mask_arr = np.asarray(mask)
        if len(mask_arr.shape) > 2:
            mask_arr = np.squeeze(mask_arr)

        if mask_arr.size == 0 or not np.any(mask_arr):
            return frame

        # Resize mask to frame dimensions if necessary
        mh, mw = mask_arr.shape[:2]
        if (mw, mh) != (target_w, target_h):
            pil_mask = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255)
            resized_mask = np.array(pil_mask.resize((target_w, target_h), Image.Resampling.NEAREST)) > 0
        else:
            resized_mask = mask_arr > 0

        # Apply translucent color blend in place
        overlay = frame.copy()
        overlay[resized_mask] = COLOR_ROAD_OVERLAY
        # Blend: (1 - alpha) * frame + alpha * overlay
        alpha = float(max(0.0, min(1.0, self.road_alpha)))
        blended = (1.0 - alpha) * frame[resized_mask] + alpha * overlay[resized_mask]
        frame[resized_mask] = blended.astype(np.uint8)
        return frame

    def _draw_detection(self, draw: ImageDraw.ImageDraw, det: Any, img_w: int, img_h: int) -> None:
        """Draw bounding box and pill label for a traffic object."""
        box = det.bbox.to_pixel(img_w, img_h) if det.bbox.is_normalized else det.bbox
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax

        is_pedestrian = det.class_name in ("person", "rider", "animal")
        if is_pedestrian:
            color = (255, 0, 255)
        elif det.road_association >= 0.25:
            color = (0, 255, 0)
        else:
            color = (255, 140, 0)

        # Draw box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=self.box_thickness)

        # Build label text
        track_str = f"#{det.track_id} " if det.track_id is not None else ""
        assoc_str = " (Road)" if det.road_association >= 0.25 else ""
        label = f"{track_str}{det.class_name} {det.confidence:.2f}{assoc_str}"

        # Draw label background pill
        label_y = max(0, ymin - 16)
        label_w = len(label) * 7 + 6
        draw.rectangle([xmin, label_y, xmin + label_w, label_y + 16], fill=color)
        draw.text((xmin + 3, label_y + 1), label, fill=(0, 0, 0))

    def _draw_pothole(self, draw: ImageDraw.ImageDraw, pot: Any, img_w: int, img_h: int) -> None:
        """Draw bounding box and label for a pothole hazard."""
        box = pot.bbox.to_pixel(img_w, img_h) if pot.bbox.is_normalized else pot.bbox
        xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax

        sev = pot.estimated_severity.value if hasattr(pot.estimated_severity, "value") else str(pot.estimated_severity)
        if sev == "LARGE":
            color = (255, 0, 0)
        elif sev == "MEDIUM":
            color = (255, 140, 0)
        else:
            color = (255, 220, 0)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=max(2, self.box_thickness + 1))

        label = f"POTHOLE #{pot.pothole_id} [{sev}]"
        label_y = max(0, ymin - 16)
        label_w = len(label) * 7 + 6
        draw.rectangle([xmin, label_y, xmin + label_w, label_y + 16], fill=color)
        draw.text((xmin + 3, label_y + 1), label, fill=(0, 0, 0))

    def _draw_hud(self, draw: ImageDraw.ImageDraw, ws: WorldState, img_w: int, img_h: int) -> None:
        """Draw comprehensive top HUD telemetry bar."""
        hud_height = 54
        # Semi-transparent dark banner
        draw.rectangle([0, 0, img_w, hud_height], fill=(20, 20, 20, 215))

        # Line 1 metrics
        fps_val = f"{ws.metadata.fps:.1f}" if ws.metadata.fps > 0 else "N/A"
        lat_val = f"{ws.metadata.processing_time_ms:.1f}ms" if ws.metadata.processing_time_ms > 0 else "N/A"
        line1 = f"FRAME: {ws.frame_id} | TIME: {ws.timestamp:.2f}s | FPS: {fps_val} | LATENCY: {lat_val}"

        # Line 2 metrics
        road_cat = ws.road_condition.category.value if hasattr(ws.road_condition.category, "value") else str(ws.road_condition.category)
        road_score = f"{ws.road_condition.score:.1f}" if ws.road_condition.score is not None else "N/A"
        cong_level = ws.traffic.congestion_level.value if hasattr(ws.traffic.congestion_level, "value") else str(ws.traffic.congestion_level)
        v_count = ws.traffic.vehicle_count
        p_count = ws.traffic.pedestrian_count
        pot_count = ws.potholes.count
        line2 = f"ROAD: {road_cat} ({road_score}/100) | CONGESTION: {cong_level} | VEHICLES: {v_count} | PEDS: {p_count} | POTHOLES: {pot_count}"

        # Status badge
        is_ok = ws.perception_status.is_all_ok()
        status_text = "PERCEPTION: OK" if is_ok else "PERCEPTION: DEGRADED"
        status_color = (0, 255, 0) if is_ok else (255, 60, 60)

        draw.text((12, 6), line1, fill=(255, 255, 255))
        draw.text((12, 28), line2, fill=(200, 230, 255))
        draw.text((img_w - 180, 6), status_text, fill=status_color)
