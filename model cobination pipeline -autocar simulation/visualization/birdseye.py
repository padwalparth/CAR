"""
Approximate Image-Space Bird's-Eye View Renderer.

Renders an approximate 2D top-down perspective of the drivable road corridor,
traffic objects, and pothole hazards based on normalized camera coordinates.

IMPORTANT:
This view is an APPROXIMATE image-space projection for visualization and debugging.
It is NOT a physically calibrated world projection (metric scale requires camera homography).
"""

from typing import Any, Tuple
import numpy as np
from PIL import Image, ImageDraw

from perception.schemas import WorldState, PotholeSeverityCategory


class BirdseyeRenderer:
    """
    Renders an approximate top-down 2D map from WorldState.
    Operates headlessly using Pillow and NumPy.
    """
    def __init__(
        self,
        output_width: int = 400,
        output_height: int = 600,
        bg_color: Tuple[int, int, int] = (25, 28, 32),
        road_color: Tuple[int, int, int] = (45, 52, 60),
    ):
        self.width = output_width
        self.height = output_height
        self.bg_color = bg_color
        self.road_color = road_color

    def render(self, world_state: WorldState) -> np.ndarray:
        """
        Generate top-down bird's-eye view image.
        
        Args:
          world_state: WorldState perception snapshot (Read-Only)
          
        Returns:
          BGR numpy ndarray of shape (output_height, output_width, 3).
        """
        w, h = self.width, self.height
        img = Image.new("RGB", (w, h), self.bg_color)
        draw = ImageDraw.Draw(img)

        # 1. Draw Road Corridor (Trapezoid approximating perspective projection)
        # Horizon at top (narrower), ego-vehicle base at bottom (wider)
        road_ratio = world_state.road.road_area_ratio if world_state.road is not None else 0.40
        road_ratio = max(0.15, min(0.95, road_ratio))

        top_half_w = int(w * 0.20 * (road_ratio / 0.40))
        bot_half_w = int(w * 0.42 * (road_ratio / 0.40))
        cx = w // 2

        corridor_pts = [
            (cx - top_half_w, 60),
            (cx + top_half_w, 60),
            (cx + bot_half_w, h - 40),
            (cx - bot_half_w, h - 40),
        ]
        draw.polygon(corridor_pts, fill=self.road_color, outline=(80, 90, 100))

        # Draw dashed center lane line
        for y in range(70, h - 40, 30):
            draw.line([(cx, y), (cx, min(y + 15, h - 40))], fill=(180, 180, 180), width=2)

        # 2. Draw Ego Vehicle Marker at bottom center
        ego_w, ego_h = 24, 38
        ego_x0 = cx - ego_w // 2
        ego_y0 = h - 40 - ego_h
        draw.rectangle([ego_x0, ego_y0, ego_x0 + ego_w, ego_y0 + ego_h], fill=(0, 200, 255), outline=(255, 255, 255))
        draw.text((ego_x0 + 4, ego_y0 + 12), "EGO", fill=(0, 0, 0))

        # 3. Draw Traffic Objects
        img_w = max(1.0, float(world_state.image_width))
        img_h = max(1.0, float(world_state.image_height))

        if world_state.traffic is not None:
            all_traffic = world_state.traffic.vehicles + world_state.traffic.pedestrians
            for det in all_traffic:
                norm_box = det.bbox.to_normalized(int(img_w), int(img_h))
                bc_x, bc_y = norm_box.bottom_center

                # Map normalized camera coordinates to top-down view
                # x maps across scene width
                # y in camera view (0 top horizon, 1 near ego) maps to top-down depth
                proj_x = int(w * bc_x)
                proj_y = int(60 + (h - 110) * bc_y)

                is_ped = det.class_name in ("person", "rider", "animal")
                color = (255, 0, 255) if is_ped else (0, 255, 120)

                # Draw agent box
                bw, bh = (10, 10) if is_ped else (18, 26)
                draw.rectangle([proj_x - bw // 2, proj_y - bh // 2, proj_x + bw // 2, proj_y + bh // 2], fill=color, outline=(255, 255, 255))

                track_label = f"#{det.track_id}" if det.track_id is not None else det.class_name[:3]
                draw.text((proj_x - 10, proj_y - bh // 2 - 12), track_label, fill=(255, 255, 255))

        # 4. Draw Pothole Hazards
        if world_state.potholes is not None:
            for pot in world_state.potholes.potholes:
                norm_box = pot.bbox.to_normalized(int(img_w), int(img_h))
                cx_n, cy_n = norm_box.center

                proj_x = int(w * cx_n)
                proj_y = int(60 + (h - 110) * cy_n)

                sev = pot.estimated_severity.value if hasattr(pot.estimated_severity, "value") else str(pot.estimated_severity)
                if sev == "LARGE":
                    color = (255, 0, 0)
                    r = 8
                elif sev == "MEDIUM":
                    color = (255, 140, 0)
                    r = 6
                else:
                    color = (255, 220, 0)
                    r = 4

                draw.ellipse([proj_x - r, proj_y - r, proj_x + r, proj_y + r], fill=color, outline=(255, 255, 255))

        # 5. Header and Documentation Labels
        draw.rectangle([0, 0, w, 44], fill=(15, 18, 22))
        draw.text((10, 6), "BIRD'S-EYE VIEW (TOP-DOWN)", fill=(240, 240, 240))
        draw.text((10, 24), "Approx. image-space projection (un-calibrated)", fill=(140, 160, 180))

        # Footer road health summary
        cat = world_state.road_condition.category.value if hasattr(world_state.road_condition.category, "value") else str(world_state.road_condition.category)
        draw.rectangle([0, h - 26, w, h], fill=(15, 18, 22))
        draw.text((10, h - 20), f"ROAD: {cat} | SCORE: {world_state.road_condition.score:.1f}", fill=(200, 220, 240))

        # Convert PIL RGB to NumPy BGR
        res_rgb = np.array(img)
        return res_rgb[:, :, ::-1].copy()
