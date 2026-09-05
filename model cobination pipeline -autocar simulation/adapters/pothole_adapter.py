"""
Pothole Model Adapter.

Converts raw pothole detector outputs (from Res2Net or future YOLO-Pothole)
into standardized List[PotholeDetection] objects.
Decouples pothole model architectures from perception fusion and simulation.
"""

from typing import Any, Dict, List, Optional
from perception.schemas import PotholeDetection, PotholeSeverityCategory, BoundingBox


class PotholeAdapter:
    """
    Adapter converting raw pothole detector output into List[PotholeDetection].
    Seamlessly handles both single-bbox outputs (Res2Net) and multi-bbox outputs (future YOLO).
    """
    def __init__(
        self,
        min_confidence: float = 0.40,
        small_max_relative_area: float = 0.015,
        medium_max_relative_area: float = 0.045,
    ):
        self.min_confidence = min_confidence
        self.small_max_relative_area = small_max_relative_area
        self.medium_max_relative_area = medium_max_relative_area

    def _estimate_severity(self, relative_area: float) -> PotholeSeverityCategory:
        """Categorize estimated severity based on relative surface area in the camera view."""
        if relative_area <= self.small_max_relative_area:
            return PotholeSeverityCategory.SMALL
        elif relative_area <= self.medium_max_relative_area:
            return PotholeSeverityCategory.MEDIUM
        else:
            return PotholeSeverityCategory.LARGE

    def convert(self, raw_output: Any) -> List[PotholeDetection]:
        """
        Convert raw model output to List[PotholeDetection].
        Accepts:
          - Dict with 'pothole_boxes' and 'orig_shape'
          - List of pothole bounding box dictionaries
        """
        if raw_output is None:
            return []

        raw_boxes: List[Dict[str, Any]] = []
        orig_shape = (720, 1280)

        if isinstance(raw_output, dict):
            raw_boxes = raw_output.get("pothole_boxes", [])
            orig_shape = raw_output.get("orig_shape", (720, 1280))
        elif isinstance(raw_output, list):
            raw_boxes = raw_output

        img_h, img_w = orig_shape[:2]
        total_frame_area = max(1.0, float(img_w * img_h))

        potholes: List[PotholeDetection] = []
        for idx, item in enumerate(raw_boxes):
            conf = float(item.get("confidence", 1.0))
            if conf < self.min_confidence:
                continue

            raw_box = item.get("bbox")
            if raw_box is None or len(raw_box) < 4:
                continue

            xmin, ymin, xmax, ymax = raw_box[:4]
            b_xmin = min(float(xmin), float(xmax))
            b_xmax = max(float(xmin), float(xmax))
            b_ymin = min(float(ymin), float(ymax))
            b_ymax = max(float(ymin), float(ymax))

            bbox = BoundingBox(
                xmin=b_xmin,
                ymin=b_ymin,
                xmax=b_xmax,
                ymax=b_ymax,
                is_normalized=False,
            )

            area_pix = float(bbox.area)
            rel_area = float(area_pix / total_frame_area)
            severity = self._estimate_severity(rel_area)

            pothole = PotholeDetection(
                pothole_id=idx + 1,
                confidence=float(conf),
                bbox=bbox,
                area_pixels=float(round(area_pix, 1)),
                relative_area=float(round(rel_area, 6)),
                estimated_severity=severity,
                road_association=0.0,  # Computed in perception/fusion.py
                estimated_distance=None,  # 2D bbox does not provide physical distance/depth
            )
            potholes.append(pothole)

        return potholes
