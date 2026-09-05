"""
Pothole Analyzer.

Consumes standardized List[PotholeDetection] and generates PotholesState.
Calculates count, total area, relative area, spatial distribution,
severity distribution, and road association statistics.

Decoupled from ML frameworks and simulator APIs.
"""

from typing import Any, Dict, List, Optional, Tuple
from perception.schemas import (
    PotholeDetection,
    PotholesState,
    PotholeSeverityCategory,
    RoadMask,
)
from perception.association import associate_pothole_with_road


class PotholeAnalysisSummary:
    """Detailed analytical breakdown of detected potholes."""
    def __init__(
        self,
        count: int,
        total_area_pixels: float,
        total_relative_area: float,
        average_relative_area: float,
        largest_pothole: Optional[PotholeDetection],
        severity_counts: Dict[str, int],
        road_associated_count: int,
        spatial_distribution: Dict[str, int],
    ):
        self.count = count
        self.total_area_pixels = total_area_pixels
        self.total_relative_area = total_relative_area
        self.average_relative_area = average_relative_area
        self.largest_pothole = largest_pothole
        self.severity_counts = severity_counts
        self.road_associated_count = road_associated_count
        self.spatial_distribution = spatial_distribution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total_area_pixels": round(self.total_area_pixels, 1),
            "total_relative_area": round(self.total_relative_area, 6),
            "average_relative_area": round(self.average_relative_area, 6),
            "largest_pothole_id": self.largest_pothole.pothole_id if self.largest_pothole else None,
            "severity_counts": self.severity_counts,
            "road_associated_count": self.road_associated_count,
            "spatial_distribution": self.spatial_distribution,
        }


class PotholeAnalyzer:
    """
    Pure analytical engine for pothole hazard perception.
    Evaluates pothole geometry, severity categories, and road surface coverage.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Severity thresholds from config
        p_sev = self.config.get("pothole_severity", {})
        self.small_max_rel_area = float(p_sev.get("small_max_relative_area", 0.015))
        self.medium_max_rel_area = float(p_sev.get("medium_max_relative_area", 0.045))

        # Association thresholds
        r_assoc = self.config.get("road_association", {})
        self.min_overlap = float(r_assoc.get("min_overlap", 0.30))
        self.center_weight = float(r_assoc.get("center_weight", 0.50))
        self.overlap_weight = float(r_assoc.get("overlap_weight", 0.50))

    def _determine_severity(self, rel_area: float) -> PotholeSeverityCategory:
        """Categorize severity based on relative image area."""
        if rel_area <= self.small_max_rel_area:
            return PotholeSeverityCategory.SMALL
        elif rel_area <= self.medium_max_rel_area:
            return PotholeSeverityCategory.MEDIUM
        else:
            return PotholeSeverityCategory.LARGE

    def analyze(
        self,
        potholes: List[PotholeDetection],
        img_width: int,
        img_height: int,
        road_mask: Optional[RoadMask] = None,
    ) -> Tuple[PotholesState, PotholeAnalysisSummary]:
        """
        Analyze a list of pothole detections.
        
        Args:
          potholes: Standardized List[PotholeDetection] (supports 0, 1, or N potholes)
          img_width: Frame pixel width
          img_height: Frame pixel height
          road_mask: Optional RoadMask for spatial road-grounding
          
        Returns:
          Tuple of (PotholesState, PotholeAnalysisSummary)
        """
        if not potholes:
            empty_state = PotholesState(
                potholes=[],
                count=0,
                total_area_pixels=0.0,
                total_relative_area=0.0,
                severity_counts={"SMALL": 0, "MEDIUM": 0, "LARGE": 0},
            )
            empty_summary = PotholeAnalysisSummary(
                count=0,
                total_area_pixels=0.0,
                total_relative_area=0.0,
                average_relative_area=0.0,
                largest_pothole=None,
                severity_counts={"SMALL": 0, "MEDIUM": 0, "LARGE": 0},
                road_associated_count=0,
                spatial_distribution={"left": 0, "center": 0, "right": 0},
            )
            return empty_state, empty_summary

        total_area_pixels = 0.0
        total_relative_area = 0.0
        severity_counts = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0}
        spatial_dist = {"left": 0, "center": 0, "right": 0}
        road_associated_count = 0
        largest_pothole: Optional[PotholeDetection] = None
        max_area = -1.0

        for p in potholes:
            # Associate with road mask if road_mask is provided and not already associated
            if road_mask is not None and road_mask.mask is not None and p.road_association == 0.0:
                is_assoc, conf = associate_pothole_with_road(
                    p.bbox,
                    road_mask.mask,
                    img_width,
                    img_height,
                    min_overlap=self.min_overlap,
                    center_weight=self.center_weight,
                    overlap_weight=self.overlap_weight,
                )
                p.road_association = conf

            if p.road_association >= self.min_overlap or (road_mask is None):
                road_associated_count += 1

            total_area_pixels += p.area_pixels
            total_relative_area += p.relative_area

            # Severity categorization
            sev = self._determine_severity(p.relative_area)
            p.estimated_severity = sev
            severity_counts[sev.value] = severity_counts.get(sev.value, 0) + 1

            # Largest pothole
            if p.area_pixels > max_area:
                max_area = p.area_pixels
                largest_pothole = p

            # Spatial distribution across image thirds (left, center, right)
            cx = p.center_x
            norm_cx = cx / max(1.0, float(img_width)) if not p.bbox.is_normalized else cx
            if norm_cx < 0.333:
                spatial_dist["left"] += 1
            elif norm_cx < 0.666:
                spatial_dist["center"] += 1
            else:
                spatial_dist["right"] += 1

        count = len(potholes)
        avg_relative_area = total_relative_area / max(1, count)

        state = PotholesState(
            potholes=potholes,
            count=count,
            total_area_pixels=float(round(total_area_pixels, 1)),
            total_relative_area=float(round(total_relative_area, 6)),
            severity_counts=severity_counts,
        )

        summary = PotholeAnalysisSummary(
            count=count,
            total_area_pixels=float(round(total_area_pixels, 1)),
            total_relative_area=float(round(total_relative_area, 6)),
            average_relative_area=float(round(avg_relative_area, 6)),
            largest_pothole=largest_pothole,
            severity_counts=severity_counts,
            road_associated_count=road_associated_count,
            spatial_distribution=spatial_dist,
        )

        return state, summary
