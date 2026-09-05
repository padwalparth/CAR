"""
Road Condition Analyzer.

Computes a deterministic 0-100 road quality score and simulation impact factors
(estimated friction factor and speed restriction factor) by combining road mask
coverage and pothole severity.

Explicit Note:
Friction and speed restriction values are estimated simulation parameters,
NOT measured physical quantities.
"""

from typing import Any, Dict, List, Optional
from perception.schemas import (
    RoadCondition,
    RoadConditionCategory,
    RoadMask,
    PotholeDetection,
    PotholeSeverityCategory,
    TrafficState,
)


class RoadConditionAnalyzer:
    """
    Pure analytical engine for evaluating road health.
    Computes a deterministic 0 to 100 score based on configurable weights and thresholds.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Extract weights from config
        rc_cfg = self.config.get("road_condition", {})
        weights = rc_cfg.get("weights", {})
        self.base_score = float(weights.get("base_score", 100.0))
        self.pothole_count_penalty = float(weights.get("pothole_count_penalty", 5.0))
        self.pothole_area_penalty = float(weights.get("pothole_relative_area_penalty", 250.0))
        self.severe_pothole_penalty = float(weights.get("severe_pothole_penalty", 10.0))
        self.road_coverage_penalty = float(weights.get("road_coverage_penalty", 15.0))

        # Extract score thresholds from config
        thresh = rc_cfg.get("thresholds", {})
        self.good_min = float(thresh.get("good_min_score", 80.0))
        self.moderate_min = float(thresh.get("moderate_min_score", 50.0))
        self.poor_min = float(thresh.get("poor_min_score", 25.0))

        # Extract simulation parameter mapping from config
        sim_map = rc_cfg.get("simulation_mapping", {})
        self.friction_map = sim_map.get(
            "friction_factors",
            {
                "GOOD": 1.00,
                "MODERATE": 0.85,
                "POOR": 0.65,
                "CRITICAL": 0.40,
            },
        )
        self.speed_map = sim_map.get(
            "speed_restrictions",
            {
                "GOOD": 1.00,
                "MODERATE": 0.75,
                "POOR": 0.50,
                "CRITICAL": 0.25,
            },
        )

    def _determine_category(self, score: float) -> RoadConditionCategory:
        """Map 0-100 score to RoadConditionCategory."""
        if score >= self.good_min:
            return RoadConditionCategory.GOOD
        elif score >= self.moderate_min:
            return RoadConditionCategory.MODERATE
        elif score >= self.poor_min:
            return RoadConditionCategory.POOR
        else:
            return RoadConditionCategory.CRITICAL

    def analyze(
        self,
        road_mask: Optional[RoadMask],
        potholes: List[PotholeDetection],
        traffic: Optional[TrafficState] = None,
    ) -> RoadCondition:
        """
        Compute road condition score and simulation impact parameters.
        
        Formula:
          score = base_score
                - (pothole_count * pothole_count_penalty)
                - (total_relative_area * pothole_area_penalty)
                - (large_potholes_count * severe_pothole_penalty)
                - road_coverage_penalty (if drivable road is severely degraded)
          score clamped to [0.0, 100.0].
        """
        # Pothole statistics
        pothole_count = len(potholes) if potholes else 0
        total_relative_area = sum(p.relative_area for p in potholes) if potholes else 0.0
        large_potholes_count = sum(
            1 for p in potholes if p.estimated_severity == PotholeSeverityCategory.LARGE
        ) if potholes else 0

        # Calculate individual penalties
        count_pen = float(pothole_count * self.pothole_count_penalty)
        area_pen = float(total_relative_area * self.pothole_area_penalty)
        severe_pen = float(large_potholes_count * self.severe_pothole_penalty)

        # Road coverage penalty: if road mask is detected but ratio is below expected threshold (e.g. < 0.25)
        coverage_pen = 0.0
        if road_mask is not None and road_mask.mask is not None:
            if road_mask.road_area_ratio < 0.25:
                degraded_fraction = (0.25 - road_mask.road_area_ratio) / 0.25
                coverage_pen = float(degraded_fraction * self.road_coverage_penalty)

        total_penalties = count_pen + area_pen + severe_pen + coverage_pen
        raw_score = self.base_score - total_penalties
        final_score = float(max(0.0, min(100.0, round(raw_score, 2))))

        category = self._determine_category(final_score)
        friction_factor = float(self.friction_map.get(category.value, 1.00))
        speed_factor = float(self.speed_map.get(category.value, 1.00))

        details = {
            "base_score": self.base_score,
            "pothole_count": pothole_count,
            "pothole_count_penalty": round(count_pen, 2),
            "total_relative_pothole_area": round(total_relative_area, 6),
            "pothole_area_penalty": round(area_pen, 2),
            "large_potholes_count": large_potholes_count,
            "severe_pothole_penalty": round(severe_pen, 2),
            "road_coverage_penalty": round(coverage_pen, 2),
            "total_penalties": round(total_penalties, 2),
        }

        return RoadCondition(
            score=final_score,
            category=category,
            estimated_friction_factor=friction_factor,
            speed_restriction_factor=speed_factor,
            details=details,
        )
