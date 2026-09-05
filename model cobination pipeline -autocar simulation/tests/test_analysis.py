"""
Unit tests for Analytical Engines: TrafficAnalyzer, PotholeAnalyzer, RoadConditionAnalyzer.
"""

import unittest
from perception.schemas import (
    Detection,
    BoundingBox,
    PotholeDetection,
    PotholeSeverityCategory,
    RoadMask,
    TrafficState,
    RoadConditionCategory,
    CongestionLevel,
)
from analysis.traffic_analysis import TrafficAnalyzer
from analysis.pothole_analysis import PotholeAnalyzer
from analysis.road_analysis import RoadConditionAnalyzer
from config.loader import load_pipeline_config


class TestTrafficAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = load_pipeline_config("config/pipeline.yaml")
        self.analyzer = TrafficAnalyzer(self.config)
        self.w = 1000
        self.h = 1000

    def test_empty_detection_list(self):
        state = self.analyzer.analyze([], self.w, self.h)
        self.assertEqual(state.vehicle_count, 0)
        self.assertEqual(state.pedestrian_count, 0)
        self.assertEqual(state.total_objects, 0)
        self.assertEqual(state.traffic_density, 0.0)
        self.assertEqual(state.congestion_level, CongestionLevel.LOW)

    def test_single_vehicle(self):
        det = Detection(
            object_id=1,
            class_name="car",
            class_id=4,
            confidence=0.9,
            bbox=BoundingBox(100, 100, 200, 200),
        )
        state = self.analyzer.analyze([det], self.w, self.h)
        self.assertEqual(state.vehicle_count, 1)
        self.assertEqual(state.pedestrian_count, 0)
        self.assertEqual(state.total_objects, 1)
        self.assertEqual(state.vehicle_counts_by_class.get("car"), 1)

    def test_multiple_vehicle_classes_and_pedestrians(self):
        dets = [
            Detection(1, "car", 4, 0.9, BoundingBox(10, 10, 50, 50)),
            Detection(2, "bus", 3, 0.85, BoundingBox(60, 60, 150, 150)),
            Detection(3, "autorickshaw", 1, 0.8, BoundingBox(200, 200, 260, 260)),
            Detection(4, "motorcycle", 6, 0.75, BoundingBox(300, 300, 340, 340)),
            Detection(5, "person", 7, 0.92, BoundingBox(400, 400, 420, 460)),
            Detection(6, "rider", 8, 0.88, BoundingBox(450, 450, 470, 500)),
            Detection(7, "traffic light", 9, 0.95, BoundingBox(500, 50, 520, 90)),
        ]
        state = self.analyzer.analyze(dets, self.w, self.h)
        self.assertEqual(state.vehicle_count, 4)
        self.assertEqual(state.pedestrian_count, 2)
        self.assertEqual(len(state.other_objects), 1)
        self.assertEqual(state.total_objects, 7)
        self.assertEqual(state.vehicle_counts_by_class["car"], 1)
        self.assertEqual(state.vehicle_counts_by_class["bus"], 1)
        self.assertEqual(state.vehicle_counts_by_class["autorickshaw"], 1)
        self.assertEqual(state.vehicle_counts_by_class["motorcycle"], 1)

    def test_congestion_levels(self):
        # LOW congestion: 1 vehicle (capacity 12 => density ~0.08)
        low_dets = [Detection(1, "car", 4, 0.9, BoundingBox(10, 10, 50, 50))]
        state_low = self.analyzer.analyze(low_dets, self.w, self.h)
        self.assertEqual(state_low.congestion_level, CongestionLevel.LOW)

        # MODERATE congestion: 4 vehicles (4/12 ~ 0.33)
        mod_dets = [
            Detection(i, "car", 4, 0.9, BoundingBox(i * 50, 500, i * 50 + 40, 580))
            for i in range(4)
        ]
        state_mod = self.analyzer.analyze(mod_dets, self.w, self.h)
        self.assertEqual(state_mod.congestion_level, CongestionLevel.MODERATE)

        # HEAVY congestion: 8 vehicles (8/12 ~ 0.67)
        heavy_dets = [
            Detection(i, "car", 4, 0.9, BoundingBox(i * 50, 500, i * 50 + 40, 580))
            for i in range(8)
        ]
        state_heavy = self.analyzer.analyze(heavy_dets, self.w, self.h)
        self.assertEqual(state_heavy.congestion_level, CongestionLevel.HEAVY)

        # GRIDLOCK congestion: 12 vehicles (12/12 = 1.0)
        grid_dets = [
            Detection(i, "truck", 13, 0.9, BoundingBox(i * 30, 400, i * 30 + 28, 550))
            for i in range(12)
        ]
        state_grid = self.analyzer.analyze(grid_dets, self.w, self.h)
        self.assertEqual(state_grid.congestion_level, CongestionLevel.GRIDLOCK)


class TestPotholeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = load_pipeline_config("config/pipeline.yaml")
        self.analyzer = PotholeAnalyzer(self.config)
        self.w = 1000
        self.h = 1000
        self.total_area = 1000 * 1000  # 1,000,000 pixels

    def test_zero_potholes(self):
        state, summary = self.analyzer.analyze([], self.w, self.h)
        self.assertEqual(state.count, 0)
        self.assertEqual(state.total_area_pixels, 0.0)
        self.assertEqual(state.total_relative_area, 0.0)
        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.largest_pothole)

    def test_single_small_pothole(self):
        # 100x100 = 10,000 pixels => rel area 0.01 (<= 0.015 => SMALL)
        box = BoundingBox(100, 600, 200, 700)
        p = PotholeDetection(1, 0.9, box, area_pixels=10000.0, relative_area=0.01)
        state, summary = self.analyzer.analyze([p], self.w, self.h)
        self.assertEqual(state.count, 1)
        self.assertEqual(state.severity_counts["SMALL"], 1)
        self.assertEqual(summary.largest_pothole.pothole_id, 1)

    def test_medium_and_large_potholes(self):
        # Medium: rel area 0.03
        p_med = PotholeDetection(1, 0.9, BoundingBox(100, 600, 400, 700), area_pixels=30000.0, relative_area=0.03)
        # Large: rel area 0.06
        p_large = PotholeDetection(2, 0.95, BoundingBox(500, 600, 800, 800), area_pixels=60000.0, relative_area=0.06)

        state, summary = self.analyzer.analyze([p_med, p_large], self.w, self.h)
        self.assertEqual(state.count, 2)
        self.assertEqual(state.severity_counts["MEDIUM"], 1)
        self.assertEqual(state.severity_counts["LARGE"], 1)
        self.assertEqual(summary.largest_pothole.pothole_id, 2)
        self.assertAlmostEqual(summary.total_relative_area, 0.09)

    def test_spatial_distribution(self):
        # Left (cx = 150 => 0.15)
        p_left = PotholeDetection(1, 0.9, BoundingBox(100, 600, 200, 700), 10000.0, 0.01)
        # Center (cx = 500 => 0.50)
        p_center = PotholeDetection(2, 0.9, BoundingBox(450, 600, 550, 700), 10000.0, 0.01)
        # Right (cx = 850 => 0.85)
        p_right = PotholeDetection(3, 0.9, BoundingBox(800, 600, 900, 700), 10000.0, 0.01)

        _, summary = self.analyzer.analyze([p_left, p_center, p_right], self.w, self.h)
        self.assertEqual(summary.spatial_distribution["left"], 1)
        self.assertEqual(summary.spatial_distribution["center"], 1)
        self.assertEqual(summary.spatial_distribution["right"], 1)


class TestRoadConditionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = load_pipeline_config("config/pipeline.yaml")
        self.analyzer = RoadConditionAnalyzer(self.config)

    def test_flawless_road_no_potholes(self):
        road = RoadMask(mask=[[1]], confidence=1.0, road_area_ratio=0.50)
        cond = self.analyzer.analyze(road, potholes=[])
        self.assertEqual(cond.score, 100.0)
        self.assertEqual(cond.category, RoadConditionCategory.GOOD)
        self.assertEqual(cond.estimated_friction_factor, 1.00)
        self.assertEqual(cond.speed_restriction_factor, 1.00)

    def test_increasing_pothole_penalties(self):
        # 1 Small pothole (area 0.01)
        p1 = PotholeDetection(
            1, 0.9, BoundingBox(100, 100, 200, 200), 10000.0, 0.01,
            estimated_severity=PotholeSeverityCategory.SMALL
        )
        road = RoadMask(mask=[[1]], confidence=1.0, road_area_ratio=0.50)
        cond1 = self.analyzer.analyze(road, [p1])
        # Penalty: 1*5 + 0.01*250 = 5 + 2.5 = 7.5 => Score = 92.5
        self.assertAlmostEqual(cond1.score, 92.5)
        self.assertEqual(cond1.category, RoadConditionCategory.GOOD)

        # 3 Medium potholes (each area 0.03)
        # Penalty: 3*5 + 0.09*250 = 15 + 22.5 = 37.5 => Score = 62.5
        p_meds = [
            PotholeDetection(
                i, 0.9, BoundingBox(i * 100, 100, i * 100 + 100, 200), 30000.0, 0.03,
                estimated_severity=PotholeSeverityCategory.MEDIUM
            )
            for i in range(3)
        ]
        cond2 = self.analyzer.analyze(road, p_meds)
        self.assertAlmostEqual(cond2.score, 62.5)
        self.assertEqual(cond2.category, RoadConditionCategory.MODERATE)
        self.assertEqual(cond2.speed_restriction_factor, 0.75)

        # Multiple large potholes: triggers POOR and CRITICAL
        p_larges = [
            PotholeDetection(
                i, 0.9, BoundingBox(i * 100, 100, i * 100 + 100, 200), 60000.0, 0.06,
                estimated_severity=PotholeSeverityCategory.LARGE
            )
            for i in range(4)
        ]
        # Penalty: 4*5 (count) + 0.24*250 (area=60) + 4*10 (severe=40) = 20 + 60 + 40 = 120
        # Score = 100 - 120 = -20 clamped to 0.0
        cond3 = self.analyzer.analyze(road, p_larges)
        self.assertEqual(cond3.score, 0.0)
        self.assertEqual(cond3.category, RoadConditionCategory.CRITICAL)
        self.assertEqual(cond3.speed_restriction_factor, 0.25)
        self.assertEqual(cond3.estimated_friction_factor, 0.40)

    def test_score_never_exceeds_bounds(self):
        road = RoadMask(mask=[[1]], confidence=1.0, road_area_ratio=0.50)
        # Empty road condition cannot exceed 100
        cond_max = self.analyzer.analyze(road, [])
        self.assertLessEqual(cond_max.score, 100.0)

        # Huge penalties cannot go below 0
        huge_potholes = [
            PotholeDetection(
                i, 1.0, BoundingBox(0, 0, 500, 500), 250000.0, 0.25,
                estimated_severity=PotholeSeverityCategory.LARGE
            )
            for i in range(10)
        ]
        cond_min = self.analyzer.analyze(road, huge_potholes)
        self.assertEqual(cond_min.score, 0.0)
        self.assertEqual(cond_min.category, RoadConditionCategory.CRITICAL)

    def test_deterministic_output(self):
        road = RoadMask(mask=[[1]], confidence=1.0, road_area_ratio=0.30)
        p = PotholeDetection(1, 0.9, BoundingBox(10, 10, 100, 100), 8100.0, 0.0081)
        cond_a = self.analyzer.analyze(road, [p])
        cond_b = self.analyzer.analyze(road, [p])
        self.assertEqual(cond_a.score, cond_b.score)
        self.assertEqual(cond_a.category, cond_b.category)
        self.assertEqual(cond_a.estimated_friction_factor, cond_b.estimated_friction_factor)


if __name__ == "__main__":
    unittest.main()
