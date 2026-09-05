"""
Unit tests for perception schemas and data contracts.
"""

import json
import unittest
from perception.schemas import (
    BoundingBox,
    Detection,
    RoadMask,
    PotholeDetection,
    TrafficState,
    PotholesState,
    RoadCondition,
    RoadConditionCategory,
    PotholeSeverityCategory,
    CongestionLevel,
    ModelStatus,
    PerceptionStatus,
    WorldStateMetadata,
    WorldState,
    FrameData,
)


class TestSchemas(unittest.TestCase):
    def test_bounding_box_properties(self):
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=80.0, is_normalized=False)
        self.assertEqual(bbox.width, 40.0)
        self.assertEqual(bbox.height, 60.0)
        self.assertEqual(bbox.area, 2400.0)
        self.assertEqual(bbox.center, (30.0, 50.0))
        self.assertEqual(bbox.bottom_center, (30.0, 80.0))

    def test_bounding_box_normalization_roundtrip(self):
        bbox = BoundingBox(xmin=100.0, ymin=200.0, xmax=300.0, ymax=400.0)
        norm_bbox = bbox.to_normalized(1000, 1000)
        self.assertTrue(norm_bbox.is_normalized)
        self.assertAlmostEqual(norm_bbox.xmin, 0.1)
        self.assertAlmostEqual(norm_bbox.ymin, 0.2)
        self.assertAlmostEqual(norm_bbox.xmax, 0.3)
        self.assertAlmostEqual(norm_bbox.ymax, 0.4)

        pix_bbox = norm_bbox.to_pixel(1000, 1000)
        self.assertFalse(pix_bbox.is_normalized)
        self.assertAlmostEqual(pix_bbox.xmin, 100.0)
        self.assertAlmostEqual(pix_bbox.ymax, 400.0)

    def test_detection_dataclass(self):
        bbox = BoundingBox(xmin=50, ymin=60, xmax=150, ymax=200)
        det = Detection(
            object_id=1,
            class_name="car",
            class_id=4,
            confidence=0.92,
            bbox=bbox,
            track_id=101,
            estimated_distance=15.5,
            road_association=0.95,
        )
        self.assertEqual(det.class_name, "car")
        self.assertEqual(det.track_id, 101)
        self.assertEqual(det.center_x, 100.0)
        self.assertEqual(det.center_y, 130.0)

        d_dict = det.to_dict()
        self.assertEqual(d_dict["class_name"], "car")
        self.assertEqual(d_dict["track_id"], 101)
        self.assertAlmostEqual(d_dict["estimated_distance"], 15.5)

    def test_pothole_detection_dataclass(self):
        bbox = BoundingBox(xmin=120, ymin=300, xmax=220, ymax=350)
        pothole = PotholeDetection(
            pothole_id=1,
            confidence=0.88,
            bbox=bbox,
            area_pixels=5000.0,
            relative_area=0.015,
            estimated_severity=PotholeSeverityCategory.MEDIUM,
            road_association=1.0,
            estimated_distance=8.0,
        )
        self.assertEqual(pothole.estimated_severity, PotholeSeverityCategory.MEDIUM)
        p_dict = pothole.to_dict()
        self.assertEqual(p_dict["estimated_severity"], "MEDIUM")
        self.assertEqual(p_dict["pothole_id"], 1)

    def test_perception_status_isolated_failure(self):
        """Verify that a single model failure does not invalidate overall perception status tracking."""
        status = PerceptionStatus(
            road="ok",
            traffic="ok",
            pothole="error",
            errors={"pothole": "CUDA out of memory in pothole detector"},
        )
        self.assertFalse(status.is_all_ok())
        self.assertEqual(status.road, "ok")
        self.assertEqual(status.traffic, "ok")
        self.assertEqual(status.pothole, "error")
        self.assertIn("CUDA", status.errors["pothole"])

    def test_world_state_serialization(self):
        """Verify complete WorldState creation and JSON serialization."""
        road = RoadMask(mask=None, confidence=0.98, road_area_ratio=0.45)
        road_cond = RoadCondition(
            score=82.5,
            category=RoadConditionCategory.GOOD,
            estimated_friction_factor=0.92,
            speed_restriction_factor=0.95,
        )
        traffic = TrafficState(
            vehicle_count=2,
            vehicle_counts_by_class={"car": 1, "autorickshaw": 1},
            pedestrian_count=1,
            total_objects=3,
            traffic_density=0.15,
            congestion_level=CongestionLevel.LOW,
        )
        potholes = PotholesState(
            count=1,
            total_area_pixels=1200.0,
            total_relative_area=0.003,
            severity_counts={"SMALL": 1},
        )
        meta = WorldStateMetadata(
            processing_time_ms=45.2,
            road_inference_ms=12.1,
            traffic_inference_ms=20.3,
            pothole_inference_ms=9.8,
            fusion_ms=3.0,
            fps=22.1,
        )
        ws = WorldState(
            frame_id=42,
            timestamp=1700000000.0,
            image_width=1280,
            image_height=720,
            road=road,
            road_condition=road_cond,
            traffic=traffic,
            potholes=potholes,
            perception_status=PerceptionStatus(),
            metadata=meta,
        )

        ws_dict = ws.to_dict()
        self.assertEqual(ws_dict["frame_id"], 42)
        self.assertEqual(ws_dict["road"]["condition"]["category"], "GOOD")
        self.assertEqual(ws_dict["traffic"]["vehicle_count"], 2)

        json_str = ws.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["frame_id"], 42)
        self.assertEqual(parsed["environment"]["image_width"], 1280)


if __name__ == "__main__":
    unittest.main()
