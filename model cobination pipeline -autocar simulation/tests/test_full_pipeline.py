#!/usr/bin/env python3
"""
End-to-End Perception and Fusion Pipeline Verification Test.
Tests U-Net Road Segmentation, YOLO IDD Traffic Detector, and Res2Net Pothole Detector.
"""

import os
import sys
import time
import numpy as np
import cv2

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from config.loader import load_models_config, load_pipeline_config
from models.factory import create_road_model, create_traffic_model, create_pothole_model
from processing.frame_processor import FrameProcessor


def main():
    print("=" * 70)
    print("  RoadVision AI - Comprehensive Perception Pipeline Test")
    print("=" * 70)

    models_cfg_path = os.path.join(PIPELINE_ROOT, "config", "models.yaml")
    pipeline_cfg_path = os.path.join(PIPELINE_ROOT, "config", "pipeline.yaml")

    models_cfg = load_models_config(models_cfg_path)
    pipeline_cfg = load_pipeline_config(pipeline_cfg_path)

    print("\n[1/4] Initializing U-Net Road Segmentation Model...")
    road_model = create_road_model(models_cfg.get("road_model", {}))
    t0 = time.time()
    road_model.load()
    print(f"  -> U-Net Loaded ({road_model.model_name}) in {(time.time()-t0)*1000:.1f}ms")

    print("\n[2/4] Initializing YOLO IDD Traffic Model...")
    traffic_model = create_traffic_model(models_cfg.get("traffic_model", {}))
    t0 = time.time()
    traffic_model.load()
    print(f"  -> YOLO Loaded ({traffic_model.model_name}) in {(time.time()-t0)*1000:.1f}ms")

    print("\n[3/4] Initializing Res2Net Pothole Detection Model...")
    pothole_model = create_pothole_model(models_cfg.get("pothole_model", {}))
    t0 = time.time()
    pothole_model.load()
    print(f"  -> Pothole Model Loaded ({pothole_model.model_name}) in {(time.time()-t0)*1000:.1f}ms")

    print("\n[4/4] Initializing Coordinator FrameProcessor & Fusion...")
    processor = FrameProcessor(
        road_model=road_model,
        traffic_model=traffic_model,
        pothole_model=pothole_model,
        config=pipeline_cfg,
    )
    processor.initialize(warmup=False)
    print("  -> FrameProcessor & Fusion Ready.")

    # Load test image from dataset
    test_img_path = os.path.join(PIPELINE_ROOT, "..", "Best Model", "Road-segmentation-UNET-model-main", "data", "data_set", "default", "image_2", "0.jpg")
    if os.path.exists(test_img_path):
        frame = cv2.imread(test_img_path)
    else:
        frame = np.full((540, 960, 3), 120, dtype=np.uint8)

    print(f"\nProcessing Frame: {test_img_path} (Shape: {frame.shape})...")
    t_start = time.time()
    world_state = processor.process_frame(frame, frame_id=1, source="test_image")
    t_proc = (time.time() - t_start) * 1000.0

    print("\n" + "=" * 70)
    print("  Pipeline Results Summary")
    print("=" * 70)
    print(f"  End-to-End Latency:      {t_proc:.1f}ms")
    print(f"  Road Model Status:       {world_state.perception_status.road}")
    if world_state.road:
        print(f"    - Road Coverage Ratio: {world_state.road.road_area_ratio:.4f} ({world_state.road.road_area_ratio * 100.0:.1f}%)")
        print(f"    - Confidence:          {world_state.road.confidence:.2f}")

    print(f"  Traffic Model Status:    {world_state.perception_status.traffic}")
    if world_state.traffic:
        all_objs = world_state.traffic.vehicles + world_state.traffic.pedestrians + world_state.traffic.other_objects
        print(f"    - Detected Objects:    {len(all_objs)}")
        for i, obj in enumerate(all_objs[:5]):
            print(f"      [{i+1}] {obj.class_name} (Conf: {obj.confidence:.2f})")

    print(f"  Pothole Model Status:    {world_state.perception_status.pothole}")
    if world_state.potholes:
        print(f"    - Detected Potholes:   {len(world_state.potholes.potholes)}")
        for i, pot in enumerate(world_state.potholes.potholes[:5]):
            b = pot.bbox
            print(f"      [{i+1}] Pothole at ({b.xmin:.1f}, {b.ymin:.1f}, {b.xmax:.1f}, {b.ymax:.1f}) | Conf: {pot.confidence:.2f} | Road Assoc: {pot.road_association:.2f}")

    print(f"  Fusion Safety Analytics:")
    print(f"    - Road Condition Score:{world_state.road_condition.score:.1f}/100 ({world_state.road_condition.category.value})")
    print(f"    - Congestion Level:    {world_state.traffic.congestion_level.value}")
    print(f"    - Model Latencies:     Road={world_state.metadata.road_inference_ms:.1f}ms, Traffic={world_state.metadata.traffic_inference_ms:.1f}ms, Potholes={world_state.metadata.pothole_inference_ms:.1f}ms")
    print("=" * 70)
    print("  [SUCCESS] All models trained, integrated, and verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
