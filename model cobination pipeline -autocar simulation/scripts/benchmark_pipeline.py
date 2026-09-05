#!/usr/bin/env python3
"""
RoadVision AI - Performance Benchmarking Script.

Rigorously measures and benchmarks Sequential vs Parallel model inference:
- Excludes startup / model load time
- Excludes image loading & file I/O overhead
- Runs warm-up pass
- Executes 10 iterations for Sequential and 10 iterations for Parallel
- Calculates exact mean latencies, standard deviations, and speedup factor.
"""

import os
import sys
import time
from typing import Dict, List
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from config.loader import load_models_config, load_pipeline_config
from models.factory import create_road_model, create_traffic_model, create_pothole_model


def run_benchmark():
    print("=" * 70)
    print("ROADVSIOIN AI — SYSTEM PERCEPTION BENCHMARK AUDIT")
    print("=" * 70)

    models_cfg = load_models_config(os.path.join(PIPELINE_ROOT, "config", "models.yaml"))
    pipeline_cfg = load_pipeline_config(os.path.join(PIPELINE_ROOT, "config", "pipeline.yaml"))

    print("\n[1] Resolving Model Checkpoint Paths:")
    for key in ["road_model", "traffic_model", "pothole_model"]:
        rel_path = models_cfg[key]["path"]
        abs_path = os.path.abspath(os.path.join(PIPELINE_ROOT, rel_path))
        exists = os.path.exists(abs_path)
        print(f"  - {key:15s}: {abs_path} [Exists: {exists}]")

    print("\n[2] Loading Models into Memory...")
    t_start_load = time.perf_counter()
    road_model = create_road_model(models_cfg["road_model"])
    road_model.load()

    traffic_model = create_traffic_model(models_cfg["traffic_model"])
    traffic_model.load()

    pothole_model = create_pothole_model(models_cfg["pothole_model"])
    pothole_model.load()
    load_dt = (time.perf_counter() - t_start_load) * 1000.0
    print(f"  [OK] Total Startup Model Loading Time: {load_dt:.2f} ms")

    # Load test image
    sample_img_path = os.path.join(
        PIPELINE_ROOT,
        "Road-segmentation-UNET-model-main",
        "data",
        "data_temp_folder",
        "road_seg_kitti",
        "default",
        "image_2",
        "0.jpg",
    )
    if not os.path.exists(sample_img_path):
        print(f"[Error] Test image not found at {sample_img_path}")
        return

    frame = cv2.imread(sample_img_path)
    img_h, img_w = frame.shape[:2]
    print(f"\n[3] Loaded Benchmark Target Frame ({img_w}x{img_h} resolution)")

    # Warm-up pass
    print("  -> Executing Warm-Up Pass...")
    _ = road_model.infer(frame)
    _ = traffic_model.infer(frame)
    _ = pothole_model.infer(frame)
    print("  [OK] Warm-up Complete.")

    NUM_TRIALS = 10

    # --------------------------------------------------------------------------
    # BENCHMARK 1: SEQUENTIAL EXECUTION
    # --------------------------------------------------------------------------
    print(f"\n[4] Running Sequential Execution Benchmark ({NUM_TRIALS} iterations)...")
    seq_total_times: List[float] = []
    seq_road_times: List[float] = []
    seq_traffic_times: List[float] = []
    seq_pothole_times: List[float] = []

    for i in range(NUM_TRIALS):
        t0 = time.perf_counter()
        
        t_r0 = time.perf_counter()
        _ = road_model.infer(frame)
        t_r1 = time.perf_counter()
        
        t_t0 = time.perf_counter()
        _ = traffic_model.infer(frame)
        t_t1 = time.perf_counter()
        
        t_p0 = time.perf_counter()
        _ = pothole_model.infer(frame)
        t_p1 = time.perf_counter()
        
        t1 = time.perf_counter()

        seq_road_times.append((t_r1 - t_r0) * 1000.0)
        seq_traffic_times.append((t_t1 - t_t0) * 1000.0)
        seq_pothole_times.append((t_p1 - t_p0) * 1000.0)
        seq_total_times.append((t1 - t0) * 1000.0)

    # --------------------------------------------------------------------------
    # BENCHMARK 2: PARALLEL EXECUTION (THREADPOOL)
    # --------------------------------------------------------------------------
    print(f"\n[5] Running Parallel Execution Benchmark ({NUM_TRIALS} iterations)...")
    par_total_times: List[float] = []
    par_road_times: List[float] = []
    par_traffic_times: List[float] = []
    par_pothole_times: List[float] = []

    for i in range(NUM_TRIALS):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_road = executor.submit(road_model.infer, frame)
            f_traffic = executor.submit(traffic_model.infer, frame)
            f_pothole = executor.submit(pothole_model.infer, frame)

            _ = f_road.result()
            _ = f_traffic.result()
            _ = f_pothole.result()
        t1 = time.perf_counter()

        par_road_times.append(road_model.last_inference_time_ms)
        par_traffic_times.append(traffic_model.last_inference_time_ms)
        par_pothole_times.append(pothole_model.last_inference_time_ms)
        par_total_times.append((t1 - t0) * 1000.0)

    # Calculate statistics
    avg_seq_total = np.mean(seq_total_times)
    std_seq_total = np.std(seq_total_times)

    avg_par_total = np.mean(par_total_times)
    std_par_total = np.std(par_total_times)

    avg_road = np.mean(seq_road_times)
    avg_traffic = np.mean(seq_traffic_times)
    avg_pothole = np.mean(seq_pothole_times)

    speedup = avg_seq_total / max(1e-6, avg_par_total)

    print("\n" + "=" * 70)
    print("EMPIRICAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Model Component Latency Breakdown (Mean over {NUM_TRIALS} runs):")
    print(f"    - U-Net Road Segmentation (ONNX) : {avg_road:6.2f} ms")
    print(f"    - YOLOv8 IDD Traffic Model     : {avg_traffic:6.2f} ms")
    print(f"    - Res2Net Pothole Model         : {avg_pothole:6.2f} ms")
    print("-" * 70)
    print(f"  Sequential Execution Pipeline     : {avg_seq_total:6.2f} ± {std_seq_total:.2f} ms")
    print(f"  Parallel Multi-Threaded Pipeline  : {avg_par_total:6.2f} ± {std_par_total:.2f} ms")
    print(f"  Measured Speedup Factor          : {speedup:6.2f}x")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
