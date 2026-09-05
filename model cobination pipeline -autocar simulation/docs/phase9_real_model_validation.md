# Phase 9: Real Model Validation & Integrated Inference Report

## 1. Executive Summary

This report documents the empirical operational validation, latency benchmarking, and qualitative output inspection of the three trained perception models connected directly to the end-to-end pipeline:
1. **Drivable Road Segmentation**: U-Net ONNX Model (`road_seg_160_160.onnx`)
2. **Heterogeneous Traffic Detection**: YOLOv8.3 IDD PyTorch Checkpoint (`best.pt`)
3. **Pothole Hazard Localization**: Res2Net-50d PyTorch Checkpoint (`best_model.pt`)

### Core Verification Invariant
- **Operational Validation**: Answers *"Does the pipeline successfully load, infer, adapt, and fuse real model weights without fallback?"* (Confirmed: **YES**)
- **Qualitative Output Inspection**: Answers *"Do the bounding boxes, segmentation masks, and telemetry visually correspond to expected scene features?"* (Confirmed: **YES**)
- **Formal Metric Evaluation**: Requires labelled test datasets with ground-truth annotations to measure quantitative mAP, IoU, and Dice scores; documented separately from operational execution.

---

## 2. Environment & Checkpoint Snapshot

### Runtime Environment
- **Python**: 3.11.16 (64bit (AMD64))
- **OS**: Windows 10
- **PyTorch**: 2.14.0+cpu
- **TorchVision**: 0.29.0+cpu
- **timm**: 1.0.29
- **Ultralytics**: 8.4.140
- **ONNX Runtime**: 1.29.0 (Providers: AzureExecutionProvider, CPUExecutionProvider)
- **OpenCV**: 5.0.0
- **Hardware Acceleration**: CUDA Not Enabled (CPU Runtime)

### Model Checkpoints
| Model | Framework | Path | Input Shape | Output Shape / Classes | Device |
|---|---|---|---|---|---|
| **Road Segmentation** | ONNX | `Road-segmentation-UNET-model-main/models/onnx_models/road_seg_160_160.onnx` | `(1, 160, 160, 3)` | `(1, 160, 160, 3)` (Softmax over 3 classes: [0, 1, 2]) | CPU |
| **Traffic Detection** | Ultralytics PyTorch | `YOLOv8.3.0 Train on IDD/runs/idd_yolov8_training/weights/best.pt` | `(1, 3, 640, 640)` | 15 IDD Classes, Pixel BBoxes `[x1, y1, x2, y2]` | CPU |
| **Pothole Detection** | PyTorch / timm | `pothole detection model/best_model.pt` | `(1, 3, 128, 128)` | 4 Regressed Coordinates on 128x128 | CPU |

---

## 3. Stage 1: Individual Model Validation Results

### 3.1 Road Segmentation (U-Net)
- **Status**: PASS
- **Raw Mask Shape**: [160, 160]
- **Observed Unique Classes**: [0, 2] (Matches 3-class contract: background=0, lane=1, road=2)
- **Standalone Latency**: 28.8 ms

### 3.2 Traffic Detection (YOLO IDD)
- **Status**: PASS
- **Detections Observed**: 5
- **Class Mapping**: Confirmed all predicted class IDs map within `[0, 14]` across the 15 IDD categories.
- **Standalone Latency**: 490.85 ms

### 3.3 Pothole Detection (Res2Net)
- **Status**: PASS
- **Architecture**: Single-bbox regression model (`res2net50d.in1k`).
- **Confidence Handling**: Explicitly assigned as **synthetic/estimated confidence (1.0)** by design; the model architecture regresses 4 spatial coordinates directly without an independent classification branch.
- **Standalone Latency**: 70.19 ms

---

## 4. Stage 2: Adapter Contract & Coordinate Verification

- **RoadAdapter**: Converts raw argmax mask into `RoadMask`. Confidence is appropriately `None` (pixel-wise argmax has no global score). `road_area_ratio` validated in range `[0.0, 1.0]`.
- **TrafficAdapter**: Converts raw pixel-space coordinates `[x1, y1, x2, y2]` into normalized `[0.0, 1.0]` coordinates. All detections verified to have finite, sorted coordinates (`xmin <= xmax`, `ymin <= ymax`).
- **PotholeAdapter**: Scales 128x128 bounding box to original frame dimensions, normalizes coordinates, calculates relative scene area, and estimates severity category (SMALL, MEDIUM, LARGE).

---

## 5. Stage 3: Integrated Pipeline & Simulation Ingestion

- **Pipeline Coordination**: `FrameProcessor` successfully loads all three models once during initialization and processes frames synchronously without silent mock fallback.
- **Perception Fusion**: `PerceptionFusion` aggregates road mask, traffic detections, and pothole hazards into an immutable `WorldState` snapshot.
- **Simulation Coupling**: `SimulationAdapter` consumes `WorldState`, constructs a `SimulationScenario`, and steps `MockSimulator` cleanly.
- **Perception Status**: `road=OK`, `traffic=OK`, `pothole=OK`.

---

## 6. Stage 4: Performance & Latency Benchmarks

Measured over **20 iterations** (preceded by **5 warm-up iterations** excluded from timing):

| Subsystem | Mean Latency (ms) | Min Latency (ms) | Max Latency (ms) | Std Dev (ms) |
|---|---|---|---|---|
| **Road Segmentation (U-Net ONNX)** | 29.69 | 26.91 | 42.13 | 3.85 |
| **Traffic Detection (YOLO IDD)** | 441.07 | 337.75 | 580.8 | 68.72 |
| **Pothole Detection (Res2Net)** | 57.57 | 36.81 | 95.13 | 19.33 |
| **Perception Fusion** | 0.48 | 0.34 | 1.01 | 0.13 |
| **End-to-End Total** | **529.68** | **461.84** | **664.68** | **58.21** |

### Effective Throughput
- **End-to-End CPU Framerate**: **1.89 FPS**
- **Hardware Profile**: AMD64 CPU execution.

---

## 7. Stage 5: Qualitative Visual Output Inspection

- **Artifact**: `output/real_model_validation/real_model_annotated_sample.jpg`
- **WorldState JSON**: `output/real_model_validation/world_state_sample.json`
- **Visual Checks**:
  1. The green drivable road overlay aligns with the road surface corridor.
  2. Detected traffic objects show IDD class labels and confidence percentages.
  3. The pothole bounding box is highlighted and mapped into the scene.
  4. The top HUD displays real-time latency breakdowns, road health score, and congestion levels.

---

## 8. Limitations & Future Scope

1. **Pothole Model Architecture**: The Res2Net model predicts exactly one bounding box per frame without learned confidence. Future phases should train a multi-box detector (e.g. YOLO-Pothole) which drops into `PotholeAdapter` seamlessly.
2. **Simulation Parameters**: Friction factors and speed restriction factors are heuristic estimates derived from perception, not physically measured tire-road interaction.
3. **Formal Evaluation**: Dataset-level mAP / IoU / Dice evaluation requires labelled test annotations, kept strictly distinct from operational pipeline verification.
