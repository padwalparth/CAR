# SIH 2026: Intelligent Road-Scene Perception & Traffic Simulation System

A modular, simulator-agnostic computer vision and simulation pipeline for road-scene intelligence, drivable area segmentation, Indian Driving Dataset (IDD) traffic object detection, and road hazard (pothole) localization.

---

## 1. Project Purpose

This project extracts road-scene intelligence from camera streams or video files and bridges perception directly into autonomous vehicle / traffic simulation backends without coupling model internals to simulator physics engines.

### Key Capabilities
- **Drivable Road Segmentation**: U-Net model segmenting road surfaces and drivable boundaries.
- **Heterogeneous Traffic Detection & Tracking**: YOLOv8 model trained on 15 Indian Driving Dataset (IDD) vehicle and pedestrian classes, with centroid tracking and persistent track IDs.
- **Pothole Hazard Localization**: Res2Net bounding-box regression model localizing road damage with calibrated severity categorization.
- **Perception Fusion & Intermediate WorldState**: Standardized, simulator-agnostic perception contract decoupling perception models from simulators.
- **Simulation Adaptation**: Scenario synthesis translating WorldState into simulation scenarios for MockSimulator, CARLA, or SUMO.
- **Real-Time Visualization**: Headless overlays, bird's-eye view projection, and telemetry dashboards.

---

## 2. System Architecture

```
                      Raw Video / Camera / Image / Mock
                                     │
                                     ▼
                                FrameSource
                     (Synthetic, Image, Video, Camera)
                                     │
                                     ▼
                              FrameProcessor
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
   Road Model                  Traffic Model                Pothole Model
   (U-Net ONNX)               (YOLOv8.3 IDD)              (Res2Net-50d)
        │                            │                            │
        ▼                            ▼                            ▼
   RoadAdapter                 TrafficAdapter              PotholeAdapter
  (RoadMask)                (List[Detection])         (List[PotholeDetection])
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                              PerceptionFusion
                       (Associates & Analyzes State)
                                     │
                                     ▼
                                 WorldState
                      (Canonical Intermediate Contract)
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   Visualization                                         SimulationAdapter
(Overlay / Bird's-eye / Dashboard)                              │
         │                                                       ▼
 Rendered Frame / HUD                                    SimulationScenario
                                                                 │
                                                                 ▼
                                                  MockSimulator / CARLA / SUMO
```

---

## 3. Directory Structure

```
├── config/
│   ├── models.yaml          # Model paths, backends (onnx, ultralytics, res2net, mock)
│   ├── pipeline.yaml        # Confidence thresholds, tracking params, severity rules
│   └── simulation.yaml      # Simulator settings (time step, scenario parameters)
├── perception/
│   ├── schemas.py           # Canonical data contracts (WorldState, Detection, RoadMask, etc.)
│   ├── coordinates.py       # Bounding box math, IoU, normalization, clipping
│   ├── association.py       # Spatial road association (point-in-mask, bbox overlap)
│   ├── tracker.py           # Centroid-based persistent track ID manager
│   ├── fusion.py            # PerceptionFusion engine assembling WorldState
│   └── world_state.py       # Temporal rolling history buffer & integrity validators
├── models/
│   ├── base.py              # BasePerceptionModel abstract lifecycle interface
│   ├── factory.py           # Model factory creating runners from config
│   ├── road_segmentation/   # UNetRoadModel & MockRoadModel
│   ├── traffic_detection/   # YOLOIDDTrafficModel & MockTrafficModel
│   └── pothole_detection/   # Res2NetPotholeModel & MockPotholeModel
├── adapters/
│   ├── road_adapter.py      # Raw segmentation -> RoadMask
│   ├── traffic_adapter.py   # Raw YOLO results -> List[Detection]
│   └── pothole_adapter.py   # Raw Res2Net bbox -> List[PotholeDetection]
├── analysis/
│   ├── traffic_analysis.py  # Congestion level & vehicle class breakdown
│   ├── pothole_analysis.py  # Severity distribution & total relative hazard area
│   └── road_analysis.py     # Drivable area ratio, condition score, friction factors
├── processing/
│   ├── frame_source.py      # SyntheticSource, ImageSource, VideoSource, CameraSource
│   ├── preprocessing.py     # Frame validation & color space checks
│   ├── postprocessing.py    # Non-maximum suppression & confidence filtering
│   ├── frame_processor.py   # Main coordinator (loads models ONCE, isolates failures)
│   ├── statistics.py        # RuntimeStatistics (latency, FPS, failure counts)
│   └── pipeline_runner.py   # Core execution loop, summary generator, file saving
├── simulation/
│   ├── simulator_interface.py # SimulatorInterface abstract base class
│   ├── scenario_builder.py    # WorldState -> SimulationScenario builder
│   ├── mock_simulator.py      # In-memory mock simulator with telemetry logging
│   ├── simulation_adapter.py  # Bridge pushing scenarios to simulator backends
│   ├── carla_adapter.py       # CARLA simulator integration stub
│   └── sumo_adapter.py        # SUMO / TraCI co-simulation stub
├── visualization/
│   ├── overlay.py           # HUD overlay, bounding boxes, drivable road mask
│   ├── birdseye.py          # Top-down road occupancy view
│   └── dashboard.py         # Formatted text & graphic telemetry dashboard
├── scripts/
│   ├── run_pipeline.py      # User-facing CLI entry point (argparse)
│   └── validate_real_models.py # Comprehensive real model validation & benchmark script
├── docs/
│   └── phase9_real_model_validation.md # Benchmark report & operational validation data
├── tests/                   # 104 unit & integration tests (100 core + 4 real model)
├── requirements-core.txt    # Minimal dependencies for mock mode & tests
├── requirements-ml.txt      # Real neural network dependencies (ONNX, PyTorch, YOLO)
├── requirements-sim.txt     # Optional CARLA / SUMO packages
└── requirements.txt         # Points to core requirements by default
```

---

## 4. Installation & Dependency Modes

The codebase is engineered with strict dependency isolation. It can run in three distinct tiers:

### Tier 1: Core / Test Mode (Zero ML Packages)
Ideal for testing, CI/CD, and lightweight laptops without GPU or heavy C-wheels:
```bash
pip install -r requirements-core.txt
```
*Dependencies: Python 3.9+, NumPy, Pillow, PyYAML.*

### Tier 2: Real Model Mode (Full ML Inference)
Enables running the actual trained neural network weights:
```bash
pip install -r requirements-ml.txt
```
*Dependencies: onnxruntime, ultralytics, torch, torchvision, timm, opencv-python.*

### Tier 3: External Simulation Mode (Optional)
Required only when coupling directly to external traffic simulators:
```bash
# For SUMO:
pip install traci
# For CARLA: install the Python wheel matching your CARLA server build.
```

---

## 5. Running the Pipeline (CLI Guide)

The unified CLI entry point is `scripts/run_pipeline.py`.

```bash
py scripts/run_pipeline.py --help
```

### Supported Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | string | `mock` | Input frame source (`mock`, path to image/video, or `camera:0`) |
| `--config` | string | `config` | Path to directory containing `models.yaml`, `pipeline.yaml`, etc. |
| `--mock-models` | flag | `False` | Force deterministic mock perception runners (no ML packages required) |
| `--visualize` | flag | `False` | Enable HUD and bounding-box overlay rendering |
| `--simulate` | flag | `False` | Enable `SimulationAdapter` and advance `MockSimulator` |
| `--save-output` | string | `None` | Directory path to save rendered frame images and `run_summary.json` |
| `--save-json` | string | `None` | Directory path to save per-frame `WorldState` JSON snapshots |
| `--max-frames` | int | `None` | Maximum number of frames to process |
| `--display` | flag | `False` | Open live GUI window (requires display server & OpenCV) |
| `--no-display` | flag | `False` | Explicit headless mode (default behavior) |
| `--verbose` | flag | `False` | Enable detailed DEBUG logging |

---

## 6. Execution Modes & Examples

### 1. Mock Mode (Primary Smoke Test)
Runs fully in-memory with zero GPU or ML requirements:
```bash
py scripts/run_pipeline.py --source mock --mock-models --simulate --max-frames 10 --no-display
```

### 2. Single Image Mode
Processes a single still frame and outputs a structured inspection report:
```bash
py scripts/run_pipeline.py --source path/to/image.jpg --mock-models --visualize --save-output output/
```
*Sample Output:*
```text
========================================
FRAME RESULT
========================================
Frame ID: 1
Resolution: 1280x720

Road:
  Condition: GOOD
  Score: 92.4

Traffic:
  Objects: 8
  Vehicles: 6
  Pedestrians: 2
  Congestion: MODERATE

Potholes:
  Count: 2
  Large: 1

Processing:
  Total: 42.1 ms
  FPS: 23.8
========================================
```

### 3. Streaming Video Mode
Streams video frames on-demand without buffering entire files in memory:
```bash
py scripts/run_pipeline.py --source road_scene.mp4 --mock-models --simulate --max-frames 100 --save-output output/
```

### 4. Camera Ingestion Mode
Streams frames directly from a connected USB webcam or RTSP feed:
```bash
py scripts/run_pipeline.py --source camera:0 --mock-models --max-frames 50
```

### 5. Real Model Mode
Executes inference with real neural networks when ML dependencies are installed:
```bash
py scripts/run_pipeline.py --source road_scene.mp4 --visualize --simulate
```
*Note: If ML dependencies or checkpoints are missing, the pipeline will halt with an actionable error rather than silently degrading to mock models.*

---

## 7. Saving Outputs & Data Export

### Rendered Frames
When `--save-output <dir>` is provided, annotated frames are saved with sequential zero-padded filenames:
```
output/
  ├── frame_000001.jpg
  ├── frame_000002.jpg
  └── run_summary.json
```

### WorldState Snapshots
When `--save-json <dir>` is provided, canonical perception snapshots are exported directly from `WorldState.to_json()`:
```
output/
  ├── world_state_000001.json
  └── world_state_000002.json
```

---

## 8. Testing & Validation

The test suite covers unit logic, coordinate math, spatial association, tracker persistence, model adapters, perception fusion, simulation adaptation, visualization renderers, CLI options, and end-to-end runner lifecycles.

To run all 100 tests:
```bash
py -m unittest discover -s tests -p "test_*.py" -v
```

---

## 9. Known Limitations & Clarifications

1. **Pothole Detection**: The current Res2Net model (`res2net50d.in1k`) regresses a single bounding box on 128x128 images. The `PotholeAdapter` supports multi-pothole output for seamless drop-in of future YOLO pothole detectors.
2. **Bird's-Eye View**: The top-down projection is an approximate image-space transformation for spatial debugging and does not claim metric centimeter-level ground truth without camera extrinsic matrix calibration.
3. **Simulation Parameters**: Road condition scores, estimated friction factors ($0.0 - 1.0$), and speed restriction factors ($0.0 - 1.0$) are heuristic perception parameters designed for simulator consumption, not physical contact-patch friction measurements.
4. **Failure Isolation**: If a single perception model fails during streaming, the pipeline logs the failure, marks `perception_status`, and continues executing remaining models without crashing the process.

---

## 10. Real Model Validation & Benchmarks (Phase 9)

Phase 9 operationalizes and benchmarks the actual trained neural networks on CPU:
- **Road U-Net ONNX**: 175.0 ms
- **Traffic YOLOv8 IDD**: 1730.7 ms
- **Pothole Res2Net-50d**: 211.3 ms
- **Perception Fusion**: 0.28 ms
- **End-to-End Latency**: 2119.3 ms (~0.47 FPS on CPU)

### Running Real Model Validation
```bash
# Execute multi-stage validation and benchmarking script:
.venv\Scripts\python.exe scripts/validate_real_models.py

# Run real-model integration tests:
.venv\Scripts\python.exe -m unittest tests/test_real_models_integration.py -v
```
Full technical findings and environment snapshots are documented in [`docs/phase9_real_model_validation.md`](docs/phase9_real_model_validation.md).
