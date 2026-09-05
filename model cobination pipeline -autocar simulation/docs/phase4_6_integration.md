# Phase 4–6: Integrated Perception-to-Simulation Pipeline

## 1. System Architecture Overview

```text
CAMERA / VIDEO / SYNTHETIC
          │
          ▼
     FrameSource (Synthetic / Image / Video / Camera)
          │
          ▼ (Raw Frame, FrameData)
    FrameProcessor
          │
    ┌─────┴─────────────────────┐
    │ Independent Model Runners │
    │ ├── UNetRoadModel         │
    │ ├── YOLOIDDTrafficModel   │
    │ └── Res2NetPotholeModel   │
    └─────┬─────────────────────┘
          │ (Raw outputs)
    ┌─────┴─────────────────────┐
    │ Standardized Adapters     │
    │ ├── RoadAdapter           │
    │ ├── TrafficAdapter        │
    │ └── PotholeAdapter        │
    └─────┬─────────────────────┘
          │ (RoadMask, Detection[], PotholeDetection[])
          ▼
   PerceptionFusion
     ├── Coordinate Normalization & Clipping
     ├── Spatial Association (Road Grounding & Pothole Containment)
     ├── SimpleTracker (Persistent track IDs)
     ├── TrafficAnalyzer
     ├── PotholeAnalyzer
     └── RoadConditionAnalyzer
          │
          ▼
      WorldState (Immutable Snapshot)
          │
          ▼
  SimulationAdapter
          │
          ▼ (ScenarioBuilder)
  SimulationScenario
          │
          ▼
  SimulatorInterface
     ├── MockSimulator   (In-memory telemetry & validation)
     ├── CarlaAdapter    (Future CARLA autonomous vehicle integration)
     └── SumoAdapter     (Future SUMO macroscopic traffic flow integration)
```

---

## 2. Component Details

### 2.1 PerceptionFusion (`perception/fusion.py`)
- Central aggregator combining multi-model perception outputs into an immutable `WorldState` snapshot.
- Evaluates spatial membership via `perception/association.py`:
  - Determines vehicle road grounding via bottom-center point evaluation.
  - Determines pothole containment within drivable road surface.
- Assigns persistent `track_id` values across frames via `SimpleTracker`.
- Measures inference and fusion latencies and populates `WorldStateMetadata`.
- Preserves model health status (`PerceptionStatus`). If one model fails, the other perception streams are still processed.

### 2.2 WorldState Snapshot Contract (`perception/schemas.py`, `perception/world_state.py`)
`WorldState` is the single source of truth intermediate representation consumed by downstream simulation:
- **Environment**: Image resolution $(W, H)$, source name, frame ID, timestamp.
- **Road**: Binary drivable mask, area ratio, 0–100 road condition score, category (`GOOD`, `MODERATE`, `POOR`, `CRITICAL`), estimated friction and speed restriction factors.
- **Traffic**: Full vehicle and pedestrian detection lists, per-class counts, spatial density, estimated congestion level (`LOW`, `MODERATE`, `HEAVY`, `GRIDLOCK`).
- **Potholes**: Pothole detections, count, total relative area, severity counts (`SMALL`, `MEDIUM`, `LARGE`).
- **PerceptionStatus**: Health tracking (`road`, `traffic`, `pothole`, `errors`).
- **Metadata**: Latency metrics (road, traffic, pothole, fusion, total ms, FPS).

Individual `WorldState` snapshots are strictly immutable. Historical buffering and rolling telemetry are handled separately by `WorldStateHistory`.

### 2.3 Frame Ingestion (`processing/frame_source.py`)
Unified streaming interface:
- **`SyntheticSource`**: Deterministic frame generator for tests and CI without external dependencies.
- **`ImageSource`**: Single image loader using PIL or OpenCV.
- **`VideoSource`**: Sequential frame reader streaming without loading full video into RAM.
- **`CameraSource`**: Live camera/webcam capture with lazy OpenCV initialization.

### 2.4 FrameProcessor Lifecycle (`processing/frame_processor.py`)
- Models are loaded **ONCE** during `processor.initialize()`. They are **NEVER** reloaded inside `process_frame()`.
- Provides **Failure Isolation**: Exceptions in one model runner are caught, recorded in `PerceptionStatus.errors`, and safe empty outputs are passed downstream without crashing the pipeline.
- Releases resources upon `processor.close()`.

### 2.5 Simulation Abstraction & Mapping (`simulation/`)
- **`SimulatorInterface`**: Abstract contract (`initialize`, `reset`, `update`, `step`, `get_state`, `close`).
- **`ScenarioBuilder`**: Translates `WorldState` into `SimulationScenario`:
  - Dynamic vehicles & pedestrians $\to$ `SimulationAgent`
  - Road potholes $\to$ `SimulationHazard`
  - Road condition $\to$ Speed restriction factor & estimated friction factor
- **`SimulationAdapter`**: Connects the perception pipeline to any simulator implementing `SimulatorInterface`.
- **`MockSimulator`**: Lightweight in-memory simulator tracking agent counts, road friction, speed restrictions, and telemetry for debugging without external simulator requirements.
- **`CarlaAdapter` / `SumoAdapter`**: Pluggable stubs ready for future deployment when CARLA or SUMO is available.
