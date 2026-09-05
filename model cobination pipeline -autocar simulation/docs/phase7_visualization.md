# Phase 7: Visualization Layer Specification

## 1. Architectural Role

Visualization is a strictly **read-only** inspection and telemetry layer:

```text
Raw Frame (BGR) ──────┐
                      ▼
                 Visualization (Read-Only)
                      ▲
                      │
                 WorldState
```

### Core Invariants
- **No Side Effects**: Visualization **never** modifies `WorldState`, triggers model inference, performs object tracking, alters road condition scores, or communicates with the simulator.
- **No Hard GUI Dependency**: All renderers support 100% headless operation using Pillow and NumPy without requiring an active display or X11/Wayland context.

---

## 2. Overlay Visualization (`visualization/overlay.py`)

### Visual Annotations
1. **Drivable Road Mask**:
   - Translucent emerald green blend (`alpha = 0.30`) over pixels where the binary road mask is active.
   - Rescales mask to frame resolution using nearest-neighbor interpolation without modifying the underlying `WorldState`.
2. **Traffic Detections**:
   - Bounding boxes color-coded by road association:
     - **Green**: Vehicle grounded on road surface (`road_association >= 0.25`).
     - **Orange**: Off-road / shoulder object.
     - **Magenta**: Pedestrian / vulnerable road user.
   - Text banner above each box: `#<track_id> <class_name> <conf> (Road)`.
3. **Potholes**:
   - Bounding boxes color-coded by estimated severity:
     - **Yellow**: `SMALL`
     - **Deep Orange**: `MEDIUM`
     - **Red**: `LARGE`
   - Label: `POTHOLE #<id> [<severity>]`.
4. **HUD Telemetry Bar**:
   - Semi-transparent top banner displaying:
     - Line 1: `FRAME`, `TIME`, `FPS`, `LATENCY (ms)`.
     - Line 2: `ROAD QUALITY (score/100)`, `CONGESTION LEVEL`, `VEHICLES`, `PEDS`, `POTHOLES`.
     - System Status Badge: `PERCEPTION: OK` or `PERCEPTION: DEGRADED`.

---

## 3. Bird's-Eye View (`visualization/birdseye.py`)

> [!WARNING]
> The bird's-eye view is an **approximate image-space top-down visualization**. It maps normalized camera coordinates $(x_{norm}, y_{norm})$ into a 2D top-down perspective. It is **NOT** a physically calibrated ground-plane metric projection. Metric distance requires calibrated camera homography or depth sensors.

### Spatial Rendering
- **Road Corridor**: Perspective trapezoid matching the camera horizon and drivable road area ratio.
- **Ego Vehicle**: Fixed marker at the bottom center.
- **Traffic Objects**: Projected onto top-down grid at their bottom-center road grounding coordinates.
- **Potholes**: Circles color-coded by severity category.

---

## 4. Telemetry Dashboard (`visualization/dashboard.py`)

### Modes of Operation
1. **Text Console Dashboard (`render_text`)**:
   - Generates structured ASCII logs suitable for stdout, terminal streaming, or logging files.
2. **Graphic Telemetry Card (`render_image`)**:
   - Produces a $640 \times 480$ BGR image panel summarizing frame metrics, road health, traffic breakdown, and active errors.

---

## 5. Performance & Headless Execution

- **Zero Unnecessary Copies**: Visual renderers operate directly on frame copies without redundant resize chains.
- **Optional Execution**: The perception pipeline can run in high-throughput headless mode with visualization completely disabled.
- **No GUI Initialization on Import**: `cv2.imshow` is never invoked during import or headless tests.
