# Phase 3: Analytical Engines & Mathematical Formulations

## 1. Architectural Role

The Analysis Layer is a pure mathematical and heuristic processing stage. It sits strictly between the standardized model adapters and the upcoming perception fusion layer:

```text
RoadMask ──┐
Detection[] ┼──> [ ANALYSIS LAYER ] ──> Standardized Analytical State
Potholes[] ─┘    ├── TrafficAnalyzer
                 ├── PotholeAnalyzer
                 └── RoadConditionAnalyzer
```

The analysis layer does **NOT** depend on ML frameworks (PyTorch, TensorFlow, ONNX, Ultralytics), model files, hardware devices, or simulator engines.

---

## 2. Traffic Analysis Metrics & Formulations

### 2.1 Object Classification & IDD Mapping
Detections are categorized according to verified Indian Driving Dataset (IDD) semantic classes:
- **Vehicles**: `{"autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle", "trailer", "train", "truck", "vehicle fallback"}`
- **Pedestrians / Vulnerable Road Users**: `{"person", "rider", "animal"}`
- **Other Objects**: `{"traffic light", "traffic sign"}`

### 2.2 Traffic Density Definition
$$\text{Density} \in [0.0, 1.0]$$

> [!IMPORTANT]
> Traffic density in this system is an **estimated camera-view spatial occupancy metric**. It represents how densely occupied the observable road surface is. It is **NOT** a physical traffic flow rate (vehicles/hour) or physical speed measurement.

When a valid `RoadMask` (with $\text{road\_area\_ratio} > 0.05$) is available:
$$\text{EffectiveCapacity} = \max(1.0, \text{SceneCapacity} \times \text{road\_area\_ratio})$$
$$\text{CountDensity} = \frac{\text{RoadAssociatedVehicles}}{\text{EffectiveCapacity}}$$
$$\text{AreaOccupancy} = \frac{\sum \text{VehicleBBoxArea}}{\text{RoadMaskAreaPixels}}$$
$$\text{Density} = \min\left(1.0, 0.60 \times \text{CountDensity} + 0.40 \times \text{AreaOccupancy}\right)$$

When no road mask is available:
$$\text{Density} = \min\left(1.0, \frac{\text{VehicleCount}}{\text{SceneCapacity}}\right)$$

### 2.3 Congestion Level Mapping
Configured in `config/pipeline.yaml`:
- $\text{Density} < 0.25$: **`LOW`**
- $0.25 \le \text{Density} < 0.60$: **`MODERATE`**
- $0.60 \le \text{Density} < 0.85$: **`HEAVY`**
- $\text{Density} \ge 0.85$: **`GRIDLOCK`**

---

## 3. Pothole Analysis Metrics & Formulations

### 3.1 Relative Area
For a pothole bounding box with width $w$ and height $h$ in a frame of dimensions $W \times H$:
$$\text{AreaPixels} = w \times h$$
$$\text{RelativeArea} = \frac{\text{AreaPixels}}{W \times H}$$

### 3.2 Estimated Severity Categories
> [!IMPORTANT]
> 2D camera bounding boxes alone **cannot directly measure physical pothole depth**. Therefore, severity categories represent **apparent surface footprint severity**, not volumetric cavity depth.

- $\text{RelativeArea} \le 0.015$: **`SMALL`**
- $0.015 < \text{RelativeArea} \le 0.045$: **`MEDIUM`**
- $\text{RelativeArea} > 0.045$: **`LARGE`**

### 3.3 Spatial Distribution
Potholes are mapped into horizontal scene corridors based on normalized centroid $c_x \in [0.0, 1.0]$:
- $c_x < 0.333$: `left`
- $0.333 \le c_x < 0.666$: `center`
- $c_x \ge 0.666$: `right`

---

## 4. Road Condition Scoring & Simulation Impact

### 4.1 Scoring Formula
$$\text{Score} \in [0.0, 100.0]$$

$$\text{Score} = \text{BaseScore} - (\text{Penalty}_{\text{count}} + \text{Penalty}_{\text{area}} + \text{Penalty}_{\text{severe}} + \text{Penalty}_{\text{coverage}})$$

Where weights are sourced from `config/pipeline.yaml`:
- $\text{BaseScore} = 100.0$
- $\text{Penalty}_{\text{count}} = \text{PotholeCount} \times 5.0$
- $\text{Penalty}_{\text{area}} = \text{TotalRelativeArea} \times 250.0$
- $\text{Penalty}_{\text{severe}} = \text{LargePotholesCount} \times 10.0$
- $\text{Penalty}_{\text{coverage}} = \max\left(0, \frac{0.25 - \text{road\_area\_ratio}}{0.25}\right) \times 15.0$

The final score is clamped strictly: $\max(0.0, \min(100.0, \text{Score}))$.

### 4.2 Road Quality Categories
- $\text{Score} \ge 80.0$: **`GOOD`**
- $50.0 \le \text{Score} < 80.0$: **`MODERATE`**
- $25.0 \le \text{Score} < 50.0$: **`POOR`**
- $\text{Score} < 25.0$: **`CRITICAL`**

### 4.3 Simulation Parameters Interpretation

> [!WARNING]
> Friction and speed restriction factors are **estimated simulation parameters** designed to inform vehicle dynamic controllers in CARLA, SUMO, or the MockSimulator. They are **NOT** physical friction coefficients measured with a skid-resistance tester.

| Category | Estimated Friction Factor | Speed Restriction Factor | Simulation Behavior |
| :--- | :---: | :---: | :--- |
| **`GOOD`** | $1.00$ | $1.00$ | Full operational vehicle speed; normal dry asphalt friction. |
| **`MODERATE`** | $0.85$ | $0.75$ | Minor speed reduction (75%); slight grip degradation warning. |
| **`POOR`** | $0.65$ | $0.50$ | Significant speed limit (50%); vehicles steer around detected pothole coordinates. |
| **`CRITICAL`** | $0.40$ | $0.25$ | Emergency hazard crawling speed (25%); vehicle routing avoidance triggered. |

---

## 5. Known Limitations & Constraints

1. **Camera Geometry**: Monocular cameras produce 2D projections without metric scale. Distance and depth cannot be measured directly without stereo calibration, lidar, or structure-from-motion.
2. **Res2Net Single-Box Limitation**: The current Res2Net checkpoint predicts a single pothole box per frame. The `PotholeAnalyzer` is designed to accept `List[PotholeDetection]` of arbitrary length so that when a multi-pothole YOLO detector is plugged in, no analyzer code will change.
3. **Engineering Heuristics**: The road condition score is a deterministic engineering heuristic based on camera visibility, not an official civil engineering pavement condition index (PCI).
