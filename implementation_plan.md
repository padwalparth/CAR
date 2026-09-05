# Implementation Plan - RoadVision AI (Traffic Safety Analysis Platform)

Build a production-quality, modern, AI-powered Road & Traffic Safety Analysis web application (**RoadVision AI**). The platform visualizes computer vision outputs for Indian road scenes combining YOLO Object Detection, U-Net Road Segmentation, Pothole Detection, and a Fusion Safety Engine.

> [!IMPORTANT]
> **No Hardcoded ML Results in Core Logic**: The frontend will rely exclusively on dynamic API structures (`src/services/api/`) and normalized response models. When backend ML models are disconnected, clean empty, loading, error, and "Model Not Connected" states will be displayed. Demo data is strictly isolated in `src/services/demoData.ts` and can be toggled on/off in Settings or via `.env`.

---

## User Review Required

> [!NOTE]
> The app will be scaffolded using **Vite + React + TypeScript + Tailwind CSS + Lucide Icons + Recharts**.
> A custom dark modern theme will be established with deep slate/charcoal backgrounds, glassmorphism, electric blue/cyan/violet accents, and clear semantic status indicators.

---

## Proposed Changes

### Project Setup & Configuration

#### [NEW] [package.json](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/package.json)
#### [NEW] [vite.config.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/vite.config.ts)
#### [NEW] [tailwind.config.js](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/tailwind.config.js)
#### [NEW] [index.html](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/index.html)
#### [NEW] [src/index.css](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/index.css)

---

### Type Definitions & Data Layer

#### [NEW] [src/types/inference.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/types/inference.ts)
- Defines schema for YOLO detected objects (`id`, `class`, `confidence`, `bbox`: `x1,y1,x2,y2`).
- Defines U-Net road segmentation schema (`mask_url`, `coverage`).
- Defines Pothole detection schema (`id`, `confidence`, `bbox`, optional `severity`).
- Defines Fusion Engine safety output (`risk_level`, `risk_score`, `warnings`).
- Defines Pipeline execution state per model stage (Waiting, Processing, Completed, Failed, Not Connected) and timing metrics.

#### [NEW] [src/types/models.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/types/models.ts)
- Model health status (Connected, Loading, Ready, Error, Offline), version, latency, endpoint URL.

#### [NEW] [src/types/analytics.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/types/analytics.ts)
- Object distribution, confidence histogram, pothole frequency timeline, road coverage trends, latency metrics.

---

### API & Service Layer

#### [NEW] [src/services/api/client.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/services/api/client.ts)
- Configurable fetch HTTP wrapper handling timeouts, retries, and clean error formatting.

#### [NEW] [src/services/api/inference.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/services/api/inference.ts)
- Async job management: `uploadMedia()`, `startAnalysis()`, `getJobStatus()`, `cancelJob()`.

#### [NEW] [src/services/api/health.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/services/api/health.ts)
- Microservices health checker for YOLO, U-Net, Pothole Model, and Fusion Engine.

#### [NEW] [src/services/api/analytics.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/services/api/analytics.ts)
- Aggregated analytics API consumer.

#### [NEW] [src/services/api/websocket.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/services/api/websocket.ts)
- Reconnecting WebSocket manager for live camera streams and real-time AI events.

#### [NEW] [src/services/demoData.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/services/demoData.ts)
- Isolated mock dataset for offline demonstration mode, strictly separated from inference logic.

---

### Hooks & State Management

#### [NEW] [src/hooks/useInference.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/hooks/useInference.ts)
- Manages active analysis state, file upload, pipeline stages, layer visibility, confidence filtering, and bi-directional element selection.

#### [NEW] [src/hooks/useModelHealth.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/hooks/useModelHealth.ts)
- Polls model microservices health status with connection state management.

#### [NEW] [src/hooks/useLiveMonitor.ts](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/hooks/useLiveMonitor.ts)
- Stream subscription and live event queue for Live Monitor page.

---

### UI Components

#### [NEW] [src/components/layout/Sidebar.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/layout/Sidebar.tsx)
- Left sidebar with RoadVision AI logo, radar/vision icon, navigation links (Dashboard, Analyze, Live Monitor, Detections, Segmentation, Potholes, Traffic Objects, Analytics, Model Status, Settings), collapse toggle, and system health status footer.

#### [NEW] [src/components/layout/TopBar.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/layout/TopBar.tsx)
- Top navbar with current page title, description, real-time backend pipeline connection indicator pill (● AI Pipeline Connected / ○ Models Not Connected), notifications, and settings menu.

#### [NEW] [src/components/common/CanvasViewer.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/common/CanvasViewer.tsx)
- High-performance HTML5 canvas visualization engine for images & videos.
- Zoom, pan, reset, fullscreen controls.
- Dynamic layer toggling (Original, Objects, Road Mask, Potholes, Combined, Confidence).
- U-Net segmentation mask rendering with smooth opacity slider (0–100%).
- YOLO & Pothole bounding boxes with dynamic class labels + confidence scores.
- Bi-directional selection: Clicking box highlights item in list; selecting item in list highlights box on canvas.

#### [NEW] [src/components/analysis/MediaUpload.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/analysis/MediaUpload.tsx)
- Drag-and-drop media uploader supporting JPG, PNG, MP4, WebM with file validation.

#### [NEW] [src/components/analysis/PipelineProgress.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/analysis/PipelineProgress.tsx)
- Interactive visual model pipeline flow chart (Input Media -> YOLO -> U-Net -> Pothole -> Fusion -> Safety Analysis) showing real-time execution state per stage.

#### [NEW] [src/components/analysis/FusionSummary.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/analysis/FusionSummary.tsx)
- AI Safety Summary panel displaying dynamic risk level (Safe, Caution, High Risk, Unknown), risk score meter, road coverage %, and warning messages.

#### [NEW] [src/components/analysis/DetectionList.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/analysis/DetectionList.tsx)
- Right panel displaying detected objects & potholes with search, class filtering dynamically built from API responses, visibility toggles, and box coordinates.

#### [NEW] [src/components/model-status/ArchitectureDiagram.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/components/model-status/ArchitectureDiagram.tsx)
- Interactive architectural dataflow diagram. Node click displays microservice status & configuration.

---

### Pages

#### [NEW] [src/pages/Dashboard.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/Dashboard.tsx)
- Dynamic KPI cards (Objects Detected, Potholes Detected, Road Coverage %, Processing Status) displaying `—` / "Waiting for inference data" when unpopulated. Recent activity feed and model readiness overview.

#### [NEW] [src/pages/Analyze.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/Analyze.tsx)
- Main workspace uniting MediaUpload, PipelineProgress, CanvasViewer, FusionSummary, DetectionList, and ConfidenceSlider.

#### [NEW] [src/pages/LiveMonitor.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/LiveMonitor.tsx)
- Live streaming viewport with real-time AI event ticker and camera connection state ("No live stream connected" when inactive).

#### [NEW] [src/pages/DetectionResults.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/DetectionResults.tsx)
- Grid & table view of detected objects with class breakdown and bounding box inspection.

#### [NEW] [src/pages/RoadSegmentation.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/RoadSegmentation.tsx)
- U-Net road coverage breakdown, drivable area mask inspector, and mask opacity controls.

#### [NEW] [src/pages/Potholes.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/Potholes.tsx)
- Pothole inventory, density metrics, severity breakdown, and hazard alerts.

#### [NEW] [src/pages/TrafficObjects.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/TrafficObjects.tsx)
- Indian road traffic objects workspace (vehicles, pedestrians, tractors, animals, signs, obstacles).

#### [NEW] [src/pages/Analytics.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/Analytics.tsx)
- Dynamic Recharts analytics (Object Distribution, Confidence Histogram, Pothole Frequency, Road Coverage, Latency) with dedicated empty states.

#### [NEW] [src/pages/ModelStatus.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/ModelStatus.tsx)
- Health cards for YOLO, U-Net, Potholes, and Fusion Engine microservices plus interactive Architecture Diagram.

#### [NEW] [src/pages/Settings.tsx](file:///c:/Users/Parth%20Padwal/Downloads/SIH%202026/src/pages/Settings.tsx)
- Analysis threshold configurations, backend API URL setup, WebSocket streaming options, and Demo Mode toggle.

---

## Verification Plan

### Automated Tests / Build Checks
- Run `npm run build` to ensure error-free TypeScript compilation and bundle generation.
- Run `npm run lint` or type-checker to verify component signatures and clean imports.

### Manual Verification
- Test Drag & Drop upload workflow with sample road image/video.
- Toggle between live API empty states and Demo Mode to verify seamless transition.
- Test Canvas layer toggles (Original, Objects, Mask, Potholes, Combined, Opacity Slider).
- Test bi-directional box selection between Canvas overlay and Detection List.
- Verify responsive layout on Desktop, Tablet, and Mobile viewports.
- Check Model Status page health checkers and interactive Architecture Diagram.
