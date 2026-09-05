import { InferenceSession, UnifiedInferenceResult } from '../types/inference';
import { ModelHealthInfo } from '../types/models';
import { AnalyticsData } from '../types/analytics';

// High quality SVG Data URI representing an Indian road scene for offline visual demonstration
export const SAMPLE_ROAD_IMAGE_URL = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="sky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="%230f172a"/>
      <stop offset="100%" stop-color="%231e293b"/>
    </linearGradient>
    <linearGradient id="road" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="%23334155"/>
      <stop offset="100%" stop-color="%230f172a"/>
    </linearGradient>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
    </pattern>
  </defs>
  <!-- Background & Sky -->
  <rect width="1280" height="720" fill="url(%23sky)"/>
  <rect width="1280" height="720" fill="url(%23grid)"/>
  
  <!-- Horizon & Scenery -->
  <path d="M 0 320 Q 320 280 640 310 T 1280 290 L 1280 720 L 0 720 Z" fill="%231e293b" opacity="0.6"/>
  
  <!-- Indian Two-Lane Asphalt Road Perspective -->
  <polygon points="100,720 1180,720 720,340 560,340" fill="url(%23road)"/>
  
  <!-- Road Edges & Center Marking -->
  <line x1="560" y1="340" x2="100" y2="720" stroke="%23f59e0b" stroke-width="4" stroke-dasharray="15,15"/>
  <line x1="720" y1="340" x2="1180" y2="720" stroke="%23ffffff" stroke-width="4"/>
  <line x1="640" y1="340" x2="640" y2="720" stroke="%23ffffff" stroke-dasharray="20,20" stroke-width="4"/>
  
  <!-- Simulated Objects in Road Scene -->
  <!-- Tractor / Commercial Vehicle (Left Lane) -->
  <rect x="240" y="440" width="180" height="130" rx="8" fill="%231e3a8a" stroke="%233b82f6" stroke-width="2" opacity="0.8"/>
  <text x="330" y="510" fill="%2393c5fd" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Tractor / Vehicle</text>
  
  <!-- Autorickshaw / Compact Vehicle (Center-Right) -->
  <rect x="680" y="460" width="140" height="110" rx="6" fill="%23854d0e" stroke="%23eab308" stroke-width="2" opacity="0.8"/>
  <text x="750" y="520" fill="%23fef08a" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Autorickshaw</text>
  
  <!-- Pedestrian (Road Shoulder Right) -->
  <circle cx="980" cy="450" r="14" fill="%23065f46" stroke="%2310b981" stroke-width="2"/>
  <rect x="970" y="464" width="20" height="40" rx="4" fill="%23065f46"/>
  
  <!-- Simulated Pothole 1 (Foreground Center) -->
  <ellipse cx="580" cy="610" rx="65" ry="32" fill="%23090d14" stroke="%23ef4444" stroke-width="3" stroke-dasharray="6,4"/>
  <text x="580" y="615" fill="%23fca5a5" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Pothole Hazard</text>

  <!-- Simulated Pothole 2 (Mid-ground Left) -->
  <ellipse cx="410" cy="510" rx="45" ry="20" fill="%23090d14" stroke="%23ef4444" stroke-width="2" stroke-dasharray="4,3"/>

  <!-- Watermark HUD -->
  <text x="30" y="40" fill="%2300f2fe" font-family="monospace" font-size="16" font-weight="bold">ROADVISION AI — SAMPLE SCENE (INDIAN HIGHWAY)</text>
</svg>`;

export const DEMO_HEALTH_STATUS: ModelHealthInfo[] = [
  {
    id: "yolo",
    name: "YOLO Object Detector",
    description: "Detects vehicles, pedestrians, animals, tractors & traffic obstacles",
    status: "Connected",
    version: "v8.3.4-onnx",
    latency_ms: 124,
    endpoint: "/api/models/yolo",
    last_check_at: new Date().toISOString()
  },
  {
    id: "unet",
    name: "U-Net Road Segmentation",
    description: "Segments drivable road boundaries & computes coverage percentage",
    status: "Connected",
    version: "v2.1.0-torch",
    latency_ms: 186,
    endpoint: "/api/models/unet",
    last_check_at: new Date().toISOString()
  },
  {
    id: "pothole",
    name: "Pothole Detector",
    description: "Identifies road surface hazards, deep cracks & pothole bounding boxes",
    status: "Connected",
    version: "v1.4.2-tensorrt",
    latency_ms: 95,
    endpoint: "/api/models/potholes",
    last_check_at: new Date().toISOString()
  },
  {
    id: "fusion",
    name: "Fusion Engine",
    description: "Fuses multi-model detections to generate safety scores & warnings",
    status: "Connected",
    version: "v3.0.1-rule-ml",
    latency_ms: 32,
    endpoint: "/api/models/fusion",
    last_check_at: new Date().toISOString()
  }
];

export const DEMO_UNIFIED_RESULT: UnifiedInferenceResult = {
  session_id: "sess_demo_8f92a10c",
  input: {
    media_url: SAMPLE_ROAD_IMAGE_URL,
    media_type: "image",
    width: 1280,
    height: 720
  },
  yolo: {
    status: "completed",
    processing_time_ms: 124,
    detections: [
      {
        id: "det_yolo_1",
        class_name: "tractor",
        confidence: 0.94,
        bbox: { x1: 240, y1: 440, x2: 420, y2: 570 }
      },
      {
        id: "det_yolo_2",
        class_name: "autorickshaw",
        confidence: 0.91,
        bbox: { x1: 680, y1: 460, x2: 820, y2: 570 }
      },
      {
        id: "det_yolo_3",
        class_name: "pedestrian",
        confidence: 0.88,
        bbox: { x1: 960, y1: 430, x2: 1000, y2: 510 }
      }
    ]
  },
  road_segmentation: {
    status: "completed",
    processing_time_ms: 186,
    coverage_ratio: 0.68,
    coverage_percent: 68.0,
    mask_url: SAMPLE_ROAD_IMAGE_URL
  },
  potholes: {
    status: "completed",
    processing_time_ms: 95,
    detections: [
      {
        id: "det_pot_1",
        confidence: 0.93,
        bbox: { x1: 515, y1: 578, x2: 645, y2: 642 },
        severity: "High"
      },
      {
        id: "det_pot_2",
        confidence: 0.86,
        bbox: { x1: 365, y1: 490, x2: 455, y2: 530 },
        severity: "Moderate"
      }
    ]
  },
  fusion: {
    status: "completed",
    processing_time_ms: 32,
    risk_level: "caution",
    risk_score: 68,
    warnings: [
      "2 Pothole surface hazards detected in active driving lane",
      "Heterogeneous Indian traffic (tractor & autorickshaw in close proximity)",
      "Pedestrian present near right road shoulder"
    ]
  },
  total_processing_time_ms: 437
};

export const DEMO_INFERENCE_SESSION: InferenceSession = {
  id: "sess_demo_8f92a10c",
  media_id: "media_img_001",
  media_type: "image",
  media_name: "indian_highway_nh44.jpg",
  created_at: new Date().toISOString(),
  status: "completed",
  models: {
    yolo: "completed",
    unet: "completed",
    pothole: "completed",
    fusion: "completed"
  },
  result: DEMO_UNIFIED_RESULT
};

export const DEMO_ANALYTICS: AnalyticsData = {
  total_sessions: 142,
  total_objects_detected: 854,
  total_potholes_detected: 196,
  average_road_coverage: 0.72,
  object_distribution: [
    { class_name: "vehicle", count: 342 },
    { class_name: "autorickshaw", count: 184 },
    { class_name: "pedestrian", count: 128 },
    { class_name: "tractor", count: 94 },
    { class_name: "animal", count: 62 },
    { class_name: "traffic_sign", count: 44 }
  ],
  confidence_distribution: [
    { range: "50-60%", count: 48 },
    { range: "60-70%", count: 112 },
    { range: "70-80%", count: 230 },
    { range: "80-90%", count: 320 },
    { range: "90-100%", count: 144 }
  ],
  pothole_timeline: [
    { timestamp: "08:00", count: 12 },
    { timestamp: "10:00", count: 28 },
    { timestamp: "12:00", count: 45 },
    { timestamp: "14:00", count: 32 },
    { timestamp: "16:00", count: 54 },
    { timestamp: "18:00", count: 25 }
  ],
  road_coverage_trend: [
    { timestamp: "08:00", coverage_percentage: 76 },
    { timestamp: "10:00", coverage_percentage: 71 },
    { timestamp: "12:00", coverage_percentage: 68 },
    { timestamp: "14:00", coverage_percentage: 74 },
    { timestamp: "16:00", coverage_percentage: 70 },
    { timestamp: "18:00", coverage_percentage: 73 }
  ],
  latency_metrics: [
    { model: "YOLO Detector", latency_ms: 124 },
    { model: "U-Net Segmenter", latency_ms: 186 },
    { model: "Pothole Detector", latency_ms: 95 },
    { model: "Fusion Engine", latency_ms: 32 }
  ]
};
