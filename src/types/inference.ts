export type ModelStageStatus =
  | "not_connected"
  | "waiting"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  id: string;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface PotholeDetection {
  id: string;
  confidence: number;
  bbox: BoundingBox;
  severity?: string; // Optional field, only rendered if backend supplies it
}

export interface UNetSegmentationResult {
  mask_url?: string;
  coverage_ratio?: number; // 0.0 to 1.0
  coverage_percent?: number; // 0.0 to 100.0
  processing_time_ms?: number;
  status: ModelStageStatus;
}

export interface YoloDetectionResult {
  detections: Detection[];
  processing_time_ms?: number;
  status: ModelStageStatus;
}

export interface PotholeDetectionResult {
  detections: PotholeDetection[];
  processing_time_ms?: number;
  status: ModelStageStatus;
}

export interface FusionResult {
  risk_level?: "safe" | "caution" | "critical" | "unknown";
  risk_score?: number; // e.g. 0 to 100 or 0.0 to 1.0
  warnings: string[];
  processing_time_ms?: number;
  status: ModelStageStatus;
}

export interface UnifiedInferenceResult {
  session_id: string;
  input: {
    media_url: string;
    media_type: "image" | "video";
    width?: number;
    height?: number;
    duration?: number;
  };
  yolo: YoloDetectionResult;
  road_segmentation: UNetSegmentationResult;
  potholes: PotholeDetectionResult;
  fusion: FusionResult;
  total_processing_time_ms?: number;
}

export interface InferenceSession {
  id: string;
  media_id: string;
  media_type: "image" | "video";
  media_name: string;
  created_at: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  models: {
    yolo: ModelStageStatus;
    unet: ModelStageStatus;
    pothole: ModelStageStatus;
    fusion: ModelStageStatus;
  };
  result?: UnifiedInferenceResult;
  error?: {
    code: string;
    message: string;
  };
}

export type LayerType = "original" | "objects" | "mask" | "potholes" | "combined" | "confidence";
