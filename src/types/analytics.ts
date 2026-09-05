export interface ObjectDistributionItem {
  class_name: string;
  count: number;
}

export interface ConfidenceDistributionBucket {
  range: string; // e.g. "50-60%"
  count: number;
}

export interface PotholeFrequencyTimelineItem {
  timestamp: string;
  count: number;
}

export interface RoadCoverageTrendItem {
  timestamp: string;
  coverage_percentage: number;
}

export interface ModelProcessingLatencyItem {
  model: string;
  latency_ms: number;
}

export interface AnalyticsData {
  total_sessions: number;
  total_objects_detected: number;
  total_potholes_detected: number;
  average_road_coverage: number;
  object_distribution: ObjectDistributionItem[];
  confidence_distribution: ConfidenceDistributionBucket[];
  pothole_timeline: PotholeFrequencyTimelineItem[];
  road_coverage_trend: RoadCoverageTrendItem[];
  latency_metrics: ModelProcessingLatencyItem[];
}
