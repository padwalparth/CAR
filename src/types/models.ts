export type ModelHealthStatus =
  | "Connected"
  | "Connecting"
  | "Not Connected"
  | "Ready"
  | "Processing"
  | "Error"
  | "Offline";

export interface ModelHealthInfo {
  id: "yolo" | "unet" | "pothole" | "fusion";
  name: string;
  description: string;
  status: ModelHealthStatus;
  version?: string;
  latency_ms?: number;
  endpoint?: string;
  last_check_at?: string;
  error?: string;
}

export interface ArchitectureNode {
  id: string;
  label: string;
  type: "input" | "manager" | "model" | "fusion" | "output";
  status: ModelHealthStatus;
  version?: string;
  endpoint?: string;
  latency_ms?: number;
  description: string;
}
