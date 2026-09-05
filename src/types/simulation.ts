import { UnifiedInferenceResult } from './inference';

export type VehicleState =
  | 'GO'
  | 'SLOW'
  | 'BRAKE'
  | 'STOP'
  | 'AVOID_LEFT'
  | 'AVOID_RIGHT';

export interface SimulationState {
  isPlaying: boolean;
  playbackTime: number; // in seconds
  speed: number; // 0.5, 1, 2, 4
  vehicleState: VehicleState;
  selectedObjectId?: string;
  showObjects: boolean;
  showRoad: boolean;
  showPotholes: boolean;
  showRiskZones: boolean;
  showLabels: boolean;
  showTrajectory: boolean;
}

export interface SimulationTimelineEvent {
  id: string;
  timeStr: string;
  timeSec: number;
  type: 'info' | 'warning' | 'hazard' | 'action';
  title: string;
  details: string;
}

export interface SimObjectPosition {
  id: string;
  type: 'traffic' | 'pothole';
  className: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  simX: number; // 0.0 (left) to 1.0 (right)
  simY: number; // 0.0 (near ego) to 1.0 (horizon)
  severity?: string;
  sourceModel: string;
}
