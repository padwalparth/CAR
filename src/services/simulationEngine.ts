import { UnifiedInferenceResult } from '../types/inference';
import { VehicleState, SimObjectPosition, SimulationTimelineEvent } from '../types/simulation';

/**
 * Projects 2D image bounding box into normalized simulation spatial coordinates.
 * x: 0.0 (left shoulder) .. 0.5 (center lane) .. 1.0 (right shoulder)
 * y: 0.0 (closest to Ego vehicle) .. 1.0 (far horizon)
 */
export function projectDetectionsToSimulation(
  result: UnifiedInferenceResult
): SimObjectPosition[] {
  const items: SimObjectPosition[] = [];
  const imgW = result.input?.width || 1920;
  const imgH = result.input?.height || 1080;

  // 1. YOLO Traffic Objects
  if (result.yolo?.detections) {
    result.yolo.detections.forEach((det, idx) => {
      const { x1, y1, x2, y2 } = det.bbox;
      const xCenter = (x1 + x2) / 2;
      const yBottom = y2; // Bottom edge of bbox corresponds to ground contact point

      // Map X: 0..imgW -> 0.0..1.0
      const simX = Math.max(0.05, Math.min(0.95, xCenter / imgW));

      // Map Y: In perspective image, y=imgH is bottom (close), y=0 is top (far)
      // Standard perspective mapping along road corridor
      const yNorm = yBottom / imgH; // 0.0 top .. 1.0 bottom
      const simY = Math.max(0.05, Math.min(0.95, 1.0 - yNorm)); // 0.0 close .. 1.0 far

      items.push({
        id: det.id || `yolo_${idx}`,
        type: 'traffic',
        className: det.class_name || 'Object',
        confidence: det.confidence,
        bbox: det.bbox,
        simX,
        simY,
        sourceModel: 'YOLOv8 IDD Traffic Detector'
      });
    });
  }

  // 2. Potholes
  if (result.potholes?.detections) {
    result.potholes.detections.forEach((pot, idx) => {
      const { x1, y1, x2, y2 } = pot.bbox;
      const xCenter = (x1 + x2) / 2;
      const yCenter = (y1 + y2) / 2;

      const simX = Math.max(0.1, Math.min(0.9, xCenter / imgW));
      const yNorm = yCenter / imgH;
      const simY = Math.max(0.05, Math.min(0.95, 1.0 - yNorm));

      items.push({
        id: pot.id || `pothole_${idx}`,
        type: 'pothole',
        className: 'Pothole',
        confidence: pot.confidence,
        bbox: pot.bbox,
        simX,
        simY,
        severity: pot.severity || 'Not provided',
        sourceModel: 'Res2Net Pothole Detector'
      });
    });
  }

  return items;
}

/**
 * Deterministic vehicle response state machine based on Fusion risk output and hazard spatial arrangement.
 */
export function determineVehicleState(
  result: UnifiedInferenceResult,
  simObjects: SimObjectPosition[]
): VehicleState {
  const riskLevel = result.fusion?.risk_level || 'safe';
  const riskScore = result.fusion?.risk_score ?? 0;
  const potholes = result.potholes?.detections || [];
  const trafficCount = result.yolo?.detections?.length || 0;

  // 1. Check for immediate critical hazards in ego lane (center corridor: simX 0.35..0.65, simY < 0.45)
  const immediateHazards = simObjects.filter(
    (obj) => obj.simX >= 0.30 && obj.simX <= 0.70 && obj.simY <= 0.45
  );

  const hasPotholeInLane = immediateHazards.some((h) => h.type === 'pothole');
  const hasVehicleInLane = immediateHazards.some((h) => h.type === 'traffic');

  if (riskLevel === 'critical' || riskScore >= 70.0 || (hasPotholeInLane && potholes.length >= 2)) {
    if (hasPotholeInLane || hasVehicleInLane) {
      // Evaluate lateral clearance for hazard avoidance
      const leftClear = !simObjects.some((o) => o.simX < 0.35 && o.simY <= 0.45);
      const rightClear = !simObjects.some((o) => o.simX > 0.65 && o.simY <= 0.45);

      if (leftClear) return 'AVOID_LEFT';
      if (rightClear) return 'AVOID_RIGHT';
      return 'STOP';
    }
    return 'BRAKE';
  }

  if (riskLevel === 'caution' || riskScore >= 35.0 || potholes.length > 0 || trafficCount >= 5) {
    if (hasPotholeInLane) {
      const leftClear = !simObjects.some((o) => o.simX < 0.35 && o.simY <= 0.45);
      if (leftClear) return 'AVOID_LEFT';
    }
    return 'SLOW';
  }

  return 'GO';
}

/**
 * Generate timeline events from actual perception and state transitions.
 */
export function generateSimulationTimeline(
  result: UnifiedInferenceResult,
  vehicleState: VehicleState
): SimulationTimelineEvent[] {
  const events: SimulationTimelineEvent[] = [];

  events.push({
    id: 'evt_1',
    timeStr: '00:00',
    timeSec: 0,
    type: 'info',
    title: 'AI Perception Scene Ingested',
    details: `Session ID: ${result.session_id} (${result.input.media_type.toUpperCase()} ${result.input.width}x${result.input.height})`
  });

  const objCount = result.yolo?.detections?.length || 0;
  events.push({
    id: 'evt_2',
    timeStr: '00:01',
    timeSec: 1,
    type: objCount > 0 ? 'info' : 'info',
    title: `YOLOv8 Detection Completed`,
    details: objCount > 0 ? `Identified ${objCount} traffic objects in travel corridor.` : 'No traffic objects detected.'
  });

  const roadCov = result.road_segmentation?.coverage_ratio;
  const covPct = result.road_segmentation?.coverage_percent ?? (roadCov !== undefined ? roadCov * 100 : undefined);
  events.push({
    id: 'evt_3',
    timeStr: '00:02',
    timeSec: 2,
    type: 'info',
    title: `U-Net Road Segmentation Active`,
    details: covPct !== undefined ? `Drivable road coverage estimated at ${covPct.toFixed(1)}%` : 'Road segmentation ready.'
  });

  const potCount = result.potholes?.detections?.length || 0;
  events.push({
    id: 'evt_4',
    timeStr: '00:03',
    timeSec: 3,
    type: potCount > 0 ? 'hazard' : 'info',
    title: `Pothole Inspection Completed`,
    details: potCount > 0 ? `Detected ${potCount} road surface defect(s).` : 'No surface defects detected.'
  });

  const riskLevel = result.fusion?.risk_level || 'safe';
  const riskScore = result.fusion?.risk_score ?? 0;
  events.push({
    id: 'evt_5',
    timeStr: '00:04',
    timeSec: 4,
    type: riskLevel === 'critical' ? 'hazard' : riskLevel === 'caution' ? 'warning' : 'info',
    title: `Fusion Risk Evaluated: ${riskLevel.toUpperCase()}`,
    details: `Risk Index: ${riskScore.toFixed(1)} | Warnings: ${result.fusion?.warnings?.length || 0}`
  });

  events.push({
    id: 'evt_6',
    timeStr: '00:05',
    timeSec: 5,
    type: 'action',
    title: `Vehicle Response State: ${vehicleState}`,
    details: `Vehicle action dispatched based on perception estimates.`
  });

  return events;
}
