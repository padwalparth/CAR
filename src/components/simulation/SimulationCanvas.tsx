import React, { useEffect, useRef, useCallback } from 'react';
import { UnifiedInferenceResult } from '../../types/inference';
import { VehicleState, SimObjectPosition } from '../../types/simulation';

interface SimulationCanvasProps {
  inferenceResult: UnifiedInferenceResult;
  simObjects: SimObjectPosition[];
  vehicleState: VehicleState;
  isPlaying: boolean;
  speed: number;
  selectedObjectId?: string;
  onSelectObject: (id?: string) => void;
  showObjects: boolean;
  showRoad: boolean;
  showPotholes: boolean;
  showRiskZones: boolean;
  showLabels: boolean;
  showTrajectory: boolean;
}

export const SimulationCanvas: React.FC<SimulationCanvasProps> = ({
  inferenceResult,
  simObjects,
  vehicleState,
  isPlaying,
  speed,
  selectedObjectId,
  onSelectObject,
  showObjects,
  showRoad,
  showPotholes,
  showRiskZones,
  showLabels,
  showTrajectory
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number | null>(null);

  // Animation offsets & vehicle steering state
  const offsetRef = useRef<number>(0);
  const egoLateralOffsetRef = useRef<number>(0); // -1.0 (left) .. 0.0 (center) .. 1.0 (right)

  // Target lateral offset based on vehicle state
  let targetLateral = 0.0;
  if (vehicleState === 'AVOID_LEFT') targetLateral = -0.32;
  if (vehicleState === 'AVOID_RIGHT') targetLateral = 0.32;

  // Speed factor for scrolling road lines
  let speedMult = 1.0;
  if (vehicleState === 'SLOW') speedMult = 0.5;
  if (vehicleState === 'BRAKE') speedMult = 0.25;
  if (vehicleState === 'STOP') speedMult = 0.0;

  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Smooth lateral steering motion
    egoLateralOffsetRef.current += (targetLateral - egoLateralOffsetRef.current) * 0.05;

    // Update line scrolling offset
    if (isPlaying) {
      offsetRef.current = (offsetRef.current + 2.5 * speed * speedMult) % 60;
    }

    // 1. Clear & Paint Dark Tactical Background
    ctx.fillStyle = '#0D1117';
    ctx.fillRect(0, 0, width, height);

    // Grid lines for engineering aesthetic
    ctx.strokeStyle = '#1F2937';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // 2. Road Geometry (Bird's-Eye Perspective)
    const roadWidth = width * 0.55;
    const roadLeft = (width - roadWidth) / 2;
    const roadRight = roadLeft + roadWidth;

    // Draw Asphalt Carriageway
    ctx.fillStyle = '#161B22';
    ctx.fillRect(roadLeft, 0, roadWidth, height);

    // Draw U-Net Road Segmentation Overlay if enabled
    if (showRoad) {
      const coverageRatio = inferenceResult.road_segmentation?.coverage_ratio ?? 0.30;
      const drivableWidth = roadWidth * Math.min(1.0, Math.max(0.3, coverageRatio * 2.2));
      const drivableLeft = (width - drivableWidth) / 2;

      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.fillRect(drivableLeft, 0, drivableWidth, height);

      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.beginPath();
      ctx.moveTo(drivableLeft, 0);
      ctx.lineTo(drivableLeft, height);
      ctx.moveTo(drivableLeft + drivableWidth, 0);
      ctx.lineTo(drivableLeft + drivableWidth, height);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Solid Outer Road Shoulders
    ctx.strokeStyle = '#FFD84D';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(roadLeft, 0);
    ctx.lineTo(roadLeft, height);
    ctx.moveTo(roadRight, 0);
    ctx.lineTo(roadRight, height);
    ctx.stroke();

    // Center Dashed Lane Lines (Scrolling animation)
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([25, 25]);
    ctx.lineDashOffset = -offsetRef.current;

    const lane1 = roadLeft + roadWidth * 0.33;
    const lane2 = roadLeft + roadWidth * 0.66;

    ctx.beginPath();
    ctx.moveTo(lane1, 0);
    ctx.lineTo(lane1, height);
    ctx.moveTo(lane2, 0);
    ctx.lineTo(lane2, height);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // 3. Risk Zones Overlay
    if (showRiskZones) {
      simObjects.forEach((obj) => {
        const x = roadLeft + obj.simX * roadWidth;
        const y = height - obj.simY * (height * 0.8) - 60;

        ctx.fillStyle = obj.type === 'pothole' ? 'rgba(239, 68, 68, 0.18)' : 'rgba(245, 158, 11, 0.14)';
        ctx.strokeStyle = obj.type === 'pothole' ? '#EF4444' : '#F59E0B';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);

        const radius = obj.type === 'pothole' ? 38 : 45;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);

        if (showLabels) {
          ctx.font = '700 9px monospace';
          ctx.fillStyle = obj.type === 'pothole' ? '#EF4444' : '#F59E0B';
          ctx.textAlign = 'center';
          ctx.fillText('Estimated Risk Zone', x, y - radius - 4);
        }
      });
    }

    // 4. Potholes Overlay
    if (showPotholes) {
      simObjects
        .filter((o) => o.type === 'pothole')
        .forEach((pot) => {
          const x = roadLeft + pot.simX * roadWidth;
          const y = height - pot.simY * (height * 0.8) - 60;
          const isSelected = selectedObjectId === pot.id;

          // Outer pulsing warning circle
          ctx.fillStyle = '#EF4444';
          ctx.beginPath();
          ctx.arc(x, y, isSelected ? 18 : 14, 0, Math.PI * 2);
          ctx.fill();

          // Inner dark core
          ctx.fillStyle = '#111111';
          ctx.beginPath();
          ctx.arc(x, y, isSelected ? 12 : 9, 0, Math.PI * 2);
          ctx.fill();

          // Hazard Crosshair Lines
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(x - 6, y);
          ctx.lineTo(x + 6, y);
          ctx.moveTo(x, y - 6);
          ctx.lineTo(x, y + 6);
          ctx.stroke();

          if (showLabels) {
            ctx.font = '700 10px monospace';
            ctx.fillStyle = '#EF4444';
            ctx.textAlign = 'center';
            ctx.fillText(
              `POTHOLE (${(pot.confidence * 100).toFixed(0)}%)`,
              x,
              y + 24
            );
          }
        });
    }

    // 5. Traffic Objects Overlay (YOLO)
    if (showObjects) {
      simObjects
        .filter((o) => o.type === 'traffic')
        .forEach((obj) => {
          const x = roadLeft + obj.simX * roadWidth;
          const y = height - obj.simY * (height * 0.8) - 60;
          const isSelected = selectedObjectId === obj.id;

          const objW = obj.className.toLowerCase().includes('truck') || obj.className.toLowerCase().includes('bus') ? 44 : 32;
          const objH = obj.className.toLowerCase().includes('truck') || obj.className.toLowerCase().includes('bus') ? 64 : 48;

          ctx.save();
          ctx.translate(x, y);

          // Selection highlight halo
          if (isSelected) {
            ctx.strokeStyle = '#FFD84D';
            ctx.lineWidth = 3;
            ctx.strokeRect(-objW / 2 - 4, -objH / 2 - 4, objW + 8, objH + 8);
          }

          // Object box representation
          ctx.fillStyle = obj.className.toLowerCase().includes('person') || obj.className.toLowerCase().includes('rider') ? '#3B82F6' : '#10B981';
          ctx.strokeStyle = '#111111';
          ctx.lineWidth = 2.5;

          ctx.fillRect(-objW / 2, -objH / 2, objW, objH);
          ctx.strokeRect(-objW / 2, -objH / 2, objW, objH);

          // Rear red tail lights for vehicles
          ctx.fillStyle = '#EF4444';
          ctx.fillRect(-objW / 2 + 2, objH / 2 - 4, 6, 3);
          ctx.fillRect(objW / 2 - 8, objH / 2 - 4, 6, 3);

          ctx.restore();

          if (showLabels) {
            ctx.font = '700 10px monospace';
            ctx.fillStyle = '#10B981';
            ctx.textAlign = 'center';
            ctx.fillText(
              `${obj.className.toUpperCase()} (${(obj.confidence * 100).toFixed(0)}%)`,
              x,
              y - objH / 2 - 6
            );
          }
        });
    }

    // 6. Ego Vehicle Rendering
    const egoX = roadLeft + (0.5 + egoLateralOffsetRef.current * 0.35) * roadWidth;
    const egoY = height - 90;
    const egoW = 38;
    const egoH = 60;

    // Trajectory Vector Line if enabled
    if (showTrajectory) {
      ctx.strokeStyle = vehicleState.includes('AVOID') ? '#F59E0B' : '#10B981';
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(egoX, egoY - egoH / 2);
      ctx.lineTo(egoX, egoY - 200);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Headlight cones
    const lightGrad = ctx.createLinearGradient(egoX, egoY - egoH / 2, egoX, egoY - 220);
    lightGrad.addColorStop(0, 'rgba(255, 255, 255, 0.45)');
    lightGrad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');

    ctx.fillStyle = lightGrad;
    ctx.beginPath();
    ctx.moveTo(egoX - 12, egoY - egoH / 2);
    ctx.lineTo(egoX - 45, egoY - 220);
    ctx.lineTo(egoX + 45, egoY - 220);
    ctx.lineTo(egoX + 12, egoY - egoH / 2);
    ctx.closePath();
    ctx.fill();

    // Vehicle Body (Bright Neobrutalist Yellow)
    ctx.fillStyle = '#FFD84D';
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 3;
    ctx.fillRect(egoX - egoW / 2, egoY - egoH / 2, egoW, egoH);
    ctx.strokeRect(egoX - egoW / 2, egoY - egoH / 2, egoW, egoH);

    // Windshield Cabin
    ctx.fillStyle = '#111111';
    ctx.fillRect(egoX - egoW / 2 + 5, egoY - 14, egoW - 10, 18);

    // Brake Lights
    const isBraking = vehicleState === 'BRAKE' || vehicleState === 'STOP';
    ctx.fillStyle = isBraking ? '#FF0000' : '#990000';
    ctx.fillRect(egoX - egoW / 2 + 2, egoY + egoH / 2 - 5, 8, 4);
    ctx.fillRect(egoX + egoW / 2 - 10, egoY + egoH / 2 - 5, 8, 4);

    if (isBraking) {
      // Glow effect for braking
      ctx.fillStyle = 'rgba(255, 0, 0, 0.4)';
      ctx.beginPath();
      ctx.arc(egoX - egoW / 2 + 6, egoY + egoH / 2 - 3, 10, 0, Math.PI * 2);
      ctx.arc(egoX + egoW / 2 - 6, egoY + egoH / 2 - 3, 10, 0, Math.PI * 2);
      ctx.fill();
    }

    // Vehicle State Label Badge
    ctx.font = '800 11px monospace';
    ctx.fillStyle = '#111111';
    const labelText = `EGO: ${vehicleState}`;
    const textWidth = ctx.measureText(labelText).width;

    ctx.fillStyle = '#FFD84D';
    ctx.fillRect(egoX - textWidth / 2 - 6, egoY + egoH / 2 + 8, textWidth + 12, 18);
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 2;
    ctx.strokeRect(egoX - textWidth / 2 - 6, egoY + egoH / 2 + 8, textWidth + 12, 18);

    ctx.fillStyle = '#111111';
    ctx.textAlign = 'center';
    ctx.fillText(labelText, egoX, egoY + egoH / 2 + 21);

    // 7. Tactical Overlays & Disclaimers
    // Top-left orientation watermark
    ctx.font = '700 10px monospace';
    ctx.fillStyle = '#9CA3AF';
    ctx.textAlign = 'left';
    ctx.fillText('2D SCENE PROJECTION — TOP-DOWN SCHEMATIC', 14, 22);

    // Bottom-right mandatory disclaimer
    ctx.font = '700 9px monospace';
    ctx.fillStyle = '#6B7280';
    ctx.textAlign = 'right';
    ctx.fillText('Schematic simulation — not to scale.', width - 14, height - 14);

  }, [
    inferenceResult,
    simObjects,
    vehicleState,
    isPlaying,
    speed,
    speedMult,
    targetLateral,
    selectedObjectId,
    showObjects,
    showRoad,
    showPotholes,
    showRiskZones,
    showLabels,
    showTrajectory
  ]);

  // Click handler for canvas hit-testing
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const roadWidth = canvas.width * 0.55;
    const roadLeft = (canvas.width - roadWidth) / 2;

    let clickedId: string | undefined = undefined;

    simObjects.forEach((obj) => {
      const x = roadLeft + obj.simX * roadWidth;
      const y = canvas.height - obj.simY * (canvas.height * 0.8) - 60;
      const dist = Math.hypot(clickX - x, clickY - y);

      if (dist <= 30) {
        clickedId = obj.id;
      }
    });

    onSelectObject(clickedId);
  };

  // Resize Canvas to container bounds
  useEffect(() => {
    const handleResize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;

      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      drawScene();
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawScene]);

  // Continuous animation loop using requestAnimationFrame
  useEffect(() => {
    let animId: number;

    const renderLoop = () => {
      drawScene();
      animId = requestAnimationFrame(renderLoop);
    };

    animId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animId);
  }, [drawScene]);

  return (
    <div ref={containerRef} className="relative w-full h-[540px] bg-[#0D1117] rounded-lg border-3 border-[#111111] overflow-hidden shadow-[5px_5px_0px_#111111]">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full h-full cursor-crosshair block"
      />
    </div>
  );
};
