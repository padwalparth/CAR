import React from 'react';
import { UnifiedInferenceResult } from '../../types/inference';
import { VehicleState } from '../../types/simulation';
import { ShieldCheck, AlertTriangle, ShieldAlert, Car, AlertCircle, Layers } from 'lucide-react';

interface SimulationStatusProps {
  inferenceResult: UnifiedInferenceResult;
  vehicleState: VehicleState;
}

export const SimulationStatus: React.FC<SimulationStatusProps> = ({
  inferenceResult,
  vehicleState
}) => {
  const fusion = inferenceResult.fusion;
  const riskLevel = fusion?.risk_level || 'safe';
  const riskScore = fusion?.risk_score ?? 0;

  const yoloCount = inferenceResult.yolo?.detections?.length || 0;
  const potholeCount = inferenceResult.potholes?.detections?.length || 0;
  const roadCovRatio = inferenceResult.road_segmentation?.coverage_ratio;
  const roadCovPct = inferenceResult.road_segmentation?.coverage_percent ?? (roadCovRatio !== undefined ? roadCovRatio * 100 : 0);

  // Badge colors for vehicle response state
  const getVehicleStateBadge = (state: VehicleState) => {
    switch (state) {
      case 'GO':
        return { bg: 'bg-[#10B981]', text: 'text-[#FFFFFF]', label: 'GO (NORMAL SPEEDS)' };
      case 'SLOW':
        return { bg: 'bg-[#FFD84D]', text: 'text-[#111111]', label: 'SLOW (CAUTION)' };
      case 'BRAKE':
        return { bg: 'bg-[#F59E0B]', text: 'text-[#FFFFFF]', label: 'BRAKE (DECELERATING)' };
      case 'STOP':
        return { bg: 'bg-[#EF4444]', text: 'text-[#FFFFFF]', label: 'STOP (HAZARD AHEAD)' };
      case 'AVOID_LEFT':
        return { bg: 'bg-[#3B82F6]', text: 'text-[#FFFFFF]', label: 'AVOID LEFT (SWERVE LEFT)' };
      case 'AVOID_RIGHT':
        return { bg: 'bg-[#8B5CF6]', text: 'text-[#FFFFFF]', label: 'AVOID RIGHT (SWERVE RIGHT)' };
      default:
        return { bg: 'bg-[#111111]', text: 'text-[#FFFFFF]', label: state };
    }
  };

  const vBadge = getVehicleStateBadge(vehicleState);

  const getRiskBadge = (level: string) => {
    switch (level) {
      case 'critical':
        return { bg: 'bg-[#EF4444]', text: 'text-[#FFFFFF]', icon: ShieldAlert, label: 'CRITICAL HAZARD' };
      case 'caution':
        return { bg: 'bg-[#FFD84D]', text: 'text-[#111111]', icon: AlertTriangle, label: 'CAUTION ADVISED' };
      case 'safe':
      default:
        return { bg: 'bg-[#10B981]', text: 'text-[#FFFFFF]', icon: ShieldCheck, label: 'CLEAR ROAD' };
    }
  };

  const rBadge = getRiskBadge(riskLevel);
  const RiskIcon = rBadge.icon;

  return (
    <div className="neo-card p-5 bg-[#FFFFFF] space-y-5">
      {/* 1. Header Title */}
      <div className="flex items-center justify-between border-b-3 border-[#111111] pb-3">
        <h3 className="text-sm font-bold text-[#111111] font-display uppercase tracking-wider">
          Live Simulation State
        </h3>
        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-[#F7F7F2] border border-[#111111]">
          REAL-TIME TELEMETRY
        </span>
      </div>

      {/* 2. Vehicle State Banner */}
      <div className="p-3.5 rounded-md bg-[#F7F7F2] border-2 border-[#111111] space-y-1.5">
        <span className="text-[10px] font-mono font-bold text-[#555555] tracking-wider uppercase block">
          RECOMMENDED VEHICLE ACTION
        </span>
        <div className={`px-3 py-2 rounded-md font-mono font-bold text-xs border-2 border-[#111111] shadow-[2px_2px_0px_#111111] uppercase flex items-center justify-between ${vBadge.bg} ${vBadge.text}`}>
          <span>{vBadge.label}</span>
          <span className="text-[10px] opacity-80">STATE: {vehicleState}</span>
        </div>
      </div>

      {/* 3. Fusion Risk Badge */}
      <div className="p-3.5 rounded-md bg-[#F7F7F2] border-2 border-[#111111] space-y-2">
        <span className="text-[10px] font-mono font-bold text-[#555555] tracking-wider uppercase block">
          FUSION PERCEPTION RISK
        </span>
        <div className={`p-3 rounded-md font-mono font-bold border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-between ${rBadge.bg} ${rBadge.text}`}>
          <div className="flex items-center gap-2">
            <RiskIcon className="w-5 h-5 stroke-[2.5]" />
            <span className="text-xs uppercase">{rBadge.label}</span>
          </div>
          <span className="text-sm font-extrabold">{riskScore.toFixed(1)} / 100</span>
        </div>
      </div>

      {/* 4. Perception Metrics Grid */}
      <div className="grid grid-cols-3 gap-2">
        <div className="p-2.5 rounded bg-[#FFFFFF] border-2 border-[#111111] text-center font-mono">
          <Car className="w-4 h-4 mx-auto mb-1 text-[#10B981] stroke-[2.5]" />
          <span className="text-[10px] text-[#555555] font-bold block uppercase">OBJECTS</span>
          <span className="text-sm font-extrabold text-[#111111]">{yoloCount}</span>
        </div>

        <div className="p-2.5 rounded bg-[#FFFFFF] border-2 border-[#111111] text-center font-mono">
          <AlertCircle className="w-4 h-4 mx-auto mb-1 text-[#EF4444] stroke-[2.5]" />
          <span className="text-[10px] text-[#555555] font-bold block uppercase">POTHOLES</span>
          <span className="text-sm font-extrabold text-[#111111]">{potholeCount}</span>
        </div>

        <div className="p-2.5 rounded bg-[#FFFFFF] border-2 border-[#111111] text-center font-mono">
          <Layers className="w-4 h-4 mx-auto mb-1 text-[#38BDF8] stroke-[2.5]" />
          <span className="text-[10px] text-[#555555] font-bold block uppercase">COVERAGE</span>
          <span className="text-sm font-extrabold text-[#111111]">{roadCovPct.toFixed(1)}%</span>
        </div>
      </div>

      {/* 5. Warnings List */}
      <div className="space-y-1.5 font-mono">
        <span className="text-[10px] font-bold text-[#555555] tracking-wider uppercase block">
          SAFETY WARNINGS ({fusion?.warnings?.length || 0})
        </span>
        {fusion?.warnings && fusion.warnings.length > 0 ? (
          <ul className="space-y-1 text-xs">
            {fusion.warnings.map((warn, i) => (
              <li key={i} className="p-2 rounded bg-[#FFFBEB] border border-[#F59E0B] text-[#92400E] flex items-start gap-1.5 font-medium">
                <span className="font-bold">•</span>
                <span>{warn}</span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="p-2.5 rounded bg-[#F0FDF4] border border-[#10B981] text-[#166534] text-xs font-medium">
            No critical hazards identified in travel corridor.
          </div>
        )}
      </div>
    </div>
  );
};
