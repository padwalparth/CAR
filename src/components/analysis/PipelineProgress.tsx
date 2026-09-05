import React from 'react';
import { ModelStageStatus } from '../../types/inference';
import { CheckCircle2, AlertCircle, Clock, Loader2 } from 'lucide-react';

interface PipelineProgressProps {
  stages: {
    yolo: ModelStageStatus;
    unet: ModelStageStatus;
    pothole: ModelStageStatus;
    fusion: ModelStageStatus;
  };
  latencies?: {
    yolo?: number;
    unet?: number;
    pothole?: number;
    fusion?: number;
  };
  totalLatencyMs?: number;
}

export const PipelineProgress: React.FC<PipelineProgressProps> = ({
  stages,
  latencies,
  totalLatencyMs
}) => {
  const stageList = [
    { key: 'yolo', label: 'YOLO Detector', description: 'Objects & Vehicles', status: stages.yolo, latency: latencies?.yolo },
    { key: 'unet', label: 'U-Net Segmenter', description: 'Drivable Road Mask', status: stages.unet, latency: latencies?.unet },
    { key: 'pothole', label: 'Pothole Detector', description: 'Surface Hazards', status: stages.pothole, latency: latencies?.pothole },
    { key: 'fusion', label: 'Fusion Engine', description: 'Safety Assessment', status: stages.fusion, latency: latencies?.fusion }
  ];

  const getStatusBadge = (status: ModelStageStatus) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase text-[#111111] bg-[#53D769] px-2 py-0.5 border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
            <CheckCircle2 className="w-3 h-3 stroke-[3]" /> Completed
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase text-[#FFFFFF] bg-[#4D7CFE] px-2 py-0.5 border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
            <Loader2 className="w-3 h-3 animate-spin stroke-[3]" /> Processing
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase text-[#FFFFFF] bg-[#FF5A5F] px-2 py-0.5 border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
            <AlertCircle className="w-3 h-3 stroke-[3]" /> Failed
          </span>
        );
      case 'not_connected':
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase text-[#555555] bg-[#EBEBE5] px-2 py-0.5 border-2 border-[#111111]">
            <AlertCircle className="w-3 h-3 stroke-[3]" /> Offline
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase text-[#111111] bg-[#FFFFFF] px-2 py-0.5 border-2 border-[#111111]">
            <Clock className="w-3 h-3 stroke-[3]" /> Waiting
          </span>
        );
    }
  };

  return (
    <div className="neo-card p-5 mb-6 bg-[#FFFFFF]">
      <div className="flex items-center justify-between mb-4 pb-2 border-b-2 border-[#111111]">
        <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider font-display flex items-center gap-2">
          <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
          AI MODEL INFERENCE PIPELINE
        </h3>
        {totalLatencyMs ? (
          <span className="text-xs font-mono font-bold text-[#111111] bg-[#FFD84D] px-2.5 py-1 rounded border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
            TOTAL PIPELINE LATENCY: {totalLatencyMs} MS
          </span>
        ) : (
          <span className="text-xs text-[#555555] font-mono font-bold uppercase">Decoupled Microservices</span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {stageList.map((stage, idx) => (
          <div
            key={stage.key}
            className={`p-3.5 rounded-md border-2 border-[#111111] transition-all ${
              stage.status === 'completed'
                ? 'bg-[#FFFFFF] shadow-[4px_4px_0px_#53D769]'
                : stage.status === 'processing'
                ? 'bg-[#FFD84D] shadow-[4px_4px_0px_#111111]'
                : stage.status === 'failed'
                ? 'bg-[#FFFFFF] shadow-[4px_4px_0px_#FF5A5F]'
                : 'bg-[#F7F7F2] shadow-[3px_3px_0px_#111111]'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5 font-mono">
              <span className="text-[10px] font-bold text-[#555555] uppercase">STAGE 0{idx + 1}</span>
              {getStatusBadge(stage.status)}
            </div>

            <h4 className="text-xs font-bold text-[#111111] font-display uppercase">{stage.label}</h4>
            <p className="text-[11px] text-[#555555] font-medium mt-0.5 truncate">{stage.description}</p>

            <div className="mt-3 pt-2 border-t-2 border-[#111111] flex items-center justify-between text-[11px] font-mono font-bold">
              <span className="text-[#555555]">LATENCY</span>
              <span className="text-[#111111]">{stage.latency ? `${stage.latency} ms` : '—'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
