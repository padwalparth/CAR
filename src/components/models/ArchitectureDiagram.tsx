import React from 'react';
import { ModelHealthInfo } from '../../types/models';
import { ArrowDown, Cpu, ShieldCheck, FileText, Layers, AlertCircle, CheckCircle2 } from 'lucide-react';

interface ArchitectureDiagramProps {
  modelsHealth: ModelHealthInfo[];
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
}

export const ArchitectureDiagram: React.FC<ArchitectureDiagramProps> = ({
  modelsHealth,
  selectedModelId,
  onSelectModel
}) => {
  const getStatus = (id: string) => {
    const found = modelsHealth.find((m) => m.id === id);
    return found?.status || 'Not Connected';
  };

  const isLive = (id: string) => {
    const s = getStatus(id);
    return s === 'Connected' || s === 'Ready';
  };

  return (
    <div className="neo-card-lg p-6 space-y-6 bg-[#FFFFFF]">
      <div className="flex items-center justify-between pb-3 border-b-2 border-[#111111]">
        <div>
          <h3 className="text-sm font-bold text-[#111111] flex items-center gap-2 font-display uppercase tracking-wider">
            <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
            RoadVision AI Multi-Model Architecture
          </h3>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Click any model node to view microservice health &amp; endpoints
          </p>
        </div>
        <span className="text-[11px] font-mono font-bold text-[#111111] bg-[#FFD84D] px-3 py-1 border-2 border-[#111111] shadow-[2px_2px_0px_#111111] uppercase tracking-wider">
          Decoupled Microservices
        </span>
      </div>

      {/* Visual Flow Topology */}
      <div className="flex flex-col items-center gap-3 py-4">
        {/* Step 1: Input Media */}
        <div className="px-6 py-3 bg-[#F7F7F2] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] text-[#111111] text-xs font-bold flex items-center gap-2">
          <FileText className="w-4 h-4 stroke-[2.5]" />
          INPUT IMAGE / VIDEO MEDIA
        </div>

        <ArrowDown className="w-4 h-4 text-[#111111] stroke-[2.5]" />

        {/* Step 2: Inference Manager */}
        <div className="px-6 py-3 bg-[#4D7CFE] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] text-[#FFFFFF] text-xs font-bold flex items-center gap-2">
          <Cpu className="w-4 h-4 stroke-[2.5]" />
          INFERENCE SESSION MANAGER
        </div>

        <ArrowDown className="w-4 h-4 text-[#111111] stroke-[2.5]" />

        {/* Step 3: Parallel Independent Model Microservices */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-2xl">
          {/* YOLO Node */}
          <button
            onClick={() => onSelectModel('yolo')}
            className={`p-4 border-3 border-[#111111] transition-all text-center flex flex-col items-center ${
              selectedModelId === 'yolo'
                ? 'bg-[#FFD84D] shadow-[6px_6px_0px_#111111] -translate-y-1'
                : isLive('yolo')
                ? 'bg-[#53D769] shadow-[4px_4px_0px_#111111]'
                : 'bg-[#FFFFFF] shadow-[4px_4px_0px_#111111]'
            }`}
          >
            <Cpu className="w-6 h-6 mb-2 text-[#111111] stroke-[2.5]" />
            <h5 className="text-xs font-bold text-[#111111] font-display uppercase">YOLO Detector</h5>
            <span className="text-[10px] text-[#555555] mt-1 font-mono font-bold">Objects &amp; Vehicles</span>
            <span className="mt-2 text-[10px] font-mono font-bold px-2 py-0.5 bg-[#FFFFFF] border border-[#111111] uppercase">
              {getStatus('yolo')}
            </span>
          </button>

          {/* U-Net Node */}
          <button
            onClick={() => onSelectModel('unet')}
            className={`p-4 border-3 border-[#111111] transition-all text-center flex flex-col items-center ${
              selectedModelId === 'unet'
                ? 'bg-[#FFD84D] shadow-[6px_6px_0px_#111111] -translate-y-1'
                : isLive('unet')
                ? 'bg-[#53D769] shadow-[4px_4px_0px_#111111]'
                : 'bg-[#FFFFFF] shadow-[4px_4px_0px_#111111]'
            }`}
          >
            <Layers className="w-6 h-6 mb-2 text-[#111111] stroke-[2.5]" />
            <h5 className="text-xs font-bold text-[#111111] font-display uppercase">U-Net Segmenter</h5>
            <span className="text-[10px] text-[#555555] mt-1 font-mono font-bold">Road Coverage</span>
            <span className="mt-2 text-[10px] font-mono font-bold px-2 py-0.5 bg-[#FFFFFF] border border-[#111111] uppercase">
              {getStatus('unet')}
            </span>
          </button>

          {/* Pothole Node */}
          <button
            onClick={() => onSelectModel('pothole')}
            className={`p-4 border-3 border-[#111111] transition-all text-center flex flex-col items-center ${
              selectedModelId === 'pothole'
                ? 'bg-[#FFD84D] shadow-[6px_6px_0px_#111111] -translate-y-1'
                : isLive('pothole')
                ? 'bg-[#53D769] shadow-[4px_4px_0px_#111111]'
                : 'bg-[#FFFFFF] shadow-[4px_4px_0px_#111111]'
            }`}
          >
            <AlertCircle className="w-6 h-6 mb-2 text-[#111111] stroke-[2.5]" />
            <h5 className="text-xs font-bold text-[#111111] font-display uppercase">Pothole Detector</h5>
            <span className="text-[10px] text-[#555555] mt-1 font-mono font-bold">Surface Hazards</span>
            <span className="mt-2 text-[10px] font-mono font-bold px-2 py-0.5 bg-[#FFFFFF] border border-[#111111] uppercase">
              {getStatus('pothole')}
            </span>
          </button>
        </div>

        <ArrowDown className="w-4 h-4 text-[#111111] stroke-[2.5]" />

        {/* Step 4: Fusion Engine Node */}
        <button
          onClick={() => onSelectModel('fusion')}
          className={`w-full max-w-sm p-3.5 border-3 border-[#111111] transition-all flex items-center justify-between ${
            selectedModelId === 'fusion'
              ? 'bg-[#FFD84D] shadow-[6px_6px_0px_#111111] -translate-y-1'
              : isLive('fusion')
              ? 'bg-[#53D769] shadow-[4px_4px_0px_#111111]'
              : 'bg-[#FFFFFF] shadow-[4px_4px_0px_#111111]'
          }`}
        >
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-[#111111] stroke-[2.5]" />
            <div className="text-left">
              <h5 className="text-xs font-bold text-[#111111] font-display uppercase">Fusion Safety Engine</h5>
              <span className="text-[10px] text-[#555555] font-mono font-bold">Unified Assessment &amp; Warnings</span>
            </div>
          </div>
          <span className="text-[10px] font-mono font-bold px-2 py-0.5 bg-[#FFFFFF] border border-[#111111] uppercase">
            {getStatus('fusion')}
          </span>
        </button>

        <ArrowDown className="w-4 h-4 text-[#111111] stroke-[2.5]" />

        {/* Step 5: Unified Result */}
        <div className="px-6 py-3 bg-[#FFD84D] border-3 border-[#111111] shadow-[4px_4px_0px_#111111] text-[#111111] text-xs font-bold flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 stroke-[2.5]" />
          UNIFIED ROADVISION AI SAFETY RESULT
        </div>
      </div>
    </div>
  );
};
