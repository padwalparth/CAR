import React from 'react';
import { ModelHealthInfo } from '../../types/models';
import { Cpu, CheckCircle2, AlertCircle, ExternalLink } from 'lucide-react';

interface ModelCardProps {
  model: ModelHealthInfo;
  isSelected?: boolean;
  onSelect?: () => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  isSelected = false,
  onSelect
}) => {
  const isConnected = model.status === 'Connected' || model.status === 'Ready';

  return (
    <div
      onClick={onSelect}
      className={`p-5 rounded-md border-3 border-[#111111] transition-all cursor-pointer ${
        isSelected
          ? 'bg-[#FFD84D] shadow-[6px_6px_0px_#111111] -translate-y-1'
          : 'bg-[#FFFFFF] hover:bg-[#F7F7F2] shadow-[4px_4px_0px_#111111]'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center shrink-0 ${
              isConnected ? 'bg-[#53D769] text-[#111111]' : 'bg-[#FFFFFF] text-[#FF5A5F]'
            }`}
          >
            <Cpu className="w-5 h-5 stroke-[2.5]" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-[#111111] uppercase font-display">{model.name}</h4>
            <p className="text-[11px] text-[#555555] font-medium mt-0.5 line-clamp-1">{model.description}</p>
          </div>
        </div>

        {/* Status Badge */}
        <span
          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold border-2 border-[#111111] uppercase shadow-[2px_2px_0px_#111111] ${
            isConnected ? 'bg-[#53D769] text-[#111111]' : 'bg-[#FF5A5F] text-[#FFFFFF]'
          }`}
        >
          {isConnected ? (
            <>
              <CheckCircle2 className="w-3 h-3 stroke-[3]" /> Connected
            </>
          ) : (
            <>
              <AlertCircle className="w-3 h-3 stroke-[3]" /> Offline
            </>
          )}
        </span>
      </div>

      {/* Details Grid */}
      <div className="grid grid-cols-2 gap-2 pt-3 border-t-2 border-[#111111] text-xs font-mono font-bold">
        <div>
          <span className="text-[#555555] text-[9px] uppercase block">VERSION</span>
          <span className="text-[#111111]">{model.version || '—'}</span>
        </div>
        <div>
          <span className="text-[#555555] text-[9px] uppercase block">LATENCY</span>
          <span className="text-[#111111]">{model.latency_ms ? `${model.latency_ms} ms` : '—'}</span>
        </div>
        <div className="col-span-2 pt-1">
          <span className="text-[#555555] text-[9px] uppercase block">ENDPOINT</span>
          <span className="text-[#111111] text-[10px] truncate block flex items-center gap-1">
            {model.endpoint || '/api/models'}
            <ExternalLink className="w-3 h-3 text-[#111111] stroke-[2.5]" />
          </span>
        </div>
      </div>

      {model.error && (
        <div className="mt-3 p-2 rounded bg-[#FF5A5F] text-[#FFFFFF] border-2 border-[#111111] text-[10px] font-mono font-bold">
          {model.error}
        </div>
      )}
    </div>
  );
};
