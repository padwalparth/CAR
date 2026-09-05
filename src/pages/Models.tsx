import React, { useState } from 'react';
import { useModelHealth } from '../hooks/useModelHealth';
import { ModelCard } from '../components/models/ModelCard';
import { ArchitectureDiagram } from '../components/models/ArchitectureDiagram';
import { RefreshCw, Cpu } from 'lucide-react';

export const ModelsPage: React.FC = () => {
  const { modelsHealth, isLoading, refreshHealth } = useModelHealth();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const selectedModel = modelsHealth.find((m) => m.id === selectedModelId);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="neo-card-lg p-5 flex items-center justify-between bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight">
            ML Microservice Models &amp; Architecture
          </h2>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Monitor backend ML service endpoints, health metrics, and dataflow topology.
          </p>
        </div>

        <button
          onClick={refreshHealth}
          disabled={isLoading}
          className="neo-btn-secondary inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-wider disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 stroke-[2.5] ${isLoading ? 'animate-spin' : ''}`} />
          Refresh Health Check
        </button>
      </div>

      {/* Model Health Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {modelsHealth.map((m) => (
          <ModelCard
            key={m.id}
            model={m}
            isSelected={selectedModelId === m.id}
            onSelect={() => setSelectedModelId(m.id)}
          />
        ))}
      </div>

      {/* Interactive Topology Diagram */}
      <ArchitectureDiagram
        modelsHealth={modelsHealth}
        selectedModelId={selectedModelId}
        onSelectModel={(id) => setSelectedModelId(id)}
      />

      {/* Selected Model Detail Panel */}
      {selectedModel && (
        <div className="neo-card-lg p-6 border-l-4 border-l-[#FFD84D] space-y-4 bg-[#FFFFFF]">
          <div className="flex items-center justify-between pb-3 border-b-2 border-[#111111]">
            <h3 className="text-sm font-bold text-[#111111] flex items-center gap-2 font-display uppercase">
              <Cpu className="w-4 h-4 stroke-[2.5]" />
              Detailed Inspector: {selectedModel.name}
            </h3>
            <span className="text-xs font-mono font-bold text-[#555555] bg-[#F7F7F2] px-2 py-0.5 border border-[#111111]">
              Endpoint: {selectedModel.endpoint}
            </span>
          </div>
          <p className="text-xs text-[#555555] font-medium">{selectedModel.description}</p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 bg-[#F7F7F2] border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
              <span className="text-[#555555] text-[10px] uppercase block font-bold mb-1">Status</span>
              <span className={`font-bold text-sm ${selectedModel.status === 'Connected' ? 'text-[#53D769]' : 'text-[#FF5A5F]'}`}>
                {selectedModel.status}
              </span>
            </div>
            <div className="p-3 bg-[#F7F7F2] border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
              <span className="text-[#555555] text-[10px] uppercase block font-bold mb-1">Version</span>
              <span className="text-[#111111] font-bold">{selectedModel.version || '—'}</span>
            </div>
            <div className="p-3 bg-[#F7F7F2] border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
              <span className="text-[#555555] text-[10px] uppercase block font-bold mb-1">Latency</span>
              <span className="text-[#4D7CFE] font-bold">{selectedModel.latency_ms ? `${selectedModel.latency_ms} ms` : '—'}</span>
            </div>
            <div className="p-3 bg-[#F7F7F2] border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
              <span className="text-[#555555] text-[10px] uppercase block font-bold mb-1">Last Checked</span>
              <span className="text-[#555555] text-[11px]">{selectedModel.last_check_at ? new Date(selectedModel.last_check_at).toLocaleTimeString() : '—'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
