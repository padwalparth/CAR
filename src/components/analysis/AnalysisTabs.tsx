import React from 'react';
import { LayoutGrid, Car, Layers, AlertCircle, ShieldCheck, PlayCircle } from 'lucide-react';

export type AnalysisSubTab = 'overview' | 'traffic' | 'segmentation' | 'potholes' | 'safety';

interface AnalysisTabsProps {
  activeTab: AnalysisSubTab;
  onTabChange: (tab: AnalysisSubTab) => void;
  counts?: {
    objects?: number;
    potholes?: number;
    coverage?: number;
  };
  onNavigateToSimulation?: () => void;
}

export const AnalysisTabs: React.FC<AnalysisTabsProps> = ({
  activeTab,
  onTabChange,
  counts,
  onNavigateToSimulation
}) => {
  const tabs: { key: AnalysisSubTab; label: string; icon: any; count?: string | number }[] = [
    { key: 'overview', label: 'OVERVIEW', icon: LayoutGrid },
    { key: 'traffic', label: 'TRAFFIC OBJECTS', icon: Car, count: counts?.objects },
    { key: 'segmentation', label: 'ROAD SEGMENTATION', icon: Layers, count: counts?.coverage ? `${(counts.coverage * 100).toFixed(0)}%` : undefined },
    { key: 'potholes', label: 'POTHOLES', icon: AlertCircle, count: counts?.potholes },
    { key: 'safety', label: 'SAFETY / FUSION', icon: ShieldCheck }
  ];

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-b-3 border-[#111111] mb-6 pb-2">
      <div className="flex items-center gap-2 overflow-x-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => onTabChange(tab.key)}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-mono font-bold uppercase transition-all whitespace-nowrap rounded-t-md border-2 border-[#111111] ${
                isActive
                  ? 'bg-[#FFD84D] text-[#111111] shadow-[3px_3px_0px_#111111] -translate-y-1'
                  : 'bg-[#FFFFFF] text-[#111111] hover:bg-[#F7F7F2]'
              }`}
            >
              <Icon className="w-4 h-4 stroke-[2.5]" />
              <span>{tab.label}</span>
              {tab.count !== undefined && (
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] border border-[#111111] font-mono ${
                    isActive ? 'bg-[#FFFFFF] text-[#111111]' : 'bg-[#F7F7F2] text-[#555555]'
                  }`}
                >
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {onNavigateToSimulation && (
        <button
          onClick={onNavigateToSimulation}
          className="flex items-center gap-1.5 px-4 py-2 text-xs font-mono font-bold uppercase neo-btn-primary shadow-[3px_3px_0px_#111111]"
          title="Launch 2D Vehicle Safety Simulation using current AI inference results"
        >
          <PlayCircle className="w-4 h-4 stroke-[2.5]" />
          <span>LAUNCH SIMULATION</span>
        </button>
      )}
    </div>
  );
};
