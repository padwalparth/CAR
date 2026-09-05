import React from 'react';
import { SimulationTimelineEvent } from '../../types/simulation';
import { Clock, Info, AlertTriangle, AlertCircle, PlayCircle } from 'lucide-react';

interface SimulationTimelineProps {
  events: SimulationTimelineEvent[];
}

export const SimulationTimeline: React.FC<SimulationTimelineProps> = ({ events }) => {
  const getEventIcon = (type: SimulationTimelineEvent['type']) => {
    switch (type) {
      case 'hazard':
        return <AlertCircle className="w-3.5 h-3.5 text-[#EF4444] stroke-[2.5]" />;
      case 'warning':
        return <AlertTriangle className="w-3.5 h-3.5 text-[#F59E0B] stroke-[2.5]" />;
      case 'action':
        return <PlayCircle className="w-3.5 h-3.5 text-[#10B981] stroke-[2.5]" />;
      case 'info':
      default:
        return <Info className="w-3.5 h-3.5 text-[#38BDF8] stroke-[2.5]" />;
    }
  };

  return (
    <div className="neo-card p-4 bg-[#FFFFFF] space-y-3">
      <div className="flex items-center justify-between border-b-2 border-[#111111] pb-2 font-mono">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-[#111111] stroke-[2.5]" />
          <h4 className="text-xs font-bold text-[#111111] uppercase tracking-wider">
            Simulation Timeline
          </h4>
        </div>
        <span className="text-[10px] text-[#555555] font-bold">
          {events.length} EVENTS LOGGED
        </span>
      </div>

      <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
        {events.map((evt) => (
          <div
            key={evt.id}
            className="p-2.5 rounded-md bg-[#F7F7F2] border border-[#111111] flex items-start gap-2.5 font-mono text-xs"
          >
            <span className="px-1.5 py-0.5 rounded bg-[#111111] text-[#FFFFFF] font-bold text-[10px]">
              {evt.timeStr}
            </span>
            <div className="flex-1 space-y-0.5">
              <div className="flex items-center gap-1.5 font-bold text-[#111111]">
                {getEventIcon(evt.type)}
                <span>{evt.title}</span>
              </div>
              <p className="text-[11px] text-[#555555] font-medium leading-tight">
                {evt.details}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
