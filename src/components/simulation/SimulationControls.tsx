import React from 'react';
import { Play, Pause, RotateCcw, Eye, Layers, AlertCircle, ShieldAlert, Tag, Navigation2 } from 'lucide-react';

interface SimulationControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  onRestart: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
  showObjects: boolean;
  onToggleObjects: () => void;
  showRoad: boolean;
  onToggleRoad: () => void;
  showPotholes: boolean;
  onTogglePotholes: () => void;
  showRiskZones: boolean;
  onToggleRiskZones: () => void;
  showLabels: boolean;
  onToggleLabels: () => void;
  showTrajectory: boolean;
  onToggleTrajectory: () => void;
  simMode: 'scene' | 'timeline' | 'live';
  onSimModeChange: (mode: 'scene' | 'timeline' | 'live') => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  isPlaying,
  onTogglePlay,
  onRestart,
  speed,
  onSpeedChange,
  showObjects,
  onToggleObjects,
  showRoad,
  onToggleRoad,
  showPotholes,
  onTogglePotholes,
  showRiskZones,
  onToggleRiskZones,
  showLabels,
  onToggleLabels,
  showTrajectory,
  onToggleTrajectory,
  simMode,
  onSimModeChange
}) => {
  const speeds = [0.5, 1, 2, 4];

  return (
    <div className="neo-card p-4 bg-[#FFFFFF] space-y-4">
      {/* 1. Mode Switcher & Primary Playback Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b-2 border-[#111111] pb-3">
        {/* Mode Selector */}
        <div className="flex items-center gap-1.5 bg-[#F7F7F2] p-1 rounded-md border-2 border-[#111111]">
          <button
            onClick={() => onSimModeChange('scene')}
            className={`px-3 py-1 text-xs font-mono font-bold uppercase rounded transition-all ${
              simMode === 'scene' ? 'bg-[#FFD84D] text-[#111111] border border-[#111111] shadow-[1px_1px_0px_#111111]' : 'text-[#555555] hover:text-[#111111]'
            }`}
          >
            Scene Simulation
          </button>
          <button
            onClick={() => onSimModeChange('timeline')}
            className={`px-3 py-1 text-xs font-mono font-bold uppercase rounded transition-all ${
              simMode === 'timeline' ? 'bg-[#FFD84D] text-[#111111] border border-[#111111] shadow-[1px_1px_0px_#111111]' : 'text-[#555555] hover:text-[#111111]'
            }`}
          >
            Timeline Playback
          </button>
          <button
            onClick={() => onSimModeChange('live')}
            className={`px-3 py-1 text-xs font-mono font-bold uppercase rounded transition-all ${
              simMode === 'live' ? 'bg-[#FFD84D] text-[#111111] border border-[#111111] shadow-[1px_1px_0px_#111111]' : 'text-[#555555] hover:text-[#111111]'
            }`}
          >
            Live Stream
          </button>
        </div>

        {/* Playback Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={onTogglePlay}
            className="flex items-center gap-1.5 px-4 py-1.5 neo-btn-primary text-xs uppercase"
          >
            {isPlaying ? <Pause className="w-4 h-4 stroke-[2.5]" /> : <Play className="w-4 h-4 stroke-[2.5]" />}
            <span>{isPlaying ? 'PAUSE' : 'PLAY'}</span>
          </button>
          <button
            onClick={onRestart}
            className="flex items-center gap-1.5 px-3 py-1.5 neo-btn-secondary text-xs uppercase"
            title="Restart Simulation"
          >
            <RotateCcw className="w-4 h-4 stroke-[2.5]" />
            <span>RESTART</span>
          </button>
        </div>

        {/* Speed Controls */}
        <div className="flex items-center gap-1 bg-[#F7F7F2] p-1 rounded-md border-2 border-[#111111]">
          <span className="text-[10px] font-mono font-bold text-[#555555] px-1.5 uppercase">SPEED:</span>
          {speeds.map((s) => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`px-2 py-0.5 text-xs font-mono font-bold rounded transition-all ${
                speed === s ? 'bg-[#111111] text-[#FFFFFF]' : 'text-[#111111] hover:bg-[#FFFFFF]'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* 2. Interactive Layer & Overlay Toggles */}
      <div>
        <span className="text-[10px] font-mono font-bold text-[#555555] tracking-wider uppercase block mb-2">
          VISUALIZATION LAYERS & TOGGLES
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={onToggleObjects}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold uppercase border-2 border-[#111111] transition-all ${
              showObjects ? 'bg-[#10B981] text-[#FFFFFF] shadow-[2px_2px_0px_#111111]' : 'bg-[#FFFFFF] text-[#555555] hover:bg-[#F7F7F2]'
            }`}
          >
            <Eye className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>OBJECTS (YOLO)</span>
          </button>

          <button
            onClick={onToggleRoad}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold uppercase border-2 border-[#111111] transition-all ${
              showRoad ? 'bg-[#38BDF8] text-[#111111] shadow-[2px_2px_0px_#111111]' : 'bg-[#FFFFFF] text-[#555555] hover:bg-[#F7F7F2]'
            }`}
          >
            <Layers className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>ROAD SEG. (U-NET)</span>
          </button>

          <button
            onClick={onTogglePotholes}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold uppercase border-2 border-[#111111] transition-all ${
              showPotholes ? 'bg-[#EF4444] text-[#FFFFFF] shadow-[2px_2px_0px_#111111]' : 'bg-[#FFFFFF] text-[#555555] hover:bg-[#F7F7F2]'
            }`}
          >
            <AlertCircle className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>POTHOLES</span>
          </button>

          <button
            onClick={onToggleRiskZones}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold uppercase border-2 border-[#111111] transition-all ${
              showRiskZones ? 'bg-[#F59E0B] text-[#111111] shadow-[2px_2px_0px_#111111]' : 'bg-[#FFFFFF] text-[#555555] hover:bg-[#F7F7F2]'
            }`}
          >
            <ShieldAlert className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>RISK ZONES</span>
          </button>

          <button
            onClick={onToggleLabels}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold uppercase border-2 border-[#111111] transition-all ${
              showLabels ? 'bg-[#FFD84D] text-[#111111] shadow-[2px_2px_0px_#111111]' : 'bg-[#FFFFFF] text-[#555555] hover:bg-[#F7F7F2]'
            }`}
          >
            <Tag className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>LABELS</span>
          </button>

          <button
            onClick={onToggleTrajectory}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono font-bold uppercase border-2 border-[#111111] transition-all ${
              showTrajectory ? 'bg-[#111111] text-[#FFFFFF] shadow-[2px_2px_0px_#111111]' : 'bg-[#FFFFFF] text-[#555555] hover:bg-[#F7F7F2]'
            }`}
          >
            <Navigation2 className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>TRAJECTORY</span>
          </button>
        </div>
      </div>
    </div>
  );
};
