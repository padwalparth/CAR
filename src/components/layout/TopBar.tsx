import React from 'react';
import { Bell, Sliders, CheckCircle2, AlertCircle, Zap } from 'lucide-react';
import { isDemoModeEnabled } from '../../services/api/inference';

interface TopBarProps {
  title: string;
  description: string;
  isPipelineConnected: boolean;
  onOpenSettings: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  title,
  description,
  isPipelineConnected,
  onOpenSettings
}) => {
  const isDemo = isDemoModeEnabled();

  return (
    <header className="h-16 border-b-3 border-[#111111] bg-[#FFFFFF] sticky top-0 z-30 flex items-center justify-between px-6">
      {/* Title & Subtitle */}
      <div>
        <h1 className="text-base font-bold text-[#111111] uppercase tracking-wide font-display flex items-center gap-2">
          {title}
          {isDemo && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#FFD84D] text-[#111111] border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
              <Zap className="w-3 h-3 stroke-[2.5]" />
              DEMO MODE
            </span>
          )}
        </h1>
        <p className="text-xs text-[#555555] font-medium">{description}</p>
      </div>

      {/* Connection Status Pill & Actions */}
      <div className="flex items-center gap-3">
        <div
          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono font-bold border-2 border-[#111111] shadow-[2px_2px_0px_#111111] ${
            isPipelineConnected ? 'bg-[#53D769] text-[#111111]' : 'bg-[#FFFFFF] text-[#FF5A5F]'
          }`}
        >
          {isPipelineConnected ? (
            <>
              <CheckCircle2 className="w-4 h-4 stroke-[2.5]" />
              <span>● AI PIPELINE CONNECTED</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-4 h-4 stroke-[2.5]" />
              <span>○ MODELS NOT CONNECTED</span>
            </>
          )}
        </div>

        {/* Notifications */}
        <button
          className="p-2 rounded-md bg-[#FFFFFF] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] hover:bg-[#FFD84D] transition-colors relative"
          title="Notifications"
        >
          <Bell className="w-4 h-4 text-[#111111] stroke-[2.5]" />
          <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-[#FF5A5F] border border-[#111111]" />
        </button>

        {/* Settings */}
        <button
          onClick={onOpenSettings}
          className="p-2 rounded-md bg-[#FFFFFF] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] hover:bg-[#FFD84D] transition-colors"
          title="Settings"
        >
          <Sliders className="w-4 h-4 text-[#111111] stroke-[2.5]" />
        </button>
      </div>
    </header>
  );
};
