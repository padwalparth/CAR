import React, { useState } from 'react';
import { isDemoModeEnabled, setDemoModeEnabled } from '../services/api/inference';
import { Sliders, Server, Radio, Zap, CheckCircle2, RefreshCw } from 'lucide-react';
import { fetchModelHealth } from '../services/api/health';

export const SettingsPage: React.FC = () => {
  const [isDemo, setIsDemo] = useState<boolean>(isDemoModeEnabled());
  const [apiUrl, setApiUrl] = useState<string>(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000');
  const [wsUrl, setWsUrl] = useState<string>(import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live');
  const [defaultConfidence, setDefaultConfidence] = useState<number>(0.50);
  const [isTestingApi, setIsTestingApi] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const handleDemoToggle = (enabled: boolean) => {
    setIsDemo(enabled);
    setDemoModeEnabled(enabled);
  };

  const handleTestConnection = async () => {
    setIsTestingApi(true);
    setTestResult(null);
    try {
      await fetchModelHealth();
      setTestResult('Connection successful! Backend models reachable.');
    } catch {
      setTestResult('Backend endpoint unreachable. Ensure server is running.');
    } finally {
      setIsTestingApi(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header Banner */}
      <div className="neo-card-lg p-5 flex items-center justify-between bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight">
            Platform Settings &amp; Configurations
          </h2>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Configure backend endpoints, threshold defaults, and demonstration mode.
          </p>
        </div>
      </div>

      {/* SECTION 1: DEMO MODE TOGGLE */}
      <div className="neo-card-lg p-6 space-y-4 bg-[#FFFFFF]">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded bg-[#FFD84D] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#111111] shrink-0">
              <Zap className="w-6 h-6 stroke-[2.5]" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-[#111111] font-display uppercase">Offline Demonstration Mode</h3>
              <p className="text-xs text-[#555555] mt-0.5 max-w-lg font-medium">
                Toggles pre-loaded sample road scene results for offline evaluation. Production API mode is default when OFF.
              </p>
            </div>
          </div>

          {/* Toggle Switch */}
          <button
            onClick={() => handleDemoToggle(!isDemo)}
            className={`w-16 h-9 border-3 border-[#111111] p-1 transition-colors duration-200 ease-in-out relative shrink-0 shadow-[3px_3px_0px_#111111] ${
              isDemo ? 'bg-[#FFD84D]' : 'bg-[#F7F7F2]'
            }`}
          >
            <div
              className={`w-6 h-6 bg-[#111111] transition-transform duration-200 ease-in-out ${
                isDemo ? 'translate-x-7' : 'translate-x-0'
              }`}
            />
          </button>
        </div>

        <div className="pt-3 border-t-2 border-[#111111] flex items-center gap-2 text-xs">
          <span className="text-[#555555] font-bold uppercase">Status:</span>
          <span className={`font-mono font-bold text-sm px-2 py-0.5 border-2 border-[#111111] ${isDemo ? 'bg-[#FFD84D] text-[#111111]' : 'bg-[#111111] text-[#FFFFFF]'}`}>
            {isDemo ? 'DEMO MODE ACTIVE' : 'PRODUCTION API MODE ACTIVE'}
          </span>
        </div>
      </div>

      {/* SECTION 2: BACKEND API CONFIGURATION */}
      <div className="neo-card-lg p-6 space-y-4 bg-[#FFFFFF]">
        <div className="flex items-center gap-4 pb-3 border-b-2 border-[#111111]">
          <div className="w-12 h-12 rounded bg-[#4D7CFE] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#FFFFFF] shrink-0">
            <Server className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-[#111111] font-display uppercase">Backend API Endpoints</h3>
            <p className="text-xs text-[#555555] mt-0.5 font-medium">
              Specify the base URL for REST inference services and WebSocket camera streaming.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-mono font-bold text-[#111111] uppercase tracking-wider mb-2">
              REST API Base URL
            </label>
            <div className="flex gap-3">
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="neo-input flex-1 px-3.5 py-2.5 text-xs font-mono"
              />
              <button
                onClick={handleTestConnection}
                disabled={isTestingApi}
                className="neo-btn-secondary px-4 py-2 text-xs uppercase tracking-wider flex items-center gap-1.5 shrink-0 disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 stroke-[2.5] ${isTestingApi ? 'animate-spin' : ''}`} />
                Test Connection
              </button>
            </div>
            {testResult && (
              <p className="text-xs mt-2 font-mono font-bold flex items-center gap-1.5 text-[#111111] bg-[#F7F7F2] p-2 border-2 border-[#111111] mt-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#53D769] stroke-[3]" />
                {testResult}
              </p>
            )}
          </div>

          <div>
            <label className="block text-xs font-mono font-bold text-[#111111] uppercase tracking-wider mb-2">
              WebSocket Live Stream URL
            </label>
            <input
              type="text"
              value={wsUrl}
              onChange={(e) => setWsUrl(e.target.value)}
              className="neo-input w-full px-3.5 py-2.5 text-xs font-mono"
            />
          </div>
        </div>
      </div>

      {/* SECTION 3: DEFAULT INFERENCE PARAMETERS */}
      <div className="neo-card-lg p-6 space-y-4 bg-[#FFFFFF]">
        <div className="flex items-center gap-4 pb-3 border-b-2 border-[#111111]">
          <div className="w-12 h-12 rounded bg-[#F7F7F2] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#111111] shrink-0">
            <Sliders className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-[#111111] font-display uppercase">Inference Parameters Defaults</h3>
            <p className="text-xs text-[#555555] mt-0.5 font-medium">
              Set default thresholds and mask rendering properties.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-[#111111] font-bold uppercase tracking-wider">Default Confidence Threshold</span>
              <span className="font-mono font-bold text-sm text-[#111111] bg-[#FFD84D] px-2 py-0.5 border-2 border-[#111111]">
                {(defaultConfidence * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={defaultConfidence}
              onChange={(e) => setDefaultConfidence(Number(e.target.value))}
              className="w-full cursor-pointer accent-[#111111] h-2 bg-[#F7F7F2] border border-[#111111] rounded-none appearance-none"
            />
            <div className="flex justify-between text-[10px] font-mono font-bold text-[#555555] mt-1">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 4: LIVE STREAM CONFIGURATION */}
      <div className="neo-card-lg p-6 space-y-4 bg-[#FFFFFF]">
        <div className="flex items-center gap-4 pb-3 border-b-2 border-[#111111]">
          <div className="w-12 h-12 rounded bg-[#F7F7F2] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#111111] shrink-0">
            <Radio className="w-6 h-6 stroke-[2.5]" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-[#111111] font-display uppercase">Live Monitor Configuration</h3>
            <p className="text-xs text-[#555555] mt-0.5 font-medium">
              Configure real-time camera stream parameters for live traffic monitoring.
            </p>
          </div>
        </div>

        <div className="p-4 bg-[#F7F7F2] border-2 border-[#111111] shadow-[3px_3px_0px_#111111]">
          <p className="text-xs text-[#555555] font-medium leading-relaxed">
            <span className="font-bold text-[#111111] uppercase">Note:</span> WebSocket stream endpoint configured above. Real-time processing requires an active RTSP/WebSocket server with ML inference backend connected and running.
          </p>
        </div>
      </div>
    </div>
  );
};
