import React from 'react';
import { useLiveMonitor } from '../hooks/useLiveMonitor';
import { Radio, Video, Play, Square, Activity, Clock } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export const LiveMonitorPage: React.FC = () => {
  const { connectionState, events, connectStream, disconnectStream } = useLiveMonitor();

  const isConnected = connectionState === 'connected';

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="neo-card-lg p-5 flex items-center justify-between bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight flex items-center gap-3">
            Real-Time Live Traffic Monitor
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono font-bold border-2 border-[#111111] uppercase ${
                isConnected
                  ? 'bg-[#53D769] text-[#111111] shadow-[2px_2px_0px_#111111]'
                  : 'bg-[#F7F7F2] text-[#555555]'
              }`}
            >
              <span className={`w-2 h-2 rounded-full border border-[#111111] ${isConnected ? 'bg-[#111111] animate-pulse' : 'bg-[#888888]'}`} />
              {connectionState.toUpperCase()}
            </span>
          </h2>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Connect live CCTV / RTSP camera streams for real-time WebSocket traffic intelligence.
          </p>
        </div>

        {/* Connect / Disconnect Action */}
        <div>
          {!isConnected ? (
            <button
              onClick={connectStream}
              className="neo-btn-primary inline-flex items-center gap-2 px-5 py-2.5 text-xs uppercase tracking-wider"
            >
              <Play className="w-4 h-4 fill-[#111111] stroke-[2.5]" />
              Connect Camera Stream
            </button>
          ) : (
            <button
              onClick={disconnectStream}
              className="inline-flex items-center gap-2 px-5 py-2.5 text-xs font-bold uppercase tracking-wider bg-[#FF5A5F] text-[#FFFFFF] border-3 border-[#111111] shadow-[4px_4px_0px_#111111] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#111111] transition-all"
            >
              <Square className="w-4 h-4 fill-[#FFFFFF] stroke-[2.5]" />
              Disconnect Camera
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Stream Viewport */}
        <div className="lg:col-span-2 neo-card-lg p-5 flex flex-col justify-between min-h-[480px] bg-[#FFFFFF]">
          <div className="flex items-center justify-between mb-4 pb-3 border-b-2 border-[#111111]">
            <span className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
              <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
              <Video className="w-4 h-4 stroke-[2.5]" />
              CCTV Camera Feed #01 — Indian Highway Junction
            </span>
            <span className="text-xs font-mono font-bold text-[#555555] bg-[#F7F7F2] px-2 py-0.5 border border-[#111111]">WebSocket / RTSP</span>
          </div>

          {!isConnected ? (
            <EmptyState
              icon={Radio}
              title="No Live Stream Connected"
              description="Connect a CCTV camera feed or live streaming server to start monitoring real-time road events, pothole alerts, and traffic flow."
              actionLabel="Connect Stream"
              onAction={connectStream}
              className="my-auto py-12"
            />
          ) : (
            <div className="relative bg-[#111111] rounded-md overflow-hidden flex items-center justify-center h-[380px] border-2 border-[#111111]">
              <div className="text-center p-6 space-y-3">
                <Activity className="w-12 h-12 text-[#FFD84D] animate-pulse mx-auto stroke-[2.5]" />
                <h4 className="text-sm font-bold text-[#FFFFFF] font-display uppercase tracking-wide">Live AI Stream Active</h4>
                <p className="text-xs text-[#888888] font-mono">Receiving real-time inference frames</p>
              </div>

              <div className="absolute top-3 left-3 px-2.5 py-1 bg-[#FF5A5F] text-white font-mono font-bold text-[10px] uppercase flex items-center gap-1.5 animate-pulse border border-[#FFFFFF]">
                <span className="w-2 h-2 rounded-full bg-white" />
                LIVE REC
              </div>
            </div>
          )}
        </div>

        {/* Right Event Ticker Panel */}
        <div className="neo-card-lg p-5 flex flex-col bg-[#FFFFFF]">
          <div className="flex items-center justify-between mb-4 pb-3 border-b-2 border-[#111111]">
            <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
              <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
              <Activity className="w-4 h-4 stroke-[2.5]" />
              Live AI Event Log
            </h3>
            <span className="text-[10px] font-mono font-bold text-[#555555] bg-[#F7F7F2] px-1.5 py-0.5 border border-[#111111]">
              {events.length} Events
            </span>
          </div>

          <div className="flex-1 overflow-y-auto space-y-2 max-h-[420px] pr-1 font-mono text-xs">
            {events.length === 0 ? (
              <div className="p-8 text-center text-[#555555] text-xs font-medium bg-[#F7F7F2] rounded border-2 border-[#111111]">
                No live events received yet. Events will stream automatically when camera feed is connected.
              </div>
            ) : (
              events.map((evt) => (
                <div
                  key={evt.id}
                  className="p-3 rounded bg-[#F7F7F2] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-start gap-3"
                >
                  <Clock className="w-3.5 h-3.5 text-[#4D7CFE] shrink-0 mt-0.5 stroke-[2.5]" />
                  <div>
                    <div className="flex items-center gap-2 text-[10px] text-[#555555]">
                      <span>{evt.timestamp}</span>
                      <span className="uppercase text-[#111111] font-bold px-1 bg-[#FFD84D] border border-[#111111]">
                        {evt.type}
                      </span>
                    </div>
                    <p className="text-[#111111] text-xs font-sans font-medium mt-1">{evt.message}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
