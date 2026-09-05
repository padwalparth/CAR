import React, { useState, useEffect } from 'react';
import { UnifiedInferenceResult, InferenceSession } from '../types/inference';
import { SimulationState, VehicleState } from '../types/simulation';
import {
  projectDetectionsToSimulation,
  determineVehicleState,
  generateSimulationTimeline
} from '../services/simulationEngine';
import { getInferenceHistory } from '../services/api/inference';
import { SimulationCanvas } from '../components/simulation/SimulationCanvas';
import { SimulationControls } from '../components/simulation/SimulationControls';
import { SimulationStatus } from '../components/simulation/SimulationStatus';
import { SimulationTimeline } from '../components/simulation/SimulationTimeline';
import { ObjectInspector } from '../components/simulation/ObjectInspector';
import { ShieldAlert, PlayCircle, Sparkles, AlertTriangle, ArrowRight } from 'lucide-react';

interface SimulationPageProps {
  currentSession?: InferenceSession | null;
  onNavigateToAnalyze?: () => void;
}

export const SimulationPage: React.FC<SimulationPageProps> = ({
  currentSession,
  onNavigateToAnalyze
}) => {
  const [historySessions, setHistorySessions] = useState<InferenceSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(
    currentSession?.id || null
  );

  // Simulation State
  const [simMode, setSimMode] = useState<'scene' | 'timeline' | 'live'>('scene');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(1.0);

  // Visualization Toggles
  const [showObjects, setShowObjects] = useState<boolean>(true);
  const [showRoad, setShowRoad] = useState<boolean>(true);
  const [showPotholes, setShowPotholes] = useState<boolean>(true);
  const [showRiskZones, setShowRiskZones] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showTrajectory, setShowTrajectory] = useState<boolean>(true);

  const [selectedObjectId, setSelectedObjectId] = useState<string | undefined>(undefined);

  // Fetch session history on mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const history = await getInferenceHistory();
        setHistorySessions(history);
        if (!selectedSessionId && history.length > 0) {
          const completed = history.find((s: InferenceSession) => s.status === 'completed' && s.result);
          if (completed) {
            setSelectedSessionId(completed.id);
          }
        }
      } catch (err) {
        console.error('Failed to load session history for simulation', err);
      }
    };
    fetchHistory();
  }, []);

  // Update selected session if currentSession prop changes
  useEffect(() => {
    if (currentSession?.id) {
      setSelectedSessionId(currentSession.id);
    }
  }, [currentSession]);

  // Find active session object & inference result
  const activeSession =
    currentSession?.id === selectedSessionId
      ? currentSession
      : historySessions.find((s) => s.id === selectedSessionId) || currentSession;

  const result: UnifiedInferenceResult | undefined = activeSession?.result;

  // Process simulation structures
  const simObjects = result ? projectDetectionsToSimulation(result) : [];
  const vehicleState: VehicleState = result ? determineVehicleState(result, simObjects) : 'GO';
  const timelineEvents = result ? generateSimulationTimeline(result, vehicleState) : [];

  const selectedObject = simObjects.find((o) => o.id === selectedObjectId);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="neo-card-lg p-5 flex flex-wrap items-center justify-between gap-4 bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight flex items-center gap-2">
            <PlayCircle className="w-6 h-6 text-[#111111] stroke-[2.5]" />
            Road Safety Intelligent Simulation
          </h2>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Converts multi-model computer vision perception outputs into an interactive 2D vehicle safety simulation.
          </p>
        </div>

        {/* Session Selector */}
        {historySessions.length > 0 && (
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="font-bold text-[#555555] uppercase">INFERENCE SESSION:</span>
            <select
              value={selectedSessionId || ''}
              onChange={(e) => setSelectedSessionId(e.target.value)}
              className="px-3 py-1.5 rounded-md bg-[#FFFFFF] border-2 border-[#111111] font-bold text-[#111111] shadow-[2px_2px_0px_#111111]"
            >
              {historySessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} ({s.media_name || 'Image'}) - {s.status.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Mandatory UI Disclaimer Banner */}
      <div className="p-3.5 rounded-md bg-[#FFFBEB] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center gap-3 text-xs font-mono text-[#92400E]">
        <ShieldAlert className="w-5 h-5 shrink-0 text-[#D97706] stroke-[2.5]" />
        <p className="font-semibold leading-relaxed">
          <span className="font-extrabold uppercase text-[#B45309]">DISCLAIMER: </span>
          Simulation based on AI perception estimates. Spatial positions and vehicle responses are illustrative and are not certified for real-world autonomous driving.
        </p>
      </div>

      {/* Main Simulation View or Empty State */}
      {!result ? (
        <div className="neo-card-lg p-12 text-center bg-[#FFFFFF]">
          <div className="w-14 h-14 rounded-md bg-[#FFD84D] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#111111] mx-auto mb-4">
            <Sparkles className="w-7 h-7 stroke-[2.5]" />
          </div>
          <h3 className="text-base font-bold text-[#111111] font-display uppercase tracking-wide">
            No Active Inference Results
          </h3>
          <p className="text-xs text-[#555555] max-w-md mx-auto mt-2 font-medium leading-relaxed">
            Run an image analysis first in the Analyze Workspace to load perception targets into the simulation engine.
          </p>
          {onNavigateToAnalyze && (
            <button
              onClick={onNavigateToAnalyze}
              className="mt-5 inline-flex items-center gap-2 px-5 py-2.5 neo-btn-primary text-xs uppercase font-mono"
            >
              <span>Go to Analyze Workspace</span>
              <ArrowRight className="w-4 h-4 stroke-[2.5]" />
            </button>
          )}
        </div>
      ) : (
        <>
          {/* Live Video Mode Notice if mode is 'live' */}
          {simMode === 'live' && (
            <div className="p-4 rounded-md bg-[#EFF6FF] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] text-xs font-mono text-[#1E40AF] flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-[#3B82F6] stroke-[2.5]" />
              <p className="font-semibold">
                Video simulation will be available when frame-by-frame inference is enabled.
              </p>
            </div>
          )}

          {/* Simulation Workspace Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left 2 Columns: Canvas, Controls & Timeline */}
            <div className="lg:col-span-2 space-y-6">
              <SimulationCanvas
                inferenceResult={result}
                simObjects={simObjects}
                vehicleState={vehicleState}
                isPlaying={isPlaying}
                speed={speed}
                selectedObjectId={selectedObjectId}
                onSelectObject={setSelectedObjectId}
                showObjects={showObjects}
                showRoad={showRoad}
                showPotholes={showPotholes}
                showRiskZones={showRiskZones}
                showLabels={showLabels}
                showTrajectory={showTrajectory}
              />

              <SimulationControls
                isPlaying={isPlaying}
                onTogglePlay={() => setIsPlaying(!isPlaying)}
                onRestart={() => {
                  setIsPlaying(true);
                  setSelectedObjectId(undefined);
                }}
                speed={speed}
                onSpeedChange={setSpeed}
                showObjects={showObjects}
                onToggleObjects={() => setShowObjects(!showObjects)}
                showRoad={showRoad}
                onToggleRoad={() => setShowRoad(!showRoad)}
                showPotholes={showPotholes}
                onTogglePotholes={() => setShowPotholes(!showPotholes)}
                showRiskZones={showRiskZones}
                onToggleRiskZones={() => setShowRiskZones(!showRiskZones)}
                showLabels={showLabels}
                onToggleLabels={() => setShowLabels(!showLabels)}
                showTrajectory={showTrajectory}
                onToggleTrajectory={() => setShowTrajectory(!showTrajectory)}
                simMode={simMode}
                onSimModeChange={setSimMode}
              />

              <SimulationTimeline events={timelineEvents} />
            </div>

            {/* Right Column: Status & Object Inspector */}
            <div className="lg:col-span-1 space-y-6">
              <SimulationStatus
                inferenceResult={result}
                vehicleState={vehicleState}
              />

              <ObjectInspector
                selectedObject={selectedObject}
                onClose={() => setSelectedObjectId(undefined)}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};
