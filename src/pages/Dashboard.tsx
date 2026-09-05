import React, { useEffect, useState } from 'react';
import { InferenceSession } from '../types/inference';
import { getInferenceHistory } from '../services/api/inference';
import { Car, AlertTriangle, Layers, Activity, Play, ArrowRight } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

interface DashboardProps {
  onNavigateToAnalyze: () => void;
}

export const DashboardPage: React.FC<DashboardProps> = ({ onNavigateToAnalyze }) => {
  const [sessions, setSessions] = useState<InferenceSession[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      try {
        const history = await getInferenceHistory();
        setSessions(history);
      } catch {
        setSessions([]);
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, []);

  const latestSession = sessions[0] || null;
  const result = latestSession?.result;

  const objectsCount = result?.yolo?.detections?.length;
  const potholesCount = result?.potholes?.detections?.length;
  const roadCoveragePercent = result?.road_segmentation?.coverage_percent ?? (result?.road_segmentation?.coverage_ratio !== undefined ? result.road_segmentation.coverage_ratio * 100 : undefined);
  const pipelineStatus = latestSession?.status || null;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="neo-card-lg p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight">
            Road Intelligence Dashboard
          </h2>
          <p className="text-xs text-[#555555] mt-1 max-w-xl leading-relaxed font-medium">
            Monitor real-time road conditions, traffic objects, road hazards, and multi-model AI pipeline activity.
          </p>
        </div>
        <button
          onClick={onNavigateToAnalyze}
          className="neo-btn-primary inline-flex items-center gap-2 px-5 py-2.5 text-xs uppercase tracking-wider shrink-0"
        >
          <Play className="w-4 h-4 fill-[#111111] stroke-[2.5]" />
          Launch New Analysis
        </button>
      </div>

      {/* KPI Cards Grid (Renders '—' when no data exists) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: Objects Detected */}
        <div className="neo-card p-5 space-y-2 bg-[#FFFFFF]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#555555] uppercase tracking-wider">Objects Detected</span>
            <div className="w-9 h-9 rounded bg-[#FFD84D] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center text-[#111111]">
              <Car className="w-4 h-4 stroke-[2.5]" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-[#111111]">
            {objectsCount !== undefined ? objectsCount : '—'}
          </div>
          <p className="text-[11px] text-[#555555] font-medium">
            {objectsCount !== undefined ? 'Returned from YOLO service' : 'Waiting for inference data'}
          </p>
        </div>

        {/* KPI 2: Potholes Detected */}
        <div className="neo-card p-5 space-y-2 bg-[#FFFFFF]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#555555] uppercase tracking-wider">Potholes Detected</span>
            <div className="w-9 h-9 rounded bg-[#FF5A5F] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center text-[#FFFFFF]">
              <AlertTriangle className="w-4 h-4 stroke-[2.5]" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-[#111111]">
            {potholesCount !== undefined ? potholesCount : '—'}
          </div>
          <p className="text-[11px] text-[#555555] font-medium">
            {potholesCount !== undefined ? 'Returned from Pothole detector' : 'Waiting for inference data'}
          </p>
        </div>

        {/* KPI 3: Road Coverage */}
        <div className="neo-card p-5 space-y-2 bg-[#FFFFFF]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#555555] uppercase tracking-wider">Road Coverage</span>
            <div className="w-9 h-9 rounded bg-[#53D769] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center text-[#111111]">
              <Layers className="w-4 h-4 stroke-[2.5]" />
            </div>
          </div>
          <div className="text-3xl font-bold font-mono text-[#111111]">
            {roadCoveragePercent !== undefined ? `${roadCoveragePercent.toFixed(0)}%` : '—'}
          </div>
          <p className="text-[11px] text-[#555555] font-medium">
            {roadCoveragePercent !== undefined ? 'Computed by U-Net segmenter' : 'Waiting for inference data'}
          </p>
        </div>

        {/* KPI 4: Processing Status */}
        <div className="neo-card p-5 space-y-2 bg-[#FFFFFF]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#555555] uppercase tracking-wider">Processing Status</span>
            <div className="w-9 h-9 rounded bg-[#4D7CFE] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center text-[#FFFFFF]">
              <Activity className="w-4 h-4 stroke-[2.5]" />
            </div>
          </div>
          <div className="text-sm font-bold font-mono text-[#111111] capitalize">
            {pipelineStatus || '—'}
          </div>
          <p className="text-[11px] text-[#555555] font-medium">
            {pipelineStatus ? 'Active inference pipeline' : 'Waiting for inference data'}
          </p>
        </div>
      </div>

      {/* Recent Inference Sessions History */}
      <div className="neo-card-lg p-6 bg-[#FFFFFF]">
        <div className="flex items-center justify-between mb-4 pb-3 border-b-2 border-[#111111]">
          <h3 className="text-sm font-bold text-[#111111] flex items-center gap-2 font-display uppercase tracking-wider">
            <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
            Recent Analysis Sessions
          </h3>
          <button
            onClick={onNavigateToAnalyze}
            className="text-xs font-bold text-[#111111] hover:text-[#4D7CFE] flex items-center gap-1 uppercase tracking-wider transition-colors"
          >
            View Workspace <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {sessions.length === 0 ? (
          <EmptyState
            icon={Car}
            title="No Analysis Sessions Found"
            description="No inference data exists in the backend history yet. Upload a road image or video to begin safety analysis."
            actionLabel="Start First Analysis"
            onAction={onNavigateToAnalyze}
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b-2 border-[#111111] text-[#555555] font-mono uppercase text-[10px]">
                  <th className="py-3 px-3 font-bold">Session ID</th>
                  <th className="py-3 px-3 font-bold">Media Name</th>
                  <th className="py-3 px-3 font-bold">Timestamp</th>
                  <th className="py-3 px-3 font-bold">Status</th>
                  <th className="py-3 px-3 font-bold">Objects</th>
                  <th className="py-3 px-3 font-bold">Potholes</th>
                  <th className="py-3 px-3 font-bold">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#EBEBE5] font-mono">
                {sessions.map((sess) => (
                  <tr key={sess.id} className="hover:bg-[#F7F7F2] transition-colors">
                    <td className="py-3 px-3 text-[#4D7CFE] font-bold">{sess.id}</td>
                    <td className="py-3 px-3 text-[#111111] font-sans font-medium">{sess.media_name}</td>
                    <td className="py-3 px-3 text-[#555555]">{new Date(sess.created_at).toLocaleTimeString()}</td>
                    <td className="py-3 px-3">
                      <span className="px-2 py-0.5 rounded text-[10px] uppercase font-bold bg-[#53D769] text-[#111111] border-2 border-[#111111]">
                        {sess.status}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-[#111111] font-bold">{sess.result?.yolo?.detections?.length ?? '—'}</td>
                    <td className="py-3 px-3 text-[#FF5A5F] font-bold">{sess.result?.potholes?.detections?.length ?? '—'}</td>
                    <td className="py-3 px-3 text-[#555555]">{sess.result?.total_processing_time_ms ? `${sess.result.total_processing_time_ms} ms` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
