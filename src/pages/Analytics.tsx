import React, { useEffect, useState } from 'react';
import { AnalyticsData } from '../types/analytics';
import { fetchAnalyticsData } from '../services/api/analytics';
import { EmptyState } from '../components/common/EmptyState';
import { BarChart3, Clock, AlertTriangle, Layers, Car } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  AreaChart,
  Area
} from 'recharts';

export const AnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadAnalytics() {
      setIsLoading(true);
      try {
        const result = await fetchAnalyticsData();
        setData(result);
      } catch {
        setData(null);
      } finally {
        setIsLoading(false);
      }
    }
    loadAnalytics();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="neo-card-lg p-5 flex items-center justify-between bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight">
            Traffic Safety &amp; Model Analytics
          </h2>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Statistical insights derived from multi-model inference sessions.
          </p>
        </div>
      </div>

      {!data ? (
        <EmptyState
          icon={BarChart3}
          title="No Analytics Data Available Yet"
          description="Analytics charts populate automatically based on real inference sessions. Run an analysis to start generating insights."
        />
      ) : (
        <>
          {/* Top Summary Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="neo-card p-4 space-y-1 bg-[#FFFFFF]">
              <span className="text-[#555555] text-xs font-bold uppercase tracking-wider">Total Sessions</span>
              <div className="text-3xl font-bold font-mono text-[#111111]">{data.total_sessions}</div>
            </div>
            <div className="neo-card p-4 space-y-1 bg-[#FFFFFF]">
              <span className="text-[#555555] text-xs font-bold uppercase tracking-wider">Objects Detected</span>
              <div className="text-3xl font-bold font-mono text-[#111111]">{data.total_objects_detected}</div>
            </div>
            <div className="neo-card p-4 space-y-1 bg-[#FFFFFF]">
              <span className="text-[#555555] text-xs font-bold uppercase tracking-wider">Potholes Detected</span>
              <div className="text-3xl font-bold font-mono text-[#FF5A5F]">{data.total_potholes_detected}</div>
            </div>
            <div className="neo-card p-4 space-y-1 bg-[#FFFFFF]">
              <span className="text-[#555555] text-xs font-bold uppercase tracking-wider">Avg Road Coverage</span>
              <div className="text-3xl font-bold font-mono text-[#53D769]">
                {(data.average_road_coverage * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Object Taxonomy Distribution */}
            <div className="neo-card p-5 space-y-4 bg-[#FFFFFF]">
              <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
                <Car className="w-4 h-4 stroke-[2.5]" />
                Detected Object Distribution
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.object_distribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EBEBE5" />
                    <XAxis dataKey="class_name" stroke="#555555" fontSize={11} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <YAxis stroke="#555555" fontSize={11} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#111111', borderWidth: '2px', borderRadius: '6px', color: '#111111', fontWeight: 700 }}
                    />
                    <Bar dataKey="count" fill="#FFD84D" stroke="#111111" strokeWidth={2} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: Pothole Frequency Timeline */}
            <div className="neo-card p-5 space-y-4 bg-[#FFFFFF]">
              <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 bg-[#FF5A5F] border border-[#111111]" />
                <AlertTriangle className="w-4 h-4 stroke-[2.5]" />
                Pothole Detection Frequency Timeline
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.pothole_timeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EBEBE5" />
                    <XAxis dataKey="timestamp" stroke="#555555" fontSize={11} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <YAxis stroke="#555555" fontSize={11} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#111111', borderWidth: '2px', borderRadius: '6px', color: '#111111', fontWeight: 700 }}
                    />
                    <Area type="monotone" dataKey="count" stroke="#FF5A5F" strokeWidth={2} fill="rgba(255, 90, 95, 0.15)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 3: Road Coverage Trend */}
            <div className="neo-card p-5 space-y-4 bg-[#FFFFFF]">
              <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 bg-[#53D769] border border-[#111111]" />
                <Layers className="w-4 h-4 stroke-[2.5]" />
                Road Segmentation Coverage Trend (%)
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.road_coverage_trend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EBEBE5" />
                    <XAxis dataKey="timestamp" stroke="#555555" fontSize={11} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <YAxis stroke="#555555" fontSize={11} domain={[0, 100]} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#111111', borderWidth: '2px', borderRadius: '6px', color: '#111111', fontWeight: 700 }}
                    />
                    <Line type="monotone" dataKey="coverage_percentage" stroke="#53D769" strokeWidth={3} dot={{ fill: '#53D769', stroke: '#111111', strokeWidth: 2 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 4: Model Latency Metrics */}
            <div className="neo-card p-5 space-y-4 bg-[#FFFFFF]">
              <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
                <span className="w-2.5 h-2.5 bg-[#4D7CFE] border border-[#111111]" />
                <Clock className="w-4 h-4 stroke-[2.5]" />
                Model Inference Latency Comparison (ms)
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.latency_metrics} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#EBEBE5" />
                    <XAxis type="number" stroke="#555555" fontSize={11} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <YAxis type="category" dataKey="model" stroke="#555555" fontSize={11} width={110} tick={{ fontWeight: 700, fill: '#111111' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#111111', borderWidth: '2px', borderRadius: '6px', color: '#111111', fontWeight: 700 }}
                    />
                    <Bar dataKey="latency_ms" fill="#4D7CFE" stroke="#111111" strokeWidth={2} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
