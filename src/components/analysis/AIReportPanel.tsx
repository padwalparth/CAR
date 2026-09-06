import React, { useState, useCallback } from 'react';
import {
  Sparkles,
  Bot,
  Loader2,
  RefreshCw,
  Copy,
  CheckCheck,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Zap,
  BrainCircuit,
  Sliders,
  ShieldCheck
} from 'lucide-react';
import { requestAIAnalysis, AIAnalysisResponse } from '../../services/api/aiAnalysis';

interface AIReportPanelProps {
  sessionId: string;
}

function parseMarkdownSections(text: string): { heading: string; body: string }[] {
  const sections: { heading: string; body: string }[] = [];
  const lines = text.split('\n');
  let current: { heading: string; body: string } | null = null;

  for (const line of lines) {
    const headingMatch = line.match(/^##\s+(.+)/);
    if (headingMatch) {
      if (current) sections.push(current);
      current = { heading: headingMatch[1].trim(), body: '' };
    } else if (current) {
      current.body += line + '\n';
    }
  }
  if (current) sections.push(current);

  // If no sections parsed, treat entire text as one block
  if (sections.length === 0 && text.trim()) {
    sections.push({ heading: '📋 Safety Assessment Report', body: text });
  }
  return sections;
}

function getSectionColor(heading: string): string {
  if (heading.includes('Hazard') || heading.includes('⚠️')) return '#FF5A5F';
  if (heading.includes('Risk') || heading.includes('📊')) return '#FF8C00';
  if (heading.includes('Road') || heading.includes('🛣️')) return '#4D7CFE';
  if (heading.includes('Traffic') || heading.includes('🚗')) return '#53D769';
  if (heading.includes('Safety') || heading.includes('🛡️')) return '#A259FF';
  return '#FFD84D';
}

export const AIReportPanel: React.FC<AIReportPanelProps> = ({ sessionId }) => {
  const [report, setReport] = useState<AIAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<'report' | 'xai' | 'maneuvers'>('report');

  const handleGenerate = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await requestAIAnalysis(sessionId);
      if (result.error && !result.report) {
        setError(result.error);
      } else {
        setReport(result);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to generate safety assessment.');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const handleCopy = useCallback(() => {
    if (report?.report) {
      navigator.clipboard.writeText(report.report).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  }, [report]);

  const handleRegenerate = useCallback(async () => {
    setReport(null);
    setError(null);
    await handleGenerate();
  }, [handleGenerate]);

  const sections = report?.report ? parseMarkdownSections(report.report) : [];
  const xai = report?.explainability;

  return (
    <div className="neo-card-lg bg-[#0A0A0F] border-2 border-[#FFD84D] overflow-hidden text-white">
      {/* Header */}
      <div className="p-5 border-b-2 border-[#1A1A2E] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {/* Glowing AI badge */}
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center border-2 border-[#FFD84D] shadow-[0_0_12px_rgba(255,216,77,0.5)]"
            style={{ background: 'linear-gradient(135deg, #1A1A2E 0%, #16213E 100%)' }}
          >
            <Bot className="w-5 h-5 text-[#FFD84D]" strokeWidth={2} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white font-display uppercase tracking-wider flex items-center gap-2">
              Road Safety Intelligence &amp; Reasoning Engine
              <span className="text-[9px] font-mono font-bold text-[#FFD84D] bg-[#1A1A2E] px-1.5 py-0.5 rounded border border-[#FFD84D]/40">
                MULTI-MODEL PERCEPTION
              </span>
            </h3>
            <p className="text-[10px] text-[#888899] font-medium mt-0.5">
              Automated safety assessment, causal decision logic, and actionable vehicle telemetry
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {report && (
            <>
              {/* Internal Tab Selector */}
              <div className="flex bg-[#161626] rounded-md p-0.5 border border-[#2A2A44] mr-2">
                <button
                  onClick={() => setActiveTab('report')}
                  className={`px-2.5 py-1 text-[11px] font-mono font-bold uppercase rounded transition-all ${
                    activeTab === 'report' ? 'bg-[#FFD84D] text-[#111111]' : 'text-[#888899] hover:text-white'
                  }`}
                >
                  Briefing
                </button>
                <button
                  onClick={() => setActiveTab('xai')}
                  className={`px-2.5 py-1 text-[11px] font-mono font-bold uppercase rounded transition-all ${
                    activeTab === 'xai' ? 'bg-[#FFD84D] text-[#111111]' : 'text-[#888899] hover:text-white'
                  }`}
                >
                  XAI Weights
                </button>
                <button
                  onClick={() => setActiveTab('maneuvers')}
                  className={`px-2.5 py-1 text-[11px] font-mono font-bold uppercase rounded transition-all ${
                    activeTab === 'maneuvers' ? 'bg-[#FFD84D] text-[#111111]' : 'text-[#888899] hover:text-white'
                  }`}
                >
                  Maneuvers
                </button>
              </div>

              <button
                onClick={handleCopy}
                title="Copy report"
                className="p-1.5 rounded border border-[#2A2A4E] text-[#888899] hover:text-white hover:border-[#FFD84D]/50 transition-all"
              >
                {copied ? <CheckCheck className="w-3.5 h-3.5 text-[#53D769]" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
              <button
                onClick={handleRegenerate}
                disabled={isLoading}
                title="Regenerate"
                className="p-1.5 rounded border border-[#2A2A4E] text-[#888899] hover:text-white hover:border-[#FFD84D]/50 transition-all disabled:opacity-40"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setIsCollapsed(c => !c)}
                className="p-1.5 rounded border border-[#2A2A4E] text-[#888899] hover:text-white transition-all"
              >
                {isCollapsed ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Content */}
      {!isCollapsed && (
        <div className="p-5">
          {/* Idle state — not yet generated */}
          {!report && !isLoading && !error && (
            <div className="text-center py-8">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 border-2 border-[#FFD84D]/30"
                style={{ background: 'linear-gradient(135deg, #1A1A2E 0%, #0D0D1A 100%)' }}
              >
                <Sparkles className="w-8 h-8 text-[#FFD84D]" strokeWidth={1.5} />
              </div>
              <p className="text-[#888899] text-xs font-medium max-w-sm mx-auto leading-relaxed mb-5">
                Generate an expert road safety assessment combining multi-model perception outputs with automated decision explainability and vehicle hazard telemetry.
              </p>
              <button
                onClick={handleGenerate}
                className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg border-2 border-[#FFD84D] text-[#111111] bg-[#FFD84D] font-bold text-xs uppercase tracking-wider shadow-[0_0_16px_rgba(255,216,77,0.4)] hover:shadow-[0_0_24px_rgba(255,216,77,0.6)] transition-all hover:scale-105 active:scale-95 cursor-pointer"
              >
                <Zap className="w-4 h-4" strokeWidth={2.5} />
                Generate AI Safety Report &amp; Explainability
              </button>
            </div>
          )}

          {/* Loading state */}
          {isLoading && (
            <div className="flex flex-col items-center gap-4 py-10">
              <div className="relative">
                <div className="w-14 h-14 rounded-xl border-2 border-[#FFD84D]/20 flex items-center justify-center">
                  <Loader2 className="w-7 h-7 text-[#FFD84D] animate-spin" strokeWidth={2} />
                </div>
                <div className="absolute inset-0 rounded-xl border-2 border-[#FFD84D]/40 animate-ping" />
              </div>
              <div className="text-center">
                <p className="text-white text-sm font-bold font-display">Synthesizing Scene Perception…</p>
                <p className="text-[#888899] text-[11px] mt-1">Multi-model safety engine is fusing perception data and computing attribution weights</p>
              </div>
              {/* Animated dots */}
              <div className="flex gap-1.5">
                {[0, 1, 2].map(i => (
                  <div
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-[#FFD84D]"
                    style={{ animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite` }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Error state */}
          {error && !isLoading && (
            <div className="rounded-lg border-2 border-[#FF5A5F]/40 bg-[#1A0A0A] p-4 flex gap-3">
              <AlertCircle className="w-5 h-5 text-[#FF5A5F] shrink-0 mt-0.5" strokeWidth={2} />
              <div className="flex-1 min-w-0">
                <p className="text-[#FF5A5F] text-xs font-bold uppercase tracking-wider mb-1">Analysis Notice</p>
                <p className="text-[#CC8888] text-[11px] font-mono break-all">{error}</p>
                <button
                  onClick={handleGenerate}
                  className="mt-3 text-xs font-bold text-[#FFD84D] hover:underline flex items-center gap-1 cursor-pointer"
                >
                  <RefreshCw className="w-3 h-3" /> Retry Generation
                </button>
              </div>
            </div>
          )}

          {/* Report Content Tabs */}
          {report && !isLoading && (
            <div className="space-y-4">
              {/* Meta Info Bar */}
              <div className="flex items-center justify-between text-[10px] text-[#666688] font-mono pb-3 border-b border-[#1A1A2E]">
                <span>Engine: {report.model_used}</span>
                <span>Generated: {new Date(report.generated_at).toLocaleTimeString()}</span>
              </div>

              {/* TAB 1: REPORT BRIEFING SECTIONS */}
              {activeTab === 'report' && (
                <div className="space-y-3">
                  {sections.map((section, idx) => {
                    const accentColor = getSectionColor(section.heading);
                    return (
                      <div
                        key={idx}
                        className="rounded-xl border-2 p-4 transition-all"
                        style={{
                          borderColor: `${accentColor}30`,
                          background: `linear-gradient(135deg, ${accentColor}08 0%, #0A0A0F 100%)`,
                        }}
                      >
                        <h4
                          className="text-xs font-bold uppercase tracking-wider mb-2 font-display"
                          style={{ color: accentColor }}
                        >
                          {section.heading}
                        </h4>
                        <p className="text-[#BBBBCC] text-[12px] leading-relaxed font-medium whitespace-pre-line">
                          {section.body.trim()}
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* TAB 2: XAI FACTOR WEIGHTS */}
              {activeTab === 'xai' && xai && (
                <div className="space-y-3">
                  <p className="text-xs text-[#888899]">
                    Quantitative decision factor attribution (SHAP scores) computed across multi-model sensor inputs:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {xai.feature_importance.map((f, idx) => {
                      const isNegative = f.impact === 'negative';
                      const color = isNegative ? '#FF5A5F' : f.impact === 'neutral' ? '#FFD84D' : '#53D769';
                      return (
                        <div key={idx} className="p-3 rounded-lg bg-[#141420] border border-[#2A2A40] space-y-1.5">
                          <div className="flex justify-between text-xs font-mono font-bold">
                            <span className="text-white">{f.factor}</span>
                            <span style={{ color }}>{f.score}/100</span>
                          </div>
                          <div className="w-full h-2 bg-[#222235] rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{ width: `${Math.max(10, f.score)}%`, backgroundColor: color }}
                            />
                          </div>
                          <p className="text-[11px] text-[#888899]">{f.description}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* TAB 3: ACTIONABLE MANEUVERS */}
              {activeTab === 'maneuvers' && xai && (
                <div className="space-y-3">
                  <p className="text-xs text-[#888899]">
                    Real-time vehicle actuator recommendations derived from computer vision hazard analysis:
                  </p>
                  <div className="grid grid-cols-1 gap-3">
                    {xai.safety_actions.map((a, idx) => {
                      const isHigh = a.priority === 'HIGH';
                      const badgeColor = isHigh ? 'bg-[#FF5A5F] text-white' : a.priority === 'MEDIUM' ? 'bg-[#FF8C00] text-white' : 'bg-[#53D769] text-white';
                      return (
                        <div key={idx} className="p-3.5 rounded-lg bg-[#141420] border border-[#2A2A40] space-y-1.5">
                          <div className="flex items-center justify-between">
                            <span className={`text-[10px] font-mono font-black px-2 py-0.5 rounded ${badgeColor}`}>
                              {a.priority} PRIORITY
                            </span>
                            <span className="text-[11px] font-mono text-[#FFD84D] bg-[#1E1E30] px-2 py-0.5 rounded border border-[#FFD84D]/30">
                              {a.actuator}
                            </span>
                          </div>
                          <h5 className="text-xs font-bold text-white uppercase">{a.action}</h5>
                          <p className="text-xs text-[#AAAAAA]">{a.details}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Footer */}
              <div className="flex items-center gap-2 pt-2 border-t border-[#1A1A2E]">
                <div className="w-1.5 h-1.5 rounded-full bg-[#53D769] animate-pulse" />
                <p className="text-[10px] text-[#666688] font-mono">
                  RoadVision AI Multi-Model Reasoning Engine · Session {sessionId}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
};
