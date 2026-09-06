import { apiClient } from './client';

export interface FeatureImportance {
  factor: string;
  weight: number;
  score: number;
  impact: 'positive' | 'negative' | 'neutral';
  description: string;
}

export interface CausalRule {
  rule_id: string;
  category: string;
  condition: string;
  status: 'PASS' | 'WARN' | 'FAIL';
  decision: string;
  impact_on_risk: number;
}

export interface CounterfactualScenario {
  scenario: string;
  action: string;
  original_risk: number;
  projected_risk: number;
  delta_risk: number;
  feasibility: string;
}

export interface SafetyAction {
  action: string;
  priority: 'HIGH' | 'MEDIUM' | 'LOW';
  details: string;
  actuator: string;
}

export interface ExplainabilityData {
  feature_importance: FeatureImportance[];
  causal_rules: CausalRule[];
  counterfactuals: CounterfactualScenario[];
  safety_actions: SafetyAction[];
  surface_integrity_score?: number;
  drivable_clearance_score?: number;
  risk_score?: number;
  risk_level?: string;
}

export interface AIAnalysisRequest {
  session_id: string;
}

export interface AIAnalysisResponse {
  session_id: string;
  model_used: string;
  report: string;
  explainability?: ExplainabilityData;
  generated_at: string;
  tokens_used?: number;
  error?: string;
}

export async function requestAIAnalysis(session_id: string): Promise<AIAnalysisResponse> {
  return apiClient<AIAnalysisResponse>('/api/ai/analyze', {
    method: 'POST',
    body: JSON.stringify({ session_id }),
  });
}

export async function requestExplainability(session_id: string): Promise<{ session_id: string; engine: string; explainability: ExplainabilityData; generated_at: string }> {
  return apiClient<{ session_id: string; engine: string; explainability: ExplainabilityData; generated_at: string }>(`/api/ai/explainability/${session_id}`);
}
