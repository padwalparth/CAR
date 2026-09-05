import { apiClient } from './client';
import { InferenceSession, UnifiedInferenceResult } from '../../types/inference';
import { DEMO_INFERENCE_SESSION, DEMO_UNIFIED_RESULT } from '../demoData';

export function isDemoModeEnabled(): boolean {
  return localStorage.getItem('rv_demo_mode') === 'true';
}

export function setDemoModeEnabled(enabled: boolean): void {
  localStorage.setItem('rv_demo_mode', enabled ? 'true' : 'false');
}

export async function createInferenceSession(file: File): Promise<InferenceSession> {
  if (isDemoModeEnabled()) {
    // Return isolated demo session in Demo Mode
    return {
      ...DEMO_INFERENCE_SESSION,
      id: `sess_demo_${Math.random().toString(36).substring(2, 9)}`,
      media_name: file.name,
      created_at: new Date().toISOString()
    };
  }

  // Real Production API Call
  const formData = new FormData();
  formData.append('file', file);
  return apiClient<InferenceSession>('/api/inference/upload', {
    method: 'POST',
    body: formData
  });
}

export async function startSessionAnalysis(sessionId: string): Promise<InferenceSession> {
  if (isDemoModeEnabled()) {
    return DEMO_INFERENCE_SESSION;
  }

  return apiClient<InferenceSession>(`/api/inference/${sessionId}/start`, {
    method: 'POST'
  });
}

export async function getInferenceSessionStatus(sessionId: string): Promise<InferenceSession> {
  if (isDemoModeEnabled()) {
    return DEMO_INFERENCE_SESSION;
  }

  return apiClient<InferenceSession>(`/api/inference/${sessionId}`);
}

export async function cancelInferenceSession(sessionId: string): Promise<void> {
  if (isDemoModeEnabled()) {
    return;
  }

  return apiClient<void>(`/api/inference/${sessionId}/cancel`, {
    method: 'POST'
  });
}

export async function getInferenceHistory(): Promise<InferenceSession[]> {
  if (isDemoModeEnabled()) {
    return [DEMO_INFERENCE_SESSION];
  }

  try {
    return await apiClient<InferenceSession[]>('/api/inference/history');
  } catch {
    // Return empty history array when backend is unpopulated or disconnected
    return [];
  }
}
