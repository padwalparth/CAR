import { apiClient } from './client';
import { InferenceSession } from '../../types/inference';

// ── Demo mode is permanently disabled — always runs real models ───────────
/** @deprecated Demo mode has been removed. Always returns false. */
export function isDemoModeEnabled(): boolean {
  // Clear any stale flag that may have been set previously
  localStorage.removeItem('rv_demo_mode');
  return false;
}
/** @deprecated Demo mode has been removed. This is a no-op. */
export function setDemoModeEnabled(_enabled: boolean): void {
  localStorage.removeItem('rv_demo_mode');
}

export async function createInferenceSession(file: File): Promise<InferenceSession> {
  const formData = new FormData();
  formData.append('file', file);
  return apiClient<InferenceSession>('/api/inference/upload', {
    method: 'POST',
    body: formData
  });
}

export async function startSessionAnalysis(sessionId: string): Promise<InferenceSession> {
  return apiClient<InferenceSession>(`/api/inference/${sessionId}/start`, {
    method: 'POST'
  });
}

export async function getInferenceSessionStatus(sessionId: string): Promise<InferenceSession> {
  return apiClient<InferenceSession>(`/api/inference/${sessionId}`);
}

export async function cancelInferenceSession(sessionId: string): Promise<void> {
  return apiClient<void>(`/api/inference/${sessionId}/cancel`, {
    method: 'POST'
  });
}

export async function getInferenceHistory(): Promise<InferenceSession[]> {
  try {
    return await apiClient<InferenceSession[]>('/api/inference/history');
  } catch {
    return [];
  }
}
