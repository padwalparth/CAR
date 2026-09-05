import { useState, useEffect, useCallback } from 'react';
import { ModelHealthInfo } from '../types/models';
import { fetchModelHealth } from '../services/api/health';

export function useModelHealth(pollIntervalMs = 15000) {
  const [modelsHealth, setModelsHealth] = useState<ModelHealthInfo[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const refreshHealth = useCallback(async () => {
    setIsLoading(true);
    try {
      const healthData = await fetchModelHealth();
      setModelsHealth(healthData);
      setLastCheck(new Date());
    } catch {
      // Ignored - health endpoint handles fallbacks
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const timer = setInterval(refreshHealth, pollIntervalMs);
    return () => clearInterval(timer);
  }, [refreshHealth, pollIntervalMs]);

  const isPipelineConnected = modelsHealth.length > 0 && modelsHealth.every(m => m.status === 'Connected' || m.status === 'Ready');

  return {
    modelsHealth,
    isLoading,
    lastCheck,
    isPipelineConnected,
    refreshHealth
  };
}
