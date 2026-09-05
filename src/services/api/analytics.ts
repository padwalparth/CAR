import { apiClient } from './client';
import { AnalyticsData } from '../../types/analytics';
import { isDemoModeEnabled } from './inference';
import { DEMO_ANALYTICS } from '../demoData';

export async function fetchAnalyticsData(): Promise<AnalyticsData | null> {
  if (isDemoModeEnabled()) {
    return DEMO_ANALYTICS;
  }

  try {
    return await apiClient<AnalyticsData>('/api/analytics');
  } catch {
    // Return null when no backend API or analytics data exists
    return null;
  }
}
