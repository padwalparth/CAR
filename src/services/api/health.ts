import { apiClient } from './client';
import { ModelHealthInfo } from '../../types/models';
import { isDemoModeEnabled } from './inference';
import { DEMO_HEALTH_STATUS } from '../demoData';

export async function fetchModelHealth(): Promise<ModelHealthInfo[]> {
  if (isDemoModeEnabled()) {
    return DEMO_HEALTH_STATUS;
  }

  try {
    return await apiClient<ModelHealthInfo[]>('/api/models/health');
  } catch {
    // Return explicit Offline / Not Connected status for each service when backend is unreachable
    return [
      {
        id: "yolo",
        name: "YOLO Object Detector",
        description: "Detects vehicles, pedestrians, animals, tractors & traffic obstacles",
        status: "Not Connected",
        endpoint: "/api/models/yolo",
        last_check_at: new Date().toISOString(),
        error: "Service unreachable"
      },
      {
        id: "unet",
        name: "U-Net Road Segmentation",
        description: "Segments drivable road boundaries & computes coverage percentage",
        status: "Not Connected",
        endpoint: "/api/models/unet",
        last_check_at: new Date().toISOString(),
        error: "Service unreachable"
      },
      {
        id: "pothole",
        name: "Pothole Detector",
        description: "Identifies road surface hazards, deep cracks & pothole bounding boxes",
        status: "Not Connected",
        endpoint: "/api/models/potholes",
        last_check_at: new Date().toISOString(),
        error: "Service unreachable"
      },
      {
        id: "fusion",
        name: "Fusion Engine",
        description: "Fuses multi-model detections to generate safety scores & warnings",
        status: "Not Connected",
        endpoint: "/api/models/fusion",
        last_check_at: new Date().toISOString(),
        error: "Service unreachable"
      }
    ];
  }
}
