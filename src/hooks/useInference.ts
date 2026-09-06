import { useState, useCallback, useMemo } from 'react';
import {
  InferenceSession,
  LayerType,
  Detection,
  PotholeDetection
} from '../types/inference';
import {
  createInferenceSession,
  startSessionAnalysis,
  cancelInferenceSession
} from '../services/api/inference';
import { AnalysisSubTab } from '../components/analysis/AnalysisTabs';

export function useInference() {
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);


  // Active sub-tab inside Analyze workspace
  const [activeTab, setActiveTab] = useState<AnalysisSubTab>('overview');

  // Visualization settings
  const [activeLayer, setActiveLayer] = useState<LayerType>('combined');
  const [maskOpacity, setMaskOpacity] = useState<number>(60);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.50);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(null);

  // Filtered Detections based on confidence threshold (Visualization Filter)
  const filteredYoloDetections = useMemo<Detection[]>(() => {
    if (!session?.result?.yolo?.detections) return [];
    return session.result.yolo.detections.filter(d => d.confidence >= confidenceThreshold);
  }, [session?.result?.yolo?.detections, confidenceThreshold]);

  const filteredPotholeDetections = useMemo<PotholeDetection[]>(() => {
    if (!session?.result?.potholes?.detections) return [];
    return session.result.potholes.detections.filter(p => p.confidence >= confidenceThreshold);
  }, [session?.result?.potholes?.detections, confidenceThreshold]);

  // Dynamically derived class list from returned YOLO detections (NO HARDCODING)
  const availableYoloClasses = useMemo<string[]>(() => {
    if (!session?.result?.yolo?.detections) return [];
    const classes = new Set<string>();
    session.result.yolo.detections.forEach(d => classes.add(d.class_name));
    return Array.from(classes);
  }, [session?.result?.yolo?.detections]);

  // Upload file & initialize session
  const uploadMedia = useCallback(async (file: File) => {
    setIsUploading(true);
    setError(null);
    try {
      const newSession = await createInferenceSession(file);
      setSession(newSession);
      return newSession;
    } catch (err: any) {
      setError(err?.message || 'Failed to upload media.');
      return null;
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Run ML inference session
  const runAnalysis = useCallback(async () => {
    if (!session) return;
    setIsAnalyzing(true);
    setError(null);

    // Update status to processing
    setSession(prev => prev ? {
      ...prev,
      status: 'processing',
      models: {
        yolo: 'processing',
        unet: 'processing',
        pothole: 'processing',
        fusion: 'waiting'
      }
    } : null);

    try {
      const updatedSession = await startSessionAnalysis(session.id);
      setSession(updatedSession);
    } catch (err: any) {
      setError(err?.message || 'Analysis failed. Model service may be offline.');
      setSession(prev => prev ? {
        ...prev,
        status: 'failed',
        error: { code: 'INFERENCE_ERROR', message: err?.message || 'Inference failed' }
      } : null);
    } finally {
      setIsAnalyzing(false);
    }
  }, [session]);

  const cancelAnalysis = useCallback(async () => {
    if (!session) return;
    try {
      await cancelInferenceSession(session.id);
      setSession(prev => prev ? { ...prev, status: 'cancelled' } : null);
    } catch {
      // Ignored
    } finally {
      setIsAnalyzing(false);
    }
  }, [session]);

  const resetSession = useCallback(() => {
    setSession(null);
    setError(null);
    setSelectedDetectionId(null);
  }, []);

  return {
    session,
    isUploading,
    isAnalyzing,
    error,
    activeTab,
    setActiveTab,
    activeLayer,
    setActiveLayer,
    maskOpacity,
    setMaskOpacity,
    confidenceThreshold,
    setConfidenceThreshold,
    selectedDetectionId,
    setSelectedDetectionId,
    filteredYoloDetections,
    filteredPotholeDetections,
    availableYoloClasses,
    uploadMedia,
    runAnalysis,
    cancelAnalysis,
    resetSession
  };
}
