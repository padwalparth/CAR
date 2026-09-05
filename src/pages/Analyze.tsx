import React, { useState } from 'react';
import { useInference } from '../hooks/useInference';
import { MediaUpload } from '../components/analysis/MediaUpload';
import { PipelineProgress } from '../components/analysis/PipelineProgress';
import { CanvasViewer } from '../components/analysis/CanvasViewer';
import { AnalysisTabs, AnalysisSubTab } from '../components/analysis/AnalysisTabs';
import { FusionSummary } from '../components/analysis/FusionSummary';
import { DetectionList } from '../components/analysis/DetectionList';
import { ErrorBanner } from '../components/common/ErrorBanner';
import { Layers, Sparkles } from 'lucide-react';

interface AnalyzePageProps {
  onNavigateToSimulation?: () => void;
}

export const AnalyzePage: React.FC<AnalyzePageProps> = ({ onNavigateToSimulation }) => {
  const {
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
    resetSession
  } = useInference();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleMediaSelect = async (file: File) => {
    setSelectedFile(file);
    await uploadMedia(file);
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    resetSession();
  };

  const result = session?.result;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="neo-card-lg p-5 flex items-center justify-between bg-[#FFFFFF]">
        <div>
          <h2 className="text-xl font-bold text-[#111111] font-display uppercase tracking-tight flex items-center gap-2">
            Road Scene AI Analysis Workspace
            {session && (
              <span className="text-xs font-mono font-bold text-[#111111] bg-[#FFD84D] px-2.5 py-0.5 rounded border-2 border-[#111111] shadow-[2px_2px_0px_#111111]">
                Session: {session.id}
              </span>
            )}
          </h2>
          <p className="text-xs text-[#555555] mt-0.5 font-medium">
            Upload media and inspect multi-model computer vision outputs across independent AI models.
          </p>
        </div>
      </div>

      {error && <ErrorBanner message={error} onRetry={runAnalysis} />}

      {/* 1. Media Upload Zone */}
      <MediaUpload
        onMediaSelect={handleMediaSelect}
        onStartAnalysis={runAnalysis}
        isUploading={isUploading}
        isAnalyzing={isAnalyzing}
        selectedFile={selectedFile}
        onClearFile={handleClearFile}
      />

      {/* 2. Visual Model Pipeline Stage Progress */}
      {session && (
        <PipelineProgress
          stages={session.models}
          latencies={{
            yolo: result?.yolo?.processing_time_ms,
            unet: result?.road_segmentation?.processing_time_ms,
            pothole: result?.potholes?.processing_time_ms,
            fusion: result?.fusion?.processing_time_ms
          }}
          totalLatencyMs={result?.total_processing_time_ms}
        />
      )}

      {/* 3. Workspace Sub-Navigation Tabs */}
      <AnalysisTabs
        activeTab={activeTab}
        onTabChange={setActiveTab}
        counts={{
          objects: result?.yolo?.detections?.length,
          potholes: result?.potholes?.detections?.length,
          coverage: result?.road_segmentation?.coverage_ratio ?? (result?.road_segmentation?.coverage_percent !== undefined ? result.road_segmentation.coverage_percent / 100 : undefined)
        }}
        onNavigateToSimulation={result ? onNavigateToSimulation : undefined}
      />

      {/* 4. Tab Content Area */}
      {!result ? (
        <div className="neo-card-lg p-12 text-center bg-[#FFFFFF]">
          <div className="w-14 h-14 rounded-md bg-[#FFD84D] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#111111] mx-auto mb-4">
            <Sparkles className="w-7 h-7 stroke-[2.5]" />
          </div>
          <h3 className="text-base font-bold text-[#111111] font-display uppercase tracking-wide">No Inference Results Available</h3>
          <p className="text-xs text-[#555555] max-w-md mx-auto mt-2 font-medium leading-relaxed">
            Select a road image or video above and click <span className="text-[#111111] font-bold underline underline-offset-2">Start AI Analysis</span> to generate model outputs.
          </p>
        </div>
      ) : (
        <>
          {/* TAB 1: OVERVIEW WORKSPACE */}
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-6">
                <CanvasViewer
                  mediaUrl={result.input.media_url}
                  mediaType={result.input.media_type}
                  yoloDetections={filteredYoloDetections}
                  potholeDetections={filteredPotholeDetections}
                  roadMaskUrl={result.road_segmentation.mask_url}
                  activeLayer={activeLayer}
                  onLayerChange={setActiveLayer}
                  maskOpacity={maskOpacity}
                  onOpacityChange={setMaskOpacity}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                />
                <FusionSummary fusion={result.fusion} />
              </div>
              <div className="lg:col-span-1">
                <DetectionList
                  yoloDetections={filteredYoloDetections}
                  potholeDetections={filteredPotholeDetections}
                  availableClasses={availableYoloClasses}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                  confidenceThreshold={confidenceThreshold}
                  onConfidenceChange={setConfidenceThreshold}
                />
              </div>
            </div>
          )}

          {/* TAB 2: TRAFFIC OBJECTS VIEW */}
          {activeTab === 'traffic' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <CanvasViewer
                    mediaUrl={result.input.media_url}
                    mediaType={result.input.media_type}
                    yoloDetections={filteredYoloDetections}
                    potholeDetections={[]}
                    activeLayer="objects"
                    onLayerChange={setActiveLayer}
                    maskOpacity={maskOpacity}
                    onOpacityChange={setMaskOpacity}
                    selectedDetectionId={selectedDetectionId}
                    onSelectDetection={setSelectedDetectionId}
                  />
                </div>
                <div>
                  <DetectionList
                    yoloDetections={filteredYoloDetections}
                    potholeDetections={[]}
                    availableClasses={availableYoloClasses}
                    selectedDetectionId={selectedDetectionId}
                    onSelectDetection={setSelectedDetectionId}
                    confidenceThreshold={confidenceThreshold}
                    onConfidenceChange={setConfidenceThreshold}
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: ROAD SEGMENTATION VIEW */}
          {activeTab === 'segmentation' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <CanvasViewer
                  mediaUrl={result.input.media_url}
                  mediaType={result.input.media_type}
                  yoloDetections={[]}
                  potholeDetections={[]}
                  roadMaskUrl={result.road_segmentation.mask_url}
                  activeLayer="mask"
                  onLayerChange={setActiveLayer}
                  maskOpacity={maskOpacity}
                  onOpacityChange={setMaskOpacity}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                />
              </div>
              <div className="neo-card p-5 bg-[#FFFFFF] space-y-4">
                <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider flex items-center gap-2 font-display">
                  <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
                  <Layers className="w-4 h-4 stroke-[2.5]" />
                  U-Net Segmentation Metrics
                </h3>
                <div className="p-4 rounded-md bg-[#F7F7F2] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] space-y-2">
                  <span className="text-xs text-[#555555] font-medium">Calculated Road Coverage</span>
                  <div className="text-4xl font-bold font-mono text-[#111111]">
                    {result.road_segmentation.coverage_percent !== undefined
                      ? `${result.road_segmentation.coverage_percent.toFixed(1)}%`
                      : result.road_segmentation.coverage_ratio !== undefined
                      ? `${(result.road_segmentation.coverage_ratio * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                  <p className="text-[11px] text-[#555555] font-medium">
                    Percentage of drivable asphalt surface identified in scene.
                  </p>
                </div>
                <div className="p-3 rounded-md bg-[#F7F7F2] border-2 border-[#111111] text-xs space-y-1">
                  <span className="text-[#555555] font-mono text-[10px] uppercase block font-bold">Model Service</span>
                  <p className="font-bold text-[#111111]">U-Net Semantic Segmenter</p>
                  <p className="text-[#555555] text-[11px] font-medium">Latency: {result.road_segmentation.processing_time_ms ?? '—'} ms</p>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: POTHOLES VIEW */}
          {activeTab === 'potholes' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <CanvasViewer
                  mediaUrl={result.input.media_url}
                  mediaType={result.input.media_type}
                  yoloDetections={[]}
                  potholeDetections={filteredPotholeDetections}
                  activeLayer="potholes"
                  onLayerChange={setActiveLayer}
                  maskOpacity={maskOpacity}
                  onOpacityChange={setMaskOpacity}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                />
              </div>
              <div>
                <DetectionList
                  yoloDetections={[]}
                  potholeDetections={filteredPotholeDetections}
                  availableClasses={[]}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                  confidenceThreshold={confidenceThreshold}
                  onConfidenceChange={setConfidenceThreshold}
                />
              </div>
            </div>
          )}

          {/* TAB 5: SAFETY / FUSION VIEW */}
          {activeTab === 'safety' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <CanvasViewer
                  mediaUrl={result.input.media_url}
                  mediaType={result.input.media_type}
                  yoloDetections={filteredYoloDetections}
                  potholeDetections={filteredPotholeDetections}
                  roadMaskUrl={result.road_segmentation.mask_url}
                  activeLayer="combined"
                  onLayerChange={setActiveLayer}
                  maskOpacity={maskOpacity}
                  onOpacityChange={setMaskOpacity}
                  selectedDetectionId={selectedDetectionId}
                  onSelectDetection={setSelectedDetectionId}
                />
              </div>
              <div>
                <FusionSummary fusion={result.fusion} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
