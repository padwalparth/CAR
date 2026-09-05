import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  Sparkles
} from 'lucide-react';
import { LayerType, Detection, PotholeDetection } from '../../types/inference';

interface CanvasViewerProps {
  mediaUrl: string;
  mediaType: 'image' | 'video';
  yoloDetections: Detection[];
  potholeDetections: PotholeDetection[];
  roadMaskUrl?: string;
  activeLayer: LayerType;
  onLayerChange: (layer: LayerType) => void;
  maskOpacity: number;
  onOpacityChange: (opacity: number) => void;
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  isUnetConnected?: boolean;
}

export const CanvasViewer: React.FC<CanvasViewerProps> = ({
  mediaUrl,
  mediaType,
  yoloDetections,
  potholeDetections,
  roadMaskUrl,
  activeLayer,
  onLayerChange,
  maskOpacity,
  onOpacityChange,
  selectedDetectionId,
  onSelectDetection
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [startPan, setStartPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [loadedImage, setLoadedImage] = useState<HTMLImageElement | null>(null);
  const [loadedMask, setLoadedMask] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (mediaType === 'image' && mediaUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = mediaUrl;
      img.onload = () => {
        setLoadedImage(img);
      };
    }
  }, [mediaUrl, mediaType]);

  useEffect(() => {
    if (roadMaskUrl) {
      const maskImg = new Image();
      maskImg.crossOrigin = 'anonymous';
      maskImg.src = roadMaskUrl;
      maskImg.onload = () => {
        setLoadedMask(maskImg);
      };
      maskImg.onerror = () => {
        setLoadedMask(null);
      };
    } else {
      setLoadedMask(null);
    }
  }, [roadMaskUrl]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = containerRef.current?.clientWidth || 800;
    const height = 500;
    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // Compute aspect-ratio preserving image display rect
    const imgNaturalW = loadedImage?.naturalWidth || 1280;
    const imgNaturalH = loadedImage?.naturalHeight || 720;
    const imgAspect = imgNaturalW / imgNaturalH;
    const canvasAspect = width / height;

    let drawW: number, drawH: number, drawX: number, drawY: number;
    if (imgAspect > canvasAspect) {
      // Image is wider — fit width
      drawW = width;
      drawH = width / imgAspect;
      drawX = 0;
      drawY = (height - drawH) / 2;
    } else {
      // Image is taller — fit height
      drawH = height;
      drawW = height * imgAspect;
      drawX = (width - drawW) / 2;
      drawY = 0;
    }

    if (mediaType === 'image' && loadedImage) {
      ctx.drawImage(loadedImage, drawX, drawY, drawW, drawH);
    }

    // U-Net Road Segmentation Mask Layer — drawn over exact same rect as image
    if ((activeLayer === 'mask' || activeLayer === 'combined') && roadMaskUrl) {
      if (loadedMask) {
        ctx.save();
        ctx.globalAlpha = maskOpacity / 100;
        ctx.drawImage(loadedMask, drawX, drawY, drawW, drawH);
        ctx.restore();
      } else {
        // Fallback synthetic trapezoid within image display rect
        ctx.fillStyle = `rgba(77, 124, 254, ${maskOpacity / 100})`;
        ctx.beginPath();
        ctx.moveTo(drawX + drawW * 0.44, drawY + drawH * 0.47);
        ctx.lineTo(drawX + drawW * 0.56, drawY + drawH * 0.47);
        ctx.lineTo(drawX + drawW * 0.92, drawY + drawH);
        ctx.lineTo(drawX + drawW * 0.08, drawY + drawH);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#4D7CFE';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    }

    // Scale factors from model bbox coords (in natural image pixels) to canvas display rect
    const scaleX = drawW / imgNaturalW;
    const scaleY = drawH / imgNaturalH;
    const offsetX = drawX;
    const offsetY = drawY;

    // YOLO Detections — bbox coords are in natural image pixel space
    if (activeLayer === 'objects' || activeLayer === 'combined' || activeLayer === 'confidence') {
      yoloDetections.forEach((det) => {
        const isSelected = selectedDetectionId === det.id;
        const x = offsetX + det.bbox.x1 * scaleX;
        const y = offsetY + det.bbox.y1 * scaleY;
        const w = (det.bbox.x2 - det.bbox.x1) * scaleX;
        const h = (det.bbox.y2 - det.bbox.y1) * scaleY;

        ctx.strokeStyle = isSelected ? '#FFFFFF' : '#FFD84D';
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.strokeRect(x, y, w, h);

        const labelText = `${det.class_name.toUpperCase()} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = isSelected ? '#111111' : '#FFD84D';
        ctx.fillRect(x, Math.max(offsetY, y - 22), textWidth + 12, 22);

        ctx.fillStyle = isSelected ? '#FFFFFF' : '#111111';
        ctx.fillText(labelText, x + 6, Math.max(offsetY + 14, y - 6));
      });
    }

    // Pothole Detections
    if (activeLayer === 'potholes' || activeLayer === 'combined' || activeLayer === 'confidence') {
      potholeDetections.forEach((pot) => {
        const isSelected = selectedDetectionId === pot.id;
        const x = offsetX + pot.bbox.x1 * scaleX;
        const y = offsetY + pot.bbox.y1 * scaleY;
        const w = (pot.bbox.x2 - pot.bbox.x1) * scaleX;
        const h = (pot.bbox.y2 - pot.bbox.y1) * scaleY;

        ctx.strokeStyle = '#FF5A5F';
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        const labelText = `POTHOLE ${(pot.confidence * 100).toFixed(0)}%${pot.severity ? ` · ${pot.severity.toUpperCase()}` : ''}`;
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = '#FF5A5F';
        ctx.fillRect(x, Math.max(offsetY, y - 20), textWidth + 10, 20);

        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(labelText, x + 5, Math.max(offsetY + 13, y - 5));
      });
    }

    ctx.restore();
  }, [
    mediaType,
    loadedImage,
    activeLayer,
    maskOpacity,
    yoloDetections,
    potholeDetections,
    selectedDetectionId,
    roadMaskUrl,
    loadedMask,
    zoom,
    pan
  ]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left - pan.x) / zoom;
      const clickY = (e.clientY - rect.top - pan.y) / zoom;

      const canvasW = canvas.width;
      const canvasH = canvas.height;
      const imgNatW = loadedImage?.naturalWidth || 1280;
      const imgNatH = loadedImage?.naturalHeight || 720;
      const imgAsp = imgNatW / imgNatH;
      const canAsp = canvasW / canvasH;
      let dW: number, dH: number, dX: number, dY: number;
      if (imgAsp > canAsp) { dW = canvasW; dH = canvasW / imgAsp; dX = 0; dY = (canvasH - dH) / 2; }
      else { dH = canvasH; dW = canvasH * imgAsp; dX = (canvasW - dW) / 2; dY = 0; }
      const sX = dW / imgNatW;
      const sY = dH / imgNatH;

      let foundId: string | null = null;
      for (const det of yoloDetections) {
        const x = dX + det.bbox.x1 * sX;
        const y = dY + det.bbox.y1 * sY;
        const w = (det.bbox.x2 - det.bbox.x1) * sX;
        const h = (det.bbox.y2 - det.bbox.y1) * sY;
        if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
          foundId = det.id;
          break;
        }
      }

      if (!foundId) {
        for (const pot of potholeDetections) {
          const x = dX + pot.bbox.x1 * sX;
          const y = dY + pot.bbox.y1 * sY;
          const w = (pot.bbox.x2 - pot.bbox.x1) * sX;
          const h = (pot.bbox.y2 - pot.bbox.y1) * sY;
          if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
            foundId = pot.id;
            break;
          }
        }
      }

      if (foundId) {
        onSelectDetection(foundId);
      } else {
        onSelectDetection(null);
        setIsPanning(true);
        setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPanning) {
      setPan({
        x: e.clientX - startPan.x,
        y: e.clientY - startPan.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const layers: { key: LayerType; label: string }[] = [
    { key: 'original', label: 'ORIGINAL' },
    { key: 'objects', label: 'OBJECTS' },
    { key: 'mask', label: 'ROAD MASK' },
    { key: 'potholes', label: 'POTHOLES' },
    { key: 'combined', label: 'COMBINED' },
    { key: 'confidence', label: 'CONFIDENCE' }
  ];

  return (
    <div ref={containerRef} className="neo-card-lg overflow-hidden bg-[#FFFFFF] relative">
      {/* Top Layer Switches Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b-3 border-[#111111] bg-[#FFFFFF]">
        <div className="flex items-center gap-1.5 flex-wrap">
          {layers.map((l) => (
            <button
              key={l.key}
              onClick={() => onLayerChange(l.key)}
              className={`px-3 py-1.5 text-xs font-mono font-bold uppercase transition-all rounded ${
                activeLayer === l.key
                  ? 'bg-[#FFD84D] text-[#111111] border-2 border-[#111111] shadow-[2px_2px_0px_#111111]'
                  : 'bg-[#FFFFFF] text-[#111111] border-2 border-transparent hover:border-[#111111]'
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>

        {/* Mask Opacity Slider */}
        {(activeLayer === 'mask' || activeLayer === 'combined') && (
          <div className="flex items-center gap-2 px-3 py-1 rounded bg-[#FFFFFF] border-2 border-[#111111] text-xs font-mono font-bold">
            <Sliders className="w-3.5 h-3.5 stroke-[2.5]" />
            <span>OPACITY</span>
            <input
              type="range"
              min="0"
              max="100"
              value={maskOpacity}
              onChange={(e) => onOpacityChange(Number(e.target.value))}
              className="w-20 accent-[#111111] cursor-pointer"
            />
            <span className="w-8">{maskOpacity}%</span>
          </div>
        )}

        {/* Zoom Controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoom((z) => Math.min(z + 0.2, 3))}
            className="p-1.5 rounded bg-[#FFFFFF] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] hover:bg-[#FFD84D]"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4 stroke-[2.5]" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(z - 0.2, 0.5))}
            className="p-1.5 rounded bg-[#FFFFFF] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] hover:bg-[#FFD84D]"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4 stroke-[2.5]" />
          </button>
          <button
            onClick={resetView}
            className="p-1.5 rounded bg-[#FFFFFF] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] hover:bg-[#FFD84D]"
            title="Reset View"
          >
            <RotateCcw className="w-4 h-4 stroke-[2.5]" />
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="relative bg-[#0F172A] flex items-center justify-center min-h-[480px]">
        {mediaType === 'video' && mediaUrl ? (
          <video
            ref={videoRef}
            src={mediaUrl}
            controls
            className="w-full max-h-[480px] object-contain"
          />
        ) : (
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className="w-full h-[480px] cursor-crosshair block"
          />
        )}

        {/* HUD Info Badge */}
        <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-1">
          <span className="px-2.5 py-1 rounded bg-[#FFD84D] border-2 border-[#111111] text-[10px] font-mono font-bold text-[#111111] shadow-[2px_2px_0px_#111111] flex items-center gap-1">
            <Sparkles className="w-3 h-3 stroke-[2.5]" />
            LAYER: <span className="uppercase">{activeLayer}</span>
          </span>
          {selectedDetectionId && (
            <span className="px-2.5 py-1 rounded bg-[#FFFFFF] border-2 border-[#111111] text-[10px] font-mono font-bold text-[#111111] shadow-[2px_2px_0px_#111111]">
              SELECTED TARGET: {selectedDetectionId}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
