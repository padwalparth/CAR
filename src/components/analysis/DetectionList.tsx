import React, { useState } from 'react';
import { Detection, PotholeDetection } from '../../types/inference';
import { Search, Sliders, Car, AlertTriangle, User, Box } from 'lucide-react';

interface DetectionListProps {
  yoloDetections: Detection[];
  potholeDetections: PotholeDetection[];
  availableClasses: string[];
  selectedDetectionId: string | null;
  onSelectDetection: (id: string | null) => void;
  confidenceThreshold: number;
  onConfidenceChange: (val: number) => void;
}

export const DetectionList: React.FC<DetectionListProps> = ({
  yoloDetections,
  potholeDetections,
  availableClasses,
  selectedDetectionId,
  onSelectDetection,
  confidenceThreshold,
  onConfidenceChange
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedClassFilter, setSelectedClassFilter] = useState<string>('all');

  const filteredYolo = yoloDetections.filter((item) => {
    const matchesSearch = item.class_name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesClass = selectedClassFilter === 'all' || selectedClassFilter === item.class_name;
    return matchesSearch && matchesClass;
  });

  const filteredPotholes = potholeDetections.filter((item) => {
    const matchesSearch = 'pothole'.includes(searchQuery.toLowerCase());
    const matchesClass = selectedClassFilter === 'all' || selectedClassFilter === 'potholes';
    return matchesSearch && matchesClass;
  });

  const totalDetectionsCount = filteredYolo.length + filteredPotholes.length;

  const getClassIcon = (className: string) => {
    const lower = className.toLowerCase();
    if (lower.includes('pedestrian') || lower.includes('person')) return User;
    if (lower.includes('pothole')) return AlertTriangle;
    if (lower.includes('vehicle') || lower.includes('car') || lower.includes('tractor') || lower.includes('autorickshaw')) return Car;
    return Box;
  };

  return (
    <div className="neo-card p-5 bg-[#FFFFFF] flex flex-col h-full">
      {/* Header & Controls */}
      <div className="mb-4 space-y-3">
        <div className="flex items-center justify-between pb-2 border-b-2 border-[#111111]">
          <h3 className="text-xs font-bold text-[#111111] uppercase tracking-wider font-display flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-[#FFD84D] border border-[#111111]" />
            DETECTED TARGETS ({totalDetectionsCount})
          </h3>
        </div>

        {/* Confidence Filter Control */}
        <div className="p-3 rounded bg-[#FFFFFF] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] space-y-1.5 font-mono">
          <div className="flex items-center justify-between text-xs font-bold">
            <span className="text-[#111111] uppercase flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 stroke-[2.5]" />
              CONFIDENCE THRESHOLD
            </span>
            <span className="text-[#111111] bg-[#FFD84D] px-1.5 py-0.5 rounded border border-[#111111]">
              {(confidenceThreshold * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidenceThreshold}
            onChange={(e) => onConfidenceChange(Number(e.target.value))}
            className="w-full accent-[#111111] cursor-pointer"
          />
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="w-4 h-4 text-[#111111] absolute left-3 top-2.5 stroke-[2.5]" />
          <input
            type="text"
            placeholder="SEARCH TARGETS..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="neo-input w-full pl-9 pr-3 py-2 text-xs font-mono font-bold uppercase placeholder-[#888888]"
          />
        </div>

        {/* Dynamic Class Filters */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          <button
            onClick={() => setSelectedClassFilter('all')}
            className={`px-2.5 py-1 text-[10px] font-mono font-bold uppercase rounded transition-all border-2 border-[#111111] ${
              selectedClassFilter === 'all'
                ? 'bg-[#FFD84D] text-[#111111] shadow-[2px_2px_0px_#111111]'
                : 'bg-[#FFFFFF] text-[#111111] hover:bg-[#F7F7F2]'
            }`}
          >
            ALL
          </button>
          {availableClasses.map((cls) => (
            <button
              key={cls}
              onClick={() => setSelectedClassFilter(cls)}
              className={`px-2.5 py-1 text-[10px] font-mono font-bold uppercase rounded transition-all border-2 border-[#111111] ${
                selectedClassFilter === cls
                  ? 'bg-[#FFD84D] text-[#111111] shadow-[2px_2px_0px_#111111]'
                  : 'bg-[#FFFFFF] text-[#111111] hover:bg-[#F7F7F2]'
              }`}
            >
              {cls}
            </button>
          ))}
          {potholeDetections.length > 0 && (
            <button
              onClick={() => setSelectedClassFilter('potholes')}
              className={`px-2.5 py-1 text-[10px] font-mono font-bold uppercase rounded transition-all border-2 border-[#111111] ${
                selectedClassFilter === 'potholes'
                  ? 'bg-[#FF5A5F] text-[#FFFFFF] shadow-[2px_2px_0px_#111111]'
                  : 'bg-[#FFFFFF] text-[#111111] hover:bg-[#F7F7F2]'
              }`}
            >
              POTHOLES
            </button>
          )}
        </div>
      </div>

      {/* Targets List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-[300px] max-h-[420px]">
        {totalDetectionsCount === 0 ? (
          <div className="p-8 text-center text-[#555555] font-mono text-xs italic bg-[#F7F7F2] rounded border-2 border-[#111111]">
            No detections matching selected filters or confidence threshold.
          </div>
        ) : (
          <>
            {/* YOLO Objects */}
            {filteredYolo.map((det) => {
              const Icon = getClassIcon(det.class_name);
              const isSelected = selectedDetectionId === det.id;
              return (
                <div
                  key={det.id}
                  onClick={() => onSelectDetection(isSelected ? null : det.id)}
                  className={`p-3 rounded border-2 border-[#111111] transition-all cursor-pointer flex items-center justify-between font-mono ${
                    isSelected
                      ? 'bg-[#FFD84D] text-[#111111] shadow-[3px_3px_0px_#111111] -translate-y-0.5'
                      : 'bg-[#FFFFFF] hover:bg-[#F7F7F2] shadow-[2px_2px_0px_#111111]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-[#FFFFFF] border-2 border-[#111111] flex items-center justify-center text-[#111111] shrink-0 font-bold">
                      <Icon className="w-4 h-4 stroke-[2.5]" />
                    </div>
                    <div>
                      <h4 className="text-xs font-bold uppercase">{det.class_name}</h4>
                      <p className="text-[10px] text-[#555555] font-bold mt-0.5">
                        BBOX: [{det.bbox.x1}, {det.bbox.y1}, {det.bbox.x2}, {det.bbox.y2}]
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-xs font-bold text-[#111111] bg-[#FFFFFF] px-1.5 py-0.5 border border-[#111111] rounded">
                      {(det.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Potholes */}
            {filteredPotholes.map((pot) => {
              const isSelected = selectedDetectionId === pot.id;
              return (
                <div
                  key={pot.id}
                  onClick={() => onSelectDetection(isSelected ? null : pot.id)}
                  className={`p-3 rounded border-2 border-[#111111] transition-all cursor-pointer flex items-center justify-between font-mono ${
                    isSelected
                      ? 'bg-[#FF5A5F] text-[#FFFFFF] shadow-[3px_3px_0px_#111111] -translate-y-0.5'
                      : 'bg-[#FFFFFF] hover:bg-[#F7F7F2] shadow-[2px_2px_0px_#111111]'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-[#FF5A5F] text-[#FFFFFF] border-2 border-[#111111] flex items-center justify-center shrink-0 font-bold">
                      <AlertTriangle className="w-4 h-4 stroke-[2.5]" />
                    </div>
                    <div>
                      <h4 className="text-xs font-bold uppercase">POTHOLE HAZARD</h4>
                      <p className="text-[10px] text-[#555555] font-bold mt-0.5">
                        {pot.severity ? `SEVERITY: ${pot.severity.toUpperCase()}` : `BBOX: [${pot.bbox.x1}, ${pot.bbox.y1}]`}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-xs font-bold text-[#111111] bg-[#FFFFFF] px-1.5 py-0.5 border border-[#111111] rounded">
                      {(pot.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </>
        )}
      </div>
    </div>
  );
};
