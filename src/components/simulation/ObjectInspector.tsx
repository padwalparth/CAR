import React from 'react';
import { SimObjectPosition } from '../../types/simulation';
import { Target, X, Cpu, Compass, Layers, AlertCircle } from 'lucide-react';

interface ObjectInspectorProps {
  selectedObject?: SimObjectPosition;
  onClose: () => void;
}

export const ObjectInspector: React.FC<ObjectInspectorProps> = ({
  selectedObject,
  onClose
}) => {
  if (!selectedObject) {
    return (
      <div className="neo-card p-4 bg-[#FFFFFF] text-center font-mono">
        <Target className="w-6 h-6 mx-auto mb-2 text-[#888888] stroke-[2]" />
        <h4 className="text-xs font-bold text-[#111111] uppercase">No Object Selected</h4>
        <p className="text-[11px] text-[#555555] mt-1">
          Click any vehicle or pothole on the simulation canvas to inspect its parameters.
        </p>
      </div>
    );
  }

  const isPothole = selectedObject.type === 'pothole';

  return (
    <div className="neo-card p-4 bg-[#FFFFFF] space-y-3 font-mono">
      {/* Header */}
      <div className="flex items-center justify-between border-b-2 border-[#111111] pb-2">
        <div className="flex items-center gap-2">
          {isPothole ? (
            <AlertCircle className="w-4 h-4 text-[#EF4444] stroke-[2.5]" />
          ) : (
            <Target className="w-4 h-4 text-[#10B981] stroke-[2.5]" />
          )}
          <h4 className="text-xs font-bold text-[#111111] uppercase tracking-wider">
            {selectedObject.className.toUpperCase()} ({selectedObject.id})
          </h4>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded bg-[#F7F7F2] border border-[#111111] hover:bg-[#FFD84D] transition-colors"
          title="Close Inspector"
        >
          <X className="w-3.5 h-3.5 stroke-[2.5]" />
        </button>
      </div>

      {/* Details Grid */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="p-2 rounded bg-[#F7F7F2] border border-[#111111] space-y-0.5">
          <span className="text-[10px] text-[#555555] font-bold block uppercase">CONFIDENCE</span>
          <span className="font-extrabold text-[#111111]">
            {(selectedObject.confidence * 100).toFixed(1)}%
          </span>
        </div>

        <div className="p-2 rounded bg-[#F7F7F2] border border-[#111111] space-y-0.5">
          <span className="text-[10px] text-[#555555] font-bold block uppercase">SOURCE MODEL</span>
          <span className="font-bold text-[#111111] truncate block text-[10px]">
            {selectedObject.sourceModel}
          </span>
        </div>

        <div className="p-2 rounded bg-[#F7F7F2] border border-[#111111] space-y-0.5">
          <span className="text-[10px] text-[#555555] font-bold block uppercase">POSITION</span>
          <span className="font-bold text-[#111111] text-[10px]">
            Estimated 2D Projection
          </span>
        </div>

        <div className="p-2 rounded bg-[#F7F7F2] border border-[#111111] space-y-0.5">
          <span className="text-[10px] text-[#555555] font-bold block uppercase">SEVERITY</span>
          <span className={`font-bold text-[10px] ${selectedObject.severity && selectedObject.severity !== 'Not provided' ? 'text-[#EF4444]' : 'text-[#555555]'}`}>
            {selectedObject.severity || 'Not provided'}
          </span>
        </div>
      </div>

      {/* Bounding Box Coordinates */}
      <div className="p-2.5 rounded bg-[#F7F7F2] border border-[#111111] text-[11px] space-y-1">
        <span className="text-[10px] font-bold text-[#555555] uppercase block">
          IMAGE BOUNDING BOX (PIXELS)
        </span>
        <div className="grid grid-cols-4 gap-1 text-center font-bold text-[#111111]">
          <span>X1: {selectedObject.bbox.x1}</span>
          <span>Y1: {selectedObject.bbox.y1}</span>
          <span>X2: {selectedObject.bbox.x2}</span>
          <span>Y2: {selectedObject.bbox.y2}</span>
        </div>
      </div>
    </div>
  );
};
