import React, { useState, useRef } from 'react';
import { Upload, FileImage, FileVideo, X, Play, CheckCircle2 } from 'lucide-react';

interface MediaUploadProps {
  onMediaSelect: (file: File) => void;
  onStartAnalysis: () => void;
  isUploading: boolean;
  isAnalyzing: boolean;
  selectedFile: File | null;
  onClearFile: () => void;
}

export const MediaUpload: React.FC<MediaUploadProps> = ({
  onMediaSelect,
  onStartAnalysis,
  isUploading,
  isAnalyzing,
  selectedFile,
  onClearFile
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (isValidMedia(file)) {
        onMediaSelect(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (isValidMedia(file)) {
        onMediaSelect(file);
      }
    }
  };

  const isValidMedia = (file: File) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'video/mp4', 'video/webm'];
    return validTypes.includes(file.type);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="neo-card p-5 mb-6 bg-[#FFFFFF]">
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-3 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-150 flex flex-col items-center justify-center ${
            isDragging
              ? 'border-[#111111] bg-[#FFD84D] translate-x-1 translate-y-1 shadow-[2px_2px_0px_#111111]'
              : 'border-[#111111] bg-[#F7F7F2] hover:bg-[#FFD84D]/30 shadow-[4px_4px_0px_#111111]'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,video/mp4,video/webm"
            className="hidden"
          />
          <div className="w-14 h-14 rounded-md bg-[#FFD84D] border-3 border-[#111111] shadow-[3px_3px_0px_#111111] flex items-center justify-center text-[#111111] mb-3">
            <Upload className="w-7 h-7 stroke-[2.5]" />
          </div>
          <h3 className="text-base font-bold text-[#111111] font-display uppercase tracking-wide mb-1">
            DROP ROAD IMAGE OR VIDEO MEDIA HERE
          </h3>
          <p className="text-xs text-[#555555] font-medium mb-4">
            Supports JPG, JPEG, PNG, MP4 and WebM formats up to 50 MB
          </p>
          <button
            type="button"
            className="neo-btn-secondary px-4 py-2 text-xs uppercase tracking-wider"
          >
            BROWSE MEDIA FILES
          </button>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-4 p-4 rounded-md bg-[#FFFFFF] border-3 border-[#111111] shadow-[4px_4px_0px_#111111]">
          <div className="flex items-center gap-4 min-w-0">
            <div className="w-12 h-12 rounded-md bg-[#FFD84D] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center text-[#111111] shrink-0">
              {selectedFile.type.startsWith('video') ? (
                <FileVideo className="w-6 h-6 stroke-[2.5]" />
              ) : (
                <FileImage className="w-6 h-6 stroke-[2.5]" />
              )}
            </div>
            <div className="truncate">
              <h4 className="text-sm font-bold text-[#111111] font-display uppercase truncate">{selectedFile.name}</h4>
              <div className="flex items-center gap-3 text-xs font-mono font-bold text-[#555555] mt-1">
                <span>{formatFileSize(selectedFile.size)}</span>
                <span>•</span>
                <span className="uppercase text-[#111111] px-1.5 py-0.5 rounded bg-[#FFD84D] border border-[#111111]">
                  {selectedFile.type.split('/')[1]}
                </span>
                <span>•</span>
                <span className="text-[#111111] flex items-center gap-1 font-bold">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#53D769] stroke-[3]" /> READY
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={onClearFile}
              disabled={isAnalyzing}
              className="p-2 rounded bg-[#FFFFFF] border-2 border-[#111111] text-[#111111] hover:bg-[#FF5A5F] hover:text-[#FFFFFF] transition-colors disabled:opacity-50"
              title="Remove File"
            >
              <X className="w-5 h-5 stroke-[2.5]" />
            </button>
            <button
              onClick={onStartAnalysis}
              disabled={isAnalyzing || isUploading}
              className="neo-btn-primary px-6 py-3 text-xs uppercase tracking-wider flex items-center gap-2"
            >
              <Play className="w-4 h-4 fill-[#111111] stroke-[2.5]" />
              {isAnalyzing ? 'RUNNING INFERENCE...' : 'START AI ANALYSIS'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
