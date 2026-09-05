import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBannerProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title = 'SERVICE ERROR',
  message,
  onRetry,
  className = ''
}) => {
  return (
    <div className={`p-4 rounded-lg bg-[#FF5A5F] text-[#FFFFFF] border-3 border-[#111111] shadow-[5px_5px_0px_#111111] flex items-start justify-between gap-4 ${className}`}>
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-6 h-6 shrink-0 mt-0.5 stroke-[2.5]" />
        <div>
          <h4 className="text-sm font-bold uppercase tracking-wider font-display">{title}</h4>
          <p className="text-xs font-medium mt-1 leading-relaxed">{message}</p>
        </div>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="neo-btn-secondary px-3 py-1.5 text-xs font-bold bg-[#FFFFFF] text-[#111111] shrink-0 flex items-center gap-1.5 uppercase"
        >
          <RefreshCw className="w-3.5 h-3.5 stroke-[2.5]" />
          Retry
        </button>
      )}
    </div>
  );
};
