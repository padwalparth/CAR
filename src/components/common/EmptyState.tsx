import React from 'react';
import { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon: Icon,
  title,
  description,
  actionLabel,
  onAction,
  className = ''
}) => {
  return (
    <div className={`neo-card p-8 text-center flex flex-col items-center justify-center bg-[#FFFFFF] ${className}`}>
      <div className="p-4 rounded-lg bg-[#FFD84D] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] mb-4 text-[#111111]">
        <Icon className="w-8 h-8 stroke-[2.5]" />
      </div>
      <h3 className="text-lg font-bold text-[#111111] uppercase tracking-wide mb-1 font-display">{title}</h3>
      <p className="text-xs text-[#555555] max-w-sm mb-6 leading-relaxed font-medium">{description}</p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="neo-btn-primary px-5 py-2.5 text-xs font-bold uppercase tracking-wider"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
};
