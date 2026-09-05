import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Processing...',
  size = 'md',
  className = ''
}) => {
  const sizeMap = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-10 h-10'
  };

  return (
    <div className={`flex flex-col items-center justify-center gap-3 p-6 text-[#111111] ${className}`}>
      <Loader2 className={`${sizeMap[size]} animate-spin text-[#111111] stroke-[2.5]`} />
      {label && <span className="text-xs font-mono font-bold tracking-wider uppercase text-[#111111]">{label}</span>}
    </div>
  );
};
