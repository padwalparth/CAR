import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'safe' | 'caution' | 'critical' | 'info' | 'neutral' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  dot?: boolean;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'neutral',
  size = 'md',
  dot = false,
  className = ''
}) => {
  const variantStyles = {
    safe: 'bg-[#53D769] text-[#111111]',
    caution: 'bg-[#FFD84D] text-[#111111]',
    critical: 'bg-[#FF5A5F] text-[#FFFFFF]',
    info: 'bg-[#4D7CFE] text-[#FFFFFF]',
    purple: 'bg-[#A78BFA] text-[#111111]',
    neutral: 'bg-[#FFFFFF] text-[#111111]'
  };

  const dotColors = {
    safe: 'bg-[#111111]',
    caution: 'bg-[#111111]',
    critical: 'bg-[#FFFFFF]',
    info: 'bg-[#FFFFFF]',
    purple: 'bg-[#111111]',
    neutral: 'bg-[#111111]'
  };

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-[10px] font-mono font-bold',
    md: 'px-2.5 py-1 text-xs font-mono font-bold',
    lg: 'px-3 py-1.5 text-xs font-mono font-bold'
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 border-2 border-[#111111] rounded-md uppercase tracking-wider shadow-[2px_2px_0px_#111111] ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {dot && <span className={`h-2 w-2 rounded-full ${dotColors[variant]}`} />}
      {children}
    </span>
  );
};
