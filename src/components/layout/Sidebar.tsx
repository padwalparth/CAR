import React, { useState } from 'react';
import {
  LayoutDashboard,
  Scan,
  PlayCircle,
  Radio,
  BarChart3,
  Cpu,
  Settings,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
  User,
  Activity
} from 'lucide-react';

export type NavItemKey = 'dashboard' | 'analyze' | 'simulation' | 'live' | 'analytics' | 'models' | 'settings';

interface SidebarProps {
  currentNav: NavItemKey;
  onNavigate: (key: NavItemKey) => void;
  isPipelineConnected?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentNav,
  onNavigate,
  isPipelineConnected = false
}) => {
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const navItems = [
    { key: 'dashboard' as NavItemKey, label: 'Dashboard', icon: LayoutDashboard },
    { key: 'analyze' as NavItemKey, label: 'Analyze', icon: Scan },
    { key: 'simulation' as NavItemKey, label: 'Simulation', icon: PlayCircle },
    { key: 'live' as NavItemKey, label: 'Live Monitor', icon: Radio },
    { key: 'analytics' as NavItemKey, label: 'Analytics', icon: BarChart3 },
    { key: 'models' as NavItemKey, label: 'Models', icon: Cpu },
    { key: 'settings' as NavItemKey, label: 'Settings', icon: Settings }
  ];

  return (
    <aside
      className={`fixed left-0 top-0 bottom-0 z-40 bg-[#FFFFFF] border-r-3 border-[#111111] flex flex-col transition-all duration-200 ease-in-out ${
        isCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b-3 border-[#111111] shrink-0 bg-[#FFFFFF]">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-9 h-9 rounded-md bg-[#FFD84D] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-center text-[#111111] shrink-0">
            <ShieldAlert className="w-5 h-5 stroke-[2.5]" />
          </div>
          {!isCollapsed && (
            <div className="flex flex-col">
              <span className="font-bold text-base tracking-tight text-[#111111] font-display flex items-center gap-1.5 uppercase">
                RoadVision <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#FFD84D] border border-[#111111] font-mono">AI</span>
              </span>
              <span className="text-[9px] text-[#555555] font-mono font-bold tracking-wider uppercase">
                Safety Intelligence
              </span>
            </div>
          )}
        </div>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1 rounded bg-[#FFFFFF] border-2 border-[#111111] text-[#111111] hover:bg-[#FFD84D] transition-colors"
          title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4 stroke-[2.5]" /> : <ChevronLeft className="w-4 h-4 stroke-[2.5]" />}
        </button>
      </div>

      {/* Primary Navigation */}
      <nav className="flex-1 py-4 px-3 space-y-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentNav === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              title={isCollapsed ? item.label : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md font-bold text-xs uppercase tracking-wider transition-all duration-100 ${
                isActive
                  ? 'bg-[#FFD84D] text-[#111111] border-2 border-[#111111] shadow-[3px_3px_0px_#111111] translate-x-0.5'
                  : 'text-[#111111] bg-[#FFFFFF] hover:bg-[#F7F7F2] border-2 border-transparent hover:border-[#111111]'
              }`}
            >
              <Icon className="w-4 h-4 shrink-0 stroke-[2.5]" />
              {!isCollapsed && <span className="truncate font-display">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Bottom Footer Section */}
      <div className="p-3 border-t-3 border-[#111111] space-y-2 shrink-0 bg-[#F7F7F2]">
        {/* System Health Status Indicator */}
        {!isCollapsed ? (
          <div className="p-2.5 rounded-md bg-[#FFFFFF] border-2 border-[#111111] shadow-[2px_2px_0px_#111111] flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#111111] stroke-[2.5]" />
              <span className="text-[#111111] font-bold">PIPELINE</span>
            </div>
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold border-2 border-[#111111] uppercase ${
              isPipelineConnected ? 'bg-[#53D769] text-[#111111]' : 'bg-[#FFFFFF] text-[#555555]'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${isPipelineConnected ? 'bg-[#111111] animate-ping' : 'bg-[#888888]'}`} />
              {isPipelineConnected ? 'ACTIVE' : 'OFFLINE'}
            </span>
          </div>
        ) : (
          <div className="flex justify-center p-2" title={isPipelineConnected ? 'AI Pipeline Active' : 'AI Pipeline Offline'}>
            <span className={`w-3 h-3 rounded-full border-2 border-[#111111] ${isPipelineConnected ? 'bg-[#53D769]' : 'bg-[#888888]'}`} />
          </div>
        )}

        {/* User Profile */}
        <div className="flex items-center justify-between pt-1">
          <button
            onClick={() => onNavigate('settings')}
            className="flex items-center gap-2 p-1.5 rounded-md bg-[#FFFFFF] border-2 border-[#111111] hover:bg-[#FFD84D] transition-colors w-full text-left"
          >
            <div className="w-7 h-7 rounded bg-[#111111] flex items-center justify-center text-[#FFFFFF] shrink-0 font-bold">
              <User className="w-4 h-4 stroke-[2.5]" />
            </div>
            {!isCollapsed && (
              <div className="flex flex-col truncate font-mono">
                <span className="text-[11px] font-bold text-[#111111]">RESEARCHER</span>
                <span className="text-[9px] text-[#555555]">SAFETY ENG</span>
              </div>
            )}
          </button>
        </div>
      </div>
    </aside>
  );
};
