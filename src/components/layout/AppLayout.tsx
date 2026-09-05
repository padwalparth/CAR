import React from 'react';
import { Sidebar, NavItemKey } from './Sidebar';
import { TopBar } from './TopBar';

interface AppLayoutProps {
  currentNav: NavItemKey;
  onNavigate: (key: NavItemKey) => void;
  isPipelineConnected: boolean;
  pageTitle: string;
  pageDescription: string;
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({
  currentNav,
  onNavigate,
  isPipelineConnected,
  pageTitle,
  pageDescription,
  children
}) => {
  return (
    <div className="min-h-screen bg-[#F7F7F2] text-[#111111] flex font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        currentNav={currentNav}
        onNavigate={onNavigate}
        isPipelineConnected={isPipelineConnected}
      />

      {/* Main Workspace Area */}
      <div className="flex-1 flex flex-col min-w-0 pl-64 transition-all duration-200">
        <TopBar
          title={pageTitle}
          description={pageDescription}
          isPipelineConnected={isPipelineConnected}
          onOpenSettings={() => onNavigate('settings')}
        />
        <main className="flex-1 p-6 overflow-y-auto max-w-7xl mx-auto w-full space-y-6">
          {children}
        </main>
      </div>
    </div>
  );
};
