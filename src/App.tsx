import React, { useState } from 'react';
import { AppLayout } from './components/layout/AppLayout';
import { NavItemKey } from './components/layout/Sidebar';
import { useModelHealth } from './hooks/useModelHealth';

import { DashboardPage } from './pages/Dashboard';
import { AnalyzePage } from './pages/Analyze';
import { SimulationPage } from './pages/Simulation';
import { LiveMonitorPage } from './pages/LiveMonitor';
import { AnalyticsPage } from './pages/Analytics';
import { ModelsPage } from './pages/Models';
import { SettingsPage } from './pages/Settings';

export function App() {
  const [currentNav, setCurrentNav] = useState<NavItemKey>('dashboard');
  const { isPipelineConnected } = useModelHealth(10000);

  const getPageHeaderInfo = (key: NavItemKey) => {
    switch (key) {
      case 'dashboard':
        return {
          title: 'Road Intelligence Dashboard',
          description: 'Monitor road conditions, traffic objects, hazards and AI model activity from one workspace.'
        };
      case 'analyze':
        return {
          title: 'Analyze Workspace',
          description: 'Upload road media, execute ML pipeline, and inspect multi-layer computer vision overlays.'
        };
      case 'simulation':
        return {
          title: 'Road Safety Simulation',
          description: 'Interactive vehicle safety simulation driven by real AI perception results.'
        };
      case 'live':
        return {
          title: 'Live Traffic Monitor',
          description: 'Future-ready real-time RTSP/WebSocket camera stream processing.'
        };
      case 'analytics':
        return {
          title: 'Safety & Model Analytics',
          description: 'Statistical distributions, pothole timelines, and latency metrics.'
        };
      case 'models':
        return {
          title: 'ML Microservices & Architecture',
          description: 'Inspect backend ML microservices health, versioning, latencies, and topology.'
        };
      case 'settings':
        return {
          title: 'System Settings',
          description: 'Configure API base endpoints, confidence defaults, and Demo Mode.'
        };
      default:
        return {
          title: 'RoadVision AI',
          description: 'Intelligent Road & Traffic Safety Analysis'
        };
    }
  };

  const headerInfo = getPageHeaderInfo(currentNav);

  return (
    <AppLayout
      currentNav={currentNav}
      onNavigate={setCurrentNav}
      isPipelineConnected={isPipelineConnected}
      pageTitle={headerInfo.title}
      pageDescription={headerInfo.description}
    >
      {currentNav === 'dashboard' && (
        <DashboardPage onNavigateToAnalyze={() => setCurrentNav('analyze')} />
      )}
      {currentNav === 'analyze' && <AnalyzePage onNavigateToSimulation={() => setCurrentNav('simulation')} />}
      {currentNav === 'simulation' && <SimulationPage onNavigateToAnalyze={() => setCurrentNav('analyze')} />}
      {currentNav === 'live' && <LiveMonitorPage />}
      {currentNav === 'analytics' && <AnalyticsPage />}
      {currentNav === 'models' && <ModelsPage />}
      {currentNav === 'settings' && <SettingsPage />}
    </AppLayout>
  );
}

export default App;
