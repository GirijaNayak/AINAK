import React, { useState, useEffect } from 'react';
import { Topbar } from './components/layout/Topbar';
import { Sidebar, ViewType } from './components/layout/Sidebar';
import { OverviewView } from './components/views/OverviewView';
import { VisionView } from './components/views/VisionView';
import { AudioView } from './components/views/AudioView';
import { GestureView } from './components/views/GestureView';
import { FusionView } from './components/views/FusionView';
import { DecisionView } from './components/views/DecisionView';
import { ReplayView } from './components/views/ReplayView';
import { BenchmarksView } from './components/views/BenchmarksView';
import { SystemInfoView } from './components/views/SystemInfoView';
import { SystemStatus, ModelsStatus, ReportsSummary } from './types';
import { fetchSystemStatus, fetchModelsStatus, fetchReportsSummary } from './services/api';

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ViewType>('overview');
  const [isPresentationMode, setIsPresentationMode] = useState<boolean>(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null);
  const [reportsSummary, setReportsSummary] = useState<ReportsSummary | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [sys, mod, rep] = await Promise.all([
        fetchSystemStatus(),
        fetchModelsStatus(),
        fetchReportsSummary(),
      ]);
      setSystemStatus(sys);
      setModelsStatus(mod);
      setReportsSummary(rep);
      setBackendOnline(true);
    } catch (e) {
      console.warn('Backend currently unreachable, using fallback telemetry.', e);
      setBackendOnline(false);
    }
  };

  const renderActiveView = () => {
    switch (activeView) {
      case 'overview':
        return (
          <OverviewView
            systemStatus={systemStatus}
            modelsStatus={modelsStatus}
            reportsSummary={reportsSummary}
            onNavigate={(v) => setActiveView(v)}
          />
        );
      case 'vision':
        return <VisionView />;
      case 'audio':
        return <AudioView />;
      case 'gesture':
        return <GestureView />;
      case 'fusion':
        return <FusionView />;
      case 'decision':
        return <DecisionView />;
      case 'replay':
        return <ReplayView />;
      case 'benchmarks':
        return <BenchmarksView />;
      case 'system':
        return <SystemInfoView systemStatus={systemStatus} modelsStatus={modelsStatus} />;
      default:
        return <OverviewView systemStatus={systemStatus} modelsStatus={modelsStatus} reportsSummary={reportsSummary} onNavigate={setActiveView} />;
    }
  };

  return (
    <div className="min-h-screen bg-background text-slate-100 flex flex-col font-sans">
      <Topbar
        systemStatus={systemStatus}
        isPresentationMode={isPresentationMode}
        onTogglePresentation={() => setIsPresentationMode(!isPresentationMode)}
        backendOnline={backendOnline}
      />

      <div className="flex-1 flex overflow-hidden">
        {!isPresentationMode && (
          <Sidebar activeView={activeView} onSelectView={(v) => setActiveView(v)} />
        )}

        <main className={`flex-1 overflow-y-auto p-6 ${isPresentationMode ? 'max-w-7xl mx-auto w-full' : ''}`}>
          {renderActiveView()}
        </main>
      </div>
    </div>
  );
};

export default App;
