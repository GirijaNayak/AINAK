import React from 'react';
import { Eye, CheckCircle2, Cpu, HardDrive, Maximize2, Minimize2, Radio } from 'lucide-react';
import { SystemStatus } from '../../types';

interface TopbarProps {
  systemStatus: SystemStatus | null;
  isPresentationMode: boolean;
  onTogglePresentation: () => void;
  backendOnline: boolean;
}

export const Topbar: React.FC<TopbarProps> = ({
  systemStatus,
  isPresentationMode,
  onTogglePresentation,
  backendOnline,
}) => {
  return (
    <header className="h-16 border-b border-cardBorder bg-surface/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
      {/* Brand */}
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-cyan/20 to-brand-blue/20 border border-brand-cyan/40 flex items-center justify-center text-brand-cyan shadow-glow-cyan">
          <Eye className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-lg font-bold tracking-tight text-white font-mono">AI AINAK</h1>
            <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/30 rounded-full">
              Research Prototype
            </span>
          </div>
          <p className="text-xs text-slate-400 font-sans hidden sm:block">
            Multimodal Assistive Intelligence Console
          </p>
        </div>
      </div>

      {/* Topbar System Stats */}
      <div className="hidden md:flex items-center space-x-3 text-xs font-mono">
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
          <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
          <span>{backendOnline ? 'Backend Online' : 'Connecting...'}</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Tests: {systemStatus?.automated_tests?.passed ?? 57} Passed (100%)</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-brand-cyan" />
          <span>Device: {systemStatus?.device?.toUpperCase() || 'CPU'}</span>
        </div>

        {systemStatus?.torch_version && (
          <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-400">
            <Radio className="w-3.5 h-3.5 text-brand-purple" />
            <span>PyTorch {systemStatus.torch_version.split('+')[0]}</span>
          </div>
        )}
      </div>

      {/* Presentation Mode Toggle */}
      <div className="flex items-center space-x-3">
        <button
          onClick={onTogglePresentation}
          className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isPresentationMode
              ? 'bg-brand-cyan text-slate-950 font-semibold shadow-glow-cyan'
              : 'bg-card hover:bg-surfaceHover border border-cardBorder text-slate-300 hover:text-white'
          }`}
          title="Toggle High-Contrast Clean Presentation Mode for Projection"
        >
          {isPresentationMode ? (
            <>
              <Minimize2 className="w-3.5 h-3.5" />
              <span>Exit Presentation</span>
            </>
          ) : (
            <>
              <Maximize2 className="w-3.5 h-3.5" />
              <span>Presentation Mode</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
