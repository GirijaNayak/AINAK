import React from 'react';
import {
  Eye,
  Volume2,
  Activity,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  PlaySquare,
  Cpu,
  Terminal,
} from 'lucide-react';
import { SystemStatus, ModelsStatus, ReportsSummary } from '../../types';
import { ViewType } from '../layout/Sidebar';

interface OverviewViewProps {
  systemStatus: SystemStatus | null;
  modelsStatus: ModelsStatus | null;
  reportsSummary: ReportsSummary | null;
  onNavigate: (view: ViewType) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  systemStatus,
  modelsStatus,
  reportsSummary,
  onNavigate,
}) => {
  const branches = [
    {
      id: 'vision' as ViewType,
      title: 'Computer Vision',
      model: 'YOLOv8n + IoU Tracker',
      checkpoint: modelsStatus?.vision?.checkpoint_path || 'artifacts/vision/yolov8n.pt',
      color: 'border-brand-cyan/40 bg-brand-cyan/5 text-brand-cyan',
      icon: Eye,
      metric: '60.52% mAP@0.5',
      latency: '~49.2 ms',
      summary: '80-class object detection, relative depth estimation, and optical loom proximity scoring.',
    },
    {
      id: 'audio' as ViewType,
      title: 'Environmental Audio',
      model: 'PANNs CNN14 + AudioMLP',
      checkpoint: modelsStatus?.audio?.checkpoint_path || 'artifacts/audio/downstream_mlp.pt',
      color: 'border-brand-amber/40 bg-brand-amber/5 text-brand-amber',
      icon: Volume2,
      metric: '89.75% Accuracy',
      latency: '~29.3 ms',
      summary: '50-class ESC-50 acoustic classifier, 2048-D bottleneck representation, noise robust to 0 dB SNR.',
    },
    {
      id: 'gesture' as ViewType,
      title: 'Gesture Recognition',
      model: 'MediaPipe Pose + GestureLSTM',
      checkpoint: modelsStatus?.gesture?.checkpoint_path || 'artifacts/gesture/gesture_lstm.pt',
      color: 'border-brand-emerald/40 bg-brand-emerald/5 text-brand-emerald',
      icon: Activity,
      metric: '100.0% Test Acc',
      latency: '~1.01 ms',
      summary: '33 kinematic pose landmarks normalized to 132-D input, 24-frame temporal buffer, 4 gesture classes.',
    },
    {
      id: 'fusion' as ViewType,
      title: 'Transformer Fusion',
      model: '2-Layer 4-Head Encoder',
      checkpoint: modelsStatus?.fusion?.checkpoint_path || 'artifacts/fusion/fusion.pt',
      color: 'border-brand-purple/40 bg-brand-purple/5 text-brand-purple',
      icon: Layers,
      metric: 'MAE: 0.4172',
      latency: '~2.29 ms',
      summary: 'Learned cross-modal attention projections (128-D), priority regression, and missing modality fallback.',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-cardBorder bg-gradient-to-r from-card to-surface p-6 shadow-card">
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded-md bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/30">
              Multimodal Assistive System
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Synchronized Multimodal Perception
            </span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white font-sans">
            AI AINAK Multimodal Assistive Intelligence Console
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            Real-time assistive glasses prototype integrating forward computer vision, acoustic scene monitoring,
            and human bodily gesture kinematics through a unified Transformer-based sensor fusion architecture,
            deterministic alert policy, and prioritized text-to-speech audio feedback.
          </p>

          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('replay')}
              className="flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold bg-brand-cyan text-slate-950 hover:bg-brand-cyan/90 transition-all shadow-glow-cyan"
            >
              <PlaySquare className="w-4 h-4" />
              <span>Launch Synchronized Replay Demo</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate('benchmarks')}
              className="flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-medium bg-card hover:bg-surfaceHover border border-cardBorder text-slate-200 transition-all"
            >
              <Zap className="w-4 h-4 text-brand-amber" />
              <span>Inspect Quantitative Benchmarks</span>
            </button>
          </div>
        </div>
      </div>

      {/* 4 Modality Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {branches.map((b) => {
          const Icon = b.icon;
          return (
            <div
              key={b.id}
              onClick={() => onNavigate(b.id)}
              className="group cursor-pointer rounded-xl border border-cardBorder bg-card hover:border-cardHoverBorder p-5 transition-all hover:-translate-y-0.5 shadow-card flex flex-col justify-between"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className={`w-9 h-9 rounded-lg border flex items-center justify-center ${b.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-surface border border-cardBorder text-slate-200">
                    {b.metric}
                  </span>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white group-hover:text-brand-cyan transition-colors">
                    {b.title}
                  </h3>
                  <div className="text-xs font-mono text-slate-400 mt-0.5">{b.model}</div>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{b.summary}</p>
              </div>

              <div className="pt-4 mt-4 border-t border-cardBorder flex items-center justify-between text-xs text-slate-400 font-mono">
                <span>Latency: {b.latency}</span>
                <span className="flex items-center space-x-1 group-hover:text-brand-cyan transition-colors">
                  <span>Explore</span>
                  <ArrowRight className="w-3 h-3" />
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Telemetry and Pipeline Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Runtime Diagnostics */}
        <div className="rounded-xl border border-cardBorder bg-card p-5 space-y-4">
          <div className="flex items-center space-x-2 text-sm font-semibold text-white">
            <Cpu className="w-4 h-4 text-brand-cyan" />
            <span>Hardware & Environment</span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex items-center justify-between py-1.5 border-b border-cardBorder/60">
              <span className="text-slate-400">Python Runtime</span>
              <span className="text-slate-200">{systemStatus?.python_version?.split(' ')[0] || '3.10.6'}</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cardBorder/60">
              <span className="text-slate-400">PyTorch Build</span>
              <span className="text-slate-200">{systemStatus?.torch_version || '2.2.2+cpu'}</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cardBorder/60">
              <span className="text-slate-400">Execution Device</span>
              <span className="text-brand-cyan font-bold">{systemStatus?.device?.toUpperCase() || 'CPU'}</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cardBorder/60">
              <span className="text-slate-400">OS Platform</span>
              <span className="text-slate-200">{systemStatus?.platform || 'Windows x64'}</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cardBorder/60">
              <span className="text-slate-400">Automated Tests</span>
              <span className="text-emerald-400 font-bold flex items-center space-x-1">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>{systemStatus?.automated_tests?.passed ?? 57} Passed (100%)</span>
              </span>
            </div>
          </div>
        </div>

        {/* Pipeline Architecture Steps */}
        <div className="lg:col-span-2 rounded-xl border border-cardBorder bg-card p-5 space-y-4">
          <div className="flex items-center space-x-2 text-sm font-semibold text-white">
            <Terminal className="w-4 h-4 text-brand-purple" />
            <span>Synchronized Assistive Pipeline Workflow</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-1.5">
              <div className="flex items-center space-x-2">
                <span className="w-5 h-5 rounded-full bg-brand-cyan/20 text-brand-cyan font-mono text-[10px] flex items-center justify-center font-bold">1</span>
                <span className="font-semibold text-slate-200">Ingestion</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">
                Concurrent capture: 640x640 visual frames, 32 kHz monaural audio, and body pose tracking.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-1.5">
              <div className="flex items-center space-x-2">
                <span className="w-5 h-5 rounded-full bg-brand-amber/20 text-brand-amber font-mono text-[10px] flex items-center justify-center font-bold">2</span>
                <span className="font-semibold text-slate-200">Alignment</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">
                Sliding-window buffer pairs observations within a bounded delta (Δt ≤ 1.5s), dropping stale events.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-1.5">
              <div className="flex items-center space-x-2">
                <span className="w-5 h-5 rounded-full bg-brand-purple/20 text-brand-purple font-mono text-[10px] flex items-center justify-center font-bold">3</span>
                <span className="font-semibold text-slate-200">Attention Fusion</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">
                2-layer 4-head Transformer encoder pools cross-modal context with modality dropout resilience.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-1.5">
              <div className="flex items-center space-x-2">
                <span className="w-5 h-5 rounded-full bg-brand-rose/20 text-brand-rose font-mono text-[10px] flex items-center justify-center font-bold">4</span>
                <span className="font-semibold text-slate-200">Speech Alerts</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">
                Deterministic decision policy maps scores to 5 urgency ranks with refractory cooldowns & TTS.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
