import React from 'react';
import {
  LayoutDashboard,
  Eye,
  Volume2,
  Activity,
  Layers,
  ShieldAlert,
  PlaySquare,
  BarChart3,
  Server,
} from 'lucide-react';

export type ViewType =
  | 'overview'
  | 'vision'
  | 'audio'
  | 'gesture'
  | 'fusion'
  | 'decision'
  | 'replay'
  | 'benchmarks'
  | 'system';

interface SidebarProps {
  activeView: ViewType;
  onSelectView: (view: ViewType) => void;
  collapsed?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onSelectView }) => {
  const navSections = [
    {
      title: 'Console & Overview',
      items: [
        {
          id: 'overview' as ViewType,
          label: 'System Overview',
          icon: LayoutDashboard,
          badge: null,
          badgeColor: '',
        },
      ],
    },
    {
      title: 'Sensory Branches',
      items: [
        {
          id: 'vision' as ViewType,
          label: 'Computer Vision',
          icon: Eye,
          badge: 'YOLOv8',
          badgeColor: 'bg-brand-cyan/10 text-brand-cyan border-brand-cyan/30',
        },
        {
          id: 'audio' as ViewType,
          label: 'Environmental Audio',
          icon: Volume2,
          badge: 'PANNs',
          badgeColor: 'bg-brand-amber/10 text-brand-amber border-brand-amber/30',
        },
        {
          id: 'gesture' as ViewType,
          label: 'Gesture Recognition',
          icon: Activity,
          badge: 'LSTM',
          badgeColor: 'bg-brand-emerald/10 text-brand-emerald border-brand-emerald/30',
        },
      ],
    },
    {
      title: 'Multimodal Fusion & Policy',
      items: [
        {
          id: 'fusion' as ViewType,
          label: 'Transformer Fusion',
          icon: Layers,
          badge: '2L-4H',
          badgeColor: 'bg-brand-purple/10 text-brand-purple border-brand-purple/30',
        },
        {
          id: 'decision' as ViewType,
          label: 'Decision & Speech',
          icon: ShieldAlert,
          badge: 'SAPI',
          badgeColor: 'bg-brand-rose/10 text-brand-rose border-brand-rose/30',
        },
      ],
    },
    {
      title: 'Demonstration & Verification',
      items: [
        {
          id: 'replay' as ViewType,
          label: 'Synchronized Replay',
          icon: PlaySquare,
          badge: 'Live WS',
          badgeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
        },
        {
          id: 'benchmarks' as ViewType,
          label: 'Model Benchmarks',
          icon: BarChart3,
          badge: 'Metrics',
          badgeColor: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
        },
        {
          id: 'system' as ViewType,
          label: 'System & Checkpoints',
          icon: Server,
          badge: 'Python',
          badgeColor: 'bg-slate-500/10 text-slate-400 border-slate-500/30',
        },
      ],
    },
  ];

  return (
    <aside className="w-64 border-r border-cardBorder bg-surface flex flex-col justify-between shrink-0 select-none">
      <div className="py-4 px-3 space-y-6 overflow-y-auto">
        {navSections.map((section, sIdx) => (
          <div key={sIdx} className="space-y-1">
            <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-3 mb-2 font-mono">
              {section.title}
            </h3>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const Icon = item.icon;
                const isActive = activeView === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onSelectView(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-card text-brand-cyan font-semibold border border-brand-cyan/30 shadow-card'
                        : 'text-slate-300 hover:text-white hover:bg-surfaceHover border border-transparent'
                    }`}
                  >
                    <div className="flex items-center space-x-2.5 truncate">
                      <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-brand-cyan' : 'text-slate-400'}`} />
                      <span className="truncate">{item.label}</span>
                    </div>
                    {item.badge && (
                      <span
                        className={`text-[10px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${item.badgeColor}`}
                      >
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-cardBorder text-[11px] text-slate-400 font-mono bg-card/40">
        <div className="flex items-center justify-between text-slate-300 font-medium">
          <span>AI AINAK</span>
          <span className="text-brand-cyan">v0.1.0</span>
        </div>
        <div className="text-[10px] text-slate-400 mt-0.5 truncate">
          Offline Multimodal System
        </div>
      </div>
    </aside>
  );
};
