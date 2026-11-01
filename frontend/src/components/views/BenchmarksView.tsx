import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  FileCheck2,
  Zap,
  TrendingUp,
  ShieldCheck,
  Cpu,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { ReportsSummary } from '../../types';
import { fetchReportsSummary } from '../../services/api';

export const BenchmarksView: React.FC = () => {
  const [reports, setReports] = useState<ReportsSummary | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'vision' | 'audio' | 'gesture' | 'fusion'>('overview');

  useEffect(() => {
    fetchReportsSummary()
      .then((data) => setReports(data))
      .catch((err) => console.error(err));
  }, []);

  const ablationsData = [
    { subset: 'Vision {V}', mae: 2.504, latency: 1.8 },
    { subset: 'Audio {A}', mae: 2.505, latency: 1.8 },
    { subset: 'Gesture {G}', mae: 2.457, latency: 1.7 },
    { subset: 'Vision+Audio {V,A}', mae: 2.522, latency: 2.0 },
    { subset: 'Vision+Gesture {V,G}', mae: 2.485, latency: 1.9 },
    { subset: 'Audio+Gesture {A,G}', mae: 2.473, latency: 1.9 },
    { subset: 'All Three {V,A,G}', mae: 2.497, latency: 2.29 },
    { subset: 'MLP Baseline', mae: 2.393, latency: 0.1 },
  ];

  const noiseData = [
    { snr: 'Clean (Inf dB)', accuracy: 89.75 },
    { snr: '+20 dB SNR', accuracy: 90.00 },
    { snr: '+10 dB SNR', accuracy: 90.75 },
    { snr: '0 dB SNR', accuracy: 89.50 },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Model Evaluation & Quantitative Benchmarks
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-blue-500/10 text-blue-400 border border-blue-500/30">
              Verified Reports
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Empirical benchmark metrics evaluated across standardized datasets (COCO128, ESC-50, multi-subject gesture sequences, and paired recording sessions).
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center space-x-1 p-1 rounded-lg bg-card border border-cardBorder text-xs font-mono">
          {(['overview', 'vision', 'audio', 'gesture', 'fusion'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 rounded-md capitalize transition-all ${
                activeTab === tab
                  ? 'bg-surface text-brand-cyan font-bold border border-cardBorder shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Main Benchmarks Content */}
      <div className="space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Vision mAP@0.5</span>
              <span className="text-brand-cyan font-bold">COCO128</span>
            </div>
            <div className="text-2xl font-bold text-white font-mono">60.52%</div>
            <div className="text-[11px] text-slate-400 font-mono">Precision: 63.95% | Recall: 53.68%</div>
          </div>

          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Audio Accuracy</span>
              <span className="text-brand-amber font-bold">ESC-50 Fold 5</span>
            </div>
            <div className="text-2xl font-bold text-white font-mono">89.75%</div>
            <div className="text-[11px] text-slate-400 font-mono">Downstream MLP (LogReg: 91.00%)</div>
          </div>

          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Gesture Test Acc</span>
              <span className="text-brand-emerald font-bold">Held-Out</span>
            </div>
            <div className="text-2xl font-bold text-white font-mono">100.0%</div>
            <div className="text-[11px] text-slate-400 font-mono">32 sequences | Val Loss: 0.0003</div>
          </div>

          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-2">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400">
              <span>Transformer MAE</span>
              <span className="text-brand-purple font-bold">Ablation Split</span>
            </div>
            <div className="text-2xl font-bold text-white font-mono">0.4172</div>
            <div className="text-[11px] text-slate-400 font-mono">Validation MAE (Test MAE: 2.497)</div>
          </div>
        </div>

        {/* Detailed Benchmark Tables */}
        {(activeTab === 'overview' || activeTab === 'fusion') && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Fusion Ablation Chart */}
            <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs font-semibold text-white">
                  <BarChart3 className="w-4 h-4 text-brand-purple" />
                  <span>Transformer Modality Ablation Matrix</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400">reports/fusion-ablations.json</span>
              </div>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={ablationsData} margin={{ left: -10, right: 10, top: 10, bottom: 20 }}>
                    <XAxis dataKey="subset" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-25} textAnchor="end" />
                    <YAxis domain={[0, 3]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#101522', borderColor: '#1e2638', fontSize: 11, borderRadius: 8 }}
                      formatter={(val: any) => [`${val}`, 'Priority MAE']}
                    />
                    <Bar dataKey="mae" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Audio Noise Robustness Chart */}
            <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs font-semibold text-white">
                  <TrendingUp className="w-4 h-4 text-brand-amber" />
                  <span>Audio Classifier Noise Degradation Curve</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400">reports/audio-downstream-eval.json</span>
              </div>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={noiseData} margin={{ left: -10, right: 10, top: 10, bottom: 20 }}>
                    <XAxis dataKey="snr" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis domain={[80, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#101522', borderColor: '#1e2638', fontSize: 11, borderRadius: 8 }}
                      formatter={(val: any) => [`${val}%`, 'Accuracy']}
                    />
                    <Bar dataKey="accuracy" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Latency Comparison Table */}
        <div className="rounded-xl border border-cardBorder bg-card overflow-hidden">
          <div className="p-4 border-b border-cardBorder bg-surface/50 flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs font-semibold text-white">
              <Cpu className="w-4 h-4 text-brand-cyan" />
              <span>Empirical Execution Latency Profile (Intel CPU)</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">session-multimodal-01 Replay Benchmark</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead className="bg-surface border-b border-cardBorder text-slate-400 text-left">
                <tr>
                  <th className="p-3">Branch / Pipeline Stage</th>
                  <th className="p-3">Model Architecture</th>
                  <th className="p-3">Cold Start (ms)</th>
                  <th className="p-3">Warmed Mean Latency</th>
                  <th className="p-3">Warmed p95 Latency</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-cardBorder/60 text-slate-200">
                <tr className="hover:bg-surface/20">
                  <td className="p-3 font-semibold text-brand-cyan">Computer Vision</td>
                  <td className="p-3 text-slate-400">YOLOv8n (640x640)</td>
                  <td className="p-3 text-slate-400">2,211.60 ms</td>
                  <td className="p-3 font-bold text-white">53.69 ms</td>
                  <td className="p-3">56.35 ms</td>
                  <td className="p-3 text-emerald-400 font-bold">READY</td>
                </tr>
                <tr className="hover:bg-surface/20">
                  <td className="p-3 font-semibold text-brand-amber">Environmental Audio</td>
                  <td className="p-3 text-slate-400">PANNs CNN14 + AudioMLP</td>
                  <td className="p-3 text-slate-400">34.24 ms</td>
                  <td className="p-3 font-bold text-white">29.31 ms</td>
                  <td className="p-3">30.24 ms</td>
                  <td className="p-3 text-emerald-400 font-bold">READY</td>
                </tr>
                <tr className="hover:bg-surface/20">
                  <td className="p-3 font-semibold text-brand-emerald">Gesture Kinematics</td>
                  <td className="p-3 text-slate-400">MediaPipe + GestureLSTM</td>
                  <td className="p-3 text-slate-400">2.19 ms</td>
                  <td className="p-3 font-bold text-white">1.01 ms</td>
                  <td className="p-3">1.45 ms</td>
                  <td className="p-3 text-emerald-400 font-bold">READY</td>
                </tr>
                <tr className="hover:bg-surface/20">
                  <td className="p-3 font-semibold text-brand-purple">Transformer Fusion & Decision</td>
                  <td className="p-3 text-slate-400">2-Layer 4-Head Encoder</td>
                  <td className="p-3 text-slate-400">3.12 ms</td>
                  <td className="p-3 font-bold text-white">2.29 ms</td>
                  <td className="p-3">2.93 ms</td>
                  <td className="p-3 text-emerald-400 font-bold">READY</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
