import React, { useState } from 'react';
import {
  Layers,
  Cpu,
  Play,
  CheckCircle,
  AlertTriangle,
  ToggleLeft,
  ToggleRight,
  ShieldAlert,
  Gauge,
  Sparkles,
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
import { FusionInferenceResult } from '../../types';
import { runFusionInference } from '../../services/api';

export const FusionView: React.FC = () => {
  const [fusionResult, setFusionResult] = useState<FusionInferenceResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Active Modalities Mask (simulating missing modalities)
  const [useVision, setUseVision] = useState(true);
  const [useAudio, setUseAudio] = useState(true);
  const [useGesture, setUseGesture] = useState(true);

  // Custom simulation inputs
  const [hazardLevel, setHazardLevel] = useState(0.75); // Vision hazard
  const [soundClass, setSoundClass] = useState<'horn' | 'siren' | 'dog' | 'ambient'>('horn');
  const [gestureCommand, setGestureCommand] = useState<'stop' | 'pointing' | 'neutral'>('stop');

  const executeFusion = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Synthesize realistic feature vectors matching the selected scenario
      let vFeat: number[] | null = null;
      let aFeat: number[] | null = null;
      let gFeat: number[] | null = null;

      if (useVision) {
        vFeat = Array.from({ length: 128 }, (_, i) =>
          Math.sin(i * 0.1) * 0.5 + (hazardLevel > 0.5 ? 0.4 : 0.05)
        );
      }
      if (useAudio) {
        const audioScale = soundClass === 'horn' || soundClass === 'siren' ? 0.6 : 0.1;
        aFeat = Array.from({ length: 2048 }, (_, i) => Math.cos(i * 0.05) * audioScale);
      }
      if (useGesture) {
        const gestureBias = gestureCommand === 'stop' ? 0.7 : gestureCommand === 'pointing' ? 0.3 : 0.0;
        gFeat = Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.2) * 0.3 + gestureBias);
      }

      const result = await runFusionInference({
        vision_feature: vFeat,
        audio_feature: aFeat,
        gesture_feature: gFeat,
        timestamp: Date.now() / 1000,
      });

      setFusionResult(result);
    } catch (err: any) {
      setError(err.message || 'Transformer fusion forward pass failed');
    } finally {
      setIsLoading(false);
    }
  };

  const alertLogitsData = fusionResult?.interpreted_prediction?.alert_probabilities
    ? Object.entries(fusionResult.interpreted_prediction.alert_probabilities).map(([name, prob]) => ({
        name: name.replace(/_/g, ' ').toUpperCase(),
        prob: Math.round(prob * 100),
      }))
    : [
        { name: 'IMMEDIATE ATTENTION', prob: 78 },
        { name: 'CAUTION', prob: 14 },
        { name: 'GESTURE FEEDBACK', prob: 5 },
        { name: 'INFORMATIONAL', prob: 2 },
        { name: 'UNCERTAIN', prob: 1 },
      ];

  const priorityScore = fusionResult
    ? Math.round((fusionResult.raw_output.priority_scalar || 0) * 100)
    : 78;

  const confidenceScore = fusionResult
    ? Math.round((fusionResult.raw_output.confidence_scalar || 0) * 100)
    : 84;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Branch 4: Multimodal Transformer Attention Fusion
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-brand-purple/10 text-brand-purple border border-brand-purple/30">
              2-Layer 4-Head Encoder
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Learned modality projections (128-D), multi-head cross-modal attention, modality masking, and calibrated multi-task regression & classification heads.
          </p>
        </div>

        {fusionResult && (
          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Latency: <span className="text-brand-purple font-bold">{fusionResult.latency_ms.toFixed(2)} ms</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Contributing: <span className="text-slate-200">{fusionResult.contributing_modalities.join(', ') || 'None'}</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Modality Mask Controls & Simulation */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                Sensor Channel Inputs
              </span>
              <span className="text-[10px] text-slate-400 font-mono">Masking Supported</span>
            </div>

            {/* Vision Switch */}
            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200">
                  <span className="w-2 h-2 rounded-full bg-brand-cyan" />
                  <span>Vision Branch (128-D)</span>
                </div>
                <button onClick={() => setUseVision(!useVision)} className="text-slate-400 hover:text-white">
                  {useVision ? (
                    <ToggleRight className="w-6 h-6 text-brand-cyan" />
                  ) : (
                    <ToggleLeft className="w-6 h-6 text-slate-600" />
                  )}
                </button>
              </div>

              {useVision && (
                <div className="pt-2 text-xs font-mono space-y-1">
                  <div className="flex justify-between text-slate-400 text-[10px]">
                    <span>Obstacle Hazard: {(hazardLevel * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={hazardLevel}
                    onChange={(e) => setHazardLevel(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-brand-cyan"
                  />
                </div>
              )}
            </div>

            {/* Audio Switch */}
            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200">
                  <span className="w-2 h-2 rounded-full bg-brand-amber" />
                  <span>Audio Branch (2048-D)</span>
                </div>
                <button onClick={() => setUseAudio(!useAudio)} className="text-slate-400 hover:text-white">
                  {useAudio ? (
                    <ToggleRight className="w-6 h-6 text-brand-amber" />
                  ) : (
                    <ToggleLeft className="w-6 h-6 text-slate-600" />
                  )}
                </button>
              </div>

              {useAudio && (
                <div className="pt-2 flex flex-wrap gap-1 text-[11px] font-mono">
                  {(['horn', 'siren', 'dog', 'ambient'] as const).map((sc) => (
                    <button
                      key={sc}
                      onClick={() => setSoundClass(sc)}
                      className={`px-2 py-0.5 rounded border transition-all ${
                        soundClass === sc
                          ? 'bg-brand-amber/20 text-brand-amber border-brand-amber/40 font-bold'
                          : 'bg-card border-cardBorder text-slate-400'
                      }`}
                    >
                      {sc.toUpperCase()}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Gesture Switch */}
            <div className="p-3 rounded-lg bg-surface border border-cardBorder space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200">
                  <span className="w-2 h-2 rounded-full bg-brand-emerald" />
                  <span>Gesture Branch (128-D)</span>
                </div>
                <button onClick={() => setUseGesture(!useGesture)} className="text-slate-400 hover:text-white">
                  {useGesture ? (
                    <ToggleRight className="w-6 h-6 text-brand-emerald" />
                  ) : (
                    <ToggleLeft className="w-6 h-6 text-slate-600" />
                  )}
                </button>
              </div>

              {useGesture && (
                <div className="pt-2 flex flex-wrap gap-1 text-[11px] font-mono">
                  {(['stop', 'pointing', 'neutral'] as const).map((gc) => (
                    <button
                      key={gc}
                      onClick={() => setGestureCommand(gc)}
                      className={`px-2 py-0.5 rounded border transition-all ${
                        gestureCommand === gc
                          ? 'bg-brand-emerald/20 text-brand-emerald border-brand-emerald/40 font-bold'
                          : 'bg-card border-cardBorder text-slate-400'
                      }`}
                    >
                      {gc.toUpperCase()}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={executeFusion}
              disabled={isLoading || (!useVision && !useAudio && !useGesture)}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg text-xs font-semibold bg-brand-purple text-white hover:bg-brand-purple/90 disabled:opacity-50 transition-all shadow-glow-purple"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Forward Pass Transformer</span>
            </button>
          </div>
        </div>

        {/* Transformer Cross-Attention Architecture Details */}
        <div className="lg:col-span-2 space-y-4">
          {/* Gauges & Predictions */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Priority Score Dial */}
            <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-3">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Hazard Priority Regression</span>
                <span className="text-brand-purple font-bold text-sm">{priorityScore} / 100</span>
              </div>

              <div className="w-full bg-surface rounded-full h-3 overflow-hidden border border-cardBorder">
                <div
                  className={`h-full transition-all duration-500 rounded-full ${
                    priorityScore >= 70
                      ? 'bg-rose-500 shadow-glow-rose'
                      : priorityScore >= 40
                      ? 'bg-brand-amber'
                      : 'bg-brand-cyan'
                  }`}
                  style={{ width: `${priorityScore}%` }}
                />
              </div>

              <div className="text-[11px] text-slate-400 font-mono flex items-center justify-between">
                <span>0 (Safe Ambient)</span>
                <span>100 (Immediate Hazard)</span>
              </div>
            </div>

            {/* Confidence Score Dial */}
            <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-3">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Cross-Modal Confidence</span>
                <span className="text-emerald-400 font-bold text-sm">{confidenceScore}%</span>
              </div>

              <div className="w-full bg-surface rounded-full h-3 overflow-hidden border border-cardBorder">
                <div
                  className="h-full bg-emerald-400 transition-all duration-500 rounded-full"
                  style={{ width: `${confidenceScore}%` }}
                />
              </div>

              <div className="text-[11px] text-slate-400 font-mono flex items-center justify-between">
                <span>Calibrated Sigmoid Head</span>
                <span>Threshold: 60%</span>
              </div>
            </div>
          </div>

          {/* Alert Class Distribution */}
          <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-semibold text-white">
              <span>Transformer Alert Classification Head</span>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-brand-purple/20 text-brand-purple border border-brand-purple/30">
                {fusionResult?.interpreted_prediction?.alert_type?.toUpperCase() || 'IMMEDIATE ATTENTION'}
              </span>
            </div>

            <div className="h-44">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={alertLogitsData} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" width={140} tick={{ fill: '#cbd5e1', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#101522', borderColor: '#1e2638', fontSize: 11, borderRadius: 8 }}
                    formatter={(val: any) => [`${val}%`, 'Logit Softmax']}
                  />
                  <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                    {alertLogitsData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={index === 0 ? '#8b5cf6' : '#334155'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
