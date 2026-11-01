import React, { useState } from 'react';
import {
  ShieldAlert,
  Volume2,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  Sliders,
  Send,
} from 'lucide-react';
import { triggerSpeechTest } from '../../services/api';

export const DecisionView: React.FC = () => {
  const [speechText, setSpeechText] = useState(
    'Warning: Approaching car horn detected directly ahead. Please stop.'
  );
  const [speechPriority, setSpeechPriority] = useState(90);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastDispatched, setLastDispatched] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const urgencyBands = [
    {
      rank: 1,
      name: 'immediate_attention',
      label: 'Immediate Attention',
      threshold: '≥ 0.60',
      cooldown: '2.0s',
      preempt: 'YES (Halts active utterance)',
      color: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
      scenario: 'Approaching vehicle horn within 2 meters, high optical loom hazard.',
    },
    {
      rank: 2,
      name: 'caution',
      label: 'Caution',
      threshold: '≥ 0.50',
      cooldown: '5.0s',
      preempt: 'NO (FIFO Queue)',
      color: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
      scenario: 'Stationary pedestrian or obstacle crossing in the walking corridor.',
    },
    {
      rank: 3,
      name: 'gesture_feedback',
      label: 'Gesture Feedback',
      threshold: '≥ 0.75',
      cooldown: '3.0s',
      preempt: 'NO (FIFO Queue)',
      color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
      scenario: 'Active user gesture detected (e.g. raised palm confirming STOP).',
    },
    {
      rank: 4,
      name: 'informational',
      label: 'Informational',
      threshold: '≥ 0.50',
      cooldown: '10.0s',
      preempt: 'NO (FIFO Queue)',
      color: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
      scenario: 'Off-path landmark detected (bench, door entrance, or signage).',
    },
    {
      rank: 5,
      name: 'uncertain',
      label: 'Uncertain',
      threshold: '≥ 0.30',
      cooldown: '15.0s',
      preempt: 'NO (Lowest Priority)',
      color: 'bg-slate-500/10 text-slate-400 border-slate-500/30',
      scenario: 'Ambiguous sensory vectors requiring user manual confirmation.',
    },
  ];

  const handleTestSpeech = async () => {
    if (!speechText.trim()) return;
    setIsSpeaking(true);
    setError(null);

    try {
      const res = await triggerSpeechTest(speechText, speechPriority);
      setLastDispatched(res);

      // Web Speech API client mirror for immediate browser feedback
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.rate = 1.1;
        window.speechSynthesis.speak(utterance);
      }
    } catch (err: any) {
      setError(err.message || 'Speech dispatch failed');
    } finally {
      setIsSpeaking(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Branch 5: Deterministic Decision Engine & Prioritized Speech
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-brand-rose/10 text-brand-rose border border-brand-rose/30">
              Rule-Based Arbitration
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Separation of neural representation learning from safety policy enforcement: 5 discrete urgency ranks, refractory timers, and preemption speech queue.
          </p>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Policy Urgency Table */}
        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-xl border border-cardBorder bg-card overflow-hidden">
            <div className="p-4 border-b border-cardBorder bg-surface/50 flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-semibold text-white">
                <ShieldAlert className="w-4 h-4 text-brand-rose" />
                <span>Deterministic Alert Policy Hierarchy</span>
              </div>
              <span className="text-[10px] font-mono text-slate-400">configs/alerts.yaml</span>
            </div>

            <div className="divide-y divide-cardBorder text-xs font-mono">
              {urgencyBands.map((band) => (
                <div key={band.rank} className="p-4 space-y-2 hover:bg-surface/30 transition-colors">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center space-x-2.5">
                      <span className="w-5 h-5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 flex items-center justify-center font-bold text-[10px]">
                        {band.rank}
                      </span>
                      <span className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] border ${band.color}`}>
                        {band.name}
                      </span>
                    </div>

                    <div className="flex items-center space-x-4 text-[11px] text-slate-400">
                      <span>Cooldown: <strong className="text-white">{band.cooldown}</strong></span>
                      <span>Min Conf: <strong className="text-white">{band.threshold}</strong></span>
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-300 font-sans">{band.scenario}</p>
                  <div className="text-[10px] text-slate-400">
                    Preemption: <span className={band.rank === 1 ? 'text-rose-400 font-bold' : 'text-slate-400'}>{band.preempt}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Interactive Speech Dispatcher */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-semibold text-white">
                <Volume2 className="w-4 h-4 text-brand-rose" />
                <span>Local TTS Dispatcher</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400">pyttsx3 / SAPI</span>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-[11px] font-mono text-slate-400 mb-1 block">
                  Spoken Alert Utterance
                </label>
                <textarea
                  rows={3}
                  value={speechText}
                  onChange={(e) => setSpeechText(e.target.value)}
                  className="w-full p-2.5 rounded-lg bg-surface border border-cardBorder text-slate-200 text-xs focus:outline-none focus:border-brand-rose resize-none font-sans"
                  placeholder="Enter custom assistive alert text..."
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                  <span className="text-slate-400">Hazard Priority:</span>
                  <span className="text-brand-rose font-bold">{speechPriority} / 100</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={speechPriority}
                  onChange={(e) => setSpeechPriority(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-brand-rose"
                />
              </div>

              <button
                onClick={handleTestSpeech}
                disabled={isSpeaking || !speechText.trim()}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg text-xs font-semibold bg-brand-rose text-white hover:bg-brand-rose/90 disabled:opacity-50 transition-all shadow-glow-rose"
              >
                <Send className="w-3.5 h-3.5" />
                <span>{isSpeaking ? 'Synthesizing...' : 'Dispatch Prioritized Speech'}</span>
              </button>
            </div>

            {lastDispatched && (
              <div className="pt-3 border-t border-cardBorder text-xs font-mono space-y-1 text-slate-400">
                <div className="flex justify-between">
                  <span>Queued:</span>
                  <span className="text-emerald-400 font-bold">{lastDispatched.queued ? 'True' : 'False'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Backend Engine:</span>
                  <span className="text-slate-200">{lastDispatched.backend}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
