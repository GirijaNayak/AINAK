import React, { useState, useRef, useEffect } from 'react';
import {
  Volume2,
  Upload,
  Play,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Mic,
  VolumeX,
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
import { AudioInferenceResult } from '../../types';
import { classifyAudioClip } from '../../services/api';

export const AudioView: React.FC = () => {
  const [audioResult, setAudioResult] = useState<AudioInferenceResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentSoundName, setCurrentSoundName] = useState<string>('Car Horn');
  const [isPlaying, setIsPlaying] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Sound preset synthesis (pure PCM WAV generators for offline demonstration)
  const presets = [
    { id: 'horn', name: 'Car Horn (Hazard)', freq: 440, duration: 1.5, type: 'hazard' },
    { id: 'siren', name: 'Emergency Siren', freq: 880, duration: 2.0, type: 'hazard' },
    { id: 'dog', name: 'Dog Bark', freq: 320, duration: 1.0, type: 'caution' },
    { id: 'silence', name: 'Ambient Silence', freq: 0, duration: 1.5, type: 'silence' },
  ];

  useEffect(() => {
    // Generate initial waveform visual
    drawWaveform();
  }, []);

  const createSyntheticWavBlob = (freq: number, duration: number): Blob => {
    const sampleRate = 32000;
    const numSamples = Math.floor(sampleRate * duration);
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    // RIFF identifier
    view.setUint32(0, 0x52494646, false); // "RIFF"
    view.setUint32(4, 36 + numSamples * 2, true);
    view.setUint32(8, 0x57415645, false); // "WAVE"
    // fmt subchunk
    view.setUint32(12, 0x666d7420, false); // "fmt "
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true); // NumChannels (1 = Mono)
    view.setUint32(24, sampleRate, true); // SampleRate (32 kHz)
    view.setUint32(28, sampleRate * 2, true); // ByteRate
    view.setUint16(32, 2, true); // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample
    // data subchunk
    view.setUint32(36, 0x64617461, false); // "data"
    view.setUint32(40, numSamples * 2, true);

    // Write samples
    for (let i = 0; i < numSamples; i++) {
      let sample = 0;
      if (freq > 0) {
        // Dual-tone harmonic with modulation for realism
        const t = i / sampleRate;
        sample = Math.sin(2 * Math.PI * freq * t) * 0.6 + Math.sin(2 * Math.PI * (freq * 1.5) * t) * 0.3;
        // Envelope
        const envelope = Math.min(1.0, t * 10) * Math.max(0.0, 1.0 - t / duration);
        sample *= envelope;
      }
      view.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, sample * 32767)), true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
  };

  const drawWaveform = (data?: Float32Array) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#1e2638';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Waveform stroke
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const points = 120;
    for (let i = 0; i < points; i++) {
      const x = (i / (points - 1)) * w;
      let amp = 0;
      if (data && data.length > 0) {
        const idx = Math.floor((i / points) * data.length);
        amp = data[idx] * (h / 2) * 0.8;
      } else {
        amp = Math.sin(i * 0.2) * Math.cos(i * 0.05) * (h / 3);
      }
      const y = h / 2 + amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  const handleTestPreset = async (preset: typeof presets[0]) => {
    setCurrentSoundName(preset.name);
    setIsLoading(true);
    setError(null);

    try {
      const blob = createSyntheticWavBlob(preset.freq, preset.duration);
      // Play local audio preview
      playAudioBlob(blob);

      const result = await classifyAudioClip(blob, Date.now() / 1000);
      setAudioResult(result);
    } catch (err: any) {
      setError(err.message || 'Audio classification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCurrentSoundName(file.name);
    setIsLoading(true);
    setError(null);

    try {
      playAudioBlob(file);
      const result = await classifyAudioClip(file, Date.now() / 1000);
      setAudioResult(result);
    } catch (err: any) {
      setError(err.message || 'Custom audio classification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const playAudioBlob = (blob: Blob) => {
    try {
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      setIsPlaying(true);
      audio.onended = () => setIsPlaying(false);
      audio.play().catch(() => setIsPlaying(false));
    } catch (e) {
      setIsPlaying(false);
    }
  };

  const chartData = audioResult?.top_classes?.map((item) => ({
    name: item.class_name.replace(/_/g, ' '),
    probability: Math.round(item.probability * 100),
  })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Branch 2: Environmental Audio Intelligence
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-brand-amber/10 text-brand-amber border border-brand-amber/30">
              PANNs CNN14 + AudioMLP
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            50-class ESC-50 acoustic classification, 2,048-D bottleneck representation, proactive silence energy rejection, and noise robustness.
          </p>
        </div>

        {audioResult && (
          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Latency: <span className="text-brand-amber font-bold">{audioResult.latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              RMS Energy: <span className="text-slate-200">{audioResult.rms_energy.toFixed(4)}</span>
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
        {/* Waveform & Presets */}
        <div className="lg:col-span-2 space-y-4">
          {/* Waveform Visualizer */}
          <div className="p-5 rounded-xl border border-cardBorder bg-card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-semibold text-white">
                <Activity className="w-4 h-4 text-brand-amber" />
                <span>Audio Waveform Monitor (32 kHz Monaural)</span>
              </div>
              <span className="text-xs font-mono text-slate-400">{currentSoundName}</span>
            </div>

            <div className="relative h-32 rounded-lg bg-background border border-cardBorder overflow-hidden flex items-center justify-center">
              <canvas ref={canvasRef} width={600} height={128} className="w-full h-full" />
              {isPlaying && (
                <div className="absolute top-2 right-2 flex items-center space-x-1.5 px-2 py-0.5 rounded bg-brand-amber/20 border border-brand-amber/40 text-[10px] font-mono text-brand-amber">
                  <span className="w-1.5 h-1.5 rounded-full bg-brand-amber animate-ping" />
                  <span>PLAYING</span>
                </div>
              )}
            </div>

            {/* Quick Test Presets */}
            <div className="space-y-2">
              <label className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
                Evaluation Audio Presets
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {presets.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleTestPreset(p)}
                    disabled={isLoading}
                    className="p-3 rounded-lg bg-surface hover:bg-surfaceHover border border-cardBorder text-left transition-all hover:border-brand-amber/40 flex flex-col justify-between"
                  >
                    <div className="text-xs font-semibold text-slate-200">{p.name}</div>
                    <div className="text-[10px] font-mono text-slate-400 mt-2 flex items-center justify-between">
                      <span>{p.duration}s WAV</span>
                      <Play className="w-3 h-3 text-brand-amber" />
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Custom WAV Upload */}
            <div className="pt-2 border-t border-cardBorder flex items-center justify-between">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/wav,audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-medium bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-300"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Upload External WAV File</span>
              </button>

              <span className="text-[11px] text-slate-400 font-mono">
                Standard 32 kHz PCM supported
              </span>
            </div>
          </div>
        </div>

        {/* Top Probabilities & Telemetry */}
        <div className="space-y-4">
          {/* Status Metric */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Predicted Class</span>
              {audioResult?.is_silent ? (
                <span className="px-2 py-0.5 rounded font-bold uppercase text-[10px] bg-slate-500/20 text-slate-300 border border-slate-500/30 flex items-center space-x-1">
                  <VolumeX className="w-3 h-3" />
                  <span>Silence / Ambient</span>
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded font-bold uppercase text-[10px] bg-brand-amber/20 text-brand-amber border border-brand-amber/30">
                  {audioResult?.predicted_class || 'Awaiting Signal'}
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-surface border border-cardBorder">
                <div className="text-[10px] text-slate-400">CONFIDENCE</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {((audioResult?.probability || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-surface border border-cardBorder">
                <div className="text-[10px] text-slate-400">EMBEDDING</div>
                <div className="text-base font-bold text-brand-amber mt-0.5">2,048-D</div>
              </div>
            </div>
          </div>

          {/* Probability Distribution Chart */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-semibold text-white">
              <span>Top Acoustic Predictions</span>
              <span className="text-[10px] font-mono text-slate-400">ESC-50</span>
            </div>

            {chartData.length > 0 ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                    <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <YAxis dataKey="name" type="category" width={90} tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#101522', borderColor: '#1e2638', fontSize: 11, borderRadius: 8 }}
                      formatter={(val: any) => [`${val}%`, 'Probability']}
                    />
                    <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                      {chartData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? '#f59e0b' : '#334155'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-56 flex flex-col items-center justify-center text-xs text-slate-400 font-mono">
                Click a preset sound above to test classification.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
