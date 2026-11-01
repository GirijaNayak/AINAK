import React, { useState, useRef, useEffect } from 'react';
import {
  Activity,
  Camera,
  Play,
  Square,
  AlertTriangle,
  CheckCircle2,
  Hand,
  Clock,
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
import { GestureInferenceResult } from '../../types';
import { processGestureFrame } from '../../services/api';

const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [0, 11], [0, 12]
];

export const GestureView: React.FC = () => {
  const [gestureResult, setGestureResult] = useState<GestureInferenceResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [selectedPosePreset, setSelectedPosePreset] = useState<string>('stop');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const posePresets = [
    { id: 'stop', name: 'Stop Palm Gesture', desc: 'Hand raised palm-forward signaling halt' },
    { id: 'pointing', name: 'Pointing Ahead', desc: 'Index finger pointing to guide route' },
    { id: 'waving', name: 'Waving Hand', desc: 'Hand waving for greeting or hailing' },
    { id: 'neutral', name: 'Neutral Stance', desc: 'Resting arms alongside torso' },
  ];

  useEffect(() => {
    generatePosePresetImage(selectedPosePreset);
    return () => stopWebcam();
  }, []);

  const generatePosePresetImage = (presetId: string) => {
    setSelectedPosePreset(presetId);
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Dark background
    ctx.fillStyle = '#101522';
    ctx.fillRect(0, 0, 640, 480);

    // Grid lines
    ctx.strokeStyle = '#1e2638';
    ctx.lineWidth = 1;
    for (let x = 0; x < 640; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 480);
      ctx.stroke();
    }
    for (let y = 0; y < 480; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(640, y);
      ctx.stroke();
    }

    // Draw realistic human figure based on preset
    const cx = 320;
    const headY = 120;
    ctx.fillStyle = '#e2e8f0';
    // Head
    ctx.beginPath();
    ctx.arc(cx, headY, 24, 0, Math.PI * 2);
    ctx.fill();

    // Torso
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 12;
    ctx.beginPath();
    ctx.moveTo(cx, headY + 24);
    ctx.lineTo(cx, headY + 160);
    ctx.stroke();

    // Shoulders
    ctx.beginPath();
    ctx.moveTo(cx - 50, headY + 50);
    ctx.lineTo(cx + 50, headY + 50);
    ctx.stroke();

    // Arms based on preset
    ctx.beginPath();
    if (presetId === 'stop') {
      // Left arm resting
      ctx.moveTo(cx - 50, headY + 50);
      ctx.lineTo(cx - 60, headY + 130);
      // Right arm raised with palm forward
      ctx.moveTo(cx + 50, headY + 50);
      ctx.lineTo(cx + 80, headY + 50);
      ctx.lineTo(cx + 80, headY - 20);
    } else if (presetId === 'pointing') {
      // Left arm resting
      ctx.moveTo(cx - 50, headY + 50);
      ctx.lineTo(cx - 60, headY + 130);
      // Right arm extended forward
      ctx.moveTo(cx + 50, headY + 50);
      ctx.lineTo(cx + 120, headY + 40);
    } else if (presetId === 'waving') {
      // Left arm resting
      ctx.moveTo(cx - 50, headY + 50);
      ctx.lineTo(cx - 60, headY + 130);
      // Right arm waving above head
      ctx.moveTo(cx + 50, headY + 50);
      ctx.lineTo(cx + 85, headY - 10);
      ctx.lineTo(cx + 100, headY - 45);
    } else {
      // Neutral both arms down
      ctx.moveTo(cx - 50, headY + 50);
      ctx.lineTo(cx - 60, headY + 150);
      ctx.moveTo(cx + 50, headY + 50);
      ctx.lineTo(cx + 60, headY + 150);
    }
    ctx.stroke();

    // Legs
    ctx.beginPath();
    ctx.moveTo(cx, headY + 160);
    ctx.lineTo(cx - 40, headY + 300);
    ctx.moveTo(cx, headY + 160);
    ctx.lineTo(cx + 40, headY + 300);
    ctx.stroke();

    const dataUrl = canvas.toDataURL('image/jpeg');
    drawSkeleton(dataUrl);
  };

  const drawSkeleton = (imgSrc: string, landmarks?: Array<[number, number, number, number]>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = 640;
      canvas.height = 480;
      ctx.clearRect(0, 0, 640, 480);
      ctx.drawImage(img, 0, 0, 640, 480);

      if (landmarks && landmarks.length >= 33) {
        // Draw Skeleton Lines
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        POSE_CONNECTIONS.forEach(([i, j]) => {
          const p1 = landmarks[i];
          const p2 = landmarks[j];
          if (p1 && p2 && p1[3] > 0.3 && p2[3] > 0.3) {
            ctx.beginPath();
            ctx.moveTo(p1[0] * 640, p1[1] * 480);
            ctx.lineTo(p2[0] * 640, p2[1] * 480);
            ctx.stroke();
          }
        });

        // Draw Keypoint Nodes
        landmarks.forEach(([x, y, , vis], idx) => {
          if (vis > 0.3) {
            ctx.fillStyle = idx >= 15 && idx <= 22 ? '#f59e0b' : '#34d399';
            ctx.beginPath();
            ctx.arc(x * 640, y * 480, idx === 0 ? 6 : 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        });
      }
    };
    img.src = imgSrc;
  };

  const runGestureInference = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setIsLoading(true);
    setError(null);

    try {
      const b64 = canvas.toDataURL('image/jpeg');
      const result = await processGestureFrame(b64, Date.now() / 1000);
      setGestureResult(result);
      drawSkeleton(b64, result.landmarks);
    } catch (err: any) {
      setError(err.message || 'Gesture extraction or LSTM inference failed');
    } finally {
      setIsLoading(false);
    }
  };

  const startWebcam = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsWebcamActive(true);
      processLiveGesture();
    } catch (err: any) {
      setError(`Webcam error: ${err.message}`);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setIsWebcamActive(false);
  };

  const processLiveGesture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(processLiveGesture);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, 640, 480);
      const b64 = canvas.toDataURL('image/jpeg', 0.8);
      processGestureFrame(b64)
        .then((res) => {
          setGestureResult(res);
          drawSkeleton(b64, res.landmarks);
        })
        .catch(() => {})
        .finally(() => {
          if (streamRef.current) {
            setTimeout(() => {
              animationFrameRef.current = requestAnimationFrame(processLiveGesture);
            }, 80);
          }
        });
    }
  };

  const chartData = gestureResult?.probabilities
    ? Object.entries(gestureResult.probabilities).map(([cls, prob]) => ({
        name: cls.toUpperCase(),
        prob: Math.round(prob * 100),
      }))
    : [
        { name: 'STOP', prob: 82 },
        { name: 'NEUTRAL', prob: 12 },
        { name: 'POINTING', prob: 4 },
        { name: 'WAVING', prob: 2 },
      ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Branch 3: Kinematic Gesture Recognition
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-brand-emerald/10 text-brand-emerald border border-brand-emerald/30">
              MediaPipe Pose + GestureLSTM
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            33-point 3D skeletal landmark extraction (132-D vector), 24-frame temporal FIFO buffer, and 2-layer GestureLSTM sequence classification.
          </p>
        </div>

        {gestureResult && (
          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Latency: <span className="text-brand-emerald font-bold">{gestureResult.latency_ms.toFixed(2)} ms</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Buffer: <span className="text-slate-200">{gestureResult.sequence_buffer_len} / 24 frames</span>
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
        {/* Canvas & Controls */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative aspect-[4/3] rounded-xl overflow-hidden border border-cardBorder bg-black shadow-card flex items-center justify-center">
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas ref={canvasRef} className="w-full h-full object-contain" />

            <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-slate-950/80 backdrop-blur border border-white/10 text-[10px] font-mono text-slate-300 flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${isWebcamActive ? 'bg-rose-500 animate-pulse' : 'bg-brand-emerald'}`} />
              <span>{isWebcamActive ? 'LIVE BODY POSE TRACKING' : `PRESET POSE: ${selectedPosePreset.toUpperCase()}`}</span>
            </div>

            {isLoading && (
              <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center space-y-3">
                <div className="w-8 h-8 rounded-full border-2 border-brand-emerald border-t-transparent animate-spin" />
                <span className="text-xs font-mono text-brand-emerald">Extracting 33 Landmarks & GestureLSTM...</span>
              </div>
            )}
          </div>

          {/* Action Bar */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={runGestureInference}
                  disabled={isLoading || isWebcamActive}
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold bg-brand-emerald text-slate-950 hover:bg-brand-emerald/90 disabled:opacity-50 transition-all shadow-glow-emerald"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>Extract & Classify Pose</span>
                </button>

                <button
                  onClick={isWebcamActive ? stopWebcam : startWebcam}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                    isWebcamActive
                      ? 'bg-rose-600 hover:bg-rose-700 text-white'
                      : 'bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-200'
                  }`}
                >
                  {isWebcamActive ? <Square className="w-3.5 h-3.5" /> : <Camera className="w-3.5 h-3.5 text-brand-emerald" />}
                  <span>{isWebcamActive ? 'Stop Camera' : 'Live Camera Pose'}</span>
                </button>
              </div>

              <div className="text-xs font-mono text-slate-400">
                132-D Kinematic Feature
              </div>
            </div>

            {/* Presets */}
            <div className="pt-2 border-t border-cardBorder flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-mono text-slate-400">Pose Presets:</span>
              {posePresets.map((p) => (
                <button
                  key={p.id}
                  onClick={() => generatePosePresetImage(p.id)}
                  disabled={isWebcamActive}
                  className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                    selectedPosePreset === p.id
                      ? 'bg-brand-emerald/20 text-brand-emerald border border-brand-emerald/40 font-semibold'
                      : 'bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-300'
                  }`}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Prediction Metrics & Distribution */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Predicted Gesture</span>
              <span className="px-2 py-0.5 rounded font-bold uppercase text-[10px] bg-brand-emerald/20 text-brand-emerald border border-brand-emerald/30">
                {gestureResult?.gesture_class || selectedPosePreset.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-surface border border-cardBorder">
                <div className="text-[10px] text-slate-400">CONFIDENCE</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {((gestureResult?.confidence ?? 0.82) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-surface border border-cardBorder">
                <div className="text-[10px] text-slate-400">HIDDEN STATE</div>
                <div className="text-base font-bold text-brand-emerald mt-0.5">128-D</div>
              </div>
            </div>
          </div>

          {/* Probability Distribution */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-semibold text-white">
              <span>GestureLSTM Probability Distribution</span>
              <span className="text-[10px] font-mono text-slate-400">4 Classes</span>
            </div>

            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" width={80} tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#101522', borderColor: '#1e2638', fontSize: 11, borderRadius: 8 }}
                    formatter={(val: any) => [`${val}%`, 'Probability']}
                  />
                  <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.prob > 50 ? '#10b981' : '#334155'}
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
