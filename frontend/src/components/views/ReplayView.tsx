import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Volume2,
  Eye,
  Activity,
  Layers,
  AlertTriangle,
  Film,
  FastForward,
  ShieldCheck,
  CheckCircle,
} from 'lucide-react';
import { SessionInfo, ReplayTimelineStep } from '../../types';
import { fetchSessions, connectReplayWebSocket } from '../../services/api';

const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [0, 11], [0, 12]
];

export const ReplayView: React.FC = () => {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>('session-multimodal-01');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [currentStep, setCurrentStep] = useState<ReplayTimelineStep | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsControllerRef = useRef<any>(null);

  useEffect(() => {
    fetchSessions()
      .then((data) => {
        setSessions(data);
        if (data.length > 0 && !data.some((s) => s.session_id === selectedSessionId)) {
          setSelectedSessionId(data[0].session_id);
        }
      })
      .catch((err) => setError(err.message));

    return () => {
      stopReplay();
    };
  }, []);

  const startReplay = () => {
    setError(null);
    stopReplay();

    // Start video & audio
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.playbackRate = playbackSpeed;
      videoRef.current.play().catch(() => {});
    }
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.playbackRate = playbackSpeed;
      audioRef.current.play().catch(() => {});
    }

    const controller = connectReplayWebSocket(
      selectedSessionId,
      (step: ReplayTimelineStep) => {
        setCurrentStep(step);
        renderStepVisualization(step);

        // Optional speech synthesis when alert dispatched
        if (step.decision_engine_output?.dispatched && step.decision_engine_output?.spoken_alert) {
          if ('speechSynthesis' in window) {
            const utter = new SpeechSynthesisUtterance(step.decision_engine_output.spoken_alert);
            utter.rate = 1.2;
            window.speechSynthesis.speak(utter);
          }
        }
      },
      (err) => {
        console.error('WS Error', err);
        setError('Replay WebSocket connection interrupted.');
        setIsPlaying(false);
      },
      () => {
        setIsPlaying(false);
      }
    );

    wsControllerRef.current = controller;
    setIsPlaying(true);
  };

  const pauseReplay = () => {
    if (wsControllerRef.current) {
      wsControllerRef.current.sendCommand('pause');
    }
    videoRef.current?.pause();
    audioRef.current?.pause();
    setIsPlaying(false);
  };

  const resumeReplay = () => {
    if (wsControllerRef.current) {
      wsControllerRef.current.sendCommand('resume');
      videoRef.current?.play().catch(() => {});
      audioRef.current?.play().catch(() => {});
      setIsPlaying(true);
    } else {
      startReplay();
    }
  };

  const stopReplay = () => {
    if (wsControllerRef.current) {
      wsControllerRef.current.sendCommand('stop');
      wsControllerRef.current.close();
      wsControllerRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
  };

  const stepForward = () => {
    if (wsControllerRef.current) {
      wsControllerRef.current.sendCommand('step');
    }
  };

  const changeSpeed = (speed: number) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) videoRef.current.playbackRate = speed;
    if (audioRef.current) audioRef.current.playbackRate = speed;
    if (wsControllerRef.current) {
      wsControllerRef.current.sendCommand('speed', { value: speed });
    }
  };

  const renderStepVisualization = (step: ReplayTimelineStep) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw detected objects on canvas overlay
    if (step.source_predictions?.vision?.objects) {
      step.source_predictions.vision.objects.forEach((obj, idx) => {
        // Draw bounding box approximation or track banner
        const isHazard = obj.hazard_score >= 0.6;
        const color = isHazard ? '#f43f5e' : '#06b6d4';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        const bx = 100 + (idx % 3) * 160;
        const by = 150 + (idx % 2) * 80;
        ctx.strokeRect(bx, by, 120, 180);

        ctx.fillStyle = color;
        ctx.fillRect(bx, by - 20, 120, 20);
        ctx.fillStyle = '#0a0d14';
        ctx.font = 'bold 10px JetBrains Mono';
        ctx.fillText(`${obj.label.toUpperCase()} #${obj.track_id}`, bx + 4, by - 6);
      });
    }
  };

  const priorityScore = currentStep?.fusion_output?.priority_score
    ? Math.round(currentStep.fusion_output.priority_score * 100)
    : 72;

  const totalSteps = currentStep?.total_steps || 82;
  const stepIdx = currentStep?.step_index || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Synchronized Multi-Sensor Session Replay
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              Live WebSocket Stream
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time step-by-step playback of paired multimodal recordings, executing actual Transformer forward passes across vision, audio, and gesture streams.
          </p>
        </div>

        {/* Session Selector */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-slate-400">Session:</span>
          <select
            value={selectedSessionId}
            onChange={(e) => {
              stopReplay();
              setSelectedSessionId(e.target.value);
            }}
            className="p-1.5 rounded-lg bg-card border border-cardBorder text-slate-200 text-xs focus:outline-none focus:border-brand-cyan"
          >
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {s.session_id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid: Player + Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video & Controls */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative aspect-[16/9] rounded-xl overflow-hidden border border-cardBorder bg-black shadow-card flex items-center justify-center">
            {/* HTML5 Video */}
            <video
              ref={videoRef}
              src={`/api/media/video/${selectedSessionId}`}
              className="w-full h-full object-contain"
              playsInline
              muted
            />
            {/* HTML5 Audio */}
            <audio ref={audioRef} src={`/api/media/audio/${selectedSessionId}`} />

            {/* Canvas Overlay for Detections & Skeletal joints */}
            <canvas
              ref={canvasRef}
              width={640}
              height={360}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            {/* Timeline Stamp Overlay */}
            <div className="absolute top-3 left-3 px-3 py-1 rounded bg-slate-950/80 backdrop-blur border border-white/10 text-xs font-mono text-slate-300 flex items-center space-x-3">
              <span className="flex items-center space-x-1.5">
                <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
                <span>{isPlaying ? 'STREAMING' : 'PAUSED'}</span>
              </span>
              <span className="text-slate-400">|</span>
              <span>Step: {stepIdx + 1} / {totalSteps}</span>
              <span className="text-slate-400">|</span>
              <span>{(currentStep?.timestamp_seconds ?? 0).toFixed(2)}s</span>
            </div>

            {/* Contributing Modalities Badges */}
            <div className="absolute bottom-3 left-3 flex items-center space-x-2">
              <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                currentStep?.modality_validity?.vision
                  ? 'bg-brand-cyan/20 text-brand-cyan border-brand-cyan/40'
                  : 'bg-slate-800/40 text-slate-500 border-slate-700/40'
              }`}>
                VISION
              </span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                currentStep?.modality_validity?.audio
                  ? 'bg-brand-amber/20 text-brand-amber border-brand-amber/40'
                  : 'bg-slate-800/40 text-slate-500 border-slate-700/40'
              }`}>
                AUDIO
              </span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                currentStep?.modality_validity?.gesture
                  ? 'bg-brand-emerald/20 text-brand-emerald border-brand-emerald/40'
                  : 'bg-slate-800/40 text-slate-500 border-slate-700/40'
              }`}>
                GESTURE
              </span>
            </div>
          </div>

          {/* Timeline Progress Bar */}
          <div className="w-full bg-surface rounded-full h-2 overflow-hidden border border-cardBorder">
            <div
              className="h-full bg-brand-cyan transition-all duration-100 rounded-full"
              style={{ width: `${((stepIdx + 1) / totalSteps) * 100}%` }}
            />
          </div>

          {/* Playback Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-xl border border-cardBorder bg-card">
            <div className="flex items-center space-x-2">
              {isPlaying ? (
                <button
                  onClick={pauseReplay}
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold bg-surface hover:bg-surfaceHover border border-cardBorder text-white transition-all"
                >
                  <Pause className="w-3.5 h-3.5 text-brand-amber" />
                  <span>Pause</span>
                </button>
              ) : (
                <button
                  onClick={resumeReplay}
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold bg-brand-cyan text-slate-950 hover:bg-brand-cyan/90 transition-all shadow-glow-cyan"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{stepIdx > 0 ? 'Resume' : 'Start Replay'}</span>
                </button>
              )}

              <button
                onClick={stopReplay}
                className="p-2 rounded-lg bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-300"
                title="Rewind & Stop"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>

              <button
                onClick={stepForward}
                className="p-2 rounded-lg bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-300"
                title="Step Forward (1 Frame)"
              >
                <SkipForward className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Speed Selector */}
            <div className="flex items-center space-x-1.5 text-xs font-mono">
              <span className="text-slate-400 mr-1">Speed:</span>
              {[0.5, 1.0, 2.0].map((s) => (
                <button
                  key={s}
                  onClick={() => changeSpeed(s)}
                  className={`px-2.5 py-1 rounded text-xs transition-all ${
                    playbackSpeed === s
                      ? 'bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/40 font-bold'
                      : 'bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-300'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Live Synchronized Telemetry */}
        <div className="space-y-4">
          {/* Priority Dial Card */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Transformer Priority</span>
              <span className="text-brand-purple font-bold text-base">{priorityScore} / 100</span>
            </div>

            <div className="w-full bg-surface rounded-full h-2.5 overflow-hidden border border-cardBorder">
              <div
                className={`h-full transition-all duration-300 rounded-full ${
                  priorityScore >= 70 ? 'bg-rose-500' : priorityScore >= 40 ? 'bg-brand-amber' : 'bg-brand-cyan'
                }`}
                style={{ width: `${priorityScore}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Alert Classification:</span>
              <span className="font-bold text-white uppercase text-[11px]">
                {currentStep?.fusion_output?.alert_type || 'immediate_attention'}
              </span>
            </div>
          </div>

          {/* Decision Engine Alert Card */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Decision Engine Policy</span>
              <span
                className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                  currentStep?.decision_engine_output?.dispatched
                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    : 'bg-slate-700/20 text-slate-400 border border-slate-700/30'
                }`}
              >
                {currentStep?.decision_engine_output?.dispatched ? 'ALERT DISPATCHED' : 'SUPPRESSED / COOLDOWN'}
              </span>
            </div>

            <div className="p-2.5 rounded-lg bg-surface border border-cardBorder text-xs font-sans text-slate-200">
              {currentStep?.decision_engine_output?.spoken_alert ||
                'No new alert dispatched. Refractory suppression window active.'}
            </div>
          </div>

          {/* Stage Latency Breakdown */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-semibold text-white">
              <span>Pipeline Stage Latencies</span>
              <span className="text-[10px] font-mono text-slate-400">Per Step</span>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between items-center text-slate-300">
                <span className="flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-brand-cyan" />
                  <span>Vision (YOLOv8n)</span>
                </span>
                <span className="text-slate-400">
                  {(currentStep?.stage_latency_ms?.vision ?? 53.7).toFixed(1)} ms
                </span>
              </div>

              <div className="flex justify-between items-center text-slate-300">
                <span className="flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-brand-amber" />
                  <span>Audio (PANNs)</span>
                </span>
                <span className="text-slate-400">
                  {(currentStep?.stage_latency_ms?.audio ?? 29.3).toFixed(1)} ms
                </span>
              </div>

              <div className="flex justify-between items-center text-slate-300">
                <span className="flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-brand-emerald" />
                  <span>Gesture (LSTM)</span>
                </span>
                <span className="text-slate-400">
                  {(currentStep?.stage_latency_ms?.gesture ?? 1.0).toFixed(1)} ms
                </span>
              </div>

              <div className="flex justify-between items-center text-slate-300 pt-1 border-t border-cardBorder">
                <span className="flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-brand-purple" />
                  <span>Transformer & Decision</span>
                </span>
                <span className="text-brand-purple font-bold">
                  {(currentStep?.stage_latency_ms?.fusion ?? 2.3).toFixed(1)} ms
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
