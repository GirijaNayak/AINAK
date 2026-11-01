import React, { useState, useRef, useEffect } from 'react';
import {
  Eye,
  Camera,
  Upload,
  Play,
  Square,
  AlertTriangle,
  Layers,
  CheckCircle,
  Crosshair,
} from 'lucide-react';
import { VisionInferenceResult, VisionDetection } from '../../types';
import { processVisionFrame } from '../../services/api';

export const VisionView: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<VisionInferenceResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Default sample image (SVG mock scene or generated test image)
  useEffect(() => {
    loadSampleImage();
    return () => {
      stopWebcam();
    };
  }, []);

  const loadSampleImage = () => {
    // Generate an illustrative canvas frame with realistic obstacles (person, car, chair)
    const c = document.createElement('canvas');
    c.width = 640;
    c.height = 480;
    const ctx = c.getContext('2d');
    if (ctx) {
      // Background street / hallway
      ctx.fillStyle = '#1a2233';
      ctx.fillRect(0, 0, 640, 480);

      // Floor perspective lines
      ctx.strokeStyle = '#2d3b55';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(320, 200);
      ctx.lineTo(0, 480);
      ctx.moveTo(320, 200);
      ctx.lineTo(640, 480);
      ctx.moveTo(320, 200);
      ctx.lineTo(320, 480);
      ctx.stroke();

      // Simulated Person (standing center-left)
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(180, 180, 80, 220);
      ctx.fillStyle = '#fbcfe8';
      ctx.beginPath();
      ctx.arc(220, 150, 30, 0, Math.PI * 2);
      ctx.fill();

      // Simulated Car / Vehicle (approaching center-right)
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(360, 240, 180, 120);
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.arc(400, 360, 25, 0, Math.PI * 2);
      ctx.arc(500, 360, 25, 0, Math.PI * 2);
      ctx.fill();

      const dataUrl = c.toDataURL('image/jpeg');
      setSelectedFile(dataUrl);
      drawToCanvas(dataUrl);
    }
  };

  const drawToCanvas = (imgSrc: string, detections: VisionDetection[] = []) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width || 640;
      canvas.height = img.height || 480;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw Zone Lines
      const w = canvas.width;
      const h = canvas.height;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 3, 0);
      ctx.lineTo(w / 3, h);
      ctx.moveTo((2 * w) / 3, 0);
      ctx.lineTo((2 * w) / 3, h);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Zone Labels
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.fillText('LEFT ZONE', 10, 20);
      ctx.fillText('CENTER ZONE', w / 3 + 10, 20);
      ctx.fillText('RIGHT ZONE', (2 * w) / 3 + 10, 20);

      // Draw Detections
      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box;
        const boxW = x2 - x1;
        const boxH = y2 - y1;

        // Color coding based on hazard score
        const isHazard = det.hazard_score >= 0.6;
        const isWarning = det.hazard_score >= 0.35 && !isHazard;
        const strokeColor = isHazard ? '#f43f5e' : isWarning ? '#f59e0b' : '#06b6d4';
        const fillColor = isHazard ? 'rgba(244, 63, 94, 0.15)' : isWarning ? 'rgba(245, 158, 11, 0.1)' : 'rgba(6, 182, 212, 0.08)';

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isHazard ? 3 : 2;
        ctx.fillStyle = fillColor;

        // Box
        ctx.strokeRect(x1, y1, boxW, boxH);
        ctx.fillRect(x1, y1, boxW, boxH);

        // Corner accents
        const cLen = Math.min(16, boxW / 4, boxH / 4);
        ctx.lineWidth = 3;
        ctx.beginPath();
        // Top-left
        ctx.moveTo(x1, y1 + cLen);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + cLen, y1);
        // Top-right
        ctx.moveTo(x2 - cLen, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + cLen);
        // Bottom-left
        ctx.moveTo(x1, y2 - cLen);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + cLen, y2);
        // Bottom-right
        ctx.moveTo(x2 - cLen, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - cLen);
        ctx.stroke();

        // Label banner
        const labelText = `${det.label.toUpperCase()} #${det.track_id} (${Math.round(det.confidence * 100)}%)`;
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = strokeColor;
        ctx.fillRect(x1, Math.max(0, y1 - 20), textWidth + 12, 20);

        ctx.fillStyle = '#0a0d14';
        ctx.fillText(labelText, x1 + 6, Math.max(14, y1 - 6));
      });
    };
    img.src = imgSrc;
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    stopWebcam();
    const reader = new FileReader();
    reader.onload = (event) => {
      const b64 = event.target?.result as string;
      setSelectedFile(b64);
      drawToCanvas(b64);
      setInferenceResult(null);
    };
    reader.readAsDataURL(file);
  };

  const runInference = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);

    try {
      const result = await processVisionFrame(selectedFile, Date.now() / 1000);
      setInferenceResult(result);
      drawToCanvas(selectedFile, result.detections);
    } catch (err: any) {
      setError(err.message || 'Vision inference failed');
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
      processLiveFrames();
    } catch (err: any) {
      setError(`Webcam access denied: ${err.message}`);
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

  const processLiveFrames = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(processLiveFrames);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL('image/jpeg', 0.8);
      processVisionFrame(b64)
        .then((res) => {
          setInferenceResult(res);
          drawToCanvas(b64, res.detections);
        })
        .catch(() => {})
        .finally(() => {
          if (streamRef.current) {
            setTimeout(() => {
              animationFrameRef.current = requestAnimationFrame(processLiveFrames);
            }, 100);
          }
        });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-cardBorder pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold tracking-tight text-white font-sans">
              Branch 1: Computer Vision Subsystem
            </h2>
            <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/30">
              YOLOv8n + IoU Tracker
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time multi-object detection, trajectory tracking, ground-perspective depth estimation, and looming hazard scoring.
          </p>
        </div>

        {inferenceResult && (
          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Latency: <span className="text-brand-cyan font-bold">{inferenceResult.latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-card border border-cardBorder text-slate-300">
              Tracked: <span className="text-emerald-400 font-bold">{inferenceResult.detected_objects_count}</span>
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

      {/* Main View Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Canvas & Controls */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative aspect-[4/3] rounded-xl overflow-hidden border border-cardBorder bg-black shadow-card flex items-center justify-center">
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas ref={canvasRef} className="w-full h-full object-contain" />

            {/* Overlaid Status Banner */}
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-slate-950/80 backdrop-blur border border-white/10 text-[10px] font-mono text-slate-300 flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${isWebcamActive ? 'bg-rose-500 animate-pulse' : 'bg-brand-cyan'}`} />
              <span>{isWebcamActive ? 'LIVE WEBCAM STREAM' : 'STATIC / RECORDED FRAME'}</span>
            </div>

            {isLoading && (
              <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center space-y-3">
                <div className="w-8 h-8 rounded-full border-2 border-brand-cyan border-t-transparent animate-spin" />
                <span className="text-xs font-mono text-brand-cyan">Running YOLOv8n & Tracker...</span>
              </div>
            )}
          </div>

          {/* Action Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-xl border border-cardBorder bg-card">
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={runInference}
                disabled={isLoading || isWebcamActive}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold bg-brand-cyan text-slate-950 hover:bg-brand-cyan/90 disabled:opacity-50 transition-all shadow-glow-cyan"
              >
                <Play className="w-3.5 h-3.5" />
                <span>Run YOLOv8n Inference</span>
              </button>

              <button
                onClick={isWebcamActive ? stopWebcam : startWebcam}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  isWebcamActive
                    ? 'bg-rose-600 hover:bg-rose-700 text-white'
                    : 'bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-200'
                }`}
              >
                {isWebcamActive ? <Square className="w-3.5 h-3.5" /> : <Camera className="w-3.5 h-3.5 text-brand-cyan" />}
                <span>{isWebcamActive ? 'Stop Camera' : 'Start Live Camera'}</span>
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isWebcamActive}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg text-xs font-medium bg-surface hover:bg-surfaceHover border border-cardBorder text-slate-300 disabled:opacity-50"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Upload Image</span>
              </button>
            </div>

            <button
              onClick={loadSampleImage}
              disabled={isWebcamActive}
              className="text-xs font-mono text-brand-cyan hover:underline disabled:opacity-50"
            >
              Reset Sample Frame
            </button>
          </div>
        </div>

        {/* Inference Telemetry & Detections Table */}
        <div className="space-y-4">
          {/* Summary Card */}
          <div className="p-4 rounded-xl border border-cardBorder bg-card space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Hazard Assessment</span>
              <span
                className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                  inferenceResult?.overall_hazard_level === 'imminent' || inferenceResult?.overall_hazard_level === 'high'
                    ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    : inferenceResult?.overall_hazard_level === 'medium'
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`}
              >
                {inferenceResult?.overall_hazard_level || 'NORMAL'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-surface border border-cardBorder">
                <div className="text-[10px] text-slate-400">MAX HAZARD</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {((inferenceResult?.max_hazard_score || 0) * 100).toFixed(0)}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-surface border border-cardBorder">
                <div className="text-[10px] text-slate-400">FEATURE VECTOR</div>
                <div className="text-base font-bold text-brand-cyan mt-0.5">128-D</div>
              </div>
            </div>
          </div>

          {/* Detections List */}
          <div className="rounded-xl border border-cardBorder bg-card overflow-hidden">
            <div className="p-3 border-b border-cardBorder bg-surface/50 flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-semibold text-white">
                <Crosshair className="w-3.5 h-3.5 text-brand-cyan" />
                <span>Detected Objects ({inferenceResult?.detections?.length || 0})</span>
              </div>
            </div>

            <div className="max-h-72 overflow-y-auto divide-y divide-cardBorder/60 text-xs font-mono">
              {inferenceResult?.detections && inferenceResult.detections.length > 0 ? (
                inferenceResult.detections.map((det, idx) => (
                  <div key={idx} className="p-3 hover:bg-surface/30 transition-colors space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-white">{det.label.toUpperCase()}</span>
                        <span className="text-[10px] text-slate-400">#{det.track_id}</span>
                      </div>
                      <span className="font-semibold text-brand-cyan">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span>Proximity: {det.depth > 0.5 ? 'Close' : 'Distant'}</span>
                      <span className={det.hazard_score >= 0.6 ? 'text-rose-400 font-bold' : ''}>
                        Hazard: {(det.hazard_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="p-6 text-center text-xs text-slate-400">
                  No objects detected yet. Click "Run YOLOv8n Inference".
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
