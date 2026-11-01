import {
  SystemStatus,
  ModelsStatus,
  ReportsSummary,
  SessionInfo,
  VisionInferenceResult,
  AudioInferenceResult,
  GestureInferenceResult,
  FusionInferenceResult,
  ReplayTimelineStep,
} from '../types';

const API_BASE = '';

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/api/system/status`);
  if (!res.ok) throw new Error(`Failed to fetch system status: ${res.statusText}`);
  return res.json();
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const res = await fetch(`${API_BASE}/api/models/status`);
  if (!res.ok) throw new Error(`Failed to fetch models status: ${res.statusText}`);
  return res.json();
}

export async function fetchReportsSummary(): Promise<ReportsSummary> {
  const res = await fetch(`${API_BASE}/api/reports/summary`);
  if (!res.ok) throw new Error(`Failed to fetch reports summary: ${res.statusText}`);
  return res.json();
}

export async function fetchSessions(): Promise<SessionInfo[]> {
  const res = await fetch(`${API_BASE}/api/sessions`);
  if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.statusText}`);
  return res.json();
}

export async function fetchSessionDetails(sessionId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
  if (!res.ok) throw new Error(`Failed to fetch session details: ${res.statusText}`);
  return res.json();
}

export async function processVisionFrame(imageBase64: string, timestamp?: number): Promise<VisionInferenceResult> {
  const res = await fetch(`${API_BASE}/api/vision/process-frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64, timestamp }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Vision frame processing failed');
  }
  return res.json();
}

export async function classifyAudioClip(file: Blob | File, timestamp = 0.0): Promise<AudioInferenceResult> {
  const formData = new FormData();
  formData.append('file', file, 'audio.wav');
  formData.append('timestamp', timestamp.toString());

  const res = await fetch(`${API_BASE}/api/audio/classify-clip`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Audio classification failed');
  }
  return res.json();
}

export async function processGestureFrame(imageBase64: string, timestamp?: number): Promise<GestureInferenceResult> {
  const res = await fetch(`${API_BASE}/api/gesture/process-frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageBase64, timestamp }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Gesture frame processing failed');
  }
  return res.json();
}

export async function runFusionInference(payload: {
  vision_feature?: number[] | null;
  audio_feature?: number[] | null;
  gesture_feature?: number[] | null;
  timestamp?: number | null;
}): Promise<FusionInferenceResult> {
  const res = await fetch(`${API_BASE}/api/fusion/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Transformer fusion forward pass failed');
  }
  return res.json();
}

export async function triggerSpeechTest(text: string, priority = 80): Promise<any> {
  const res = await fetch(`${API_BASE}/api/speech/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, priority }),
  });
  if (!res.ok) throw new Error(`Speech test failed: ${res.statusText}`);
  return res.json();
}

export function connectReplayWebSocket(
  sessionId: string,
  onMessage: (step: ReplayTimelineStep) => void,
  onError: (err: Event) => void,
  onClose: (event: CloseEvent) => void
): {
  socket: WebSocket;
  sendCommand: (command: string, extra?: Record<string, any>) => void;
  close: () => void;
} {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}/ws/replay/${sessionId}`;
  
  const socket = new WebSocket(wsUrl);

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('Replay stream error:', data.error);
        return;
      }
      onMessage(data);
    } catch (e) {
      console.error('Failed to parse WebSocket message', e);
    }
  };

  socket.onerror = onError;
  socket.onclose = onClose;

  const sendCommand = (command: string, extra: Record<string, any> = {}) => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ command, ...extra }));
    }
  };

  return {
    socket,
    sendCommand,
    close: () => socket.close(),
  };
}
