"""FastAPI application for the AI AINAK Multimodal Assistive Intelligence Console."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import DashboardService

LOG = logging.getLogger("ainak.dashboard.app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIST = BASE_DIR.parents[2] / "frontend" / "dist"

app = FastAPI(
    title="AI AINAK – Multimodal Assistive Intelligence Console",
    description="Interactive laboratory console for Computer Vision, Environmental Audio, Gesture Recognition, and Transformer Fusion.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = DashboardService()


# ------------------------------------------------------------------------------
# REST API: SYSTEM, MODELS, METRICS, SESSIONS
# ------------------------------------------------------------------------------

@app.get("/api/system/status")
async def get_system_status():
    """Return runtime environment, hardware stats, and automated test readiness."""
    return service.get_system_status()


@app.get("/api/models/status")
async def get_models_status():
    """Return model status, parameters, checkpoint paths, and cryptographic digests."""
    return service.get_models_status()


@app.get("/api/reports/summary")
async def get_reports_summary():
    """Return real quantitative benchmark reports (COCO128, ESC-50, Gesture, Fusion)."""
    return service.get_evaluation_reports()


@app.get("/api/sessions")
async def get_sessions():
    """List paired multimodal recording sessions available for replay."""
    return service.get_available_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Retrieve session metadata, clock tables, and trace summary."""
    session_dir = service.paths.root / "datasets" / "recordings" / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    meta_file = session_dir / "recording.json"
    if not meta_file.is_file():
        raise HTTPException(status_code=404, detail="Missing recording.json in session")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    trace_file = session_dir / "phase3-demo.trace.jsonl"
    demo_file = session_dir / "phase3-demo.json"

    demo_data = json.loads(demo_file.read_text(encoding="utf-8")) if demo_file.is_file() else {}
    trace_summary = []
    if trace_file.is_file():
        with trace_file.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    trace_summary.append(json.loads(line))
                if idx >= 150:
                    break

    return {
        "metadata": meta,
        "demo_report": demo_data,
        "trace_records_count": len(trace_summary),
        "trace_records_sample": trace_summary[:20],
    }


# ------------------------------------------------------------------------------
# MEDIA STREAMING (HTML5 VIDEO & AUDIO)
# ------------------------------------------------------------------------------

@app.get("/api/media/video/{session_id}")
async def stream_session_video(session_id: str):
    """Serve local recorded MP4 video for HTML5 video element."""
    session_dir = service.paths.root / "datasets" / "recordings" / session_id
    video_path = session_dir / "video.mp4"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Session video not found")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/api/media/audio/{session_id}")
async def stream_session_audio(session_id: str):
    """Serve local recorded WAV audio for HTML5 audio element."""
    session_dir = service.paths.root / "datasets" / "recordings" / session_id
    audio_path = session_dir / "audio.wav"
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Session audio not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


# ------------------------------------------------------------------------------
# STANDALONE MODALITY INFERENCE ENDPOINTS
# ------------------------------------------------------------------------------

class FramePayload(BaseModel):
    image_base64: str
    timestamp: float | None = None


@app.post("/api/vision/process-frame")
async def process_vision_frame(payload: FramePayload):
    """Process a video frame through YOLOv8n, ByteTrack, relative depth, and hazard scorer."""
    try:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image frame")

        result = service.infer_vision_frame(frame, payload.timestamp)
        return result
    except Exception as exc:
        LOG.error("Vision frame processing failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/audio/classify-clip")
async def classify_audio_clip(file: UploadFile = File(...), timestamp: float = Form(0.0)):
    """Classify an uploaded WAV clip through PANNs CNN14 and AudioMLP."""
    try:
        contents = await file.read()
        import io
        waveform, sr = sf.read(io.BytesIO(contents), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != 32000:
            import scipy.signal
            gcd = np.gcd(32000, sr)
            waveform = scipy.signal.resample_poly(waveform, 32000 // gcd, sr // gcd).astype(np.float32)

        # Pad or trim to 2.0s (64,000 samples)
        target_len = 64000
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]

        result = service.infer_audio_buffer(waveform, timestamp)
        return result
    except Exception as exc:
        LOG.error("Audio classification failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/gesture/process-frame")
async def process_gesture_frame(payload: FramePayload):
    """Process a video frame through MediaPipe Pose and GestureLSTM."""
    try:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image frame")

        result = service.infer_gesture_frame(frame, payload.timestamp)
        return result
    except Exception as exc:
        LOG.error("Gesture processing failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


class FusionPayload(BaseModel):
    vision_feature: list[float] | None = None
    audio_feature: list[float] | None = None
    gesture_feature: list[float] | None = None
    timestamp: float | None = None


@app.post("/api/fusion/infer")
async def run_transformer_fusion(payload: FusionPayload):
    """Execute Multimodal Transformer forward pass over given modality features."""
    try:
        result = service.infer_fusion(
            vision_feat=payload.vision_feature,
            audio_feat=payload.audio_feature,
            gesture_feat=payload.gesture_feature,
            timestamp=payload.timestamp
        )
        return result
    except Exception as exc:
        LOG.error("Transformer fusion forward failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


class SpeechTestPayload(BaseModel):
    text: str = "AI AINAK multimodal assistive prototype is ready and operational."
    priority: int = 80


@app.post("/api/speech/test")
async def test_speech_output(payload: SpeechTestPayload):
    """Trigger speech synthesis and return local speech diagnostics."""
    return service.speak_alert(payload.text, payload.priority)


# ------------------------------------------------------------------------------
# SYNCHRONIZED MULTIMODAL REPLAY WEBSOCKET
# ------------------------------------------------------------------------------

@app.websocket("/ws/replay/{session_id}")
async def websocket_session_replay(websocket: WebSocket, session_id: str):
    """Stream synchronized multimodal events step-by-step with real Transformer invocations."""
    await websocket.accept()
    session_dir = service.paths.root / "datasets" / "recordings" / session_id
    if not session_dir.is_dir():
        await websocket.send_json({"error": f"Session not found: {session_id}"})
        await websocket.close()
        return

    trace_file = session_dir / "phase3-demo.trace.jsonl"
    if not trace_file.is_file():
        await websocket.send_json({"error": "Session trace missing; run export-features first."})
        await websocket.close()
        return

    # Read pre-aligned timeline steps
    timeline_steps = []
    with trace_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                timeline_steps.append(json.loads(line))

    try:
        current_step = 0
        is_paused = False
        speed = 1.0

        while current_step < len(timeline_steps):
            # Check client command (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                cmd = data.get("command")
                if cmd == "pause":
                    is_paused = True
                elif cmd == "resume":
                    is_paused = False
                elif cmd == "seek":
                    target_time = float(data.get("timestamp", 0.0))
                    # Find closest step
                    closest_idx = min(
                        range(len(timeline_steps)),
                        key=lambda i: abs(timeline_steps[i].get("timestamp_seconds", 0.0) - target_time)
                    )
                    current_step = closest_idx
                elif cmd == "speed":
                    speed = max(0.25, min(4.0, float(data.get("value", 1.0))))
                elif cmd == "stop":
                    break
            except asyncio.TimeoutError:
                pass

            if is_paused:
                await asyncio.sleep(0.1)
                continue

            record = timeline_steps[current_step]
            step_ts = record.get("timestamp_seconds", 0.0)

            # Package synchronized payload
            payload = {
                "step_index": current_step,
                "total_steps": len(timeline_steps),
                "timestamp_seconds": step_ts,
                "contributing_modalities": record.get("contributing_modalities", []),
                "modality_validity": record.get("modality_validity", {}),
                "transformer_used": record.get("transformer_used", True),
                "fusion_output": record.get("fusion_output"),
                "decision_engine_output": record.get("decision_engine_output"),
                "source_predictions": record.get("source_predictions", {}),
                "stage_latency_ms": record.get("stage_latency_ms", {}),
            }

            await websocket.send_json(payload)
            current_step += 1

            # Sleep synchronized interval based on speed
            delay = 0.04 / speed  # ~25 FPS timeline advance
            await asyncio.sleep(delay)

        await websocket.send_json({"status": "complete", "message": "Replay finished successfully."})
    except WebSocketDisconnect:
        LOG.info("Replay WebSocket disconnected for %s", session_id)
    except Exception as exc:
        LOG.error("WebSocket replay error: %s", exc)


# ------------------------------------------------------------------------------
# STATIC ASSETS AND SINGLE-PAGE APPLICATION
# ------------------------------------------------------------------------------

# Mount legacy static assets directory (for backward compatibility and test asserts)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount modern React + Vite production build assets when compiled
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")


@app.get("/favicon.svg")
async def serve_favicon():
    favicon = FRONTEND_DIST / "favicon.svg"
    if favicon.is_file():
        return FileResponse(str(favicon), media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/")
async def serve_index():
    """Serve the modern single-page application dashboard (falls back to legacy static if not built)."""
    modern_index = FRONTEND_DIST / "index.html"
    if modern_index.is_file():
        return FileResponse(str(modern_index))

    legacy_index = STATIC_DIR / "index.html"
    if legacy_index.is_file():
        return FileResponse(str(legacy_index))

    raise HTTPException(
        status_code=404,
        detail="Dashboard frontend not found. Please run 'npm run build' in the frontend/ directory."
    )
