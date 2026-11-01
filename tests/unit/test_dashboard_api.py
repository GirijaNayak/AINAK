"""Unit tests for the AI AINAK Dashboard REST API, static assets, and WebSocket layer."""
from __future__ import annotations

import base64
import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from ainak.interfaces.dashboard.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_system_status_endpoint(client):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert data["app_name"] == "AI AINAK"
    assert "python_version" in data
    assert "torch_version" in data
    assert "automated_tests" in data
    assert data["automated_tests"]["passed"] >= 55


def test_models_status_endpoint(client):
    response = client.get("/api/models/status")
    assert response.status_code == 200
    models = response.json()
    assert "vision" in models
    assert "audio" in models
    assert "gesture" in models
    assert "fusion" in models
    assert models["vision"]["available"] is True
    assert models["audio"]["available"] is True
    assert models["gesture"]["available"] is True
    assert models["fusion"]["available"] is True


def test_reports_summary_endpoint(client):
    response = client.get("/api/reports/summary")
    assert response.status_code == 200
    data = response.json()
    assert "vision" in data
    assert "audio" in data
    assert "gesture" in data
    assert "fusion" in data
    assert "summary_metrics" in data
    assert data["summary_metrics"]["audio"]["accuracy"] >= 0.90
    assert data["summary_metrics"]["gesture"]["accuracy"] >= 0.99
    assert data["summary_metrics"]["vision"]["map_50"] >= 0.60


def test_sessions_list_and_detail(client):
    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert isinstance(sessions, list)
    assert len(sessions) >= 4
    session_ids = [s["session_id"] for s in sessions]
    assert "session-multimodal-01" in session_ids

    detail_resp = client.get("/api/sessions/session-multimodal-01")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert "metadata" in detail
    assert "trace_records_count" in detail
    assert detail["trace_records_count"] > 0


def test_fusion_infer_endpoint(client):
    v_feat = [0.1] * 128
    a_feat = [0.05] * 2048
    g_feat = [0.0] * 128
    response = client.post(
        "/api/fusion/infer",
        json={
            "vision_feature": v_feat,
            "audio_feature": a_feat,
            "gesture_feature": g_feat,
            "timestamp": 1.5,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert "raw_output" in result
    assert "interpreted_prediction" in result
    assert "decision_engine_output" in result
    assert "priority_scalar" in result["raw_output"]
    assert "alert_type" in result["interpreted_prediction"]
    assert result["latency_ms"] > 0


def test_speech_test_endpoint(client):
    response = client.post(
        "/api/speech/test",
        json={"text": "Test alert notification", "priority": 50},
    )
    assert response.status_code == 200
    data = response.json()
    assert "queued" in data
    assert "backend" in data


def test_static_assets_and_index_endpoint(client):
    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "AI AINAK" in index_resp.text
    assert "Multimodal Assistive Intelligence Console" in index_resp.text

    css_resp = client.get("/static/css/dashboard.css")
    assert css_resp.status_code == 200
    assert "--bg-primary" in css_resp.text

    js_resp = client.get("/static/js/dashboard.js")
    assert js_resp.status_code == 200
    assert "POSE_CONNECTIONS" in js_resp.text


def test_vision_frame_processing_endpoint(client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    resp = client.post(
        "/api/vision/process-frame",
        json={"image_base64": b64, "timestamp": 0.5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "detections" in data
    assert "hazard_summary" in data
    assert "latency_ms" in data
    assert data["latency_ms"] > 0


def test_gesture_frame_processing_endpoint(client):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    resp = client.post(
        "/api/gesture/process-frame",
        json={"image_base64": b64, "timestamp": 0.5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "has_pose" in data
    assert "probabilities" in data
    assert "buffer_fill" in data
    assert "latency_ms" in data
    assert data["latency_ms"] > 0


def test_websocket_replay_stream(client):
    with client.websocket_connect("/ws/replay/session-multimodal-01") as ws:
        # Receive first step
        data = ws.receive_json()
        assert "step_index" in data
        assert "timestamp_seconds" in data
        assert "contributing_modalities" in data
        assert "transformer_used" in data
        assert data["step_index"] == 0
        # Send pause command
        ws.send_json({"command": "pause"})
        # Send stop command
        ws.send_json({"command": "stop"})
