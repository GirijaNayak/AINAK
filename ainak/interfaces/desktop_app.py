"""Responsive local vision shell with explicit multimodal model availability."""
from __future__ import annotations

import sys
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget,
    QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog)

from ..config import Paths
from ..decision.alert_engine import AlertEngine
from ..audio_output.speech import LocalSpeech


class VisionWorker(QThread):
    frame_ready = Signal(object, object, float)
    state_changed = Signal(str)

    def __init__(self, weights: str, camera: int):
        super().__init__()
        self.weights, self.camera = weights, camera
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        capture = None
        final_status = "Vision: stopped"
        try:
            from ..models.vision.detector import VisionDetector
            self.state_changed.emit("Loading YOLOv8n in the background…")
            detector = VisionDetector(self.weights)
            capture = cv2.VideoCapture(self.camera)
            if not capture.isOpened():
                final_status = f"Camera unavailable at index {self.camera}"
                return
            self.state_changed.emit("Vision: active | pretrained YOLOv8n | camera inference on worker thread")
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(.01)
                    continue
                detections, event, annotated = detector.infer(frame)
                rgb = np.ascontiguousarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                self.frame_ready.emit(rgb, [
                    {"class": item.class_name, "confidence": item.confidence,
                     "timestamp": item.timestamp, "bbox_xyxy": item.bbox_xyxy}
                    for item in detections], float(event.latency_ms))
        except Exception as exc:
            final_status = f"Vision worker error: {type(exc).__name__}: {exc}"
        finally:
            if capture is not None:
                capture.release()
            if self._stop_event.is_set():
                final_status = "Vision: stopped"
            self.state_changed.emit(final_status)


class SessionPreviewWorker(QThread):
    frame_ready = Signal(object)
    state_changed = Signal(str)

    def __init__(self, video_path: Path):
        super().__init__()
        self.video_path = Path(video_path)

    def run(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self.state_changed.emit("Prerecorded video preview unavailable")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        delay = max(1, round(1000 / min(fps if np.isfinite(fps) and fps > 0 else 10., 15.)))
        try:
            self.state_changed.emit("Playing the local prerecorded video preview")
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.frame_ready.emit(rgb)
                self.msleep(delay)
        finally:
            capture.release()
            self.state_changed.emit("Prerecorded video preview stopped")


class PrerecordedDemoWorker(QThread):
    result_ready = Signal(object)
    state_changed = Signal(str)

    def __init__(self, session: Path):
        super().__init__()
        self.session = Path(session)

    def run(self) -> None:
        try:
            from ..data.feature_export import export_session_features
            from ..runtime.prerecorded_demo import run_prerecorded_demo
            paths = Paths()
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            features = self.session / f"desktop-demo-{suffix}.features.jsonl"
            report_path = self.session / f"desktop-demo-{suffix}.report.json"
            pose_model = paths.root / "artifacts" / "gesture" / "pose_landmarker_lite.task"
            self.state_changed.emit("Exporting actual vision/audio features; gesture status depends on a verified checkpoint")
            export_session_features(self.session, features, vision_weights=paths.vision_weights,
                audio_weights=paths.audio_weights, gesture_weights=paths.gesture_weights,
                pose_model_path=pose_model if pose_model.is_file() else None)
            self.state_changed.emit("Replaying the synchronized local session")
            report = run_prerecorded_demo(self.session, features, report_path,
                fusion_weights=paths.fusion_weights, overwrite=False)
            trace = deque(maxlen=10)
            trace_path = Path(report["trace_jsonl"])
            with trace_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        trace.append(json.loads(line))
            self.result_ready.emit({"report": report, "trace_tail": list(trace),
                                    "features": str(features), "report_path": str(report_path)})
        except Exception as exc:
            self.result_ready.emit({"error": f"{type(exc).__name__}: {exc}"})


class MainWindow(QMainWindow):
    def __init__(self, weights: str = "artifacts/vision/yolov8n.pt", camera: int = 0):
        super().__init__()
        self.setWindowTitle("AI AINAK — local research prototype")
        self.resize(960, 760)
        self.weights, self.camera = weights, camera
        self.worker: VisionWorker | None = None
        self.preview_worker: SessionPreviewWorker | None = None
        self.demo_worker: PrerecordedDemoWorker | None = None
        self._close_pending = False
        self.alert_engine = AlertEngine()
        self.speech = LocalSpeech()

        self.status = QLabel("Vision: stopped")
        paths = Paths()
        self.model_status = QLabel("\n".join((
            f"Vision model: {'pretrained weights present' if paths.vision_weights.is_file() else 'checkpoint missing'}",
            f"Audio CNN14: {'pretrained checkpoint present' if paths.audio_weights.is_file() else 'checkpoint missing'}",
            f"Gesture LSTM: {'checkpoint present; provenance checked during replay' if paths.gesture_weights.is_file() else 'UNTRAINED — no reviewed gesture data'}",
            f"Fusion Transformer: {'checkpoint present; provenance checked during replay' if paths.fusion_weights.is_file() else 'UNTRAINED — no aligned reviewed session data'}",
            "Live preview: vision-only | Prerecorded replay: actual available model branches",
        )))
        self.preview = QLabel("Camera stopped")
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("background: #171a21; color: #d8dee9; padding: 10px;")
        self.recent_events = QLabel("Recent events: none")
        self.inference_timing = QLabel("Inference timing: unavailable")
        self.alert_history = QListWidget()
        self.alert_history.setMaximumHeight(120)
        self.start_button = QPushButton("Start vision preview")
        self.session_button = QPushButton("Open prerecorded session")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.demo_details = QLabel("Prerecorded inference: no session selected")
        row = QHBoxLayout(); row.addWidget(self.start_button); row.addWidget(self.session_button); row.addWidget(self.stop_button)
        layout = QVBoxLayout()
        layout.addWidget(self.status); layout.addWidget(self.model_status); layout.addWidget(self.preview)
        layout.addWidget(self.recent_events); layout.addWidget(self.inference_timing); layout.addWidget(self.demo_details)
        layout.addWidget(QLabel("Recent alert history:")); layout.addWidget(self.alert_history); layout.addLayout(row)
        root = QWidget(); root.setLayout(layout); self.setCentralWidget(root)
        self.start_button.clicked.connect(self.start)
        self.session_button.clicked.connect(self.select_prerecorded_session)
        self.stop_button.clicked.connect(self.stop)

    def start(self):
        if not self._stop_preview():
            self.status.setText("Waiting for prerecorded preview to stop")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = VisionWorker(self.weights, self.camera)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.state_changed.connect(self.on_state)
        self.worker.finished.connect(self.on_worker_finished)
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.status.setText("Starting local vision preview…")
        self.worker.start()

    def on_state(self, status: str):
        self.status.setText(status)

    def on_frame(self, rgb: np.ndarray, detections: list, latency_ms: float):
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size()))
        if detections:
            names = ", ".join(f"{item['class']} {item['confidence']:.2f}" for item in detections[:6])
            self.recent_events.setText(f"Recent vision detections: {names}")
        else:
            self.recent_events.setText("Recent vision detections: none in current frame")
        self.inference_timing.setText(f"YOLO inference: {latency_ms:.1f} ms | deterministic fallback only; no learned fusion")
        now = time.time()
        alert = self.alert_engine.decide(detections, now)
        if alert is not None:
            self.alert_history.insertItem(0, f"{alert.timestamp:.1f} | {alert.text} | {alert.priority:.2f} | fallback")
            self.speech.say(alert.text, priority=int(round(alert.priority * 100)))
            while self.alert_history.count() > 10:
                self.alert_history.takeItem(self.alert_history.count() - 1)

    def _display_rgb(self, rgb: np.ndarray):
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size()))

    def select_prerecorded_session(self):
        selected = QFileDialog.getExistingDirectory(self, "Select AI AINAK recording session", "datasets/recordings")
        if not selected:
            return
        session = Path(selected)
        metadata_path = session / "recording.json"
        if not metadata_path.is_file():
            self.status.setText("Selected folder has no recording.json; choose a paired AINAK session")
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            video_path = (session / metadata["video_file"]).resolve()
            if session.resolve() not in video_path.parents or not video_path.is_file():
                raise ValueError("recorded video is missing or outside session")
        except Exception as exc:
            self.status.setText(f"Cannot preview selected session: {type(exc).__name__}: {exc}")
            return
        if self.demo_worker is not None and self.demo_worker.isRunning():
            self.status.setText("A prerecorded session is already being processed")
            return
        if not self._stop_preview():
            self.status.setText("Waiting for current prerecorded preview to stop")
            return
        self.preview_worker = SessionPreviewWorker(video_path)
        self.preview_worker.frame_ready.connect(self._display_rgb)
        self.preview_worker.state_changed.connect(self.on_state)
        self.preview_worker.finished.connect(self._on_preview_finished)
        self.preview_worker.start()
        self.demo_worker = PrerecordedDemoWorker(session)
        self.demo_worker.state_changed.connect(self.on_state)
        self.demo_worker.result_ready.connect(self._show_demo_result)
        self.demo_worker.finished.connect(self._on_demo_finished)
        self.session_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.demo_details.setText(f"Running local replay for session {session.name}…")
        self.demo_worker.start()

    def _prediction_text(self, source: dict | None) -> str:
        if not source:
            return "unavailable"
        prediction = source.get("prediction", {})
        entries = prediction.get("detections") or prediction.get("top_classes") or []
        formatted = []
        for item in entries[:3]:
            label = item.get("class") or item.get("label") or item.get("top_class") or "event"
            confidence = item.get("confidence", item.get("score"))
            formatted.append(f"{label} {float(confidence):.2f}" if confidence is not None else str(label))
        if prediction.get("top_class") and not formatted:
            confidence = prediction.get("confidence")
            formatted.append(f"{prediction['top_class']} {float(confidence):.2f}" if confidence is not None else prediction["top_class"])
        if not formatted:
            return "no labels"
        return f"{', '.join(formatted)} @ {source.get('timestamp_seconds', 0.):.2f}s"

    def _show_demo_result(self, result: dict):
        if result.get("error"):
            self.status.setText(f"Prerecorded replay failed: {result['error']}")
            self.demo_details.setText("No replay result was produced; see the error above.")
            return
        report = result["report"]
        statuses = report.get("model_status", {})
        self.model_status.setText("\n".join(
            f"{name.title()}: {status.get('status', 'unknown')}"
            + (f" — {status.get('reason')}" if status.get("reason") else "")
            for name, status in statuses.items()))
        tail = result.get("trace_tail", [])
        latest = tail[-1] if tail else {}
        sources = latest.get("source_predictions", {})
        self.recent_events.setText(" | ".join(
            f"{name.title()}: {self._prediction_text(sources.get(name))}"
            for name in ("vision", "audio", "gesture")))
        if latest.get("transformer_invocation_successful"):
            fusion = latest.get("fusion_output") or {}
            fusion_text = (f"Transformer: invoked | {fusion.get('alert_type')} | "
                           f"priority {fusion.get('priority', 0.):.2f} | confidence {fusion.get('confidence', 0.):.2f}")
        else:
            fusion_text = f"Transformer: not invoked | {latest.get('fusion_error') or 'blocked'}"
        timing = report.get("latency_ms", {})
        per_model = timing.get("per_modality_model_stage", {})
        warmed = ", ".join(f"{name} {item.get('after_first_observation_warmup', {}).get('mean_ms')} ms warmed mean"
                            for name, item in per_model.items())
        self.inference_timing.setText(f"{fusion_text} | {warmed} | replay {report.get('demo_replay_wall_seconds')} s")
        self.demo_details.setText(
            f"Replay state: {report.get('status')} | successful Transformer calls: {report.get('transformer_invocations', 0)} "
            f"(attempts {report.get('transformer_invocation_attempts', 0)}) | missing warnings: "
            f"{' ; '.join(report.get('blockers', [])) or 'none'}")
        self.alert_history.clear()
        for row in reversed(tail):
            decision = row.get("decision_engine_output")
            if decision:
                self.alert_history.addItem(
                    f"{row.get('timestamp_seconds', 0.):.2f}s | {decision.get('text')} | "
                    f"{decision.get('priority', 0.):.2f} | {'fallback' if decision.get('fallback') else 'fusion'}")
                self.speech.say(decision.get("text", ""), priority=int(round(float(decision.get("priority", 0.)) * 100)))
        if not self.alert_history.count():
            self.alert_history.addItem("No alert in the recent replay window")

    def _stop_preview(self):
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.requestInterruption()
            if not self.preview_worker.wait(2000):
                return False
        self.preview_worker = None
        return True

    def _on_preview_finished(self):
        self.preview_worker = None
        if self._close_pending and (self.demo_worker is None or not self.demo_worker.isRunning()):
            QTimer.singleShot(0, self.close)

    def _on_demo_finished(self):
        self.demo_worker = None
        self.session_button.setEnabled(True)
        self.stop_button.setEnabled(bool(self.worker and self.worker.isRunning()))
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def stop(self):
        preview_stopped = self._stop_preview()
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.status.setText("Stopping vision preview after the active frame…")
        elif not preview_stopped:
            self.status.setText("Waiting for prerecorded preview to stop")
        else:
            self.on_worker_finished()

    def on_worker_finished(self):
        self.preview.setText("Camera stopped")
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.worker = None
        if self._close_pending and (self.demo_worker is None or not self.demo_worker.isRunning()):
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.demo_worker is not None and self.demo_worker.isRunning():
            self._close_pending = True
            self.status.setText("Finishing local model work before closing; the window will close when replay ends")
            event.ignore()
            return
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            if not worker.wait(6000):
                self._close_pending = True
                self.status.setText("Stopping camera worker before closing")
                event.ignore()
                return
        if not self._stop_preview():
            self._close_pending = True
            self.status.setText("Stopping prerecorded preview before closing")
            event.ignore()
            return
        self.speech.close()
        event.accept()


def main():
    app = QApplication(sys.argv); window = MainWindow(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
