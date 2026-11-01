from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..models.vision.detector import VisionDetector
from .features import FEATURE_SCHEMA_VERSION, PREPROCESSING_VERSIONS, validate_feature_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _safe_member(folder: Path, filename: str) -> Path:
    path = (folder / filename).resolve()
    if folder.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"recording metadata references a missing or unsafe file: {filename}")
    return path


def _read_clock_index(path: Path, time_column: str, sample_column: str | None = None) -> tuple[list[float], list[float]]:
    if not path.is_file():
        return [], []
    samples: list[float] = []
    seconds: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if sample_column is not None:
                count = int(row.get("frames") or 0)
                if count <= 0:
                    continue
                samples.append(float(row[sample_column]))
            else:
                samples.append(float(row["frame_index"]))
            seconds.append(float(row[time_column]))
    if not np.isfinite(samples).all() or not np.isfinite(seconds).all() or any(b < a for a, b in zip(seconds, seconds[1:])):
        raise ValueError(f"invalid timestamp index: {path}")
    return samples, seconds


def _clock_time(position: float, positions: list[float], seconds: list[float], offset: float,
                units_per_second: float | None = None) -> float:
    if positions:
        if position > positions[-1] and len(positions) >= 2:
            slope = (seconds[-1] - seconds[-2]) / max(positions[-1] - positions[-2], 1e-9)
            return float(seconds[-1] + (position - positions[-1]) * slope)
        if position > positions[-1] and units_per_second:
            return float(seconds[-1] + (position - positions[-1]) / units_per_second)
        return float(np.interp(position, positions, seconds))
    return offset + position / (units_per_second or 1.)


def _audio_windows(path: Path, window_seconds: float, hop_seconds: float) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Yield one bounded native-rate audio window at a time; memory is O(window size)."""
    import soundfile as sf
    import librosa

    with sf.SoundFile(str(path)) as sound:
        native_rate = int(sound.samplerate)
        total = int(len(sound))
        window = max(1, round(native_rate * window_seconds))
        hop = max(1, round(native_rate * hop_seconds))
        start = 0
        while start < total:
            sound.seek(start)
            channels = sound.read(min(window, total - start), dtype="float32", always_2d=True)
            if len(channels) >= max(1, native_rate // 4):
                mono = channels.mean(axis=1, dtype=np.float32)
                if native_rate != 32000:
                    mono = librosa.resample(mono, orig_sr=native_rate, target_sr=32000)
                yield start, start + len(channels), native_rate, mono
            if start + window >= total:
                break
            start += hop


def _record(feature_id: str, session_id: str, source_recording_id: str, modality: str,
            timestamp: float, start: float, end: float, feature: np.ndarray | None,
            prediction: dict[str, Any], confidence: float | None, feature_kind: str,
            checkpoint_id: str, preprocessing_version: str, missing_reason: str | None = None) -> dict[str, Any]:
    row = {
        "schema_version": FEATURE_SCHEMA_VERSION, "feature_id": feature_id,
        "session_id": session_id, "source_recording_id": source_recording_id,
        "modality": modality, "timestamp_seconds": float(timestamp),
        "interval_start_seconds": float(start), "interval_end_seconds": float(end),
        "feature": None if feature is None else np.asarray(feature, dtype=np.float32).reshape(-1).tolist(),
        "feature_kind": feature_kind, "prediction": prediction,
        "model_checkpoint_id": checkpoint_id, "preprocessing_version": preprocessing_version,
        "valid": feature is not None, "confidence": confidence, "missing_reason": missing_reason,
        "latency_ms": 0.,
    }
    validate_feature_record(row)
    return row


def export_session_features(session: Path, output: Path, *, vision_weights: Path,
                            audio_weights: Path, gesture_weights: Path,
                            pose_model_path: Path | None = None,
                            sample_every: int = 5, max_frames: int = 10000,
                            audio_window_seconds: float = 2., audio_hop_seconds: float = 1.,
                            device: str = "cpu", overwrite: bool = False) -> dict[str, Any]:
    """Export actual model features from a session folder with a shared session clock."""
    export_started = time.perf_counter()
    import cv2
    import soundfile as sf
    import torch

    if sample_every < 1 or max_frames < 1 or audio_window_seconds < .25 or audio_hop_seconds <= 0:
        raise ValueError("sampling values must be positive; audio windows must be at least 250 ms")
    if audio_hop_seconds > audio_window_seconds:
        raise ValueError("audio hop cannot exceed the audio window duration")
    session = Path(session).resolve()
    metadata_path = session / "recording.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing recording.json in session folder: {session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session_id = str(metadata.get("recording_id") or metadata.get("session_id") or "")
    if not session_id:
        raise ValueError("recording.json must identify this session with recording_id or session_id")
    video = _safe_member(session, str(metadata.get("video_file", "")))
    audio = _safe_member(session, str(metadata.get("audio_file", "")))
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite feature file: {output}; pass --overwrite explicitly")
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite feature summary: {summary_path}; pass --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve each branch independently so a missing checkpoint does not hide usable branch output.
    statuses: dict[str, dict[str, Any]] = {}
    vision_model = audio_model = gesture_model = None
    try:
        vision_model = VisionDetector(vision_weights, device)
        statuses["vision"] = {"status": "ready", "checkpoint_id": _sha256(Path(vision_weights))}
    except Exception as exc:
        statuses["vision"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        from ..models.audio.classifier import PannsCNN14
        audio_model = PannsCNN14(audio_weights, device)
        statuses["audio"] = {"status": "ready", "checkpoint_id": _sha256(Path(audio_weights))}
    except Exception as exc:
        statuses["audio"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    if Path(gesture_weights).is_file():
        try:
            from ..models.gesture.lstm import GestureRecognizer
            payload = torch.load(gesture_weights, map_location=device, weights_only=False)
            hidden = int(payload.get("model_config", {}).get("hidden_size", 128))
            if hidden != 128:
                raise ValueError(f"gesture checkpoint embedding dimension {hidden} is incompatible with fusion schema (128)")
            gesture_model = GestureRecognizer(gesture_weights, 24, device, .75)
            statuses["gesture"] = {"status": "ready", "checkpoint_id": _sha256(Path(gesture_weights))}
        except Exception as exc:
            statuses["gesture"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        statuses["gesture"] = {"status": "unavailable", "reason": f"gesture checkpoint missing: {gesture_weights}"}

    video_indices, video_seconds = _read_clock_index(session / "video_frames.csv", "monotonic_seconds")
    audio_indices, audio_seconds = _read_clock_index(session / "audio_blocks.csv", "monotonic_seconds", "sample_start")
    video_offset = float(metadata.get("video_offset_seconds", 0.))
    audio_offset = float(metadata.get("audio_offset_seconds", 0.))
    if not all(math.isfinite(x) and x >= 0 for x in (video_offset, audio_offset)):
        raise ValueError("video/audio offsets in recording.json must be finite nonnegative seconds")
    counts = {name: 0 for name in ("vision", "audio", "gesture")}
    invalid_audio = 0
    errors: list[str] = []
    frame_count = 0
    open_mode = "w" if overwrite else "x"
    with output.open(open_mode, encoding="utf-8", newline="\n") as sink:
        def emit(row: dict[str, Any]) -> None:
            sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            errors.append("video: could not open local recording")
        else:
            pose_context = None
            pose = None
            pose_times: list[float] = []
            last_media_seconds = -1.
            try:
                if gesture_model is not None:
                    from ..models.gesture.pose import create_pose_tracker
                    pose_context = create_pose_tracker(model_path=pose_model_path, static_image_mode=False, model_complexity=1,
                        min_detection_confidence=.5, min_tracking_confidence=.5)
                    pose = pose_context
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    fps = 30.
                while frame_count < max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    media_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
                    if not math.isfinite(media_seconds) or media_seconds < 0:
                        media_seconds = frame_count / fps
                    if video_indices and frame_count < len(video_indices):
                        timestamp = video_seconds[frame_count]
                    else:
                        if frame_count and media_seconds <= last_media_seconds:
                            media_seconds = frame_count / fps
                        timestamp = video_offset + media_seconds
                    last_media_seconds = media_seconds
                    if vision_model is not None and frame_count % sample_every == 0:
                        try:
                            detections, event, _ = vision_model.infer(frame, timestamp=timestamp)
                            row = _record(f"{session_id}:vision:{counts['vision']}", session_id, session_id, "vision",
                                timestamp, timestamp, timestamp, np.asarray(event.feature),
                                {"detections": [{"class": d.class_name, "confidence": d.confidence,
                                    "bbox_xyxy": list(d.bbox_xyxy), "track_id": d.track_id} for d in detections]},
                                event.confidence, "detection_summary", statuses["vision"]["checkpoint_id"], PREPROCESSING_VERSIONS["vision"])
                            row["latency_ms"] = event.latency_ms
                            emit(row)
                            counts["vision"] += 1
                        except Exception as exc:
                            errors.append(f"vision at {timestamp:.3f}s: {type(exc).__name__}: {exc}")
                    if gesture_model is not None and pose is not None:
                        try:
                            from ..models.gesture.lstm import extract_pose_landmarks
                            landmarks = extract_pose_landmarks(frame, pose, int(timestamp * 1000))
                            if landmarks is None:
                                gesture_model.sequence.clear(); pose_times.clear()
                            else:
                                if pose_times and timestamp - pose_times[-1] > .5:
                                    gesture_model.sequence.clear(); pose_times.clear()
                                pose_times.append(timestamp)
                                pose_times = pose_times[-gesture_model.seq_len:]
                                event = gesture_model.append(landmarks, timestamp)
                                if event is not None:
                                    row = _record(f"{session_id}:gesture:{counts['gesture']}", session_id, session_id, "gesture",
                                        timestamp, pose_times[0], timestamp, np.asarray(event.feature),
                                        {"probabilities": event.metadata.get("probabilities", {}),
                                         "top_class": event.metadata.get("top_class"),
                                         "top_classes": ([{"class": event.metadata.get("top_class"), "confidence": event.confidence}]
                                            if event.metadata.get("threshold_accepted") else []),
                                         "threshold_accepted": event.metadata.get("threshold_accepted", False)},
                                        event.confidence, "lstm_embedding", statuses["gesture"]["checkpoint_id"], PREPROCESSING_VERSIONS["gesture"])
                                    row["latency_ms"] = event.latency_ms
                                    emit(row)
                                    counts["gesture"] += 1
                        except Exception as exc:
                            errors.append(f"gesture at {timestamp:.3f}s: {type(exc).__name__}: {exc}")
                    frame_count += 1
            finally:
                capture.release()
                if pose_context is not None:
                    pose_context.close()

        if audio_model is not None:
            try:
                with sf.SoundFile(str(audio)) as sound:
                    sample_rate = int(sound.samplerate)
                    audio_duration = len(sound) / sample_rate
                for start_sample, end_sample, native_rate, waveform in _audio_windows(audio, audio_window_seconds, audio_hop_seconds):
                    start_time = _clock_time(start_sample, audio_indices, audio_seconds, audio_offset, native_rate)
                    end_time = _clock_time(end_sample, audio_indices, audio_seconds, audio_offset, native_rate) if audio_indices else audio_offset + end_sample / native_rate
                    midpoint = (start_time + end_time) / 2
                    try:
                        predictions, event = audio_model.infer(waveform, 32000, timestamp=midpoint)
                        probabilities = {name: float(score) for name, score in predictions}
                        row = _record(f"{session_id}:audio:{counts['audio']}", session_id, session_id, "audio",
                            midpoint, start_time, end_time, np.asarray(event.feature),
                            {"probabilities": probabilities, "top_classes": [
                                {"class": name, "confidence": float(score)} for name, score in predictions[:10]],
                             "sample_rate_original": native_rate, "sample_start": start_sample,
                             "sample_end": end_sample}, event.confidence, "cnn14_embedding",
                            statuses["audio"]["checkpoint_id"], PREPROCESSING_VERSIONS["audio"])
                        row["latency_ms"] = event.latency_ms
                        emit(row)
                        counts["audio"] += 1
                    except ValueError as exc:
                        invalid_audio += 1
                        emit(_record(f"{session_id}:audio-invalid:{invalid_audio}", session_id, session_id, "audio",
                            midpoint, start_time, end_time, None, {"sample_rate_original": native_rate}, None,
                            "cnn14_embedding", statuses["audio"]["checkpoint_id"], PREPROCESSING_VERSIONS["audio"],
                            f"audio window rejected: {exc}"))
                statuses["audio"]["duration_seconds"] = audio_duration
            except Exception as exc:
                errors.append(f"audio: {type(exc).__name__}: {exc}")
        sink.flush()

    summary = {"schema_version": FEATURE_SCHEMA_VERSION, "session_id": session_id,
        "source_recordings": {"video": {"path": str(video), "sha256": _sha256(video)},
                              "audio": {"path": str(audio), "sha256": _sha256(audio)}},
        "features_jsonl": str(output.resolve()), "model_status": statuses, "valid_feature_counts": counts,
        "invalid_audio_windows": invalid_audio, "decoded_video_frames": frame_count,
        "errors": errors, "timeline_clock": metadata.get("clock", "file-relative offsets from recording.json"),
        "export_wall_seconds": time.perf_counter() - export_started,
        "parameters": {"sample_every": sample_every, "max_frames": max_frames,
                       "audio_window_seconds": audio_window_seconds, "audio_hop_seconds": audio_hop_seconds,
                       "device": device}}
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary

