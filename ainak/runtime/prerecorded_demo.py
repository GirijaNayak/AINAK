from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import soundfile as sf
import torch

from ..data.features import validate_feature_record
from ..models.fusion.transformer import MultimodalTransformer
from ..data.schemas import ALERT_TYPES, FEATURE_DIMS
from .events import ModelEvent
from .pipeline import MultimodalPipeline
from .checkpoint_validation import real_training_checkpoint_issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _require_full_demo_evidence(status: str, successful_invocations: int, all_three_same_invocation: bool) -> None:
    """Guard report claims against counters and observed branch participation."""
    if status == "complete_trained_three_branch" and (successful_invocations < 1 or not all_three_same_invocation):
        raise RuntimeError("internal report invariant failed: complete fusion requires a successful three-branch Transformer invocation")


def _fusion_model(path: Path, device: str) -> tuple[MultimodalTransformer | None, dict[str, Any]]:
    if not path.is_file():
        return None, {"status": "unavailable", "reason": f"trained Transformer checkpoint missing: {path}"}
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        issues = real_training_checkpoint_issues(payload, "fusion")
        if issues:
            return None, {"status": "unavailable", "reason": "checkpoint provenance rejected: " + "; ".join(issues)}
        if payload.get("feature_dims") != FEATURE_DIMS:
            return None, {"status": "unavailable", "reason": "checkpoint feature dimensions do not match the runtime schema"}
        if payload.get("alert_types") != ALERT_TYPES:
            return None, {"status": "unavailable", "reason": "checkpoint alert labels do not match the runtime schema"}
        model = MultimodalTransformer(**payload.get("model_config", {})).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        return model, {"status": "ready", "verified_trained": True,
                       "trained_on_real_data": True, "checkpoint_id": _sha256(path),
                       "training_manifest_sha256": payload["training_metadata"]["manifest_sha256"],
                       "recording_split": payload["recording_split"],
                       "training_examples": payload["training_examples"],
                       "validation_examples": payload["validation_examples"],
                       "test_examples": payload["test_examples"]}
    except Exception as exc:
        return None, {"status": "unavailable", "reason": f"checkpoint load failed: {type(exc).__name__}: {exc}"}


def _validate_session_timing(session: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    video_path = (session / str(metadata["video_file"])).resolve()
    audio_path = (session / str(metadata["audio_file"])).resolve()
    if session.resolve() not in video_path.parents or not video_path.is_file():
        raise ValueError("recording.json video_file is missing or outside the session folder")
    if session.resolve() not in audio_path.parents or not audio_path.is_file():
        raise ValueError("recording.json audio_file is missing or outside the session folder")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open prerecorded video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError("video has no usable FPS/frame count; cannot verify timing")
    video_duration = frame_count / fps
    info = sf.info(str(audio_path))
    audio_duration = info.frames / info.samplerate
    if audio_duration <= 0:
        raise ValueError("audio recording is empty")
    video_offset = float(metadata.get("video_offset_seconds", 0.))
    audio_offset = float(metadata.get("audio_offset_seconds", 0.))
    video_interval = (video_offset, video_offset + video_duration)
    audio_interval = (audio_offset, audio_offset + audio_duration)
    # Recorder-produced clock indexes take precedence over file duration when available.
    clock_spans = {}
    for name, column in (("video_frames.csv", "monotonic_seconds"), ("audio_blocks.csv", "monotonic_seconds")):
        path = session / name
        if path.is_file():
            import csv
            with path.open(newline="", encoding="utf-8-sig") as handle:
                values = [float(row[column]) for row in csv.DictReader(handle) if row.get(column)]
            if values:
                if not all(math.isfinite(value) and value >= 0 for value in values) or any(b < a for a, b in zip(values, values[1:])):
                    raise ValueError(f"invalid session clock values in {name}")
                clock_spans[name] = [values[0], values[-1]]
    if "video_frames.csv" in clock_spans and "audio_blocks.csv" in clock_spans:
        overlap_start = max(clock_spans["video_frames.csv"][0], clock_spans["audio_blocks.csv"][0])
        overlap_end = min(clock_spans["video_frames.csv"][1], clock_spans["audio_blocks.csv"][1])
        basis = "shared capture monotonic timestamps"
    else:
        overlap_start = max(video_interval[0], audio_interval[0])
        overlap_end = min(video_interval[1], audio_interval[1])
        basis = "recording.json file offsets and media durations"
    if overlap_end <= overlap_start:
        raise ValueError("video and audio do not have a positive common session-time interval")
    return {"video_file": str(video_path), "audio_file": str(audio_path),
            "video_duration_seconds": video_duration, "audio_duration_seconds": audio_duration,
            "overlap_start_seconds": overlap_start, "overlap_end_seconds": overlap_end,
            "timing_validation_basis": basis, "clock_spans": clock_spans}


def run_prerecorded_demo(session: Path, features_jsonl: Path, output: Path, *,
                         fusion_weights: Path, device: str = "cpu", sequence_length: int = 8,
                         lookback_seconds: float = 8., overwrite: bool = False) -> dict[str, Any]:
    """Run actual available branches from a paired local recording and trace each fusion step."""
    demo_started = time.perf_counter()
    session = Path(session).resolve()
    meta_path = session / "recording.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing recording metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    session_id = str(metadata.get("recording_id") or metadata.get("session_id") or "")
    if not session_id:
        raise ValueError("recording.json must define recording_id or session_id")
    timing = _validate_session_timing(session, metadata)
    features_jsonl = Path(features_jsonl).resolve()
    summary_path = features_jsonl.with_suffix(features_jsonl.suffix + ".summary.json")
    if not features_jsonl.is_file() or not summary_path.is_file():
        raise FileNotFoundError("run export-features first; both the feature JSONL and its summary are required")
    export_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if export_summary.get("session_id") != session_id:
        raise ValueError("feature file session_id does not match the paired video/audio recording")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite demo report: {output}; pass --overwrite explicitly")
    trace_path = output.with_suffix(".trace.jsonl")
    if trace_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite demo trace: {trace_path}; pass --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)

    fusion, fusion_status = _fusion_model(Path(fusion_weights), device)
    gesture_status = export_summary.get("model_status", {}).get("gesture", {"status": "unavailable", "reason": "gesture branch status missing from exporter summary"})
    if gesture_status.get("status") == "ready" and export_summary.get("valid_feature_counts", {}).get("gesture", 0) == 0:
        gesture_status = {"status": "unavailable", "reason": "gesture checkpoint loaded, but no valid gesture sequences were exported from this session"}
    pipeline = MultimodalPipeline(fusion, trained=fusion is not None, device=device,
        sequence_length=sequence_length, lookback_seconds=lookback_seconds,
        checkpoint_metadata=fusion_status,
        blocked_reason=fusion_status.get("reason", "trained fusion checkpoint unavailable"))

    counts = Counter()
    all_three = False
    infer_latencies: list[float] = []
    modality_latencies: dict[str, list[float]] = {name: [] for name in ("vision", "audio", "gesture")}
    last_predictions: dict[str, Any] = {}
    trace_count = 0
    feature_counts = {name: 0 for name in ("vision", "audio", "gesture")}

    with tempfile.TemporaryDirectory(prefix="ainak-demo-") as temp:
        db = sqlite3.connect(str(Path(temp) / "events.sqlite3"))
        db.execute("CREATE TABLE events (timestamp REAL, modality TEXT, payload TEXT)")
        batch = []
        with features_jsonl.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_feature_record(record)
                    if record["session_id"] != session_id or record["source_recording_id"] != session_id:
                        raise ValueError("recording ID mismatch")
                    counts["feature_records"] += 1
                    if not record["valid"]:
                        counts[f"invalid_{record['modality']}"] += 1
                        continue
                    feature_counts[record["modality"]] += 1
                    modality_latencies[record["modality"]].append(float(record.get("latency_ms", 0.)))
                    batch.append((float(record["timestamp_seconds"]), record["modality"], json.dumps(record, allow_nan=False)))
                    if len(batch) >= 500:
                        db.executemany("INSERT INTO events VALUES (?,?,?)", batch)
                        batch.clear()
                except Exception as exc:
                    raise ValueError(f"feature file line {line_number}: {exc}") from exc
        if batch:
            db.executemany("INSERT INTO events VALUES (?,?,?)", batch)
        db.execute("CREATE INDEX event_timeline ON events(timestamp, modality)")
        db.commit()

        def process_group(timestamp: float, group: list[dict[str, Any]], trace) -> None:
            nonlocal all_three, trace_count
            for row in group:
                modality = row["modality"]
                prediction = row["prediction"]
                if modality == "vision":
                    labels = prediction.get("detections", [])
                else:
                    labels = prediction.get("top_classes", [])
                event = ModelEvent(modality, timestamp, np_array(row["feature"]), row.get("confidence") or 0.,
                                   labels, True, float(row.get("latency_ms", 0.)), prediction)
                try:
                    pipeline.push(event)
                    last_predictions[modality] = {"timestamp_seconds": timestamp, "prediction": prediction,
                        "feature_dimensions": len(row["feature"]),
                        "model_checkpoint_id": row.get("model_checkpoint_id"),
                        "latency_ms": row.get("latency_ms", 0.)}
                except Exception as exc:
                    counts["event_push_errors"] += 1
                    counts["errors"] += 1
            stage_start = time.perf_counter()
            try:
                result = pipeline.infer(timestamp)
                inference_ms = (time.perf_counter() - stage_start) * 1000
                infer_latencies.append(inference_ms)
                if result.transformer_used:
                    all_three = all_three or set(result.contributing_modalities) == {"vision", "audio", "gesture"}
                counts["timeline_steps"] += 1
                aligned = pipeline.sync.align(timestamp)
                available = {name: event for name, event in aligned.events.items() if event is not None and event.available}
                selected_history = pipeline.sync.history(timestamp, sequence_length, lookback_seconds)
                window_inputs = []
                for window_frame in selected_history:
                    window_modalities = {}
                    for name, event in window_frame.events.items():
                        window_modalities[name] = None if event is None or not event.available else {
                            "source_timestamp_seconds": event.timestamp,
                            "feature_dimensions": int(np.asarray(event.feature).size),
                            "prediction": event.metadata,
                        }
                    window_inputs.append({"timestamp_seconds": window_frame.timestamp,
                                          "modalities": window_modalities})
                trace_row = {"timestamp_seconds": timestamp, "input_session_id": session_id,
                    "source_predictions": {name: last_predictions.get(name) for name in available},
                    "fusion_window_inputs": window_inputs,
                    "contributing_modalities": list(result.contributing_modalities),
                    "modality_validity": {name: name in result.contributing_modalities for name in ("vision", "audio", "gesture")},
                    "selected_fusion_window_timestamps": list(result.selected_window_timestamps),
                    "fusion_checkpoint_id": result.fusion_checkpoint_id,
                    "fusion_attempted": result.fusion_attempted,
                    "transformer_invocation_successful": result.transformer_used,
                    "fusion_error": result.fusion_error,
                    "transformer_used": result.transformer_used,
                    "fusion_output": ({"priority": result.priority, "confidence": result.confidence,
                        "alert_type": result.alert_type, "modality_contributions": result.modality_contributions}
                        if result.transformer_used else None),
                    "decision_engine_output": (None if result.alert is None else {
                        "text": result.alert.text, "priority": result.alert.priority,
                        "kind": result.alert.kind, "sources": list(result.alert.sources),
                        "fallback": result.alert.fallback}),
                    "stage_latency_ms": {"model_events": {row["modality"]: row.get("latency_ms", 0.) for row in group},
                                         "fusion_and_decision": inference_ms}}
                trace.write(json.dumps(trace_row, allow_nan=False) + "\n")
                trace_count += 1
            except Exception as exc:
                counts["inference_errors"] += 1
                trace.write(json.dumps({"timestamp_seconds": timestamp, "input_session_id": session_id,
                    "error": f"{type(exc).__name__}: {exc}"}, allow_nan=False) + "\n")
                trace_count += 1

        import numpy as np
        with trace_path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as trace:
            cursor = db.execute("SELECT timestamp, payload FROM events ORDER BY timestamp, modality")
            current_timestamp = None
            group: list[dict[str, Any]] = []
            for timestamp, payload in cursor:
                record = json.loads(payload)
                if current_timestamp is not None and timestamp != current_timestamp:
                    process_group(float(current_timestamp), group, trace)
                    group = []
                current_timestamp = timestamp
                group.append(record)
            if current_timestamp is not None:
                process_group(float(current_timestamp), group, trace)
            trace.flush()
        db.close()

    blockers = []
    if gesture_status.get("status") != "ready":
        blockers.append(str(gesture_status.get("reason", "gesture LSTM unavailable")))
    if fusion_status.get("status") != "ready":
        blockers.append(str(fusion_status.get("reason", "trained Transformer unavailable")))
    if not all_three:
        blockers.append("no single trained Transformer invocation received valid vision, audio, and gesture features together")
    status = "complete_trained_three_branch" if not blockers else "partial_blocked"
    transformer_invocations = pipeline.successful_invocations
    _require_full_demo_evidence(status, transformer_invocations, all_three)
    modality_latency_report = {}
    for name, values in modality_latencies.items():
        warm = values[1:]
        modality_latency_report[name] = {"cold_first_ms": values[0] if values else None,
            "after_first_observation_warmup": {"mean_ms": float(np.mean(warm)) if warm else None,
                "p95_ms": float(np.percentile(warm, 95)) if warm else None, "count": len(warm)},
            "all_observations": len(values)}
    report = {"status": status, "full_trained_multimodal_demo": not blockers,
        "session_id": session_id, "recording_timing": timing,
        "source_recordings": export_summary.get("source_recordings"),
        "model_status": {"vision": export_summary.get("model_status", {}).get("vision"),
                         "audio": export_summary.get("model_status", {}).get("audio"),
                         "gesture": gesture_status, "fusion": fusion_status},
        "valid_feature_counts": feature_counts, "feature_records_seen": counts["feature_records"],
        "timeline_steps": counts["timeline_steps"],
        "transformer_invocation_attempts": pipeline.invocation_attempts,
        "transformer_invocations": transformer_invocations,
        "transformer_failed_invocations": pipeline.failed_invocations,
        "all_three_branches_same_invocation": all_three, "trace_records": trace_count,
        "trace_jsonl": str(trace_path.resolve()),
        "latency_ms": {"fusion_and_decision_mean": float(np.mean(infer_latencies)) if infer_latencies else None,
                       "fusion_and_decision_p95": float(np.percentile(infer_latencies, 95)) if infer_latencies else None,
                       "timed_steps": len(infer_latencies), "per_modality_model_stage": modality_latency_report},
        "export_wall_seconds": export_summary.get("export_wall_seconds"),
        "demo_replay_wall_seconds": time.perf_counter() - demo_started,
        "export_plus_replay_wall_seconds": (export_summary.get("export_wall_seconds", 0.) or 0.) + (time.perf_counter() - demo_started),
        "latency_protocol": "first feature per modality reported as cold start; warmed mean/p95 exclude that first feature; replay latency covers only synchronization/fusion/decision after features are exported",
        "errors": {key: value for key, value in counts.items() if key.endswith("errors") or key == "event_push_errors"},
        "blockers": blockers,
        "note": "A partial trace is an execution diagnostic, not evidence of trained safety performance."}
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def np_array(values: list[float]):
    import numpy as np
    return np.asarray(values, dtype=np.float32)

