"""Versioned JSONL contract for provenance-preserving model features."""
from __future__ import annotations

import math
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import FEATURE_DIMS

FEATURE_SCHEMA_VERSION = "ainak.feature.v1"
PREPROCESSING_VERSIONS = {
    "vision": "yolov8-detection-summary-128-v1",
    "audio": "panns-cnn14-32khz-mono-v1",
    "gesture": "mediapipe-pose-lstm-v1",
}


def validate_feature_record(row: dict[str, Any]) -> None:
    if row.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {FEATURE_SCHEMA_VERSION!r}")
    for key in ("feature_id", "session_id", "source_recording_id", "modality", "feature_kind",
                "model_checkpoint_id", "preprocessing_version"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    modality = row["modality"]
    if modality not in FEATURE_DIMS:
        raise ValueError(f"unsupported modality {modality!r}")
    start = row.get("interval_start_seconds")
    end = row.get("interval_end_seconds")
    timestamp = row.get("timestamp_seconds")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end, timestamp)):
        raise ValueError("timestamps must be finite numeric seconds")
    if start < 0 or end < start or not start <= timestamp <= end:
        raise ValueError("invalid or inconsistent feature time interval")
    if not isinstance(row.get("prediction"), dict):
        raise ValueError("prediction must be an object")
    if not isinstance(row.get("valid"), bool):
        raise ValueError("valid must be a boolean")
    latency = row.get("latency_ms", 0.)
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError("latency_ms must be finite and nonnegative")
    feature = row.get("feature")
    if row["valid"]:
        if not isinstance(feature, list) or len(feature) != FEATURE_DIMS[modality]:
            raise ValueError(f"valid {modality} feature must have {FEATURE_DIMS[modality]} values")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in feature):
            raise ValueError("feature values must be finite numbers")
        if row.get("missing_reason"):
            raise ValueError("valid features cannot have a missing_reason")
        if modality == "vision" and any(not 0 <= value <= 1 for value in feature):
            raise ValueError("vision detection-summary values must be normalized to [0,1]")
    elif feature is not None or not isinstance(row.get("missing_reason"), str) or not row["missing_reason"].strip():
        raise ValueError("invalid features require feature=null and a missing_reason")
    confidence = row.get("confidence")
    if confidence is not None and (not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("confidence must be finite and in [0,1]")
    for name, value in row["prediction"].get("probabilities", {}).items():
        if not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("prediction probabilities must be finite and in [0,1]")
    if modality == "gesture":
        from ..models.gesture.lstm import GESTURE_LABELS
        unknown = set(row["prediction"].get("probabilities", {})) - set(GESTURE_LABELS)
        if unknown:
            raise ValueError(f"gesture checkpoint returned labels outside the configured map: {sorted(unknown)}")
        top_class = row["prediction"].get("top_class")
        if top_class is not None and top_class not in GESTURE_LABELS:
            raise ValueError(f"unknown gesture class {top_class!r}")
    if modality == "vision":
        for detection in row["prediction"].get("detections", []):
            box = detection.get("bbox_xyxy")
            if not isinstance(detection.get("class"), str) or not detection["class"].strip():
                raise ValueError("vision detections require a class name")
            if not isinstance(detection.get("confidence"), (int, float)) or not math.isfinite(detection["confidence"]) or not 0 <= detection["confidence"] <= 1:
                raise ValueError("vision detection confidence must be in [0,1]")
            if not isinstance(box, list) or len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                raise ValueError("vision detections require four finite xyxy coordinates")
            if box[2] < box[0] or box[3] < box[1]:
                raise ValueError("vision detection box coordinates must be ordered x1,y1,x2,y2")


def validate_feature_jsonl(path: Path) -> tuple[dict[str, Any], list[str]]:
    counts = Counter()
    sessions: set[str] = set()
    checkpoints: dict[str, set[str]] = {name: set() for name in FEATURE_DIMS}
    ids: set[str] = set()
    last_timestamp: dict[tuple[str, str], float] = {}
    errors: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_feature_record(row)
                if row["feature_id"] in ids:
                    raise ValueError(f"duplicate feature_id {row['feature_id']!r}")
                ids.add(row["feature_id"])
                key = (row["session_id"], row["modality"])
                previous = last_timestamp.get(key)
                if previous is not None and float(row["timestamp_seconds"]) < previous:
                    raise ValueError("timestamps must be nondecreasing per session and modality")
                last_timestamp[key] = float(row["timestamp_seconds"])
                sessions.add(row["session_id"])
                counts[f"{row['modality']}_{'valid' if row['valid'] else 'invalid'}"] += 1
                checkpoints[row["modality"]].add(row["model_checkpoint_id"])
                counts["records"] += 1
            except Exception as exc:
                errors.append(f"line {line_number}: {exc}")
    report = {"schema_version": FEATURE_SCHEMA_VERSION, "records": counts["records"],
        "sessions": sorted(sessions), "counts": dict(counts),
        "checkpoint_ids": {name: sorted(values) for name, values in checkpoints.items()},
        "valid": not errors and counts["records"] > 0}
    if counts["records"] == 0 and not errors:
        errors.append("feature file contains no records")
        report["valid"] = False
    return report, errors

