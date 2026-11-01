from __future__ import annotations

import csv
import json
import math
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .features import validate_feature_record
from .schemas import ALERT_TYPES, FEATURE_DIMS
from .validation import parse_example


def _empty_annotation(row: dict[str, str]) -> bool:
    return not any(str(value or "").strip() for value in row.values())


def _sample_anchors(start: float, end: float, window_seconds: float, stride_seconds: float) -> list[float]:
    if end - start <= window_seconds:
        return [end]
    anchors = []
    current = start + window_seconds
    while current <= end + 1e-9:
        anchors.append(min(current, end))
        current += stride_seconds
    if not anchors or end - anchors[-1] > stride_seconds * .5:
        anchors.append(end)
    return anchors


def build_aligned_dataset(session: Path, features_jsonl: Path, output: Path, *,
                          window_seconds: float = 2., stride_seconds: float = 1.,
                          max_feature_age_seconds: float = 1.5, overwrite: bool = False) -> dict[str, Any]:
    """Join model features to explicitly reviewed annotation intervals by timestamps."""
    session = Path(session).resolve()
    meta_path = session / "recording.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing session metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    session_id = str(metadata.get("recording_id") or metadata.get("session_id") or "")
    if not session_id:
        raise ValueError("recording.json must define recording_id or session_id")
    annotation_name = str(metadata.get("annotation_template", "annotation_template.csv"))
    annotation_path = (session / annotation_name).resolve()
    if session not in annotation_path.parents or not annotation_path.is_file():
        raise ValueError(f"missing or unsafe annotation file in session: {annotation_name}")
    feature_path = Path(features_jsonl).resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"feature JSONL does not exist: {feature_path}")
    if window_seconds <= 0 or stride_seconds <= 0 or max_feature_age_seconds <= 0:
        raise ValueError("window, stride, and maximum feature age must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite aligned manifest: {output}; pass --overwrite explicitly")
    report_path = output.with_suffix(output.suffix + ".summary.json")
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite alignment summary: {report_path}; pass --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)

    with annotation_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"start_seconds", "end_seconds", "event_label", "priority", "alert_type",
                    "annotation_confidence", "annotator", "provenance", "reviewed"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"annotation file must contain reviewed human-label columns: {sorted(required)}")
        annotations = [row for row in reader if not _empty_annotation(row)]
    if not annotations:
        raise ValueError("no human annotations found; review annotation_template.csv before aligning training data")

    prepared: list[dict[str, Any]] = []
    for row_number, row in enumerate(annotations, 2):
        if row.get("reviewed", "").strip().lower() not in {"true", "yes", "1"}:
            raise ValueError(f"annotation row {row_number} is not explicitly marked reviewed")
        start, end = float(row["start_seconds"]), float(row["end_seconds"])
        priority = float(row["priority"])
        confidence = float(row["annotation_confidence"])
        if not all(math.isfinite(x) for x in (start, end, priority, confidence)):
            raise ValueError(f"annotation row {row_number} contains non-finite values")
        if start < 0 or end <= start or not 0 <= priority <= 1 or not 0 <= confidence <= 1:
            raise ValueError(f"annotation row {row_number} contains an invalid interval, priority, or confidence")
        if row["alert_type"] not in ALERT_TYPES or not row["event_label"].strip():
            raise ValueError(f"annotation row {row_number} has an invalid event label or alert_type")
        if not row["annotator"].strip() or not row["provenance"].strip():
            raise ValueError(f"annotation row {row_number} must identify annotator and review provenance")
        prepared.append({"row_number": row_number, "start": start, "end": end, "priority": priority,
                         "confidence": confidence, "alert_type": row["alert_type"],
                         "event_label": row["event_label"].strip(), "annotator": row["annotator"].strip(),
                         "provenance": row["provenance"].strip()})
    prepared.sort(key=lambda item: item["start"])
    for earlier, later in zip(prepared, prepared[1:]):
        if later["start"] < earlier["end"]:
            raise ValueError(f"reviewed annotation intervals overlap: rows {earlier['row_number']} and {later['row_number']}; resolve them before dataset generation")

    counts = Counter()
    invalid_features = Counter()
    output_count = 0
    modality_counts = Counter()
    class_counts = Counter()
    with tempfile.TemporaryDirectory(prefix="ainak-align-") as temp:
        db = sqlite3.connect(str(Path(temp) / "feature-index.sqlite3"))
        db.execute("CREATE TABLE features (modality TEXT, timestamp REAL, start REAL, end REAL, valid INTEGER, payload TEXT)")
        batch = []
        with feature_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_feature_record(record)
                    if record["session_id"] != session_id or record["source_recording_id"] != session_id:
                        raise ValueError("feature session/source recording ID does not match the session metadata")
                    batch.append((record["modality"], float(record["timestamp_seconds"]),
                                  float(record["interval_start_seconds"]), float(record["interval_end_seconds"]),
                                  int(record["valid"]), json.dumps(record, allow_nan=False)))
                    if not record["valid"]:
                        invalid_features[record["modality"]] += 1
                    counts[record["modality"]] += 1
                    if len(batch) >= 500:
                        db.executemany("INSERT INTO features VALUES (?,?,?,?,?,?)", batch)
                        batch.clear()
                except Exception as exc:
                    db.close()
                    raise ValueError(f"feature file line {line_number}: {exc}") from exc
        if batch:
            db.executemany("INSERT INTO features VALUES (?,?,?,?,?,?)", batch)
        db.execute("CREATE INDEX feature_lookup ON features(modality, timestamp)")
        db.commit()

        with output.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as sink:
            for annotation in prepared:
                for sample_index, anchor in enumerate(_sample_anchors(annotation["start"], annotation["end"], window_seconds, stride_seconds)):
                    window_start = max(annotation["start"], anchor - window_seconds)
                    observations: dict[str, dict[str, Any] | None] = {name: None for name in FEATURE_DIMS}
                    for modality in FEATURE_DIMS:
                        found = db.execute(
                            "SELECT timestamp, start, end, valid, payload FROM features "
                            "WHERE modality=? AND timestamp<=? AND timestamp>=? ORDER BY timestamp DESC LIMIT 1",
                            (modality, anchor, anchor - max_feature_age_seconds)).fetchone()
                        if found is None:
                            continue
                        _, feature_start, feature_end, is_valid, payload = found
                        if feature_end < window_start or feature_start > anchor:
                            continue
                        if not is_valid:
                            continue
                        record = json.loads(payload)
                        observations[modality] = {
                            "modality": modality, "timestamp": record["timestamp_seconds"],
                            "feature": record["feature"], "labels": record["prediction"].get("top_classes") or record["prediction"].get("detections"),
                            "confidence": record.get("confidence"),
                            "interval_start_seconds": record["interval_start_seconds"],
                            "interval_end_seconds": record["interval_end_seconds"],
                            "feature_kind": record["feature_kind"],
                            "model_checkpoint_id": record["model_checkpoint_id"],
                            "preprocessing_version": record["preprocessing_version"],
                        }
                        modality_counts[modality] += 1
                    if not any(observations.values()):
                        continue
                    example_id = f"{session_id}:{annotation['row_number']}:{sample_index}"
                    aligned = {
                        "schema_version": "ainak.aligned-example.v1", "example_id": example_id,
                        "recording_id": session_id, "session_id": session_id,
                        "participant_id": metadata.get("participant_code"),
                        "source_recording_ids": {name: session_id for name in FEATURE_DIMS},
                        "timestamp": anchor, "vision": observations["vision"],
                        "audio": observations["audio"], "gesture": observations["gesture"],
                        "modality_validity": {name: observations[name] is not None for name in FEATURE_DIMS},
                        "priority": annotation["priority"], "confidence": annotation["confidence"],
                        "alert_type": annotation["alert_type"], "event_label": annotation["event_label"],
                        "target_source": "human_reviewed",
                        "annotation_provenance": {"reviewed": True, "annotation_row": annotation["row_number"],
                            "annotator": annotation["annotator"], "provenance": annotation["provenance"],
                            "interval_start_seconds": annotation["start"], "interval_end_seconds": annotation["end"]},
                    }
                    # Run the same parser used by training before writing anything.
                    try:
                        parse_example(aligned)
                    except Exception:
                        db.close()
                        raise
                    sink.write(json.dumps(aligned, allow_nan=False) + "\n")
                    output_count += 1
                    class_counts[annotation["alert_type"]] += 1
        db.close()

    if output_count == 0:
        raise ValueError("no aligned examples could be built; check feature coverage, session clocks, and annotation intervals")
    report = {"schema_version": "ainak.aligned-example.v1", "session_id": session_id,
        "feature_records": dict(counts), "invalid_features": dict(invalid_features),
        "aligned_examples": output_count, "examples_with_valid_modality": dict(modality_counts),
        "alert_type_distribution": dict(class_counts),
        "annotations_used": len(prepared), "label_provenance": "explicitly reviewed human annotation intervals",
        "alignment": {"window_seconds": window_seconds, "stride_seconds": stride_seconds,
                      "maximum_feature_age_seconds": max_feature_age_seconds,
                      "timestamps_used": "session-relative seconds; interval overlap and staleness checked"},
        "output": str(output.resolve())}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

