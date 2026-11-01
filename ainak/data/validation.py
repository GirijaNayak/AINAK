from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any
from .schemas import AlignedExample, ModalityObservation


def parse_example(row: dict[str, Any]) -> AlignedExample:
    def obs(name: str):
        raw = row.get(name)
        if raw is None: return None
        values=dict(raw)
        stored_modality=values.pop("modality",name)
        if stored_modality != name: raise ValueError(f"{name} observation is tagged as {stored_modality!r}")
        return ModalityObservation(modality=name, **values)
    example = AlignedExample(example_id=str(row["example_id"]), timestamp=float(row["timestamp"]),
        vision=obs("vision"), audio=obs("audio"), gesture=obs("gesture"), priority=float(row["priority"]), confidence=float(row["confidence"]),
        alert_type=row["alert_type"], recording_id=str(row.get("recording_id", "")),
        schema_version=row.get("schema_version"), session_id=row.get("session_id"),
        source_recording_ids=row.get("source_recording_ids"), modality_validity=row.get("modality_validity"),
        annotation_provenance=row.get("annotation_provenance"), target_source=row.get("target_source"),
        event_label=row.get("event_label"), participant_id=row.get("participant_id"))
    example.validate()
    return example


def validate_jsonl(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    seen_ids: set[str] = set()
    last_by_recording: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example=parse_example(json.loads(line))
                if example.example_id in seen_ids:
                    raise ValueError(f"duplicate example_id {example.example_id!r}")
                seen_ids.add(example.example_id)
                if example.recording_id:
                    previous=last_by_recording.get(example.recording_id)
                    if previous is not None and example.timestamp <= previous:
                        raise ValueError(f"timestamps for recording {example.recording_id!r} must be strictly increasing")
                    last_by_recording[example.recording_id]=example.timestamp
                count += 1
            except Exception as exc:
                errors.append(f"line {line_number}: {exc}")
    if count == 0 and not errors:
        errors.append("manifest contains no examples")
    return count, errors


def validate_recording_session(folder: Path) -> list[str]:
    """Check synchronized local capture indexes and any human annotations."""
    from ..models.gesture.lstm import GESTURE_LABELS
    from .schemas import ALERT_TYPES

    errors: list[str] = []
    metadata_path = folder / "recording.json"
    if not metadata_path.is_file():
        return [f"missing recording metadata: {metadata_path}"]
    try:
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        duration=float(metadata["duration_seconds"])
        sample_rate=int(metadata["audio_sample_rate"])
        total_samples=int(metadata["audio_samples"])
        if not math.isfinite(duration) or duration <= 0 or sample_rate <= 0 or total_samples < 0:
            raise ValueError("invalid duration, sample rate, or sample count")
    except Exception as exc:
        return [f"invalid recording metadata: {exc}"]
    required_session_files=[metadata.get("video_file", ""),metadata.get("audio_file", ""),metadata.get("annotation_template", "annotation_template.csv")]
    for index_name in ("video_frames.csv","audio_blocks.csv"):
        if (folder/index_name).exists() or metadata.get(index_name.replace(".csv","_index")):
            required_session_files.append(metadata.get(index_name.replace(".csv","_index"),index_name))
    for key in required_session_files:
        target=(folder/key).resolve()
        if folder.resolve() not in target.parents or not target.is_file():
            errors.append(f"missing or unsafe session file: {key}")
        elif target.stat().st_size == 0:
            errors.append(f"empty session file: {key}")
    for name, columns in (("video_frames.csv", ("frame_index", "monotonic_seconds")),
                          ("audio_blocks.csv", ("sample_start", "frames", "monotonic_seconds"))):
        path=folder/name
        if not path.is_file(): continue
        try:
            with path.open(newline="",encoding="utf-8-sig") as handle:
                reader=csv.DictReader(handle)
                if not set(columns).issubset(reader.fieldnames or []):
                    errors.append(f"{name} missing required columns {columns}"); continue
                rows=list(reader)
            previous=-1.0
            for row_number,row in enumerate(rows,2):
                ts=float(row["monotonic_seconds"])
                if not math.isfinite(ts) or ts < 0 or ts > duration+1 or ts < previous:
                    errors.append(f"{name} row {row_number}: invalid or out-of-order timestamp")
                previous=ts
                if name=="video_frames.csv": int(row["frame_index"])
                else:
                    start=int(row["sample_start"]); count=int(row["frames"] or 0)
                    if count>0 and (start<0 or start+count>total_samples): errors.append(f"{name} row {row_number}: audio sample interval outside captured data")
        except Exception as exc:
            errors.append(f"{name} parse error: {exc}")
    annotation=folder/metadata.get("annotation_template","annotation_template.csv")
    if annotation.is_file():
        try:
            with annotation.open(newline="",encoding="utf-8-sig") as handle:
                reader=csv.DictReader(handle)
                required={"start_seconds","end_seconds","event_label","priority","alert_type","annotation_confidence","annotator","provenance","reviewed"}
                if not required.issubset(reader.fieldnames or []): errors.append(f"annotation template missing columns {sorted(required-set(reader.fieldnames or []))}")
                else:
                    for row_number,row in enumerate(reader,2):
                        if not any(str(value or "").strip() for value in row.values()): continue
                        start,end=float(row["start_seconds"]),float(row["end_seconds"])
                        priority,confidence=float(row["priority"]),float(row["annotation_confidence"])
                        if not all(map(math.isfinite,(start,end,priority,confidence))) or start<0 or end<=start or end>duration or not 0<=priority<=1 or not 0<=confidence<=1:
                            errors.append(f"annotation row {row_number}: invalid interval, priority, or confidence")
                        if row["alert_type"] not in ALERT_TYPES: errors.append(f"annotation row {row_number}: unknown alert_type")
                        if row.get("gesture_label") and row["gesture_label"] not in GESTURE_LABELS: errors.append(f"annotation row {row_number}: unknown gesture_label")
                        if not row["event_label"].strip() or not row["annotator"].strip() or not row["provenance"].strip(): errors.append(f"annotation row {row_number}: event label, annotator, and provenance are required")
                        if row.get("reviewed", "").strip().lower() not in {"true", "yes", "1"}: errors.append(f"annotation row {row_number}: label must be explicitly marked reviewed")
        except Exception as exc:
            errors.append(f"annotation template parse error: {exc}")
    return errors
