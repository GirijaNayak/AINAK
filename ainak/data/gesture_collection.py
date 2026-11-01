from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..models.gesture.lstm import FEATURES, GESTURE_LABELS, SEQUENCE_LENGTH, extract_pose_landmarks
from ..models.gesture.pose import create_pose_tracker


ANNOTATION_COLUMNS = ("start_seconds", "end_seconds", "label", "reviewed", "annotator", "confidence", "provenance")


def record_gesture_landmarks(output: Path, participant_code: str, duration_seconds: int = 60,
                             camera: int = 0, consent_confirmed: bool = False,
                             pose_model_path: Path | None = None) -> Path:
    """Capture pose landmarks only; raw frames/video are never written to disk."""
    if not consent_confirmed:
        raise ValueError("recording requires informed participant consent; pass --consent-confirmed after obtaining it")
    if not participant_code.strip() or len(participant_code) > 40:
        raise ValueError("participant_code must be a short anonymous code, not a name")
    if not 1 <= duration_seconds <= 300:
        raise ValueError("duration_seconds must be in 1..300")
    import cv2

    session_id = f"gesture_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    folder = output / session_id
    folder.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        folder.rmdir()
        raise RuntimeError(f"could not open camera index {camera}")
    fields = ["frame_index", "timestamp_seconds", "pose_visible", *[f"feature_{i:03}" for i in range(FEATURES)]]
    rows = 0
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        with (folder / "landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            with create_pose_tracker(model_path=pose_model_path, static_image_mode=False, model_complexity=1,
                                     min_detection_confidence=.5, min_tracking_confidence=.5) as pose:
                while time.perf_counter() - started < duration_seconds:
                    ok, frame = capture.read()
                    if not ok:
                        time.sleep(.01)
                        continue
                    timestamp = time.perf_counter()-started
                    landmarks = extract_pose_landmarks(frame, pose, int(timestamp * 1000))
                    row = {"frame_index": rows, "timestamp_seconds": f"{timestamp:.6f}",
                           "pose_visible": int(landmarks is not None)}
                    values = landmarks if landmarks is not None else np.zeros(FEATURES, dtype=np.float32)
                    row.update({f"feature_{i:03}": float(v) for i, v in enumerate(values)})
                    writer.writerow(row)
                    rows += 1
    finally:
        capture.release()
    metadata = {"session_id": session_id, "participant_code": participant_code,
                "recorded_utc": started_wall.isoformat(), "duration_seconds": duration_seconds,
                "frames": rows, "preprocessing_version": "mediapipe-pose-33-normalized-hip-center-shoulder-width-v1",
                "clock": "time.perf_counter relative to recording start", "raw_video_saved": False,
                "annotation_status": "unlabeled; review annotation_template.csv before feature extraction"}
    (folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (folder / "annotation_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
    return folder


def extract_reviewed_gesture_sequences(session: Path, output_manifest: Path, min_frames: int = 20,
                                       min_mean_visibility: float = .2,
                                       sequence_length: int = SEQUENCE_LENGTH) -> dict:
    """Extract fixed-label intervals only from explicitly human-reviewed annotations."""
    meta_path = session / "session.json"
    landmarks_path = session / "landmarks.csv"
    annotation_path = session / "annotation_template.csv"
    for path in (meta_path, landmarks_path, annotation_path):
        if not path.is_file():
            raise FileNotFoundError(f"gesture session input missing: {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with landmarks_path.open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    timestamps = np.array([float(row["timestamp_seconds"]) for row in frame_rows], dtype=np.float64)
    if not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("landmark timestamps are empty, non-finite, or out of order")
    matrix = np.array([[float(row[f"feature_{i:03}"]) for i in range(FEATURES)] for row in frame_rows], dtype=np.float32)
    if matrix.shape != (len(frame_rows), FEATURES) or not np.isfinite(matrix).all():
        raise ValueError(f"malformed landmark matrix; expected [frames,{FEATURES}] finite values")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two frames")

    output_dir = session / "sequences"
    output_dir.mkdir(exist_ok=True)
    accepted = []
    rejected = []
    with annotation_path.open(newline="", encoding="utf-8-sig") as handle:
        annotations = list(csv.DictReader(handle))
    intervals = []
    for index, annotation in enumerate(annotations):
        try:
            if str(annotation.get("reviewed", "")).strip().lower() not in {"true", "yes", "1"}:
                continue
            start, end = float(annotation["start_seconds"]), float(annotation["end_seconds"])
            if not np.isfinite(start) or not np.isfinite(end) or start >= end:
                continue
            intervals.append((index, start, end))
        except (KeyError, TypeError, ValueError):
            continue
    conflicting_indices = set()
    for position, (index, start, end) in enumerate(intervals):
        for other_index, other_start, other_end in intervals[position + 1:]:
            if max(start, other_start) < min(end, other_end):
                conflicting_indices.update((index, other_index))
    for index, annotation in enumerate(annotations):
        try:
            if index in conflicting_indices:
                raise ValueError("reviewed interval overlaps another reviewed interval; resolve ambiguity before extraction")
            if str(annotation.get("reviewed", "")).strip().lower() not in {"true", "yes", "1"}:
                raise ValueError("annotation is not marked reviewed")
            if annotation.get("label") not in GESTURE_LABELS:
                raise ValueError(f"unknown gesture label {annotation.get('label')!r}")
            start, end = float(annotation["start_seconds"]), float(annotation["end_seconds"])
            confidence = float(annotation["confidence"])
            if not (np.isfinite(start) and np.isfinite(end) and start < end and 0 <= confidence <= 1):
                raise ValueError("invalid interval or annotation confidence")
            if not annotation.get("annotator", "").strip() or not annotation.get("provenance", "").strip():
                raise ValueError("annotator and annotation provenance are required")
            selected = (timestamps >= start) & (timestamps <= end)
            sequence = matrix[selected]
            if len(sequence) < min_frames:
                raise ValueError(f"only {len(sequence)} frames in interval; need at least {min_frames}")
            # Visibility occupies each landmark's fourth coordinate.
            visibility = sequence[:, 3::4]
            mean_visibility = float(visibility.mean())
            if mean_visibility < min_mean_visibility:
                raise ValueError(f"mean landmark visibility {mean_visibility:.3f} is below threshold")
            positions = np.linspace(0, len(sequence) - 1, sequence_length)
            lower = np.floor(positions).astype(np.int64)
            upper = np.minimum(lower + 1, len(sequence) - 1)
            weight = (positions - lower)[:, None]
            sequence = ((1 - weight) * sequence[lower] + weight * sequence[upper]).astype(np.float32)
            rel_path = Path("sequences") / f"gesture_{index:04d}.npy"
            np.save(session / rel_path, sequence)
            accepted.append({"path": str((session / rel_path).resolve()), "label": annotation["label"],
                             "recording_id": metadata["session_id"], "subject_id": metadata["participant_code"],
                             "start_seconds": start, "end_seconds": end, "reviewed": True,
                             "annotation_confidence": confidence, "annotator": annotation["annotator"],
                             "provenance": annotation["provenance"], "frame_count": int(len(sequence)),
                             "mean_visibility": mean_visibility, "preprocessing_version": metadata["preprocessing_version"],
                             "synthetic": bool(metadata.get("synthetic", False))})
        except Exception as exc:
            rejected.append({"annotation_row": index + 2, "reason": str(exc)})
    if not accepted:
        raise ValueError(f"no valid reviewed gesture sequences; rejected {len(rejected)} annotation rows")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(accepted[0])
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(accepted)
    summary = {"session_id": metadata["session_id"], "participant_code": metadata["participant_code"],
               "accepted_sequences": len(accepted), "rejected_annotations": rejected,
               "class_counts": {label: sum(row["label"] == label for row in accepted) for label in GESTURE_LABELS},
               "manifest": str(output_manifest), "preprocessing_version": metadata["preprocessing_version"],
               "sequence_length": sequence_length}
    output_manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def import_gesture_video(video: Path, output: Path, participant_code: str, *,
                         consent_confirmed: bool = False, pose_model_path: Path | None = None) -> Path:
    """Import a phone/other-device video and locally extract timestamped pose landmarks."""
    if not consent_confirmed:
        raise ValueError("video processing requires informed participant consent; pass --consent-confirmed after obtaining it")
    if not participant_code.strip() or len(participant_code) > 40:
        raise ValueError("participant_code must be a short anonymous code, not a name")
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"gesture source video does not exist: {video}")
    import cv2

    session_id = f"gesture_import_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    folder = Path(output) / session_id
    folder.mkdir(parents=True, exist_ok=False)
    local_video = folder / ("source_video" + (video.suffix.lower() or ".mp4"))
    shutil.copy2(video, local_video)
    digest = hashlib.sha256()
    with local_video.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    capture = cv2.VideoCapture(str(local_video))
    if not capture.isOpened():
        shutil.rmtree(folder, ignore_errors=True)
        raise ValueError(f"cannot decode imported gesture video: {local_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.
    rows = 0
    started_wall = datetime.now(timezone.utc)
    last_timestamp = -1.
    fields = ["frame_index", "timestamp_seconds", "pose_visible", *[f"feature_{i:03}" for i in range(FEATURES)]]
    landmarks_path = folder / "landmarks.csv"
    try:
        with landmarks_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            with create_pose_tracker(model_path=pose_model_path, static_image_mode=False, model_complexity=1,
                                     min_detection_confidence=.5, min_tracking_confidence=.5) as pose:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
                    if not np.isfinite(timestamp) or timestamp < 0 or (rows and timestamp <= last_timestamp):
                        timestamp = rows / fps
                    last_timestamp = timestamp
                    landmarks = extract_pose_landmarks(frame, pose, int(timestamp * 1000))
                    row = {"frame_index": rows, "timestamp_seconds": f"{timestamp:.6f}",
                           "pose_visible": int(landmarks is not None)}
                    values = landmarks if landmarks is not None else np.zeros(FEATURES, dtype=np.float32)
                    row.update({f"feature_{i:03}": float(v) for i, v in enumerate(values)})
                    writer.writerow(row)
                    rows += 1
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        capture.release()
    if rows == 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise ValueError("imported gesture video contains no decodable frames")
    metadata = {"session_id": session_id, "participant_code": participant_code,
        "recorded_utc": started_wall.isoformat(), "duration_seconds": rows / fps, "frames": rows,
        "preprocessing_version": "mediapipe-pose-33-normalized-hip-center-shoulder-width-v1",
        "clock": "video media seconds from file start", "raw_video_saved": True,
        "raw_video_file": local_video.name, "video_sha256": "sha256:" + digest.hexdigest(),
        "annotation_status": "unlabeled; review annotation_template.csv before feature extraction"}
    (folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (folder / "annotation_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
    return folder


def delete_gesture_session(session: Path, expected_session_id: str,
                           manifests: list[Path] | None = None) -> dict:
    """Delete one confirmed local gesture session and remove its rows from named manifests."""
    requested_path = Path(session)
    if requested_path.is_symlink():
        raise ValueError("refusing to delete a symbolic-link gesture session")
    session = requested_path.resolve()
    if not session.is_dir():
        raise ValueError("session must be an existing, non-symlink directory")
    metadata_path = session / "session.json"
    if not metadata_path.is_file():
        raise ValueError("refusing to delete a directory without gesture session metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session_id = str(metadata.get("session_id", ""))
    if not session_id or expected_session_id != session_id:
        raise ValueError("confirmation ID does not exactly match the gesture session ID")
    manifest_changes = []
    for manifest in manifests or []:
        manifest = Path(manifest).resolve()
        if manifest == session or session in manifest.parents:
            continue
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest}")
        temp_path = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.erase-tmp")
        removed = 0
        try:
            if manifest.suffix.lower() == ".jsonl":
                with manifest.open(encoding="utf-8") as source, temp_path.open("w", encoding="utf-8", newline="\n") as target:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        source_ids = set(row.get("source_recording_ids") or [])
                        identifiers = {str(row.get(key, "")) for key in ("recording_id", "session_id")}
                        raw_path = str(row.get("source_path", ""))
                        path_in_session = False
                        if raw_path:
                            candidate = (manifest.parent / raw_path).resolve()
                            path_in_session = candidate == session or session in candidate.parents
                        if session_id in identifiers or session_id in source_ids or path_in_session:
                            removed += 1
                        else:
                            target.write(line if line.endswith("\n") else line + "\n")
            else:
                with manifest.open(newline="", encoding="utf-8-sig") as source:
                    reader = csv.DictReader(source)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
                with temp_path.open("w", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        identifiers = {str(row.get(key, "")) for key in ("recording_id", "session_id")}
                        raw_path = str(row.get("path", ""))
                        path_in_session = False
                        if raw_path:
                            candidate = (manifest.parent / raw_path).resolve()
                            path_in_session = candidate == session or session in candidate.parents
                        if session_id in identifiers or path_in_session:
                            removed += 1
                        else:
                            writer.writerow(row)
            try:
                temp_path.replace(manifest)
            except PermissionError:
                # Some Windows sandboxed filesystems deny rename of a temporary
                # file even when the target directory is writable.
                with temp_path.open("rb") as source, manifest.open("wb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                temp_path.unlink()
            summary = manifest.with_suffix(".summary.json")
            summary_removed = summary.is_file()
            if summary.is_file():
                summary.unlink()
            manifest_changes.append({"manifest": str(manifest), "rows_removed": removed,
                                     "summary_removed": summary_removed})
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    shutil.rmtree(session)
    return {"deleted_session_id": session_id, "deleted_session_path": str(session),
            "removed_manifests": manifest_changes,
            "note": "source video, landmarks, annotations, and session-local sequences were removed"}
