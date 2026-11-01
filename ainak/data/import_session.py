from __future__ import annotations

import csv
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import soundfile as sf


ANNOTATION_FIELDS = ("start_seconds", "end_seconds", "event_label", "priority", "alert_type",
    "annotation_confidence", "reviewed", "gesture_label", "annotator", "provenance",
    "vision_notes", "audio_notes", "gesture_notes")


def import_prerecorded_session(video: Path, audio: Path, output_root: Path, recording_id: str, *,
                               video_offset_seconds: float = 0., audio_offset_seconds: float = 0.,
                               annotations: Path | None = None, participant_code: str | None = None) -> Path:
    """Copy a user-paired video/audio recording into an explicit session folder."""
    video, audio, output_root = Path(video).resolve(), Path(audio).resolve(), Path(output_root).resolve()
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("both source video and audio files must exist")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", recording_id) or recording_id in {".", ".."}:
        raise ValueError("recording_id must be a safe 1–80 character session identifier")
    if not all(math.isfinite(value) and value >= 0 for value in (video_offset_seconds, audio_offset_seconds)):
        raise ValueError("file offsets must be finite and nonnegative seconds")
    if annotations is not None and not Path(annotations).is_file():
        raise FileNotFoundError(f"annotation file does not exist: {annotations}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot decode video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)); frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError("source video does not report a usable frame count and FPS")
    video_duration = frame_count / fps
    audio_info = sf.info(str(audio))
    if audio_info.frames <= 0 or audio_info.samplerate <= 0:
        raise ValueError("source audio is empty or has an invalid sample rate")
    audio_duration = audio_info.frames / audio_info.samplerate
    if min(video_offset_seconds + video_duration, audio_offset_seconds + audio_duration) <= max(video_offset_seconds, audio_offset_seconds):
        raise ValueError("source media and declared offsets do not overlap in the common session clock")

    folder = output_root / recording_id
    folder.mkdir(parents=True, exist_ok=False)
    video_name = "video" + (video.suffix.lower() or ".mp4")
    audio_name = "audio" + (audio.suffix.lower() or ".wav")
    shutil.copy2(video, folder / video_name)
    shutil.copy2(audio, folder / audio_name)
    annotation_name = "annotation_template.csv"
    if annotations is not None:
        shutil.copy2(Path(annotations), folder / annotation_name)
    else:
        with (folder / annotation_name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
    metadata = {"recording_id": recording_id, "session_id": recording_id,
        "created_utc": datetime.now(timezone.utc).isoformat(), "duration_seconds": max(
            video_offset_seconds + video_duration, audio_offset_seconds + audio_duration),
        "audio_sample_rate": int(audio_info.samplerate), "audio_samples": int(audio_info.frames),
        "video_frames": frame_count, "video_fps_header": fps, "video_file": video_name,
        "audio_file": audio_name, "annotation_template": annotation_name,
        "video_offset_seconds": float(video_offset_seconds), "audio_offset_seconds": float(audio_offset_seconds),
        "clock": "user-declared file-start offsets in common session-relative seconds",
        "pairing_assertion": "source video and audio were paired by the importer user; this metadata does not independently verify provenance",
        "source_media": {"video_filename": video.name, "audio_filename": audio.name},
        "annotations": "human review required; importer creates no labels"}
    if participant_code:
        metadata["participant_code"] = participant_code
    (folder / "recording.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder

