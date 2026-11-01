from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import platform
import hashlib
import json
import os
import shutil
import urllib.request
import uuid
import zipfile

import numpy as np

POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def acquire_pose_model(output: Path, *, overwrite: bool = False, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Download the documented Google-hosted Lite bundle and record its local hash."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite pose model: {output}; pass --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(POSE_LANDMARKER_LITE_URL,
            headers={"User-Agent": "AI-AINAK-local-research-prototype/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as target:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"pose model exceeded configured download limit {max_bytes} bytes")
                digest.update(block)
                target.write(block)
            target.flush()
        if size == 0 or not zipfile.is_zipfile(temporary):
            raise ValueError("downloaded Pose Landmarker bundle is empty or is not a readable task archive")
        with zipfile.ZipFile(temporary) as bundle:
            members = bundle.namelist()
            if not any(name.lower().endswith(".tflite") for name in members):
                raise ValueError("Pose Landmarker archive contains no TFLite model")
        try:
            temporary.replace(output)
        except PermissionError:
            with temporary.open("rb") as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            temporary.unlink()
        report = {"path": str(output.resolve()), "url": POSE_LANDMARKER_LITE_URL,
                  "bytes": size, "sha256": "sha256:" + digest.hexdigest(),
                  "bundle_valid": True, "bundle_members": members,
                  "published_checksum_available": False,
                  "license_review": "review the Google MediaPipe model-bundle terms before distribution"}
        output.with_suffix(output.suffix + ".source.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def pose_readiness(model_path: Path | None = None, *, verify_source: Path | None = None) -> dict[str, Any]:
    """Report API/model readiness, optionally verifying extraction on a real image/video frame."""
    try:
        import mediapipe as mp
    except Exception as exc:
        return {"installed": False, "version": None, "selected_api": "none",
                "model_bundle": {"path": str(model_path) if model_path else None, "available": False},
                "extraction_ready": False, "pose_extraction_verified": False,
                "error": f"{type(exc).__name__}: {exc}"}
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        api = "legacy_solutions_pose"
        model_available = True
        model_required = False
    elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        api = "tasks_pose_landmarker"
        model_required = True
        model_available = bool(model_path and Path(model_path).is_file() and Path(model_path).stat().st_size > 0)
    else:
        api = "unsupported"
        model_required = True
        model_available = False

    verified = False
    verified_details = None
    if verify_source and Path(verify_source).exists() and (api == "tasks_pose_landmarker" and model_available):
        import cv2
        tracker = TasksPoseAdapter(Path(model_path), static_image_mode=True)
        src_p = Path(verify_source)
        frame = None
        if src_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            frame = cv2.imread(str(src_p))
        elif src_p.suffix.lower() in [".mp4", ".avi", ".mov"]:
            cap = cv2.VideoCapture(str(src_p))
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = tracker.process(rgb)
            if res.pose_landmarks and hasattr(res.pose_landmarks, "landmark"):
                verified = True
                verified_details = {
                    "source": str(src_p),
                    "landmarks_extracted": len(res.pose_landmarks.landmark),
                    "visibility_mean": float(round(float(np.mean([getattr(lm, "visibility", 1.0) for lm in res.pose_landmarks.landmark])), 4)),
                }
        tracker.close()

    note = "Real landmark extraction verified on source media." if verified else "Readiness is dependency/model inspection only; process a local video to verify extraction."
    return {"installed": True, "version": getattr(mp, "__version__", "unknown"),
        "python_version": platform.python_version(), "python_architecture": platform.architecture()[0],
        "selected_api": api,
        "model_bundle": {"path": str(Path(model_path).resolve()) if model_path else None,
                         "required": model_required, "available": model_available,
                         "bytes": Path(model_path).stat().st_size if model_available and model_path else None},
        "extraction_ready": api == "legacy_solutions_pose" or (api == "tasks_pose_landmarker" and model_available),
        "pose_extraction_verified": verified,
        "verification_details": verified_details,
        "verification_note": note}


class TasksPoseAdapter:
    """Small compatibility wrapper for MediaPipe's current PoseLandmarker Tasks API."""
    def __init__(self, model_path: Path, *, static_image_mode: bool = False,
                 min_detection_confidence: float = .5, min_tracking_confidence: float = .5):
        import mediapipe as mp
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe Pose Landmarker task model not found: {model_path}")
        mode = mp.tasks.vision.RunningMode.IMAGE if static_image_mode else mp.tasks.vision.RunningMode.VIDEO
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode, num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._mode = mode
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int | None = None):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb, dtype=np.uint8))
        if self._mode == self._mp.tasks.vision.RunningMode.IMAGE:
            result = self._landmarker.detect(image)
        else:
            stamp = self._last_timestamp_ms + 1 if timestamp_ms is None else max(self._last_timestamp_ms + 1, int(timestamp_ms))
            self._last_timestamp_ms = stamp
            result = self._landmarker.detect_for_video(image, stamp)
        if not result.pose_landmarks:
            return SimpleNamespace(pose_landmarks=None)
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=result.pose_landmarks[0]))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_pose_tracker(*, model_path: Path | None = None, static_image_mode: bool = False,
                        model_complexity: int = 1, min_detection_confidence: float = .5,
                        min_tracking_confidence: float = .5) -> Any:
    """Use legacy `solutions.pose` where available, otherwise the Tasks API plus a `.task` model."""
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        return mp.solutions.pose.Pose(static_image_mode=static_image_mode,
            model_complexity=model_complexity, min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
    if model_path is None:
        raise RuntimeError(
            f"MediaPipe {mp.__version__} no longer exposes mp.solutions.pose; "
            "install the project's pinned MediaPipe version below 0.10.31, or provide a local "
            "Pose Landmarker .task model with --pose-model (no model is downloaded automatically)."
        )
    return TasksPoseAdapter(Path(model_path), static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

