"""Real-model integration gate; skips with explicit prerequisites when data is absent."""
import csv
import json
from pathlib import Path

import pytest

from ainak.config import Paths
from ainak.data.feature_export import export_session_features
from ainak.models.gesture.lstm import GestureRecognizer
from ainak.models.gesture.pose import pose_readiness
from ainak.runtime.prerecorded_demo import _fusion_model, run_prerecorded_demo


def _reviewed_recording(session: Path) -> bool:
    try:
        metadata = json.loads((session / "recording.json").read_text(encoding="utf-8"))
        if metadata.get("synthetic") is True:
            return False
        annotation = session / metadata.get("annotation_template", "annotation_template.csv")
        if not annotation.is_file():
            return False
        with annotation.open(newline="", encoding="utf-8-sig") as handle:
            return any(str(row.get("reviewed", "")).strip().lower() in {"true", "yes", "1"}
                       for row in csv.DictReader(handle))
    except Exception:
        return False


def test_real_three_branch_replay_invokes_one_trained_transformer(tmp_path):
    paths = Paths()
    missing = []
    if not paths.gesture_weights.is_file():
        missing.append("trained gesture checkpoint absent")
    else:
        try:
            GestureRecognizer(paths.gesture_weights, device="cpu")
        except Exception as exc:
            missing.append(f"gesture checkpoint rejected: {type(exc).__name__}: {exc}")
    fusion, fusion_status = _fusion_model(paths.fusion_weights, "cpu")
    if fusion is None:
        missing.append(f"fusion checkpoint unavailable: {fusion_status.get('reason')}")
    pose_model = paths.root / "artifacts" / "gesture" / "pose_landmarker_lite.task"
    readiness = pose_readiness(pose_model)
    if not readiness["extraction_ready"]:
        missing.append("MediaPipe pose extraction is not ready: " + readiness.get("selected_api", "unknown"))
    sessions = [folder for folder in (paths.root / "datasets" / "recordings").glob("*")
                if folder.is_dir() and _reviewed_recording(folder)]
    if not sessions:
        missing.append("no paired prerecorded session with at least one reviewed annotation")
    if missing:
        pytest.skip("real three-branch replay blocked: " + "; ".join(missing))

    session = sessions[0]
    features = tmp_path / "real-features.jsonl"
    export = export_session_features(session, features, vision_weights=paths.vision_weights,
        audio_weights=paths.audio_weights, gesture_weights=paths.gesture_weights,
        pose_model_path=pose_model if pose_model.is_file() else None, overwrite=False)
    assert export["valid_feature_counts"].get("vision", 0) > 0
    assert export["valid_feature_counts"].get("audio", 0) > 0
    assert export["valid_feature_counts"].get("gesture", 0) > 0
    report = run_prerecorded_demo(session, features, tmp_path / "real-demo.json",
        fusion_weights=paths.fusion_weights, device="cpu")
    assert report["status"] == "complete_trained_three_branch"
    assert report["full_trained_multimodal_demo"] is True
    assert report["transformer_invocations"] > 0
    assert report["all_three_branches_same_invocation"] is True
    trace = [json.loads(line) for line in Path(report["trace_jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row.get("transformer_invocation_successful") and
               set(row.get("contributing_modalities", [])) == {"vision", "audio", "gesture"}
               for row in trace)
