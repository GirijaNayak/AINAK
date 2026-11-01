import json

import numpy as np

from ainak.data.alignment import build_aligned_dataset
from ainak.data.features import FEATURE_SCHEMA_VERSION, validate_feature_jsonl
from ainak.data.validation import validate_jsonl


def _feature(session, modality, timestamp, start, end, dimension):
    return {"schema_version": FEATURE_SCHEMA_VERSION, "feature_id": f"{session}:{modality}:{timestamp}",
        "session_id": session, "source_recording_id": session, "modality": modality,
        "timestamp_seconds": timestamp, "interval_start_seconds": start, "interval_end_seconds": end,
        "feature": np.zeros(dimension, dtype=np.float32).tolist(), "feature_kind": f"test-{modality}",
        "prediction": {"top_classes": []}, "model_checkpoint_id": "sha256:test-fixture",
        "preprocessing_version": "test-fixture-v1", "valid": True, "confidence": .5, "missing_reason": None,
        "latency_ms": .1}


def test_timestamped_feature_export_validation(tmp_path):
    source = tmp_path / "features.jsonl"
    rows = [_feature("session-1", "vision", 1., 1., 1., 128),
            _feature("session-1", "audio", .7, .2, 1.2, 2048)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report, errors = validate_feature_jsonl(source)
    assert not errors
    assert report["valid"]
    assert report["counts"]["vision_valid"] == 1
    assert report["counts"]["audio_valid"] == 1


def test_alignment_uses_reviewed_labels_and_feature_timestamps(tmp_path):
    session = tmp_path / "session-1"
    session.mkdir()
    (session / "recording.json").write_text(json.dumps({"recording_id": "session-1",
        "annotation_template": "annotation_template.csv"}), encoding="utf-8")
    (session / "annotation_template.csv").write_text(
        "start_seconds,end_seconds,event_label,priority,alert_type,annotation_confidence,reviewed,annotator,provenance\n"
        "0,2,car_horn,0.7,caution,0.9,true,reviewer,manual review\n", encoding="utf-8")
    feature_path = session / "features.jsonl"
    rows = [_feature("session-1", "vision", 1., 1., 1., 128),
            _feature("session-1", "audio", .7, .2, 1.2, 2048)]
    feature_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = session / "aligned.jsonl"
    report = build_aligned_dataset(session, feature_path, output, window_seconds=2., max_feature_age_seconds=1.5)
    count, errors = validate_jsonl(output)
    assert count == 1 and not errors
    assert report["aligned_examples"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["modality_validity"] == {"vision": True, "audio": True, "gesture": False}
    assert row["target_source"] == "human_reviewed"
    assert row["vision"]["model_checkpoint_id"] == "sha256:test-fixture"


def test_alignment_rejects_unreviewed_target(tmp_path):
    session = tmp_path / "session-2"
    session.mkdir()
    (session / "recording.json").write_text(json.dumps({"recording_id": "session-2"}), encoding="utf-8")
    (session / "annotation_template.csv").write_text(
        "start_seconds,end_seconds,event_label,priority,alert_type,annotation_confidence,reviewed,annotator,provenance\n"
        "0,2,car_horn,0.7,caution,0.9,false,reviewer,manual review\n", encoding="utf-8")
    features = session / "features.jsonl"
    features.write_text(json.dumps(_feature("session-2", "vision", 1., 1., 1., 128)) + "\n", encoding="utf-8")
    try:
        build_aligned_dataset(session, features, session / "aligned.jsonl")
    except ValueError as exc:
        assert "not explicitly marked reviewed" in str(exc)
    else:
        raise AssertionError("unreviewed annotations must not enter a training manifest")
