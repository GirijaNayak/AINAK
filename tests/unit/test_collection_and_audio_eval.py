import csv
import json

import numpy as np
import pytest

from ainak.data.gesture_collection import delete_gesture_session, extract_reviewed_gesture_sequences
from ainak.models.gesture.lstm import FEATURES, GESTURE_LABELS
from ainak.training.audio_evaluation import ESC50_TO_AUDIOSET, validate_esc50_mapping
from ainak.data.validation import validate_recording_session
from ainak.training.gesture import train_gesture


def test_esc50_mapping_is_complete_and_requires_real_checkpoint_labels():
    categories = set(ESC50_TO_AUDIOSET)
    labels = sorted({label for aliases in ESC50_TO_AUDIOSET.values() for label in aliases})
    mapping = validate_esc50_mapping(categories, labels)
    assert mapping["dog"] == [labels.index("Dog")]
    assert mapping["can_opening"] == [labels.index("Crack")]
    with pytest.raises(ValueError, match="absent"):
        validate_esc50_mapping(categories, ["Dog"])


def test_reviewed_gesture_intervals_extract_sequences_and_preserve_synthetic_flag(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    metadata = {"session_id": "session-1", "participant_code": "p1",
                "preprocessing_version": "fixture-v1", "synthetic": True}
    (session / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
    fields = ["frame_index", "timestamp_seconds", "pose_visible", *[f"feature_{i:03}" for i in range(FEATURES)]]
    with (session / "landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in range(4):
            values = np.zeros(FEATURES, dtype=np.float32); values[3::4] = 1.
            row = {"frame_index": i, "timestamp_seconds": i * .1, "pose_visible": 1}
            row.update({f"feature_{j:03}": values[j] for j in range(FEATURES)})
            writer.writerow(row)
    with (session / "annotation_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_seconds", "end_seconds", "label", "reviewed", "annotator", "confidence", "provenance"])
        writer.writeheader()
        writer.writerow({"start_seconds": .05, "end_seconds": .25, "label": GESTURE_LABELS[0], "reviewed": "true",
                         "annotator": "reviewer-1", "confidence": .95, "provenance": "manual frame review"})
    manifest = tmp_path / "manifest.csv"
    result = extract_reviewed_gesture_sequences(session, manifest, min_frames=2)
    assert result["accepted_sequences"] == 1
    row = next(csv.DictReader(manifest.open(encoding="utf-8")))
    assert row["synthetic"] == "True"
    sequence = np.load(row["path"])
    assert sequence.shape == (24, FEATURES)


def test_unreviewed_or_unknown_gesture_annotations_are_not_accepted(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    (session / "session.json").write_text(json.dumps({"session_id": "s", "participant_code": "p",
        "preprocessing_version": "fixture-v1"}), encoding="utf-8")
    fields = ["frame_index", "timestamp_seconds", "pose_visible", *[f"feature_{i:03}" for i in range(FEATURES)]]
    with (session / "landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in range(3):
            values = np.zeros(FEATURES, dtype=np.float32); values[3::4] = 1.
            row = {"frame_index": i, "timestamp_seconds": i * .1, "pose_visible": 1}
            row.update({f"feature_{j:03}": values[j] for j in range(FEATURES)}); writer.writerow(row)
    with (session / "annotation_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_seconds", "end_seconds", "label", "reviewed", "annotator", "confidence", "provenance"])
        writer.writeheader(); writer.writerow({"start_seconds": 0, "end_seconds": .2, "label": "not-a-gesture",
            "reviewed": "false", "annotator": "", "confidence": .1, "provenance": ""})
    with pytest.raises(ValueError, match="no valid reviewed"):
        extract_reviewed_gesture_sequences(session, tmp_path / "manifest.csv", min_frames=2)


def test_synchronized_recording_validator_checks_timestamp_indexes(tmp_path):
    session = tmp_path / "recording"; session.mkdir()
    (session / "recording.json").write_text(json.dumps({"duration_seconds": 1, "audio_sample_rate": 32000,
        "audio_samples": 32000, "video_file": "video.mp4", "audio_file": "audio.wav",
        "annotation_template": "annotation_template.csv"}), encoding="utf-8")
    (session / "video.mp4").write_bytes(b"fixture")
    (session / "audio.wav").write_bytes(b"fixture")
    (session / "video_frames.csv").write_text("frame_index,monotonic_seconds,width,height\n0,0.0,640,480\n1,0.1,640,480\n", encoding="utf-8")
    (session / "audio_blocks.csv").write_text("sample_start,frames,monotonic_seconds,status\n0,3200,0.1,\n3200,3200,0.2,\n", encoding="utf-8")
    annotation_fields = "start_seconds,end_seconds,event_label,priority,alert_type,annotation_confidence,reviewed,gesture_label,annotator,provenance,vision_notes,audio_notes,gesture_notes\n"
    (session / "annotation_template.csv").write_text(annotation_fields + "0.2,0.4,car_horn,0.7,caution,0.9,true,,reviewer,manual review,,,\n", encoding="utf-8")
    assert validate_recording_session(session) == []
    (session / "video_frames.csv").write_text("frame_index,monotonic_seconds,width,height\n0,0.2,640,480\n1,0.1,640,480\n", encoding="utf-8")
    assert any("out-of-order" in error for error in validate_recording_session(session))


def test_gesture_trainer_rejects_synthetic_examples(tmp_path):
    manifest=tmp_path/"gesture.csv"
    manifest.write_text("path,label,recording_id,subject_id,synthetic\nmissing.npy,pointing,s1,p1,true\n",encoding="utf-8")
    with pytest.raises(ValueError,match="synthetic gesture sequences"):
        train_gesture(manifest,tmp_path/"model.pt")


def test_overlapping_reviewed_gesture_intervals_are_rejected(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    (session / "session.json").write_text(json.dumps({"session_id": "overlap-session", "participant_code": "p1",
        "preprocessing_version": "fixture-v1"}), encoding="utf-8")
    fields = ["frame_index", "timestamp_seconds", "pose_visible", *[f"feature_{i:03}" for i in range(FEATURES)]]
    with (session / "landmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in range(10):
            values = np.zeros(FEATURES, dtype=np.float32); values[3::4] = 1.
            row = {"frame_index": i, "timestamp_seconds": i * .1, "pose_visible": 1}
            row.update({f"feature_{j:03}": values[j] for j in range(FEATURES)}); writer.writerow(row)
    with (session / "annotation_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_seconds", "end_seconds", "label", "reviewed", "annotator", "confidence", "provenance"])
        writer.writeheader()
        for label, start, end in (("pointing", 0, .5), ("waving", .4, .9)):
            writer.writerow({"start_seconds": start, "end_seconds": end, "label": label, "reviewed": "true",
                "annotator": "reviewer", "confidence": .9, "provenance": "manual"})
    with pytest.raises(ValueError, match="no valid reviewed"):
        extract_reviewed_gesture_sequences(session, tmp_path / "manifest.csv", min_frames=2)


def test_delete_confirmed_gesture_session_removes_source_derived_files_and_manifest_rows(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    (session / "source_video.mp4").write_bytes(b"local fixture")
    (session / "sequences").mkdir(); (session / "sequences" / "one.npy").write_bytes(b"derived fixture")
    (session / "session.json").write_text(json.dumps({"session_id": "session-delete", "participant_code": "p-delete"}), encoding="utf-8")
    manifest = tmp_path / "gesture-manifest.csv"
    manifest.write_text("path,label,recording_id,subject_id\n" + str(session / "sequences" / "one.npy") + ",pointing,session-delete,p-delete\n"
                        + "other.npy,waving,other-session,p-other\n", encoding="utf-8")
    summary = manifest.with_suffix(".summary.json")
    summary.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation ID"):
        delete_gesture_session(session, "wrong-id")
    result = delete_gesture_session(session, "session-delete", [manifest])
    assert not session.exists()
    assert "other-session" in manifest.read_text(encoding="utf-8")
    assert "session-delete" not in manifest.read_text(encoding="utf-8")
    assert not summary.exists()
    assert result["removed_manifests"][0]["rows_removed"] == 1


def test_delete_gesture_session_preserves_other_sessions_by_same_participant_and_path_prefix(tmp_path):
    session = tmp_path / "session"; session.mkdir()
    (session / "session.json").write_text(json.dumps({"session_id": "session-one", "participant_code": "p1"}), encoding="utf-8")
    sibling = tmp_path / "session-archive"; sibling.mkdir()
    (sibling / "sequence.npy").write_bytes(b"keep")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"session_id": "session-one", "source_path": str(session / "sequence.npy")},
        {"session_id": "session-two", "participant_id": "p1", "source_path": str(sibling / "sequence.npy")},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = delete_gesture_session(session, "session-one", [manifest])
    kept = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1 and kept[0]["session_id"] == "session-two"
    assert sibling.exists() and (sibling / "sequence.npy").read_bytes() == b"keep"
    assert result["removed_manifests"][0]["rows_removed"] == 1
