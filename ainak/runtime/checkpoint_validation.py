"""Structural checks for checkpoints emitted by AI AINAK's real-data trainers."""
from __future__ import annotations

import re
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def real_training_checkpoint_issues(payload: Any, kind: str) -> list[str]:
    """Return reasons a payload cannot be treated as a real-data trained checkpoint.

    This checks recorded provenance and split structure; it cannot independently
    certify that human annotations or the source dataset were truthful.
    """
    if kind not in {"gesture", "fusion"}:
        raise ValueError("kind must be 'gesture' or 'fusion'")
    if not isinstance(payload, dict):
        return ["checkpoint payload is not an object"]
    issues = []
    if payload.get("checkpoint_schema") != f"ainak.{kind}-checkpoint.v1":
        issues.append("checkpoint has no supported trainer provenance schema")
    if payload.get("trained_on_real_data") is not True:
        issues.append("checkpoint is not marked as trained on real reviewed data")
    provenance = payload.get("training_data_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        issues.append("checkpoint has no training data provenance description")
    metadata = payload.get("training_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        issues.append("checkpoint has no training metadata")
    digest = metadata.get("manifest_sha256", payload.get("manifest_sha256"))
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        issues.append("checkpoint has no valid training manifest SHA-256")
    if not isinstance(metadata.get("epochs_completed"), int) or metadata["epochs_completed"] < 1:
        issues.append("checkpoint has no completed training epochs")
    split = payload.get("recording_split")
    if kind == "gesture" and not isinstance(split, dict):
        split = {"train": payload.get("train_groups"),
                 "validation": payload.get("validation_groups"),
                 "test": payload.get("test_groups")}
    if not isinstance(split, dict):
        split = {}
    groups = {name: split.get(name) for name in ("train", "validation", "test")}
    if any(not isinstance(value, list) or not value for value in groups.values()):
        issues.append("checkpoint is missing non-empty participant/session train, validation, and test groups")
    else:
        sets = {name: set(value) for name, value in groups.items()}
        if any(sets[left] & sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            issues.append("checkpoint participant/session splits overlap")
    for key in ("state_dict", "model_config"):
        if not isinstance(payload.get(key), dict) or not payload[key]:
            issues.append(f"checkpoint is missing {key}")
    for key in ("training_examples", "validation_examples", "test_examples"):
        if not isinstance(payload.get(key), int) or payload[key] < 1:
            issues.append(f"checkpoint is missing a positive {key} count")
    return issues
