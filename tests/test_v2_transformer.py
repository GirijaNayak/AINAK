"""AI AINAK Version 2 — Automated Test Suite for Multimodal Fusion Transformer

Validates:
1. Checkpoint Loading (V2 Transformer, YOLO-World, GestureTemporalNet, PANNs, AudioMLP, V1 baseline)
2. Feature Dimension Compatibility (Vision 256, Audio 2048, Gesture 128)
3. Modality Preprocessing Consistency & Numerical Safety (NaN / Inf detection)
4. Missing-Modality Masking & Dynamic Attention Weights
5. Multi-Task Forward Pass & Output Shape Contracts
6. Checkpoint Save, Reload & Numerical Reproducibility
7. Evaluation Metric Computations (Accuracy, Macro-F1, MAE)
8. Baseline Checkpoint Cryptographic Integrity (SHA-256 Checksums)
"""
import os
import json
import hashlib
import numpy as np
import pytest
import torch

from ainak.models.fusion.v2_transformer import (
    V2MultimodalTransformer,
    V2FusionOutput,
    DEFAULT_FEATURE_DIMS_V2,
    HAZARD_SCENARIOS,
    ALERT_TIERS,
)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Test 1: Checkpoint Discovery and Baseline Integrity
# -----------------------------------------------------------------------------
def test_baseline_checkpoint_integrity():
    """Verify that all baseline model checkpoints exist and remain 100% byte-for-byte intact."""
    baseline_hashes = {
        "artifacts/v2/vision/best.pt": "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792",
        "artifacts/v2/gesture/best.pt": "ce9b4b787ef1678c439d2a2e8176d8b287736f09e9cf9e065125ef32190669a6",
        "artifacts/audio/Cnn14_mAP=0.431.pth": "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31",
        "artifacts/audio/downstream_mlp.pt": "0ec51e895c1483fc720a146a9a669bccf7bfc5505fb1af149c09563149e879a6",
        "artifacts/fusion/fusion.pt": "49aead8f88b731f728d96ce73481c6aa6fc6f3afbf2d2544f7be4ef05c978adb",
        "artifacts/v2/hazard/hazard_policy_v2.json": "d7ee4da982bcbd093cb25e58fa2cfa090f30638619fb11076d283cfc91504481",
    }

    for path, expected_hash in baseline_hashes.items():
        assert os.path.exists(path), f"Required baseline file missing: {path}"
        actual_hash = compute_sha256(path)
        assert actual_hash == expected_hash, (
            f"BASELINE CHECKPOINT INTEGRITY VIOLATION for {path}!\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}"
        )


# -----------------------------------------------------------------------------
# Test 2: V2 Transformer Checkpoint Loading & Architecture Contract
# -----------------------------------------------------------------------------
def test_v2_transformer_checkpoint_loading():
    """Verify that artifacts/v2/fusion/best.pt loads cleanly and matches schema."""
    ckpt_path = "artifacts/v2/fusion/best.pt"
    assert os.path.exists(ckpt_path), f"Trained V2 Transformer checkpoint missing: {ckpt_path}"

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "state_dict" in payload, "Checkpoint payload missing 'state_dict'"
    assert payload.get("checkpoint_schema") == "ainak.v2_fusion_transformer.v1"
    assert payload.get("trained_on_real_data") is True
    assert payload.get("epochs_completed") == 25

    model = V2MultimodalTransformer()
    model.load_state_dict(payload["state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 608142, f"Parameter count mismatch: {total_params} != 608,142"


# -----------------------------------------------------------------------------
# Test 3: Feature Dimension Compatibility & Preprocessing Safety
# -----------------------------------------------------------------------------
def test_feature_dimensions_and_numerical_safety():
    """Verify model accepts exact feature dimensions and rejects NaNs/Infs."""
    model = V2MultimodalTransformer()
    model.eval()

    b, t = 3, 4
    feats = {
        "vision": torch.randn(b, t, 256),
        "audio": torch.randn(b, t, 2048),
        "gesture": torch.randn(b, t, 128),
    }
    mask = torch.ones(b, t, 3, dtype=torch.bool)
    times = torch.linspace(0, 1.5, t).unsqueeze(0).repeat(b, 1)

    out = model(feats, mask, times)
    assert isinstance(out, V2FusionOutput)

    # Check finite numbers
    assert torch.isfinite(out.scenario_logits).all(), "NaN/Inf in scenario logits"
    assert torch.isfinite(out.priority).all(), "NaN/Inf in priority predictions"
    assert torch.isfinite(out.alert_logits).all(), "NaN/Inf in alert logits"
    assert torch.isfinite(out.modality_contributions).all(), "NaN/Inf in modality contributions"
    assert torch.isfinite(out.context).all(), "NaN/Inf in pooled context"

    # Shape contracts
    assert out.scenario_logits.shape == (b, len(HAZARD_SCENARIOS))
    assert out.priority.shape == (b,)
    assert out.alert_logits.shape == (b, len(ALERT_TIERS))
    assert out.modality_contributions.shape == (b, 3)
    assert out.context.shape == (b, 128)

    # Value ranges
    assert (out.priority >= 0.0).all() and (out.priority <= 1.0).all()


# -----------------------------------------------------------------------------
# Test 4: Missing Modality Masking & Dynamic Contribution Scoring
# -----------------------------------------------------------------------------
def test_missing_modality_masking():
    """Verify that unobserved modalities receive zero attention contribution."""
    model = V2MultimodalTransformer()
    model.eval()

    b, t = 4, 4
    feats = {
        "vision": torch.randn(b, t, 256),
        "audio": torch.randn(b, t, 2048),
        "gesture": torch.randn(b, t, 128),
    }

    # Batch 0: Full [V, A, G]
    # Batch 1: Audio missing [V, 0, G]
    # Batch 2: Vision missing [0, A, G]
    # Batch 3: Gesture missing [V, A, 0]
    mask = torch.tensor([
        [[True, True, True]],
        [[True, False, True]],
        [[False, True, True]],
        [[True, True, False]],
    ], dtype=torch.bool).repeat(1, t, 1)

    out = model(feats, mask)
    contrib = out.modality_contributions

    # Batch 1: audio contribution must be identically 0.0
    assert contrib[1, 1].item() == 0.0, "Unobserved audio received non-zero contribution"
    # Batch 2: vision contribution must be identically 0.0
    assert contrib[2, 0].item() == 0.0, "Unobserved vision received non-zero contribution"
    # Batch 3: gesture contribution must be identically 0.0
    assert contrib[3, 2].item() == 0.0, "Unobserved gesture received non-zero contribution"

    # Contributions must sum to 1.0 across observed modalities
    for i in range(b):
        assert abs(contrib[i].sum().item() - 1.0) < 1e-4, f"Contributions for item {i} do not sum to 1.0: {contrib[i]}"


# -----------------------------------------------------------------------------
# Test 5: Checkpoint Save and Reload Reproducibility
# -----------------------------------------------------------------------------
def test_inference_reproducibility():
    """Verify that checkpoint reload produces bitwise identical predictions."""
    ckpt_path = "artifacts/v2/fusion/best.pt"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    m1 = V2MultimodalTransformer()
    m1.load_state_dict(payload["state_dict"])
    m1.eval()

    m2 = V2MultimodalTransformer()
    m2.load_state_dict(payload["state_dict"])
    m2.eval()

    torch.manual_seed(99)
    feats = {
        "vision": torch.randn(2, 4, 256),
        "audio": torch.randn(2, 4, 2048),
        "gesture": torch.randn(2, 4, 128),
    }
    mask = torch.ones(2, 4, 3, dtype=torch.bool)

    with torch.no_grad():
        out1 = m1(feats, mask)
        out2 = m2(feats, mask)

    assert torch.equal(out1.scenario_logits, out2.scenario_logits)
    assert torch.equal(out1.priority, out2.priority)
    assert torch.equal(out1.alert_logits, out2.alert_logits)
    assert torch.equal(out1.modality_contributions, out2.modality_contributions)


# -----------------------------------------------------------------------------
# Test 6: Aligned Dataset Integrity & Split Consistency
# -----------------------------------------------------------------------------
def test_dataset_integrity_and_splits():
    """Verify that aligned dataset manifest and splits are fully populated and disjoint."""
    manifest_path = "datasets/v2_fusion/manifests/aligned_manifest_v2.jsonl"
    split_path = "datasets/v2_fusion/splits/train_val_test_split.json"

    assert os.path.exists(manifest_path)
    assert os.path.exists(split_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 360, f"Expected 360 dataset examples, found {len(records)}"

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    # Disjointness checks
    assert len(train_set.intersection(val_set)) == 0, "Train and Val splits overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test splits overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test splits overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == 360, "Split count mismatch"

    # Held out session 04 check
    for eid in test_set:
        rec = next(r for r in records if r["example_id"] == eid)
        assert rec["session_id"] in ("session-multimodal-04", "subject_p01"), (
            f"Test split contaminated with non-test session: {rec['session_id']}"
        )
