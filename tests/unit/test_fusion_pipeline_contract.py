"""Synthetic-only software contract checks; these are not trained-model evidence."""
from pathlib import Path

import numpy as np
import pytest
import torch

from ainak.models.fusion.transformer import MultimodalTransformer
from ainak.runtime.checkpoint_validation import real_training_checkpoint_issues
from ainak.runtime.events import ModelEvent
from ainak.runtime.pipeline import MultimodalPipeline
from ainak.runtime.prerecorded_demo import _fusion_model, _require_full_demo_evidence


DIMS = {"vision": 3, "audio": 4, "gesture": 2}


def _model():
    return MultimodalTransformer(feature_dims=DIMS, dim=16, heads=4, layers=1,
                                 ff_dim=32, dropout=0., modality_dropout=0.).eval()


def _pipeline(model=None, *, stale_after=2.):
    return MultimodalPipeline(model or _model(), trained=True,
        _test_only_allow_untrained=True, sequence_length=4,
        lookback_seconds=5., stale_after=stale_after, window=stale_after)


def _event(modality, timestamp=1., *, available=True, feature=None):
    return ModelEvent(modality, timestamp,
        np.zeros(DIMS[modality], dtype=np.float32) if feature is None else feature,
        confidence=.8, available=available)


def test_three_modalities_reach_one_successful_transformer_forward():
    pipeline = _pipeline()
    for name in DIMS:
        pipeline.push(_event(name))
    result = pipeline.infer(1.1)
    assert result.fusion_attempted and result.transformer_used
    assert result.contributing_modalities == ("vision", "audio", "gesture")
    assert pipeline.invocation_attempts == pipeline.successful_invocations == 1
    assert pipeline.failed_invocations == 0


@pytest.mark.parametrize("modalities", [("vision", "gesture"), ("audio",)])
def test_fusion_runs_with_one_or_two_modalities_missing(modalities):
    pipeline = _pipeline()
    for name in modalities:
        pipeline.push(_event(name))
    result = pipeline.infer(1.1)
    assert result.transformer_used
    assert set(result.contributing_modalities) == set(modalities)
    assert sum(result.modality_validity.values()) == len(modalities)


def test_stale_observation_is_masked_and_not_counted_as_an_invocation():
    pipeline = _pipeline(stale_after=.5)
    pipeline.push(_event("vision", 1.))
    result = pipeline.infer(2.)
    assert not result.fusion_attempted and not result.transformer_used
    assert result.fusion_error.startswith("no_fresh_modalities")
    assert pipeline.invocation_attempts == 0


def test_invalid_feature_dimensions_are_rejected_before_transformer_call():
    pipeline = _pipeline()
    pipeline.push(_event("vision", feature=np.zeros((1, DIMS["vision"]), dtype=np.float32)))
    result = pipeline.infer(1.1)
    assert not result.fusion_attempted and not result.transformer_used
    assert result.fusion_error == "invalid_vision_feature_shape: expected [3], got [1, 3]"


def test_missing_checkpoint_cannot_be_marked_trained():
    with pytest.raises(ValueError, match="requires a loaded fusion model"):
        MultimodalPipeline(None, trained=True)


def test_random_model_cannot_be_marked_trained_without_loader_metadata():
    with pytest.raises(ValueError, match="metadata verified"):
        MultimodalPipeline(_model(), trained=True)


def test_untrained_or_unprovenanced_checkpoint_is_rejected(tmp_path: Path):
    path = tmp_path / "fusion.pt"
    torch.save({"trained_on_real_data": False}, path)
    model, status = _fusion_model(path, "cpu")
    assert model is None
    assert status["status"] == "unavailable"
    assert "checkpoint provenance rejected" in status["reason"]


def test_invalid_timestamps_are_rejected():
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="timestamp"):
        pipeline.push(_event("vision", float("nan")))
    with pytest.raises(ValueError, match="timestamp"):
        pipeline.infer(float("inf"))


def test_failed_transformer_forward_is_recorded_and_not_counted_as_success():
    pipeline = _pipeline()
    pipeline.fusion.forward = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test forward failure"))
    pipeline.push(_event("vision"))
    result = pipeline.infer(1.1)
    assert result.fusion_attempted and not result.transformer_used
    assert result.fusion_error == "transformer_forward_failed: RuntimeError: test forward failure"
    assert pipeline.invocation_attempts == 1
    assert pipeline.successful_invocations == 0
    assert pipeline.failed_invocations == 1


def test_complete_demo_claim_requires_same_call_three_branch_success():
    with pytest.raises(RuntimeError, match="requires a successful three-branch"):
        _require_full_demo_evidence("complete_trained_three_branch", 0, False)
    with pytest.raises(RuntimeError, match="requires a successful three-branch"):
        _require_full_demo_evidence("complete_trained_three_branch", 1, False)
    _require_full_demo_evidence("partial_blocked", 0, False)


def test_checkpoint_provenance_validator_rejects_missing_evidence():
    issues = real_training_checkpoint_issues({"trained_on_real_data": True}, "fusion")
    assert any("manifest SHA-256" in item for item in issues)
    assert any("train, validation, and test groups" in item for item in issues)
