from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch

from ..decision.alert_engine import Alert, AlertEngine
from ..models.fusion.transformer import MultimodalTransformer
from .events import ModelEvent
from .synchronization import TemporalSynchronizer


@dataclass
class PipelineResult:
    timestamp: float
    contributing_modalities: tuple[str, ...]
    transformer_used: bool
    context: list[float] | None
    priority: float
    confidence: float
    alert_type: str
    alert: Alert | None
    modality_contributions: list[float] | None = None
    fusion_attempted: bool = False
    fusion_error: str | None = None
    fusion_checkpoint_id: str | None = None
    selected_window_timestamps: tuple[float, ...] = ()
    modality_validity: dict[str, bool] = field(default_factory=dict)
    feature_dimensions: dict[str, int | None] = field(default_factory=dict)


class MultimodalPipeline:
    """Fuse timestamped branch features without treating an untrained model as ready."""

    def __init__(self, fusion: MultimodalTransformer | None, trained: bool = False,
                 device: str = "cpu", stale_after: float = 3., window: float = 2.,
                 alert_engine: AlertEngine | None = None, speech: Any = None,
                 sequence_length: int = 8, lookback_seconds: float = 8.,
                 checkpoint_metadata: dict[str, Any] | None = None,
                 blocked_reason: str | None = None,
                 _test_only_allow_untrained: bool = False):
        if trained and fusion is None:
            raise ValueError("trained=True requires a loaded fusion model")
        if trained and not _test_only_allow_untrained:
            if not isinstance(checkpoint_metadata, dict) or checkpoint_metadata.get("verified_trained") is not True:
                raise ValueError("trained fusion requires metadata verified by the real-data checkpoint loader")
            split = checkpoint_metadata.get("recording_split", {})
            groups = [split.get(name) for name in ("train", "validation", "test")] if isinstance(split, dict) else []
            if (checkpoint_metadata.get("trained_on_real_data") is not True
                    or not isinstance(checkpoint_metadata.get("checkpoint_id"), str)
                    or not checkpoint_metadata["checkpoint_id"].startswith("sha256:")
                    or not isinstance(checkpoint_metadata.get("training_manifest_sha256"), str)
                    or len(checkpoint_metadata["training_manifest_sha256"]) != 64
                    or len(groups) != 3 or any(not isinstance(value, list) or not value for value in groups)
                    or any(not isinstance(checkpoint_metadata.get(key), int) or checkpoint_metadata[key] < 1
                           for key in ("training_examples", "validation_examples", "test_examples"))):
                raise ValueError("fusion checkpoint metadata is incomplete or lacks held-out real-data provenance")
            sets = [set(value) for value in groups]
            if any(sets[left] & sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("fusion checkpoint participant/session splits overlap")
        if sequence_length < 1 or lookback_seconds <= 0:
            raise ValueError("sequence_length and lookback_seconds must be positive")
        if not math.isfinite(stale_after) or stale_after <= 0 or not math.isfinite(window) or window <= 0:
            raise ValueError("staleness and synchronization windows must be finite and positive")
        self.fusion = fusion
        self.trained = bool(trained)
        self.device = device
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.checkpoint_id = self.checkpoint_metadata.get("checkpoint_id")
        self.blocked_reason = blocked_reason or "trained fusion checkpoint unavailable"
        self.sequence_length, self.lookback_seconds = sequence_length, lookback_seconds
        self.sync = TemporalSynchronizer(stale_after, window)
        self.alerts = alert_engine or AlertEngine()
        self.speech = speech
        self.invocation_attempts = 0
        self.successful_invocations = 0
        self.failed_invocations = 0
        if self.fusion is not None:
            self.fusion.eval()

    def push(self, event: ModelEvent) -> None:
        self.sync.push(event)

    def _fallback_result(self, timestamp: float, present: dict[str, ModelEvent], alert: Alert | None,
                         reason: str | None = None, *, attempted: bool = False,
                         window_timestamps: tuple[float, ...] = (),
                         modality_validity: dict[str, bool] | None = None,
                         feature_dimensions: dict[str, int | None] | None = None) -> PipelineResult:
        if alert is not None and self.speech is not None:
            self.speech.say(alert.text, priority=int(round(alert.priority * 100)))
        return PipelineResult(timestamp, tuple(sorted(present)), False, None, 0., 0.,
            "fallback" if alert else "none", alert, fusion_attempted=attempted,
            fusion_error=reason, fusion_checkpoint_id=self.checkpoint_id,
            selected_window_timestamps=window_timestamps,
            modality_validity=modality_validity or {}, feature_dimensions=feature_dimensions or {})

    def infer(self, timestamp: float) -> PipelineResult:
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("inference timestamp must be finite nonnegative session seconds")
        frame = self.sync.align(float(timestamp))
        present = {name: event for name, event in frame.events.items()
                   if event is not None and event.available}
        observations = []
        for modality, event in present.items():
            for item in event.labels:
                observations.append({**item, "timestamp": event.timestamp, "modality": modality})
        alert = self.alerts.decide(observations, float(timestamp))

        if not self.trained or self.fusion is None:
            reason = self.blocked_reason if not self.trained else "fusion model unavailable"
            return self._fallback_result(float(timestamp), present, alert, reason)
        if not present:
            return self._fallback_result(float(timestamp), present, alert,
                "no_fresh_modalities: all observations are missing, unavailable, future-dated, or stale")

        order = ("vision", "audio", "gesture")
        history = self.sync.history(float(timestamp), self.sequence_length, self.lookback_seconds)
        window_timestamps = tuple(item.timestamp for item in history)
        modality_validity = {name: any(item.events[name] is not None and item.events[name].available for item in history)
                             for name in order}
        feature_dimensions: dict[str, int | None] = {name: None for name in order}
        features: dict[str, list[torch.Tensor]] = {name: [] for name in order}
        mask = torch.zeros(1, len(history), len(order), dtype=torch.bool, device=self.device)
        relative_times = []

        for time_index, aligned in enumerate(history):
            relative_times.append(aligned.timestamp - history[0].timestamp)
            for modality_index, name in enumerate(order):
                event = aligned.events[name]
                expected = self.fusion.projections[name][0].in_features
                if event is None or not event.available:
                    features[name].append(torch.zeros(expected, dtype=torch.float32))
                    continue
                try:
                    values = np.asarray(event.feature, dtype=np.float32)
                except Exception as exc:
                    reason = f"invalid_{name}_feature: cannot convert feature to a numeric vector ({type(exc).__name__}: {exc})"
                    return self._fallback_result(float(timestamp), present, alert, reason,
                        window_timestamps=window_timestamps, modality_validity=modality_validity,
                        feature_dimensions=feature_dimensions)
                if values.ndim != 1 or values.size != expected:
                    reason = f"invalid_{name}_feature_shape: expected [{expected}], got {list(values.shape)}"
                    return self._fallback_result(float(timestamp), present, alert, reason,
                        window_timestamps=window_timestamps, modality_validity=modality_validity,
                        feature_dimensions=feature_dimensions)
                if not np.isfinite(values).all():
                    reason = f"invalid_{name}_feature_values: feature contains NaN or Inf"
                    return self._fallback_result(float(timestamp), present, alert, reason,
                        window_timestamps=window_timestamps, modality_validity=modality_validity,
                        feature_dimensions=feature_dimensions)
                feature_dimensions[name] = int(values.size)
                features[name].append(torch.from_numpy(values.copy()))
                mask[0, time_index, modality_index] = True

        if not bool(mask.any()):
            return self._fallback_result(float(timestamp), present, alert,
                "no_valid_modalities_in_selected_fusion_window", window_timestamps=window_timestamps,
                modality_validity=modality_validity, feature_dimensions=feature_dimensions)

        feature_tensors = {name: torch.stack(values).unsqueeze(0).to(self.device)
                           for name, values in features.items()}
        times = torch.tensor([relative_times], dtype=torch.float32, device=self.device)
        self.invocation_attempts += 1
        try:
            with torch.no_grad():
                out = self.fusion(feature_tensors, mask, times)
            if not all(hasattr(out, name) for name in ("context", "priority", "confidence", "alert_logits", "modality_contributions")):
                raise TypeError("fusion forward returned an incompatible output object")
            self.successful_invocations += 1
        except Exception as exc:
            self.failed_invocations += 1
            reason = f"transformer_forward_failed: {type(exc).__name__}: {exc}"
            return self._fallback_result(float(timestamp), present, alert, reason, attempted=True,
                window_timestamps=window_timestamps, modality_validity=modality_validity,
                feature_dimensions=feature_dimensions)

        idx = int(out.alert_logits[0].argmax().item())
        alert_type = ("none", "informational", "caution", "urgent")[idx]
        learned = {"trained": True, "priority": float(out.priority[0]),
                   "confidence": float(out.confidence[0]), "alert_type": alert_type}
        alert = self.alerts.decide(observations, float(timestamp), fusion=learned)
        if alert is not None and self.speech is not None:
            self.speech.say(alert.text, priority=int(round(alert.priority * 100)))
        # Include modalities found anywhere in the selected temporal history.
        contributing = tuple(name for name in order if modality_validity[name])
        return PipelineResult(float(timestamp), contributing, True, out.context[0].cpu().tolist(),
            learned["priority"], learned["confidence"], alert_type, alert,
            out.modality_contributions[0].cpu().tolist(), fusion_attempted=True,
            fusion_checkpoint_id=self.checkpoint_id, selected_window_timestamps=window_timestamps,
            modality_validity=modality_validity, feature_dimensions=feature_dimensions)
