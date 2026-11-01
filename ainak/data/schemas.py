from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math

FEATURE_DIMS = {"vision": 128, "audio": 2048, "gesture": 128}
MAX_MODALITY_SKEW_SECONDS = 3.0


@dataclass
class ModalityObservation:
    modality: str
    timestamp: float
    feature: list[float]
    labels: list[dict[str, Any]] | None = None
    confidence: float | None = None
    interval_start_seconds: float | None = None
    interval_end_seconds: float | None = None
    feature_kind: str | None = None
    model_checkpoint_id: str | None = None
    preprocessing_version: str | None = None

    def validate(self) -> None:
        if self.modality not in {"vision", "audio", "gesture"}:
            raise ValueError(f"unsupported modality: {self.modality}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite numeric seconds")
        if not isinstance(self.feature, list) or len(self.feature) != FEATURE_DIMS[self.modality]:
            raise ValueError(f"{self.modality} feature must contain exactly {FEATURE_DIMS[self.modality]} values")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in self.feature):
            raise ValueError("feature values must be finite numbers")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("modality confidence must be finite and in [0,1]")
        if self.interval_start_seconds is not None or self.interval_end_seconds is not None:
            if self.interval_start_seconds is None or self.interval_end_seconds is None:
                raise ValueError("both interval bounds must be present")
            if not all(math.isfinite(v) for v in (self.interval_start_seconds, self.interval_end_seconds)):
                raise ValueError("interval bounds must be finite")
            if self.interval_start_seconds < 0 or self.interval_end_seconds < self.interval_start_seconds:
                raise ValueError("invalid observation interval")
            if not self.interval_start_seconds <= self.timestamp <= self.interval_end_seconds:
                raise ValueError("observation timestamp must lie within its interval")


@dataclass
class AlignedExample:
    example_id: str
    timestamp: float
    vision: ModalityObservation | None
    audio: ModalityObservation | None
    gesture: ModalityObservation | None
    priority: float
    confidence: float
    alert_type: str
    recording_id: str = ""
    schema_version: str | None = None
    session_id: str | None = None
    source_recording_ids: dict[str, str] | None = None
    modality_validity: dict[str, bool] | None = None
    annotation_provenance: dict[str, Any] | None = None
    target_source: str | None = None
    event_label: str | None = None
    participant_id: str | None = None

    def validate(self) -> None:
        if not self.example_id or not self.recording_id or not math.isfinite(self.timestamp) or not 0 <= self.priority <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("example_id and recording_id are required; timestamp must be finite; priority and annotation confidence must be in [0,1]")
        if self.alert_type not in ALERT_TYPES:
            raise ValueError(f"alert_type must be one of {ALERT_TYPES}")
        if not any((self.vision, self.audio, self.gesture)):
            raise ValueError("aligned example must contain at least one modality")
        for obs in (self.vision, self.audio, self.gesture):
            if obs:
                obs.validate()
                if abs(obs.timestamp-self.timestamp)>MAX_MODALITY_SKEW_SECONDS:
                    raise ValueError(f"{obs.modality} observation is stale or misaligned by more than {MAX_MODALITY_SKEW_SECONDS} seconds")
        if self.schema_version == "ainak.aligned-example.v1":
            if not self.session_id or not self.source_recording_ids or not self.modality_validity:
                raise ValueError("v1 aligned examples require session, source recording IDs, and modality validity")
            if self.target_source != "human_reviewed":
                raise ValueError("v1 training targets must come from explicitly human-reviewed annotations")
            if not self.annotation_provenance or not self.annotation_provenance.get("reviewed"):
                raise ValueError("v1 examples require reviewed annotation provenance")
            for modality in ("vision", "audio", "gesture"):
                if self.modality_validity.get(modality) != (getattr(self, modality) is not None):
                    raise ValueError(f"{modality} validity flag does not match feature presence")

    def as_json(self) -> dict[str, Any]:
        return {**asdict(self)}


ALERT_TYPES = ("none", "informational", "caution", "urgent")
