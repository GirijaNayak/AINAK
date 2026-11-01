"""Hazard scoring and prioritization layer for the vision branch.

Distinguishes raw object detection from assistive safety hazard policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class ScoredHazard:
    class_name: str
    track_id: int | None
    hazard_score: float  # [0.0, 1.0]
    hazard_level: str    # "immediate_attention", "caution", "informational", "none"
    relative_depth: float
    delta_depth: float   # >0 indicates looming/approaching, <0 indicates receding
    horizontal_cue: str  # "left", "center", "right"
    confidence: float
    persistence_frames: int
    rationale: str


class VisionHazardScorer:
    """Configurable hazard assessment for detected and tracked objects.

    Hazard is evaluated from:
    1. Base class risk weight (vehicles and moving hazards > static furniture).
    2. Proximity / relative depth estimate (closer objects = higher risk).
    3. Looming rate (objects expanding / approaching camera = urgent risk).
    4. Centering cue (objects directly in central path = higher risk).
    5. Track persistence (suppresses single-frame false positive spikes).
    """

    CRITICAL_RISK_WEIGHTS: dict[str, float] = {
        "car": 0.85,
        "truck": 0.90,
        "bus": 0.90,
        "motorcycle": 0.85,
        "bicycle": 0.75,
        "train": 0.95,
        "person": 0.70,
        "traffic light": 0.60,
        "stop sign": 0.65,
        "fire hydrant": 0.50,
        "chair": 0.40,
        "dog": 0.60,
    }

    def __init__(
        self,
        immediate_threshold: float = 0.68,
        caution_threshold: float = 0.42,
        min_confidence: float = 0.35,
    ):
        self.immediate_threshold = immediate_threshold
        self.caution_threshold = caution_threshold
        self.min_confidence = min_confidence

    def score_detection(
        self,
        class_name: str,
        confidence: float,
        relative_depth: float,
        delta_depth: float,
        horizontal_cue: str,
        track_age_frames: int,
        track_id: int | None = None,
    ) -> ScoredHazard:
        """Score an individual tracked detection and categorize its hazard level."""
        if confidence < self.min_confidence:
            return ScoredHazard(
                class_name=class_name,
                track_id=track_id,
                hazard_score=0.0,
                hazard_level="none",
                relative_depth=relative_depth,
                delta_depth=delta_depth,
                horizontal_cue=horizontal_cue,
                confidence=confidence,
                persistence_frames=track_age_frames,
                rationale="Below minimum detection confidence threshold",
            )

        base_weight = self.CRITICAL_RISK_WEIGHTS.get(class_name.lower(), 0.30)

        # Proximity component [0.0, 1.0]
        proximity_factor = relative_depth

        # Looming approach component: positive delta_depth indicates approaching
        looming_factor = max(0.0, min(1.0, delta_depth * 2.5))

        # Central trajectory penalty: obstacles in the central walking path are riskier
        center_factor = 1.15 if horizontal_cue == "center" else 0.90

        # Persistence bonus: stable tracks (>2 frames) reduce false alarms
        persistence_factor = min(1.0, track_age_frames / 3.0)

        # Combined hazard score calculation
        raw_score = (
            0.35 * base_weight +
            0.35 * proximity_factor +
            0.20 * looming_factor +
            0.10 * confidence
        ) * center_factor * (0.7 + 0.3 * persistence_factor)

        hazard_score = float(np.clip(raw_score, 0.0, 1.0))

        # Categorization
        if hazard_score >= self.immediate_threshold or (proximity_factor > 0.65 and base_weight >= 0.70):
            level = "immediate_attention"
            rationale = f"Close or approaching {class_name} ({horizontal_cue}, proximity={proximity_factor:.2f})"
        elif hazard_score >= self.caution_threshold:
            level = "caution"
            rationale = f"Relevant {class_name} detected ({horizontal_cue}, proximity={proximity_factor:.2f})"
        else:
            level = "informational"
            rationale = f"Observed {class_name} at safe distance ({horizontal_cue})"

        return ScoredHazard(
            class_name=class_name,
            track_id=track_id,
            hazard_score=hazard_score,
            hazard_level=level,
            relative_depth=relative_depth,
            delta_depth=delta_depth,
            horizontal_cue=horizontal_cue,
            confidence=confidence,
            persistence_frames=track_age_frames,
            rationale=rationale,
        )
