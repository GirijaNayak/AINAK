"""AI AINAK — Contextual Hazard Engine (Version 2)

Decouples raw object detection from contextual assistive hazard policy.
Evaluates proximity proxy, looming rate, lateral crossing vectors, central corridor
alignment, and track persistence to assign one of 4 standardized hazard tiers:
- INFO
- CAUTION
- WARNING
- CRITICAL

Includes scenario-specific reasoning for approaching vehicles, stairs, fires,
and walking path obstructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class ScoredHazardV2:
    class_name: str
    track_id: int | None
    hazard_score: float         # [0.0, 1.0]
    hazard_tier: str            # "INFO", "CAUTION", "WARNING", "CRITICAL"
    scenario: str               # e.g., "VEHICLE_APPROACHING", "STAIRS_AHEAD", "CLEAR_PATH"
    relative_proximity: float   # [0.0, 1.0] relative optical proxy
    looming_rate: float         # >0 indicates approaching, <0 indicates receding
    lateral_velocity: float     # Rate of horizontal cross-movement
    corridor_alignment: float   # [0.0, 1.0] alignment with user's walking path
    confidence: float
    persistence_frames: int
    rationale: str
    time_to_collision_sec: float | None  # Estimated TTC proxy, or None


class ContextualHazardEngineV2:
    """Advanced multi-factor contextual hazard engine for AINAK V2."""

    # Categorical Risk Profiles
    CLASS_RISK_PROFILES: dict[str, float] = {
        # Critical fast-moving & fire threats
        "fire": 1.00,
        "flame": 1.00,
        "open flame": 1.00,
        "smoke": 0.95,
        "train": 0.95,
        "subway train": 0.95,
        "truck": 0.90,
        "bus": 0.90,
        "city bus": 0.90,
        "semi-trailer": 0.90,
        "ambulance": 0.90,
        "fire truck": 0.90,
        "car": 0.85,
        "taxi": 0.85,
        "motorcycle": 0.85,
        "moving machinery": 0.90,
        "excavator": 0.85,
        "bulldozer": 0.85,

        # Fall and terrain drop hazards
        "stairs": 0.85,
        "staircase": 0.85,
        "step": 0.80,
        "pothole": 0.80,
        "trench": 0.85,
        "open ditch": 0.85,
        "broken glass": 0.75,

        # Path obstructions & barriers
        "road barrier": 0.70,
        "construction barrier": 0.70,
        "bollard": 0.65,
        "open door": 0.60,
        "scaffolding": 0.70,
        "traffic cone": 0.55,
        "bicycle": 0.60,
        "electric scooter": 0.65,

        # Accessibility & human navigation
        "person": 0.50,
        "pedestrian": 0.50,
        "wheelchair": 0.50,
        "white cane": 0.40,
        "dog": 0.50,
        "guide dog": 0.35,

        # Standard indoor / outdoor items
        "chair": 0.35,
        "table": 0.35,
        "desk": 0.35,
        "tree": 0.40,
        "bench": 0.35,
        "trash can": 0.30,
        "curb": 0.55,
    }

    def __init__(
        self,
        critical_threshold: float = 0.72,
        warning_threshold: float = 0.50,
        caution_threshold: float = 0.30,
        min_confidence: float = 0.25,
    ):
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.caution_threshold = caution_threshold
        self.min_confidence = min_confidence

    def score_detection(
        self,
        class_name: str,
        confidence: float,
        bbox_xyxy: tuple[float, float, float, float],
        frame_dims: tuple[int, int],
        relative_proximity: float,
        looming_rate: float,
        track_age_frames: int,
        track_id: int | None = None,
        lateral_velocity: float = 0.0,
    ) -> ScoredHazardV2:
        h_img, w_img = frame_dims
        norm_class = class_name.lower().strip()
        base_risk = self.CLASS_RISK_PROFILES.get(norm_class, 0.25)

        if confidence < self.min_confidence:
            return ScoredHazardV2(
                class_name=class_name,
                track_id=track_id,
                hazard_score=0.0,
                hazard_tier="INFO",
                scenario="LOW_CONFIDENCE",
                relative_proximity=relative_proximity,
                looming_rate=looming_rate,
                lateral_velocity=lateral_velocity,
                corridor_alignment=0.0,
                confidence=confidence,
                persistence_frames=track_age_frames,
                rationale="Below minimum detection threshold",
                time_to_collision_sec=None,
            )

        # 1. Central walking path corridor alignment [0.0, 1.0]
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / (2.0 * w_img)
        # Center of camera is 0.5. Corridor is [0.35, 0.65]
        dist_from_center = abs(cx - 0.5)
        corridor_factor = max(0.0, 1.0 - (dist_from_center / 0.4))

        # 2. Looming expansion factor
        looming_factor = max(0.0, min(1.0, looming_rate * 2.5))

        # 3. Lateral crossing factor (e.g. car cutting across walking trajectory)
        lateral_factor = min(1.0, abs(lateral_velocity) * 1.8)

        # 4. Temporal stability factor (suppress single-frame spikes)
        persistence_factor = min(1.0, track_age_frames / 3.0)

        # 5. Composite Hazard Score
        raw_score = (
            0.30 * base_risk +
            0.30 * relative_proximity +
            0.20 * looming_factor +
            0.10 * corridor_factor +
            0.10 * confidence
        ) * (0.75 + 0.25 * persistence_factor)

        # If obstacle is crossing the walking corridor at close range, apply cross penalty
        if lateral_factor > 0.4 and relative_proximity > 0.5:
            raw_score += 0.15 * lateral_factor

        hazard_score = float(np.clip(raw_score, 0.0, 1.0))

        # Estimate Time to Collision (TTC) proxy: proximity / looming_rate
        ttc = None
        if looming_rate > 0.05 and relative_proximity > 0.2:
            ttc = max(0.5, round((1.0 - relative_proximity) / (looming_rate * 2.0), 1))

        # Scenario identification
        scenario = "ENVIRONMENT_INFO"
        rationale = f"Observed {class_name} at safe distance"

        # Check critical scenarios
        if norm_class in {"fire", "flame", "smoke"}:
            scenario = "FIRE_HAZARD"
            rationale = f"Detected active {class_name} ({confidence*100:.0f}% confidence)"
            hazard_score = max(hazard_score, 0.85)

        elif norm_class in {"stairs", "staircase", "step", "pothole", "trench"}:
            scenario = "TERRAIN_DROP_AHEAD"
            rationale = f"Impending {class_name} terrain transition ahead (proximity: {relative_proximity:.2f})"
            if relative_proximity > 0.5:
                hazard_score = max(hazard_score, 0.75)

        elif norm_class in {"car", "truck", "bus", "motorcycle", "train"}:
            if looming_rate > 0.08 and corridor_factor > 0.5:
                scenario = "VEHICLE_APPROACHING"
                rationale = f"Rapidly looming {class_name} approaching central path (TTC ~{ttc}s)"
                hazard_score = max(hazard_score, 0.80)
            elif abs(lateral_velocity) > 0.15:
                scenario = "VEHICLE_CROSSING_PATH"
                rationale = f"Lateral moving {class_name} crossing travel trajectory"

        elif norm_class in {"person", "pedestrian"} and corridor_factor > 0.7 and relative_proximity > 0.6:
            scenario = "PEDESTRIAN_IN_PATH"
            rationale = f"Person directly in forward walking path (proximity: {relative_proximity:.2f})"

        elif norm_class in {"construction barrier", "road barrier", "open door", "bollard"} and corridor_factor > 0.6:
            scenario = "PATH_OBSTRUCTION"
            rationale = f"Forward walking corridor obstructed by {class_name}"

        # Assign discrete Hazard Tiers
        if hazard_score >= self.critical_threshold and persistence_factor >= 0.66:
            tier = "CRITICAL"
        elif hazard_score >= self.warning_threshold:
            tier = "WARNING"
        elif hazard_score >= self.caution_threshold:
            tier = "CAUTION"
        else:
            tier = "INFO"

        return ScoredHazardV2(
            class_name=class_name,
            track_id=track_id,
            hazard_score=hazard_score,
            hazard_tier=tier,
            scenario=scenario,
            relative_proximity=relative_proximity,
            looming_rate=looming_rate,
            lateral_velocity=lateral_velocity,
            corridor_alignment=corridor_factor,
            confidence=confidence,
            persistence_frames=track_age_frames,
            rationale=rationale,
            time_to_collision_sec=ttc,
        )
