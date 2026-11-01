from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    timestamp: float
    track_id: int | None = None
    relative_depth: float | None = None
    delta_depth: float | None = None
    track_age_frames: int = 1
    hazard_score: float = 0.0
    hazard_level: str = "none"


@dataclass
class ModelEvent:
    modality: str
    timestamp: float
    feature: Any
    confidence: float = 0.0
    labels: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
