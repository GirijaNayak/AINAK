"""Relative depth estimation and temporal motion cues for monocular vision.

NOTE: Monocular depth cues represent relative optical proximity, not metric ground-truth
physical distance or collision guarantees.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


class RelativeDepthEstimator:
    """Estimates normalized relative proximity [0.0 = distant, 1.0 = highly proximate]

    Combines:
    1. Normalized bounding box bounding area (angular extent proxy).
    2. Lower-edge vertical perspective position in the image frame.
    3. Category-specific geometric prior scale factors.
    """

    CLASS_SCALE_PRIORS: dict[str, float] = {
        "bus": 1.4,
        "truck": 1.3,
        "train": 1.5,
        "car": 1.1,
        "motorcycle": 1.0,
        "bicycle": 0.9,
        "person": 0.85,
        "traffic light": 0.7,
        "stop sign": 0.7,
        "bench": 0.7,
        "chair": 0.6,
        "fire hydrant": 0.6,
        "dog": 0.6,
    }

    def __init__(self, horizon_ratio: float = 0.40):
        self.horizon_ratio = horizon_ratio

    def estimate_proximity(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        frame_shape: tuple[int, int],
        class_name: str = "object",
    ) -> float:
        """Calculate relative proximity score in [0.0, 1.0]."""
        h, w = frame_shape[:2]
        if h <= 0 or w <= 0:
            return 0.0

        x1, y1, x2, y2 = bbox_xyxy
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area_ratio = (box_w * box_h) / (w * h)

        # Angular extent cue (square-root of area ratio is proportional to apparent angle)
        angular_cue = math.sqrt(max(0.0, area_ratio))

        # Ground plane perspective cue: objects whose bottom contact point is near frame bottom are closer
        bottom_y = y2 / h
        perspective_cue = max(0.0, (bottom_y - self.horizon_ratio) / max(1.0 - self.horizon_ratio, 1e-6))

        # Category prior
        prior = self.CLASS_SCALE_PRIORS.get(class_name.lower(), 0.75)

        # Combined normalized proximity
        combined = (0.55 * angular_cue * prior) + (0.45 * perspective_cue * angular_cue)
        return float(np.clip(combined * 1.8, 0.0, 1.0))
