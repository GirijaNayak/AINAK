"""AI AINAK — Gesture V2 Expanded Model Architecture & Inference Pipeline

Supports 25 real-world communicative and emergency gesture classes using
33-point BlazePose 3D kinematics (132 features/frame, 24 frames/sequence).

Preserves 128-dimensional latent embedding compatibility with the
AINAK Multimodal Fusion Transformer.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

from ...runtime.events import ModelEvent

GESTURE_CLASSES_V2 = (
    "idle",
    "wave",
    "stop",
    "come_here",
    "go_away",
    "thumbs_up",
    "thumbs_down",
    "ok_sign",
    "peace_sign",
    "open_palm",
    "fist",
    "pointing_left",
    "pointing_right",
    "pointing_forward",
    "pointing_upward",
    "pointing_downward",
    "help",
    "emergency",
    "call_someone",
    "follow_me",
    "wait",
    "turn_left",
    "turn_right",
    "slow_down",
    "thank_you",
)

LANDMARKS = 33
FEATURES = LANDMARKS * 4  # normalized x, y, z, visibility
SEQUENCE_LENGTH = 24


def normalize_landmarks_v2(points: np.ndarray) -> np.ndarray:
    """Normalize 33 BlazePose landmarks: center hips, scale by shoulder width, zero low-visibility."""
    p = np.asarray(points, dtype=np.float32).reshape(LANDMARKS, 4).copy()
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    # MediaPipe indices: shoulders 11/12, hips 23/24
    center = (p[23, :3] + p[24, :3]) / 2.0
    scale = max(float(np.linalg.norm(p[11, :2] - p[12, :2])), 1e-3)
    p[:, :3] = (p[:, :3] - center) / scale
    p[p[:, 3] < 0.2, :3] = 0.0
    p[:, :3] = np.clip(p[:, :3], -5.0, 5.0)
    p[:, 3] = np.clip(p[:, 3], 0.0, 1.0)
    return p.reshape(-1)


class GestureTemporalNet(nn.Module):
    """Deep 2-layer temporal kinematic neural network for 25 gesture classes.

    Emits 128-D embedding matching AINAK Multimodal Transformer specifications.
    """
    def __init__(
        self,
        feature_dim: int = FEATURES,
        hidden_size: int = 128,
        classes: int = len(GESTURE_CLASSES_V2),
        dropout: float = 0.2,
        layers: int = 2,
    ):
        super().__init__()
        if layers < 1 or hidden_size < 1:
            raise ValueError("layers and hidden_size must be positive")
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (logits [B, classes], embedding [B, 128])."""
        _, (h, _) = self.lstm(x)
        embedding = self.norm(h[-1])
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding


class GestureSequenceBufferV2:
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        if sequence_length < 2:
            raise ValueError("sequence_length must be >=2")
        self.sequence_length = sequence_length
        self._sequence: deque[np.ndarray] = deque(maxlen=sequence_length)

    def append(self, landmarks: np.ndarray) -> np.ndarray | None:
        self._sequence.append(np.asarray(landmarks, dtype=np.float32).reshape(FEATURES))
        return np.stack(self._sequence) if len(self._sequence) == self.sequence_length else None

    def clear(self) -> None:
        self._sequence.clear()


class GestureRecognizerV2:
    """Production inference controller for 25-class Gesture V2 with unknown rejection."""

    def __init__(
        self,
        checkpoint: str | Path,
        seq_len: int = SEQUENCE_LENGTH,
        device: str = "cpu",
        threshold: float = 0.70,
        entropy_threshold: float = 2.2,
    ):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Gesture V2 checkpoint missing: {path}")

        self.device = device
        self.seq_len = seq_len
        self.threshold = threshold
        self.entropy_threshold = entropy_threshold

        payload = torch.load(path, map_location=device, weights_only=False)
        self.labels = tuple(payload.get("labels", GESTURE_CLASSES_V2))
        self.model = GestureTemporalNet(
            feature_dim=FEATURES,
            hidden_size=128,
            classes=len(self.labels),
            dropout=0.2,
            layers=2,
        ).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.sequence = GestureSequenceBufferV2(seq_len)
        self._history: deque[str] = deque(maxlen=3)

    def append(self, normalized_landmarks: np.ndarray, timestamp: float | None = None) -> ModelEvent | None:
        sequence = self.sequence.append(normalized_landmarks)
        if sequence is None:
            return None

        start = time.perf_counter()
        now = time.time() if timestamp is None else timestamp

        with torch.no_grad():
            x = torch.from_numpy(sequence[None]).to(self.device)
            logits, embedding = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(probs.argmax())
        confidence = float(probs[idx])
        pred_label = self.labels[idx]

        # Calculate prediction entropy to detect uncertain / unknown gestures
        probs_clamped = np.clip(probs, 1e-8, 1.0)
        entropy = -float(np.sum(probs_clamped * np.log(probs_clamped)))
        is_unknown = entropy > self.entropy_threshold or confidence < self.threshold

        self._history.append(pred_label if not is_unknown else "uncertain")
        temporal_confirmed = (len(self._history) >= 2 and self._history[-1] == self._history[-2])

        accepted = (not is_unknown) and temporal_confirmed and (confidence >= self.threshold)

        latency = (time.perf_counter() - start) * 1000

        labels_payload = (
            [{"class": pred_label, "confidence": confidence}]
            if accepted
            else [{"class": "uncertain", "confidence": confidence}] if is_unknown else []
        )

        metadata = {
            "probabilities": {label: float(probs[i]) for i, label in enumerate(self.labels)},
            "top_class": pred_label if not is_unknown else "uncertain",
            "raw_top_class": pred_label,
            "raw_confidence": confidence,
            "entropy": entropy,
            "threshold_accepted": accepted,
            "temporal_confirmed": temporal_confirmed,
            "supported_classes_count": len(self.labels),
        }

        return ModelEvent(
            modality="gesture",
            timestamp=now,
            feature=embedding[0].cpu().numpy(),
            confidence=confidence,
            labels=labels_payload,
            available=True,
            latency_ms=latency,
            metadata=metadata,
        )
