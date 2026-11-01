"""Downstream classifier and inference module for PANNs CNN14 audio embeddings."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
from torch import nn

from ...runtime.events import ModelEvent


class AudioMLP(nn.Module):
    """Downstream Multi-Layer Perceptron trained on 2,048-D PANNs embeddings."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, num_classes: int = 50, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownstreamAudioClassifier:
    """Inference wrapper for trained downstream environmental sound classifier.

    Features:
    - Calibrated softmax probabilities.
    - Confidence thresholding and reject/uncertain state.
    - Temporal smoothing over sliding prediction window.
    - Top-k prediction ranking.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        confidence_threshold: float = 0.35,
        smoothing_window: int = 3,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Downstream audio checkpoint missing: {path}")

        payload = torch.load(path, map_location=device, weights_only=False)
        self.classes = payload["classes"]
        self.model_type = payload.get("model_type", "mlp")
        self.num_classes = len(self.classes)

        if self.model_type == "mlp":
            self.model = AudioMLP(
                input_dim=payload.get("input_dim", 2048),
                hidden_dim=payload.get("hidden_dim", 256),
                num_classes=self.num_classes,
                dropout=payload.get("dropout", 0.2),
            ).to(device)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()
        else:
            raise ValueError(f"Unsupported downstream model type: {self.model_type}")

        self._history: deque[np.ndarray] = deque(maxlen=smoothing_window)

    def predict_embedding(
        self,
        embedding: np.ndarray,
        timestamp: float | None = None,
        top_k: int = 5,
    ) -> tuple[str, float, list[dict[str, Any]], ModelEvent]:
        """Classify a 2,048-dimensional PANNs embedding."""
        now = time.time() if timestamp is None else timestamp
        start = time.perf_counter()

        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != 2048:
            raise ValueError(f"Expected 2048-D embedding, got {vec.shape[1]}")

        with torch.no_grad():
            x = torch.from_numpy(vec).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # Update smoothing history
        self._history.append(probs)
        smoothed_probs = np.mean(self._history, axis=0)

        # Ranked predictions
        ranked_indices = np.argsort(smoothed_probs)[::-1]
        top_idx = int(ranked_indices[0])
        top_class = self.classes[top_idx]
        top_conf = float(smoothed_probs[top_idx])

        # Confidence gate: reject or mark as uncertain if below threshold
        is_uncertain = top_conf < self.confidence_threshold
        effective_class = "uncertain" if is_uncertain else top_class

        top_k_list = [
            {"class": self.classes[i], "confidence": float(smoothed_probs[i])}
            for i in ranked_indices[:top_k]
        ]

        latency = (time.perf_counter() - start) * 1000

        event = ModelEvent(
            modality="audio",
            timestamp=now,
            feature=vec.reshape(-1),
            confidence=top_conf,
            labels=top_k_list,
            available=True,
            latency_ms=latency,
            metadata={
                "top_class": top_class,
                "effective_class": effective_class,
                "is_uncertain": is_uncertain,
                "raw_confidence": float(probs[top_idx]),
                "smoothed_confidence": top_conf,
                "smoothing_window_size": len(self._history),
            },
        )

        return effective_class, top_conf, top_k_list, event
