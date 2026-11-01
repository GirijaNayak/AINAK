from __future__ import annotations
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from ...runtime.events import ModelEvent
from ...runtime.checkpoint_validation import real_training_checkpoint_issues

GESTURE_LABELS = ("pointing", "waving", "stop", "neutral")
LANDMARKS = 33
FEATURES = LANDMARKS * 4  # normalized x/y/z + visibility
SEQUENCE_LENGTH = 24


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    """Input (33,4); center on hips, scale by shoulder width, zero low-visibility points."""
    p = np.asarray(points, dtype=np.float32).reshape(LANDMARKS, 4).copy()
    p = np.nan_to_num(p, nan=0., posinf=0., neginf=0.)
    # MediaPipe indices: left/right shoulder 11/12; hips 23/24.
    center = (p[23, :3] + p[24, :3]) / 2
    scale = max(float(np.linalg.norm(p[11, :2] - p[12, :2])), 1e-3)
    p[:, :3] = (p[:, :3] - center) / scale
    p[p[:, 3] < .2, :3] = 0.
    p[:, :3] = np.clip(p[:, :3], -5, 5)
    p[:, 3] = np.clip(p[:, 3], 0, 1)
    return p.reshape(-1)


class GestureLSTM(nn.Module):
    def __init__(self, feature_dim: int = FEATURES, hidden_size: int = 128, classes: int = len(GESTURE_LABELS),
                 dropout: float = .2, layers: int = 2):
        super().__init__()
        if layers < 1 or hidden_size < 1: raise ValueError("layers and hidden_size must be positive")
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, classes)
    def forward(self, x: torch.Tensor):
        _, (h, _) = self.lstm(x)
        embedding = self.norm(h[-1])
        return self.classifier(embedding), embedding


class GestureSequenceBuffer:
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        if sequence_length < 2: raise ValueError("sequence_length must be >=2")
        self.sequence_length=sequence_length;self._sequence: deque[np.ndarray]=deque(maxlen=sequence_length)
    def append(self, landmarks: np.ndarray) -> np.ndarray | None:
        self._sequence.append(np.asarray(landmarks,dtype=np.float32).reshape(FEATURES))
        return np.stack(self._sequence) if len(self._sequence)==self.sequence_length else None
    def clear(self) -> None:self._sequence.clear()


class GestureRecognizer:
    def __init__(self, checkpoint: str | Path, seq_len: int = 24, device: str = "cpu", threshold: float = .75):
        path = Path(checkpoint)
        if not path.is_file(): raise FileNotFoundError(f"gesture LSTM checkpoint missing: {path}; collect labeled data and train it first")
        self.device, self.seq_len, self.threshold = device, seq_len, threshold
        payload = torch.load(path, map_location=device, weights_only=False)
        issues = real_training_checkpoint_issues(payload, "gesture")
        if issues:
            raise ValueError("gesture checkpoint provenance rejected: " + "; ".join(issues))
        if tuple(payload.get("labels", ())) != GESTURE_LABELS: raise ValueError(f"gesture label map mismatch; expected {GESTURE_LABELS}")
        self.model = GestureLSTM(**payload.get("model_config", {})).to(device)
        self.model.load_state_dict(payload["state_dict"]); self.model.eval()
        self.sequence = GestureSequenceBuffer(seq_len)

    def append(self, normalized_landmarks: np.ndarray, timestamp: float | None = None) -> ModelEvent | None:
        sequence=self.sequence.append(normalized_landmarks)
        if sequence is None: return None
        start = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(sequence[None]).to(self.device)
            logits, embedding = self.model(x); probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax()); confidence = float(probs[idx]); accepted = confidence >= self.threshold
        return ModelEvent("gesture", time.time() if timestamp is None else timestamp, embedding[0].cpu().numpy(), confidence,
            ([{"class": GESTURE_LABELS[idx], "confidence": confidence}] if accepted else []), True,
            (time.perf_counter()-start)*1000,
            {"probabilities": {label: float(probs[i]) for i, label in enumerate(GESTURE_LABELS)},
             "top_class": GESTURE_LABELS[idx], "threshold_accepted": accepted})


def extract_pose_landmarks(frame: np.ndarray, pose_model, timestamp_ms: int | None = None) -> np.ndarray | None:
    """Use an initialized MediaPipe Pose object; returns normalized 132-vector or None."""
    import cv2
    from .pose import TasksPoseAdapter
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb, timestamp_ms=timestamp_ms) if isinstance(pose_model, TasksPoseAdapter) else pose_model.process(rgb)
    if not result.pose_landmarks: return None
    points = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark], dtype=np.float32)
    return normalize_landmarks(points)
