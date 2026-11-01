from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    artifact_dir: Path = Path("artifacts")
    dataset_dir: Path = Path("datasets")
    device: str = "auto"
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    vision_confidence: float = 0.35
    vision_iou: float = 0.45
    audio_sample_rate: int = 32000
    audio_window_seconds: float = 2.0
    audio_threshold: float = 0.35
    gesture_sequence_length: int = 24
    gesture_threshold: float = 0.75
    sync_window_seconds: float = 2.0
    stale_after_seconds: float = 3.0
    dropout: float = 0.1
    feedforward_dim: int = 256
    fusion_threshold: float = 0.7
    modality_dropout: float = 0.1

    def validate(self) -> "RuntimeConfig":
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        for name in ("vision_confidence", "vision_iou", "audio_threshold", "gesture_threshold", "fusion_threshold", "dropout", "modality_dropout"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.gesture_sequence_length < 2 or self.audio_window_seconds <= 0 or self.sync_window_seconds <= 0 or self.stale_after_seconds <= 0:
            raise ValueError("sequence length and time windows must be positive")
        if self.camera_width < 1 or self.camera_height < 1 or self.camera_fps < 1:
            raise ValueError("camera dimensions and fps must be positive")
        return self

    def resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch runtime has no available CUDA device")
        return self.device


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    @property
    def vision_weights(self) -> Path: return self.root / "artifacts" / "vision" / "yolov8n.pt"
    @property
    def audio_weights(self) -> Path: return self.root / "artifacts" / "audio" / "Cnn14_mAP=0.431.pth"
    @property
    def gesture_weights(self) -> Path: return self.root / "artifacts" / "gesture" / "gesture_lstm.pt"
    @property
    def fusion_weights(self) -> Path: return self.root / "artifacts" / "fusion" / "fusion.pt"
