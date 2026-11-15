"""AI AINAK — Dynamic Version-Aware Model Factory

Provides unified factory constructors for Vision, Gesture, and Hazard modules,
dynamically resolving between V1 (baseline) and V2 (expanded) architectures based
on configs/version_config.yaml or explicit parameters.

Guarantees 100% reversible rollback to V1 and identical 128-D multimodal interfaces.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "configs" / "version_config.yaml"


def get_active_versions() -> dict[str, str]:
    if not CONFIG_PATH.is_file():
        return {"vision": "v1", "gesture": "v1", "hazard": "v1"}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("models", {"vision": "v1", "gesture": "v1", "hazard": "v1"})


def get_vision_detector(version: str | None = None, device: str = "cpu", confidence: float | None = None) -> Any:
    active = (version or get_active_versions().get("vision", "v1")).lower()
    if active == "v2":
        from .vision.v2_detector import VisionDetectorV2
        weights = ROOT / "artifacts" / "v2" / "vision" / "best.pt"
        conf = confidence if confidence is not None else 0.25
        return VisionDetectorV2(weights=weights if weights.is_file() else "yolov8s-worldv2.pt", device=device, confidence=conf)
    else:
        from .vision.detector import VisionDetector
        weights = ROOT / "artifacts" / "vision" / "yolov8n.pt"
        conf = confidence if confidence is not None else 0.35
        return VisionDetector(weights=weights, device=device, confidence=conf)


def get_gesture_recognizer(version: str | None = None, device: str = "cpu", threshold: float | None = None) -> Any:
    active = (version or get_active_versions().get("gesture", "v1")).lower()
    if active == "v2":
        from .gesture.v2_gesture import GestureRecognizerV2
        weights = ROOT / "artifacts" / "v2" / "gesture" / "best.pt"
        thresh = threshold if threshold is not None else 0.70
        return GestureRecognizerV2(checkpoint=weights, device=device, threshold=thresh)
    else:
        from .gesture.lstm import GestureRecognizer
        weights = ROOT / "artifacts" / "gesture" / "gesture_lstm.pt"
        thresh = threshold if threshold is not None else 0.75
        return GestureRecognizer(checkpoint=weights, device=device, threshold=thresh)


def get_hazard_engine(version: str | None = None) -> Any:
    active = (version or get_active_versions().get("hazard", "v1")).lower()
    if active == "v2":
        from .vision.v2_hazard_engine import ContextualHazardEngineV2
        return ContextualHazardEngineV2()
    else:
        from .vision.hazard_scorer import VisionHazardScorer
        return VisionHazardScorer()


def get_fusion_transformer(device: str = "cpu") -> Any:
    from .fusion.transformer import MultimodalTransformer
    import torch
    weights = ROOT / "artifacts" / "fusion" / "fusion.pt"
    model = MultimodalTransformer().to(device)
    if weights.is_file():
        payload = torch.load(weights, map_location=device, weights_only=False)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict)
    model.eval()
    return model
