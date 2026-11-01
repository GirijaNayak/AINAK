from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import torch

from ..models.vision.detector import VisionDetector


def profile_vision_images(sources: list[Path], weights: Path, device: str = "cpu", image_size: int = 640,
                         warmup: int = 1, repeats: int = 3) -> dict:
    if not sources or image_size < 32 or warmup < 0 or repeats < 1:
        raise ValueError("provide images, image_size >= 32, warmup >= 0, repeats >= 1")
    images = []
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"vision image not found: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unsupported or unreadable image input: {path}")
        images.append((path, image))
    detector = VisionDetector(weights, device)
    records = []
    for path, image in images:
        for _ in range(warmup):
            detector.model.predict(image, device=device, imgsz=image_size, verbose=False)
        timings = []
        result = None
        for _ in range(repeats):
            start = time.perf_counter()
            result = detector.model.predict(image, device=device, imgsz=image_size, verbose=False)[0]
            timings.append((time.perf_counter() - start) * 1000)
        assert result is not None
        names = result.names
        detections = [{"class": str(names[int(box.cls.item())]), "confidence": float(box.conf.item()),
                       "bbox_xyxy": [float(v) for v in box.xyxy[0].detach().cpu().tolist()]}
                      for box in result.boxes]
        speed = result.speed or {}
        records.append({"image": str(path), "width": int(image.shape[1]), "height": int(image.shape[0]),
                        "detections": detections, "warmup_runs": warmup, "measured_runs": repeats,
                        "wall_latency_ms_mean": sum(timings) / len(timings),
                        "ultralytics_preprocess_ms": float(speed.get("preprocess", 0.)),
                        "ultralytics_inference_ms": float(speed.get("inference", 0.)),
                        "ultralytics_postprocess_ms": float(speed.get("postprocess", 0.))})
    checkpoint_sha = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_sha.update(chunk)
    return {"device": device, "image_size": image_size, "checkpoint": str(weights),
            "checkpoint_sha256": checkpoint_sha.hexdigest(), "python": platform.python_version(),
            "torch": torch.__version__, "results": records,
            "limitations": "Image samples only; no ground-truth labels, so no detection accuracy or mAP is computed. Wall timing includes Python/predictor overhead; Ultralytics stage times are reported separately."}
