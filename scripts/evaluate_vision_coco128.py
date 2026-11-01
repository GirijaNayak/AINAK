"""Run YOLOv8n validation on COCO128 to evaluate genuine mAP@0.5, mAP@0.5:0.95, precision, and recall."""
from __future__ import annotations

import json
import time
from pathlib import Path
from ultralytics import YOLO


def evaluate_vision_coco128(
    weights: Path = Path("artifacts/vision/yolov8n.pt"),
    output: Path = Path("reports/vision-coco128-val.json"),
    device: str = "cpu",
    imgsz: int = 640,
) -> dict:
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    start = time.perf_counter()
    metrics = model.val(data="coco128.yaml", device=device, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    # Extract metrics from Ultralytics Results object
    box = metrics.box
    report = {
        "benchmark": "COCO128 validation",
        "dataset": "coco128.yaml",
        "weights": str(weights),
        "device": device,
        "image_size": imgsz,
        "evaluation_duration_seconds": elapsed,
        "metrics": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "speed_per_image_ms": {
            "preprocess": float(metrics.speed.get("preprocess", 0.0)),
            "inference": float(metrics.speed.get("inference", 0.0)),
            "loss": float(metrics.speed.get("loss", 0.0)),
            "postprocess": float(metrics.speed.get("postprocess", 0.0)),
        },
        "summary": (
            f"COCO128 Validation: mAP@0.5={box.map50:.4f}, mAP@0.5:0.95={box.map:.4f}, "
            f"Precision={box.mp:.4f}, Recall={box.mr:.4f} across 128 instances."
        ),
    }

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = evaluate_vision_coco128()
    print(json.dumps(rep, indent=2))
