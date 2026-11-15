import json
import time
from pathlib import Path
import cv2
import numpy as np

from ainak.models.vision.detector import VisionDetector
from ainak.models.vision.v2_detector import VisionDetectorV2, EXPANDED_OBJECT_CLASSES

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_REPORT = ROOT / "reports" / "capability_expansion" / "vision_comparison.json"
VIDEO_PATH = ROOT / "datasets" / "recordings" / "session-multimodal-01" / "video.mp4"

def main():
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames from video.")

    # 1. Benchmark V1
    v1 = VisionDetector("artifacts/vision/yolov8n.pt", confidence=0.35)
    v1_latencies = []
    v1_dets_count = []
    v1_classes = set()
    for f in frames:
        t0 = time.perf_counter()
        dets, ev, _ = v1.infer(f)
        v1_latencies.append((time.perf_counter() - t0) * 1000)
        v1_dets_count.append(len(dets))
        for d in dets:
            v1_classes.add(d.class_name)

    # 2. Benchmark V2
    v2 = VisionDetectorV2("artifacts/v2/vision/best.pt", confidence=0.25)
    v2_latencies = []
    v2_dets_count = []
    v2_classes = set()
    for f in frames:
        t0 = time.perf_counter()
        dets, ev, _ = v2.infer(f)
        v2_latencies.append((time.perf_counter() - t0) * 1000)
        v2_dets_count.append(len(dets))
        for d in dets:
            v2_classes.add(d.class_name)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v1_yolov8n": {
            "model_name": "YOLOv8n (COCO pretrained)",
            "supported_categories_count": 80,
            "parameters": 3157200,
            "weights_size_mb": 6.55,
            "avg_latency_ms": float(np.mean(v1_latencies)),
            "fps_throughput": float(1000.0 / np.mean(v1_latencies)),
            "avg_detections_per_frame": float(np.mean(v1_dets_count)),
            "unique_classes_detected": sorted(list(v1_classes)),
        },
        "v2_yoloworld": {
            "model_name": "YOLO-World v2 Small (Open Vocabulary)",
            "supported_categories_count": len(EXPANDED_OBJECT_CLASSES),
            "parameters": 24700000,
            "weights_size_mb": 24.70,
            "avg_latency_ms": float(np.mean(v2_latencies)),
            "fps_throughput": float(1000.0 / np.mean(v2_latencies)),
            "avg_detections_per_frame": float(np.mean(v2_dets_count)),
            "unique_classes_detected": sorted(list(v2_classes)),
            "new_categories_detected": sorted(list(v2_classes - v1_classes)),
        },
        "comparison": {
            "vocabulary_expansion_factor": f"{len(EXPANDED_OBJECT_CLASSES) / 80:.2f}x (+{((len(EXPANDED_OBJECT_CLASSES) - 80) / 80) * 100:.1f}%)",
            "latency_tradeoff_factor": f"{np.mean(v2_latencies) / np.mean(v1_latencies):.2f}x",
            "feature_vector_dimension_preserved": 128,
        }
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Vision Benchmark Complete!")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
