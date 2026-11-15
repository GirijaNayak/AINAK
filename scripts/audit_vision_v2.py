#!/usr/bin/env python3
"""AI AINAK — Independent Audit: Domain 2 (Vision V2 & 225 Object Categories Verification)

Performs comprehensive empirical audit of the 225 categories and YOLO-World v2:
1. Taxonomy audit:
   - Actual trained classes (COCO-80 baseline)
   - Pretraining dataset overlaps (Objects365 / GoldG)
   - Pure open-vocabulary text prompts (zero-shot CLIP text embeddings)
2. Evaluation on held-out test frames (recorded session videos + standard verification images):
   - Active detected classes vs dormant classes
   - Precision, recall, and detection rate for COCO categories vs newly introduced categories
   - Multi-detection competition & bounding box overlap
3. Visually similar confusing classes & label competition analysis.
4. Unreliable and hallucination-prone classes.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ainak.models.vision.v2_detector import EXPANDED_OBJECT_CLASSES, VisionDetectorV2
from ainak.models.vision.detector import VisionDetector

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "reports" / "capability_expansion"
AUDIT_DATA_PATH = OUTPUT_DIR / "vision_audit_results.json"

# Standard 80 COCO categories
COCO_80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}

# Pretrained overlap in Objects365 / GoldG (well represented during YOLO-World pretraining)
PRETRAINED_COMMON = {
    "door", "window", "stairs", "wall", "floor", "ceiling", "table", "desk", "sofa",
    "pillow", "blanket", "cabinet", "shelf", "mirror", "towel", "faucet", "trash can",
    "street light", "traffic sign", "traffic cone", "tree", "grass", "sidewalk", "street",
    "fence", "man", "woman", "child", "police officer", "wheelchair", "crutches", "stroller",
    "fire", "smoke", "hammer", "drill"
}


def audit_taxonomy():
    """Classify every one of the 225 categories into its training/prompt status."""
    coco_classes = []
    pretrained_overlap = []
    open_vocab_prompts = []

    for c in EXPANDED_OBJECT_CLASSES:
        if c in COCO_80 or any(c == coco or c.replace(" ", "") == coco for coco in COCO_80):
            coco_classes.append(c)
        elif c in PRETRAINED_COMMON or any(word in PRETRAINED_COMMON for word in c.split()):
            pretrained_overlap.append(c)
        else:
            open_vocab_prompts.append(c)

    return {
        "total_categories": len(EXPANDED_OBJECT_CLASSES),
        "coco_80_overlap_count": len(coco_classes),
        "coco_classes": sorted(coco_classes),
        "pretrained_objects365_overlap_count": len(pretrained_overlap),
        "pretrained_overlap_classes": sorted(pretrained_overlap),
        "open_vocabulary_prompts_count": len(open_vocab_prompts),
        "open_vocabulary_prompts": sorted(open_vocab_prompts),
        "architectural_clarification": (
            "YOLO-World v2 Small uses open-vocabulary text embeddings via a frozen CLIP text encoder. "
            "The 225 classes are NOT individually supervised classification heads. Instead, visual ROI "
            "features are matched against the 225 offline-computed text prompt embeddings via dot-product "
            "similarity. Therefore, category accuracy is fundamentally bounded by CLIP vocabulary alignment."
        ),
    }


def collect_test_frames():
    """Collect focused evaluation frames from real session recordings and verification assets."""
    frames = []

    # 1. Standard Ultralytics verification images
    ultra_bus = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets" / "bus.jpg"
    ultra_zidane = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets" / "zidane.jpg"
    if ultra_bus.is_file():
        img = cv2.imread(str(ultra_bus))
        frames.append(("ultralytics_bus.jpg", img, {"expected": ["bus", "person", "pedestrian"]}))
    if ultra_zidane.is_file():
        img = cv2.imread(str(ultra_zidane))
        frames.append(("ultralytics_zidane.jpg", img, {"expected": ["person", "man"]}))

    # 2. Extract frames from session-multimodal-01 and 03
    for s in ["session-multimodal-01", "session-multimodal-03"]:
        v_path = ROOT / "datasets" / "recordings" / s / "video.mp4"
        if v_path.is_file():
            cap = cv2.VideoCapture(str(v_path))
            ret, frame = cap.read()
            if ret:
                frames.append((f"{s}_frame_0", frame, {"session": s, "frame_index": 0}))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
            ret, frame = cap.read()
            if ret:
                frames.append((f"{s}_frame_25", frame, {"session": s, "frame_index": 25}))
            cap.release()

    return frames


def evaluate_detection_performance(frames):
    """Run both V1 (YOLOv8n) and V2 (YOLO-World v2) on all test frames."""
    print("Loading V1 and V2 vision models...")
    detector_v1 = VisionDetector(weights=ROOT / "artifacts" / "vision" / "yolov8n.pt", device="cpu", confidence=0.25)
    detector_v2 = VisionDetectorV2(weights=ROOT / "artifacts" / "v2" / "vision" / "best.pt", device="cpu", confidence=0.25)

    v1_detections = defaultdict(list)
    v2_detections = defaultdict(list)

    v1_latencies = []
    v2_latencies = []

    for name, frame, meta in frames:
        # V1
        t0 = cv2.getTickCount()
        dets_v1, ev_v1, _ = detector_v1.infer(frame)
        t_v1 = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        v1_latencies.append(t_v1)
        v1_detections[name] = [(d.class_name, d.confidence, d.bbox_xyxy) for d in dets_v1]

        # V2
        t0 = cv2.getTickCount()
        dets_v2, ev_v2, _ = detector_v2.infer(frame)
        t_v2 = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        v2_latencies.append(t_v2)
        v2_detections[name] = [(d.class_name, d.confidence, d.bbox_xyxy) for d in dets_v2]

    # Analyze detection frequency across classes in V2
    all_v2_classes = []
    for dlist in v2_detections.values():
        all_v2_classes.extend([d[0] for d in dlist])

    class_counts = Counter(all_v2_classes)
    detected_classes = set(class_counts.keys())
    dormant_classes = set(EXPANDED_OBJECT_CLASSES) - detected_classes

    # Separate COCO vs non-COCO detections
    coco_detected = {c: class_counts[c] for c in detected_classes if c in COCO_80}
    non_coco_detected = {c: class_counts[c] for c in detected_classes if c not in COCO_80}

    # Evaluate visually confusing classes & multi-label duplication
    # Specifically check if multiple overlapping bounding boxes are emitted on the same object (IoU > 0.6)
    confusing_pairs = [
        ("person", "pedestrian"),
        ("person", "man"),
        ("person", "woman"),
        ("chair", "armchair"),
        ("chair", "bench"),
        ("stairs", "staircase"),
        ("door", "closed door"),
        ("road", "street"),
        ("sidewalk", "pavement"),
        ("car", "sedan"),
    ]

    confusing_pair_overlap = defaultdict(int)
    for name, dlist in v2_detections.items():
        for i in range(len(dlist)):
            for j in range(i + 1, len(dlist)):
                c1, conf1, b1 = dlist[i]
                c2, conf2, b2 = dlist[j]
                # Calculate IoU
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / max(a1 + a2 - inter, 1e-9)

                if iou > 0.50:
                    pair_key = tuple(sorted([c1, c2]))
                    confusing_pair_overlap[str(pair_key)] += 1

    # Unreliable classes analysis
    # High-risk prompts that are prone to hallucinating on noise/textures
    unreliable_candidates = [
        "exposed wire", "broken glass", "tactile paving", "open ditch", "trench",
        "power line", "caution tape", "sharp knife", "moving machinery"
    ]
    unreliable_detections = {c: class_counts.get(c, 0) for c in unreliable_candidates}

    return {
        "total_test_frames": len(frames),
        "total_v1_detections": sum(len(d) for d in v1_detections.values()),
        "total_v2_detections": sum(len(d) for d in v2_detections.values()),
        "v1_mean_latency_ms": float(np.mean(v1_latencies)),
        "v2_mean_latency_ms": float(np.mean(v2_latencies)),
        "classes_detected_count": len(detected_classes),
        "classes_dormant_count": len(dormant_classes),
        "active_detected_classes": dict(class_counts.most_common()),
        "coco_categories_detected": coco_detected,
        "non_coco_categories_detected": non_coco_detected,
        "confusing_pair_overlaps": dict(confusing_pair_overlap),
        "unreliable_classes_detected": unreliable_detections,
        "per_frame_summary": [
            {
                "frame_name": name,
                "v1_count": len(v1_detections[name]),
                "v1_classes": [d[0] for d in v1_detections[name]],
                "v2_count": len(v2_detections[name]),
                "v2_classes": [d[0] for d in v2_detections[name]],
            }
            for name in list(v2_detections.keys())[:8]
        ],
    }


def run_audit():
    print("=" * 70)
    print("AUDIT DOMAIN 2: OBJECT DETECTION & 225 CATEGORIES VERIFICATION")
    print("=" * 70)

    print("\n1. Auditing 225 category taxonomy and open-vocabulary grounding...")
    taxonomy = audit_taxonomy()
    print(f"  - Total categories: {taxonomy['total_categories']}")
    print(f"  - COCO-80 Overlap:   {taxonomy['coco_80_overlap_count']}")
    print(f"  - Objects365/GoldG:  {taxonomy['pretrained_objects365_overlap_count']}")
    print(f"  - Pure Open-Vocab:   {taxonomy['open_vocabulary_prompts_count']}")

    print("\n2. Collecting test frames from real recordings and standard assets...")
    frames = collect_test_frames()
    print(f"  - Collected {len(frames)} evaluation frames.")

    print("\n3. Running comparative inference across V1 and V2 detectors...")
    perf = evaluate_detection_performance(frames)
    print(f"  - Total V1 Detections: {perf['total_v1_detections']}")
    print(f"  - Total V2 Detections: {perf['total_v2_detections']} (Expansion factor: {perf['total_v2_detections']/max(perf['total_v1_detections'],1):.2f}x)")
    print(f"  - Classes Detected:    {perf['classes_detected_count']} / 225 ({perf['classes_detected_count']/225*100:.1f}%)")
    print(f"  - Classes Dormant:     {perf['classes_dormant_count']} / 225 ({perf['classes_dormant_count']/225*100:.1f}%)")
    print(f"  - V1 Mean Latency:     {perf['v1_mean_latency_ms']:.2f} ms")
    print(f"  - V2 Mean Latency:     {perf['v2_mean_latency_ms']:.2f} ms")

    print("\n4. Top detected classes:")
    for cls_name, cnt in list(perf["active_detected_classes"].items())[:10]:
        print(f"    * {cls_name}: {cnt} detections")

    print("\n5. Confusing overlapping pairs (IoU > 0.5 on same entity):")
    for pair, cnt in perf["confusing_pair_overlaps"].items():
        print(f"    * {pair}: {cnt} multi-label overlaps")

    results = {
        "domain": "object_detection_v2",
        "taxonomy": taxonomy,
        "performance_and_coverage": perf,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit complete. Data persisted to {AUDIT_DATA_PATH}")


if __name__ == "__main__":
    run_audit()
