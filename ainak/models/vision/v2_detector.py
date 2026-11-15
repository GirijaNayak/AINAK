"""AI AINAK — Vision V2 Expanded Object Detector (YOLO-World Open Vocabulary)

Expands the visual detection vocabulary from 80 COCO categories to 220+
accessibility, everyday navigation, infrastructure, and hazard categories.

Preserves 100% compatibility with AINAK's 128-dimensional multimodal
feature interface, tracking logic, and depth estimation.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np

from ...runtime.events import Detection, ModelEvent
from .depth import RelativeDepthEstimator
from .hazard_scorer import VisionHazardScorer, ScoredHazard

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CLASSES_JSON_PATH = ROOT / "artifacts" / "v2" / "vision" / "yolo_world_classes.json"

# 225 Curated Accessibility, Navigation, Everyday & Hazard Categories
EXPANDED_OBJECT_CLASSES: list[str] = [
    # 1. Everyday personal objects
    "keys", "wallet", "purse", "backpack", "handbag", "suitcase", "umbrella",
    "eyeglasses", "sunglasses", "cell phone", "laptop", "tablet", "charger",
    "power bank", "cable", "headphones", "earphones", "watch", "water bottle",
    "thermos", "mug", "cup", "glass", "plate", "bowl", "fork", "spoon", "knife",
    "scissors", "pen", "pencil", "notebook", "book", "paper", "toothbrush",
    "toothpaste", "soap", "towel", "comb", "hairbrush", "mirror", "tissue box",
    "pills", "medicine bottle",

    # 2. Furniture and indoor infrastructure
    "door", "closed door", "open door", "door handle", "window", "window blinds",
    "stairs", "staircase", "step", "curb", "ramp", "handrail", "railing",
    "floor", "carpet", "mat", "wall", "ceiling", "pillar", "table", "dining table",
    "coffee table", "desk", "chair", "office chair", "armchair", "stool", "bench",
    "sofa", "couch", "bed", "mattress", "pillow", "blanket", "wardrobe", "cupboard",
    "cabinet", "drawer", "shelf", "bookshelf", "refrigerator", "freezer", "microwave",
    "oven", "stove", "toaster", "sink", "faucet", "dishwasher", "washing machine",
    "dryer", "trash can", "dustbin", "recycling bin", "ceiling fan", "stand fan",
    "air conditioner", "heater", "radiator", "lamp", "floor lamp", "light switch",
    "electrical outlet", "television", "monitor", "keyboard", "mouse", "printer",
    "clock", "wall clock", "curtain",

    # 3. Outdoor streetscape and navigation
    "streetlight", "street lamp", "traffic light", "pedestrian signal", "traffic sign",
    "stop sign", "yield sign", "speed limit sign", "street sign", "signboard",
    "billboard", "traffic cone", "bollard", "road barrier", "jersey barrier",
    "guard rail", "pothole", "manhole", "drain grate", "fire hydrant", "utility pole",
    "telephone pole", "tree", "tree trunk", "bush", "hedge", "grass", "lawn",
    "flower bed", "sidewalk", "crosswalk", "pedestrian crossing", "road", "street",
    "asphalt", "pavement", "road divider", "curbstone", "fence", "wooden fence",
    "chain link fence", "gate", "turnstile", "park bench", "bus stop", "bus shelter",
    "mailbox", "bicycle rack", "parking meter",

    # 4. Vehicles and transportation
    "car", "sedan", "suv", "taxi", "police car", "ambulance", "fire truck", "bus",
    "school bus", "city bus", "truck", "delivery truck", "pickup truck", "semi-trailer",
    "motorcycle", "moped", "scooter", "electric scooter", "bicycle", "mountain bike",
    "auto-rickshaw", "rickshaw", "golf cart", "forklift", "tractor", "train",
    "subway train", "tram", "trolley", "train track", "railroad crossing",

    # 5. People and accessibility aids
    "person", "pedestrian", "man", "woman", "child", "baby", "toddler",
    "elderly person", "police officer", "construction worker", "cyclist",
    "wheelchair", "motorized wheelchair", "walking stick", "white cane", "crutches",
    "walker", "rollator", "stroller", "pram", "service dog", "guide dog", "dog",
    "cat", "pet", "elevator", "escalator", "moving walkway", "tactile paving",
    "braille sign", "accessible entrance",

    # 6. Hazards and safety threats
    "fire", "flame", "open flame", "smoke", "fire extinguisher", "broken glass",
    "sharp knife", "saw", "hammer", "drill", "power tool", "exposed wire",
    "electrical wire", "power line", "wet floor sign", "caution tape",
    "construction equipment", "excavator", "bulldozer", "crane", "scaffolding",
    "trench", "open ditch", "moving machinery", "industrial equipment"
]


class VisionDetectorV2:
    """Expanded Open-Vocabulary Vision Detector using YOLO-World v2.

    Recognizes 220+ categories, tracks objects across frames, calculates
    perspective depth and looming rates, and outputs a 128-D multimodal feature vector.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8s-worldv2.pt",
        device: str = "cpu",
        confidence: float = 0.25,
        iou: float = 0.45,
        categories: list[str] | None = None,
    ):
        path = Path(weights)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics required for Vision V2 detector") from exc

        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.categories = categories or EXPANDED_OBJECT_CLASSES

        # Save class dictionary for provenance
        CLASSES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CLASSES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "count": len(self.categories),
                "classes": self.categories,
                "framework": "ultralytics-yolo-worldv2",
                "model_size": "small",
            }, f, indent=2)

        # Initialize model
        weights_target = path if path.is_file() else "yolov8s-worldv2.pt"
        self.model = YOLO(str(weights_target))
        self.model.set_classes(self.categories)

        # State tracking
        self._tracks: list[tuple[tuple[float, float, float, float], str, int, float, float, int]] = []
        self._next_track = 0
        self.depth_estimator = RelativeDepthEstimator()
        self.hazard_scorer = VisionHazardScorer(min_confidence=confidence)

    def infer(self, frame: np.ndarray, timestamp: float | None = None) -> tuple[list[Detection], ModelEvent, Any]:
        now = time.time() if timestamp is None else timestamp
        start = time.perf_counter()

        result = self.model.predict(
            frame,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )[0]

        boxes = result.boxes
        dets: list[Detection] = []
        h, w = frame.shape[:2]

        # 128-D multimodal feature vector
        vector = np.zeros(128, dtype=np.float32)

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)
            cls_idx = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            class_name = self.categories[cls_idx] if cls_idx < len(self.categories) else f"object_{cls_idx}"
            coords = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            dets.append(Detection(class_name, conf, coords, now))

            # Feature projection: map 225 classes into the first 80 feature bins via modular projection
            vector[cls_idx % 80] += conf

            # Geometric coordinates and confidence statistics
            cx = (xyxy[0] + xyxy[2]) / (2.0 * w)
            cy = (xyxy[1] + xyxy[3]) / (2.0 * h)
            area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) / (w * h))
            vector[80:84] += np.array([cx, cy, area, conf], dtype=np.float32) / max(len(boxes), 1)

        # Tracking and relative depth estimation
        next_tracks = []
        used = set()
        scored_hazards: list[ScoredHazard] = []

        for detection in dets:
            best_iou = 0.3
            best_match = None
            for old_bbox, old_class, track_id, old_time, old_depth, old_age in self._tracks:
                if track_id in used or old_class != detection.class_name or (now - old_time) > 1.0:
                    continue
                x1 = max(old_bbox[0], detection.bbox_xyxy[0])
                y1 = max(old_bbox[1], detection.bbox_xyxy[1])
                x2 = min(old_bbox[2], detection.bbox_xyxy[2])
                y2 = min(old_bbox[3], detection.bbox_xyxy[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                a = max(0.0, old_bbox[2] - old_bbox[0]) * max(0.0, old_bbox[3] - old_bbox[1])
                b = max(0.0, detection.bbox_xyxy[2] - detection.bbox_xyxy[0]) * max(0.0, detection.bbox_xyxy[3] - detection.bbox_xyxy[1])
                overlap = inter / max(a + b - inter, 1e-9)
                if overlap > best_iou:
                    best_iou = overlap
                    best_match = (track_id, old_depth, old_time, old_age)

            rel_depth = self.depth_estimator.estimate_proximity(detection.bbox_xyxy, (h, w), detection.class_name)
            detection.relative_depth = rel_depth

            if best_match is not None:
                track_id, prev_depth, prev_time, prev_age = best_match
                dt = max(1e-3, now - prev_time)
                delta_d = (rel_depth - prev_depth) / dt
                age = prev_age + 1
            else:
                track_id = self._next_track
                self._next_track += 1
                delta_d = 0.0
                age = 1

            detection.track_id = track_id
            detection.delta_depth = delta_d
            detection.track_age_frames = age
            used.add(track_id)

            cue = self.camera_horizontal_cue(detection, w)
            hazard = self.hazard_scorer.score_detection(
                class_name=detection.class_name,
                confidence=detection.confidence,
                relative_depth=rel_depth,
                delta_depth=delta_d,
                horizontal_cue=cue,
                track_age_frames=age,
                track_id=track_id,
            )
            detection.hazard_score = hazard.hazard_score
            detection.hazard_level = hazard.hazard_level
            scored_hazards.append(hazard)

            next_tracks.append((detection.bbox_xyxy, detection.class_name, track_id, now, rel_depth, age))

        self._tracks = next_tracks

        # Normalize the first 80 histogram bins
        if len(boxes):
            vector[:80] /= max(float(vector[:80].sum()), 1e-6)

        # Contextual summary statistics in bins 84..90
        vector[84] = max((h.hazard_score for h in scored_hazards), default=0.0)
        vector[85] = float(len(dets)) / 20.0
        vector[86] = sum(1 for h in scored_hazards if h.relative_depth > 0.6) / max(len(dets), 1)

        latency = (time.perf_counter() - start) * 1000

        labels_payload = [
            {
                "class": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy,
                "track_id": d.track_id,
                "relative_depth": d.relative_depth,
                "delta_depth": d.delta_depth,
                "hazard_score": d.hazard_score,
                "hazard_level": d.hazard_level,
                "horizontal_cue": self.camera_horizontal_cue(d, w),
            }
            for d in dets
        ]

        severity_rank = {"immediate_attention": 4, "caution": 3, "informational": 2, "none": 1}
        highest_hazard = max(
            (h.hazard_level for h in scored_hazards),
            key=lambda lvl: severity_rank.get(lvl, 0),
            default="none",
        )

        metadata = {
            "tracked_objects_count": len(dets),
            "max_hazard_score": max((h.hazard_score for h in scored_hazards), default=0.0),
            "highest_hazard_level": highest_hazard,
            "urgent_hazards": [h.rationale for h in scored_hazards if h.hazard_level == "immediate_attention"],
            "supported_vocabulary_size": len(self.categories),
        }

        event = ModelEvent(
            modality="vision",
            timestamp=now,
            feature=vector,
            confidence=max((d.confidence for d in dets), default=0.0),
            labels=labels_payload,
            available=True,
            latency_ms=latency,
            metadata=metadata,
        )

        return dets, event, result.plot()

    @staticmethod
    def camera_horizontal_cue(detection: Detection, image_width: int) -> str:
        center = (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2.0
        return "left" if center < image_width / 3.0 else "right" if center > 2.0 * image_width / 3.0 else "center"
