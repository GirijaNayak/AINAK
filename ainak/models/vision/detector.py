from __future__ import annotations
import time
import os
from pathlib import Path
from typing import Any
import numpy as np
from ...runtime.events import Detection, ModelEvent


from .depth import RelativeDepthEstimator
from .hazard_scorer import VisionHazardScorer, ScoredHazard


class VisionDetector:
    """YOLOv8 detector. The 128-vector summarizes actual normalized detections, not scene geometry."""
    def __init__(self, weights: str | Path, device: str = "cpu", confidence: float = .35, iou: float = .45):
        path = Path(weights)
        if not path.is_file(): raise FileNotFoundError(f"YOLO weights missing: {path}. Run `python -m ainak.interfaces.cli acquire vision`.")
        try:
            # Keep Ultralytics configuration writable inside the project artifacts tree on locked-down Windows profiles.
            os.environ.setdefault("YOLO_CONFIG_DIR", str(path.parent / ".ultralytics"))
            from ultralytics import YOLO
        except ImportError as exc: raise RuntimeError("Install project dependencies to use vision inference") from exc
        self.model = YOLO(str(path)); self.device = device; self.confidence = confidence; self.iou = iou
        # Track state: tuple of (bbox, class_name, track_id, timestamp, relative_depth, age_frames)
        self._tracks: list[tuple[tuple[float,float,float,float], str, int, float, float, int]] = []
        self._next_track = 0
        self.depth_estimator = RelativeDepthEstimator()
        self.hazard_scorer = VisionHazardScorer(min_confidence=confidence)

    def infer(self, frame: np.ndarray, timestamp: float | None = None) -> tuple[list[Detection], ModelEvent, Any]:
        now = time.time() if timestamp is None else timestamp
        start = time.perf_counter()
        result = self.model.predict(frame, device=self.device, conf=self.confidence, iou=self.iou, verbose=False)[0]
        boxes = result.boxes
        names = result.names
        dets: list[Detection] = []
        h, w = frame.shape[:2]
        vector = np.zeros(128, dtype=np.float32)
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)
            cls = int(boxes.cls[i].item()); conf = float(boxes.conf[i].item())
            coords = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            dets.append(Detection(str(names[cls]), conf, coords, now))
            # Stable lightweight learned-input representation: COCO-class histogram + normalized geometry/confidence statistics.
            vector[cls % 80] += conf
            cx, cy = (xyxy[0] + xyxy[2]) / (2*w), (xyxy[1] + xyxy[3]) / (2*h)
            area = max(0., (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(w*h))
            vector[80:84] += np.array([cx, cy, area, conf], dtype=np.float32) / max(len(boxes), 1)

        # Greedy class-consistent IoU tracking with temporal depth and age tracking
        next_tracks = []
        used = set()
        scored_hazards: list[ScoredHazard] = []

        for detection in dets:
            best_iou = .3
            best_match = None
            for old_bbox, old_class, track_id, old_time, old_depth, old_age in self._tracks:
                if track_id in used or old_class != detection.class_name or (now - old_time) > 1.0:
                    continue
                x1 = max(old_bbox[0], detection.bbox_xyxy[0]); y1 = max(old_bbox[1], detection.bbox_xyxy[1])
                x2 = min(old_bbox[2], detection.bbox_xyxy[2]); y2 = min(old_bbox[3], detection.bbox_xyxy[3])
                inter = max(0., x2 - x1) * max(0., y2 - y1)
                a = max(0., old_bbox[2] - old_bbox[0]) * max(0., old_bbox[3] - old_bbox[1])
                b = max(0., detection.bbox_xyxy[2] - detection.bbox_xyxy[0]) * max(0., detection.bbox_xyxy[3] - detection.bbox_xyxy[1])
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
        if len(boxes): vector[:80] /= max(float(vector[:80].sum()), 1e-6)
        latency = (time.perf_counter()-start)*1000

        # Detailed event labels including depth and hazard cues
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
        }

        event = ModelEvent(
            modality="vision",
            timestamp=now,
            feature=vector,
            confidence=max((d.confidence for d in dets), default=0.),
            labels=labels_payload,
            available=True,
            latency_ms=latency,
            metadata=metadata,
        )
        return dets, event, result.plot()

    @staticmethod
    def camera_horizontal_cue(detection: Detection, image_width: int) -> str:
        center = (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2
        return "left" if center < image_width/3 else "right" if center > 2*image_width/3 else "center"

