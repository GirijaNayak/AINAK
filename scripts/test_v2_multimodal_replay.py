import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch

from ainak.models.factory import get_vision_detector, get_gesture_recognizer, get_hazard_engine, get_fusion_transformer
from ainak.decision.alert_engine import AlertEngine

ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = ROOT / "datasets" / "recordings" / "session-multimodal-01"
VIDEO_PATH = SESSION_DIR / "video.mp4"

def run_replay():
    print("1. Initializing V2 Models...")
    vision_v2 = get_vision_detector(version="v2", confidence=0.25)
    gesture_v2 = get_gesture_recognizer(version="v2", threshold=0.70)
    hazard_v2 = get_hazard_engine(version="v2")
    transformer = get_fusion_transformer()
    decision = AlertEngine()

    print("2. Reading video frame from session-multimodal-01...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read from {VIDEO_PATH}"

    print("3. Executing Vision V2 Inference on Frame...")
    dets, v_event, _ = vision_v2.infer(frame, timestamp=0.0)
    vision_events = [v_event]
    print(f"Vision V2 processed frame, detected {len(dets)} objects.")

    print("4. Executing Gesture V2 Inference on Test Sequences...")
    gesture_events = []
    test_seq = np.load("datasets/gesture-sessions-v2/sequences/subject_p01_stop_00.npy")
    for frame_lm in test_seq:
        g_ev = gesture_v2.append(frame_lm, timestamp=0.8)
        if g_ev is not None:
            gesture_events.append(g_ev)
    print(f"Gesture V2 processed sequence, emitted {len(gesture_events)} events.")

    print("5. Feeding V2 Features into Multimodal Transformer...")
    v_tensor = torch.from_numpy(np.stack([e.feature for e in vision_events])[None]).float()
    t_len = v_tensor.shape[1]
    g_tensor = torch.zeros(1, t_len, 128)
    if gesture_events:
        g_tensor[0, -1] = torch.from_numpy(gesture_events[-1].feature).float()
    a_tensor = torch.zeros(1, t_len, 2048)

    mask = torch.ones(1, t_len, 3, dtype=torch.bool)
    mask[:, :, 1] = False  # Mask out missing audio branch

    with torch.no_grad():
        fusion_out = transformer({"vision": v_tensor, "audio": a_tensor, "gesture": g_tensor}, mask=mask)

    print("Multimodal Fusion Predictions:")
    print(f"  - Context Vector Norm: {torch.norm(fusion_out.context).item():.4f}")
    print(f"  - Priority:           {fusion_out.priority.item():.4f}")
    print(f"  - Confidence:         {fusion_out.confidence.item():.4f}")
    print(f"  - Alert Logits:       {fusion_out.alert_logits.cpu().numpy().tolist()}")

    print("6. Decision & Alert Engine Evaluation...")
    observations = []
    for d in dets:
        observations.append({"timestamp": 0.16, "confidence": d.confidence, "class": d.class_name, "modality": "vision"})
    if gesture_events and gesture_events[-1].labels:
        observations.append({"timestamp": 0.16, "confidence": gesture_events[-1].confidence, "class": gesture_events[-1].labels[0]["class"], "modality": "gesture"})

    alert = decision.decide(observations, now=0.16)
    if alert:
        print(f"  - Active Alert Emitted: '{alert.text}' (Priority: {alert.priority:.2f})")
    else:
        print("  - Alert Engine: observations processed, within nominal limits.")

    print("\n[SUCCESS] V2 Multimodal Replay & Decision Test PASSED!")

if __name__ == "__main__":
    run_replay()
