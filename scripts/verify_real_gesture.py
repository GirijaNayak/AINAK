"""Real-data gesture verification script.

Loads the actual pose_landmarker_lite.task checkpoint, extracts real pose landmarks from a
real recorded video frame, runs normalization, buffers 24 real consecutive frames, and executes
inference through the actual trained GestureLSTM model.
"""
from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np

from ainak.models.gesture.pose import TasksPoseAdapter
from ainak.models.gesture.lstm import GestureRecognizer, normalize_landmarks

def run_real_gesture_verification():
    video_path = Path("datasets/recordings/session-multimodal-01/video.mp4")
    pose_model_path = Path("artifacts/gesture/pose_landmarker_lite.task")
    lstm_model_path = Path("artifacts/gesture/gesture_lstm.pt")

    assert video_path.is_file(), f"Video file not found: {video_path}"
    assert pose_model_path.is_file(), f"Pose model not found: {pose_model_path}"
    assert lstm_model_path.is_file(), f"LSTM checkpoint not found: {lstm_model_path}"

    print(f"1. Loading pose landmarker from: {pose_model_path}")
    t0_load = time.perf_counter()
    tracker = TasksPoseAdapter(pose_model_path, static_image_mode=True)
    print(f"   Pose landmarker loaded in {(time.perf_counter() - t0_load)*1000:.2f} ms")

    print(f"2. Loading trained GestureLSTM from: {lstm_model_path}")
    t0_lstm_load = time.perf_counter()
    recognizer = GestureRecognizer(lstm_model_path, seq_len=24)
    print(f"   GestureRecognizer loaded in {(time.perf_counter() - t0_lstm_load)*1000:.2f} ms")

    print(f"3. Processing real video frames from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_count = 0
    detected_count = 0
    first_landmarks_raw = None
    first_normalized_vec = None
    emitted_events = []
    total_pose_latency_ms = 0.0

    while cap.isOpened() and frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_count * 1000 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t_p_start = time.perf_counter()
        res = tracker.process(rgb, timestamp_ms=timestamp_ms)
        p_latency = (time.perf_counter() - t_p_start) * 1000
        total_pose_latency_ms += p_latency

        if res.pose_landmarks and hasattr(res.pose_landmarks, "landmark"):
            detected_count += 1
            raw_lms = res.pose_landmarks.landmark
            points = np.array(
                [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in raw_lms],
                dtype=np.float32,
            )
            if first_landmarks_raw is None:
                first_landmarks_raw = points

            # Step 5: Normalize landmarks
            norm_vec = normalize_landmarks(points)
            if first_normalized_vec is None:
                first_normalized_vec = norm_vec

            # Step 6: Feed to GestureRecognizer (buffers 24 frames)
            event = recognizer.append(norm_vec, timestamp=frame_count / fps)
            if event is not None:
                emitted_events.append(event)

        frame_count += 1

    cap.release()
    tracker.close()

    print("\n================ REAL GESTURE VERIFICATION RESULTS ================")
    print(f"Total Video Frames Processed: {frame_count}")
    print(f"Frames with Extracted Landmarks: {detected_count} ({detected_count/frame_count*100:.1f}%)")
    assert detected_count >= 24, "Did not extract enough real frames with landmarks"
    assert first_landmarks_raw is not None
    assert first_normalized_vec is not None

    print(f"Raw Pose Landmark Count: {len(first_landmarks_raw)} keypoints (33 standard MediaPipe landmarks)")
    print(f"Raw Coordinates Shape: {first_landmarks_raw.shape} [x, y, z, visibility]")
    print(f"Key Landmarks Sample:")
    print(f"  - Nose (0):         x={first_landmarks_raw[0,0]:.4f}, y={first_landmarks_raw[0,1]:.4f}, z={first_landmarks_raw[0,2]:.4f}, vis={first_landmarks_raw[0,3]:.4f}")
    print(f"  - Left Wrist (15):  x={first_landmarks_raw[15,0]:.4f}, y={first_landmarks_raw[15,1]:.4f}, z={first_landmarks_raw[15,2]:.4f}, vis={first_landmarks_raw[15,3]:.4f}")
    print(f"  - Right Wrist (16): x={first_landmarks_raw[16,0]:.4f}, y={first_landmarks_raw[16,1]:.4f}, z={first_landmarks_raw[16,2]:.4f}, vis={first_landmarks_raw[16,3]:.4f}")
    print(f"  - Left Hip (23):    x={first_landmarks_raw[23,0]:.4f}, y={first_landmarks_raw[23,1]:.4f}, z={first_landmarks_raw[23,2]:.4f}, vis={first_landmarks_raw[23,3]:.4f}")
    print(f"  - Right Hip (24):   x={first_landmarks_raw[24,0]:.4f}, y={first_landmarks_raw[24,1]:.4f}, z={first_landmarks_raw[24,2]:.4f}, vis={first_landmarks_raw[24,3]:.4f}")

    print(f"\nNormalized Feature Dimension: {first_normalized_vec.shape[0]}-D vector")
    print(f"Non-zero elements in Normalized Vector: {np.count_nonzero(first_normalized_vec)} / {first_normalized_vec.shape[0]}")
    print(f"Normalized Vector min/max/mean: min={first_normalized_vec.min():.4f}, max={first_normalized_vec.max():.4f}, mean={first_normalized_vec.mean():.4f}")

    print(f"\nAverage MediaPipe Pose Extraction Latency: {total_pose_latency_ms / frame_count:.2f} ms/frame")
    print(f"Gesture Events Emitted (after 24-frame temporal buffer): {len(emitted_events)}")

    assert len(emitted_events) > 0, "No gesture event was emitted!"
    latest_event = emitted_events[-1]

    print(f"\nTrained GestureLSTM Prediction Output:")
    print(f"  - Top Gesture Class:    {latest_event.metadata['top_class']}")
    print(f"  - Prediction Confidence: {latest_event.confidence:.4f} ({latest_event.confidence*100:.2f}%)")
    print(f"  - Threshold Accepted:   {latest_event.metadata['threshold_accepted']}")
    print(f"  - Class Probabilities:  {latest_event.metadata['probabilities']}")
    print(f"  - Output Embedding Dim: {len(latest_event.feature)} (128-D multimodal projection)")
    print(f"  - LSTM Inference Latency: {latest_event.latency_ms:.2f} ms")

    print("\n>>> COMPLETE POSE EXTRACTION -> NORMALIZATION -> GESTURE_LSTM PIPELINE CONFIRMED WORKING ON REAL DATA <<<")
    return True

if __name__ == "__main__":
    run_real_gesture_verification()
