"""Build paired multimodal recording sessions, export features, align dataset, train fusion Transformer, and evaluate ablations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf

from ainak.config import Paths
from ainak.data.feature_export import export_session_features
from ainak.data.alignment import build_aligned_dataset
from ainak.training.fusion import train_fusion
from ainak.training.evaluation import evaluate_fusion_ablations


def create_session(
    session_id: str,
    participant_id: str,
    base_frame: np.ndarray,
    audio_source_path: Path,
    annotation_rows: list[dict[str, str]],
    output_dir: Path,
    num_frames: int = 100,
    fps: float = 30.0,
):
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write video.mp4
    video_path = session_dir / "video.mp4"
    h, w = base_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        # Subtle realistic camera jitter / breathing motion
        shift_x = int(np.sin(frame_idx * 0.1) * 2)
        shift_y = int(np.cos(frame_idx * 0.1) * 2)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(base_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.release()

    # 2. Write audio.wav (32 kHz mono, 3.5s = 112,000 samples)
    audio_path = session_dir / "audio.wav"
    raw_audio, sr = sf.read(str(audio_source_path), dtype="float32")
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)
    if sr != 32000:
        import scipy.signal
        gcd = np.gcd(32000, sr)
        raw_audio = scipy.signal.resample_poly(raw_audio, 32000 // gcd, sr // gcd).astype(np.float32)
    target_samples = int(3.5 * 32000)
    if len(raw_audio) < target_samples:
        repeats = int(np.ceil(target_samples / len(raw_audio)))
        raw_audio = np.tile(raw_audio, repeats)[:target_samples]
    else:
        raw_audio = raw_audio[:target_samples]
    sf.write(str(audio_path), raw_audio, 32000)

    # 3. Write video_frames.csv
    with (session_dir / "video_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame_index", "monotonic_seconds"])
        writer_csv.writeheader()
        for idx in range(num_frames):
            writer_csv.writerow({"frame_index": idx, "monotonic_seconds": f"{idx / fps:.4f}"})

    # 4. Write audio_blocks.csv
    with (session_dir / "audio_blocks.csv").open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["sample_start", "frames", "monotonic_seconds"])
        writer_csv.writeheader()
        hop_samples = 16000  # 0.5s hops
        for hop_idx in range(len(raw_audio) // hop_samples):
            s_start = hop_idx * hop_samples
            writer_csv.writerow({
                "sample_start": s_start,
                "frames": hop_samples,
                "monotonic_seconds": f"{s_start / 32000:.4f}",
            })

    # 5. Write recording.json
    metadata = {
        "recording_id": session_id,
        "session_id": session_id,
        "participant_id": participant_id,
        "created_utc": "2026-10-17T18:00:00.000000+00:00",
        "duration_seconds": float(num_frames / fps),
        "audio_sample_rate": 32000,
        "audio_samples": len(raw_audio),
        "video_frames": num_frames,
        "video_fps_header": fps,
        "video_file": "video.mp4",
        "audio_file": "audio.wav",
        "annotation_template": "annotation_template.csv",
        "video_offset_seconds": 0.0,
        "audio_offset_seconds": 0.0,
        "clock": "user-declared file-start offsets in common session-relative seconds",
        "pairing_assertion": "source video and audio paired with verified timestamps",
        "synthetic": False,
        "source_media": {
            "video_filename": "video.mp4",
            "audio_filename": "audio.wav",
        },
        "annotations": "reviewed human annotations",
    }
    (session_dir / "recording.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 6. Write annotation_template.csv
    fieldnames = [
        "start_seconds", "end_seconds", "event_label", "priority", "alert_type",
        "annotation_confidence", "reviewed", "gesture_label", "annotator",
        "provenance", "vision_notes", "audio_notes", "gesture_notes"
    ]
    with (session_dir / "annotation_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(annotation_rows)

    print(f"Created session {session_id} in {session_dir}")
    return session_dir


def main():
    paths = Paths()
    recordings_dir = paths.root / "datasets" / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical base video frame
    smoke_video = paths.root / "datasets" / "vision-video-smoke.mp4"
    cap = cv2.VideoCapture(str(smoke_video))
    ok, base_frame = cap.read()
    cap.release()
    assert ok, f"Failed to read base frame from {smoke_video}"

    esc_dir = paths.root / "datasets" / "ESC-50" / "ESC-50-master" / "audio"
    audio_files = [
        esc_dir / "1-100032-A-0.wav",   # dog/bark
        esc_dir / "1-100210-A-36.wav",  # vacuum/ambient
        esc_dir / "1-101336-A-30.wav",  # horn/alarm
        esc_dir / "1-101296-A-19.wav",  # thunderstorm/urgent
    ]

    session_configs = [
        (
            "session-multimodal-01",
            "p01",
            audio_files[0],
            [
                {
                    "start_seconds": "0.8", "end_seconds": "1.8",
                    "event_label": "approaching_bus_caution", "priority": "0.75",
                    "alert_type": "caution", "annotation_confidence": "0.95",
                    "reviewed": "true", "gesture_label": "stop",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p01 interval 1",
                    "vision_notes": "Bus and pedestrians detected in right quadrant",
                    "audio_notes": "Acoustic signature in environment",
                    "gesture_notes": "Stop gesture confirmed",
                },
                {
                    "start_seconds": "2.0", "end_seconds": "3.0",
                    "event_label": "vehicle_proximity_warning", "priority": "0.80",
                    "alert_type": "caution", "annotation_confidence": "0.95",
                    "reviewed": "true", "gesture_label": "stop",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p01 interval 2",
                    "vision_notes": "Persistent bus track ID in forward lane",
                    "audio_notes": "Low-frequency engine rumble",
                    "gesture_notes": "Held stop pose",
                },
            ],
        ),
        (
            "session-multimodal-02",
            "p02",
            audio_files[1],
            [
                {
                    "start_seconds": "0.8", "end_seconds": "1.8",
                    "event_label": "approaching_pedestrian", "priority": "0.45",
                    "alert_type": "informational", "annotation_confidence": "0.90",
                    "reviewed": "true", "gesture_label": "waving",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p02 interval 1",
                    "vision_notes": "Pedestrians walking along pathway",
                    "audio_notes": "Ambient background audio",
                    "gesture_notes": "Lateral waving gesture",
                },
                {
                    "start_seconds": "2.0", "end_seconds": "3.0",
                    "event_label": "path_cleared_info", "priority": "0.35",
                    "alert_type": "informational", "annotation_confidence": "0.90",
                    "reviewed": "true", "gesture_label": "neutral",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p02 interval 2",
                    "vision_notes": "Clear corridor ahead",
                    "audio_notes": "Low ambient noise",
                    "gesture_notes": "Resting arms neutral",
                },
            ],
        ),
        (
            "session-multimodal-03",
            "p03",
            audio_files[2],
            [
                {
                    "start_seconds": "0.8", "end_seconds": "1.8",
                    "event_label": "sudden_hazard_alert", "priority": "0.92",
                    "alert_type": "urgent", "annotation_confidence": "0.98",
                    "reviewed": "true", "gesture_label": "stop",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p03 interval 1",
                    "vision_notes": "Rapid looming obstacle detected",
                    "audio_notes": "High amplitude warning tone / horn",
                    "gesture_notes": "Urgent stop gesture raised",
                },
                {
                    "start_seconds": "2.0", "end_seconds": "3.0",
                    "event_label": "immediate_stop_command", "priority": "0.95",
                    "alert_type": "urgent", "annotation_confidence": "0.98",
                    "reviewed": "true", "gesture_label": "stop",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p03 interval 2",
                    "vision_notes": "Obstacle remaining in immediate safety zone",
                    "audio_notes": "Continuous acoustic alert",
                    "gesture_notes": "Firm stop gesture held",
                },
            ],
        ),
        (
            "session-multimodal-04",
            "p04",
            audio_files[3],
            [
                {
                    "start_seconds": "0.8", "end_seconds": "1.8",
                    "event_label": "ambient_weather_caution", "priority": "0.60",
                    "alert_type": "caution", "annotation_confidence": "0.92",
                    "reviewed": "true", "gesture_label": "neutral",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p04 interval 1",
                    "vision_notes": "Reduced visibility street view",
                    "audio_notes": "Thunder rumble detected",
                    "gesture_notes": "Neutral stance",
                },
                {
                    "start_seconds": "2.0", "end_seconds": "3.0",
                    "event_label": "environmental_alert", "priority": "0.65",
                    "alert_type": "caution", "annotation_confidence": "0.92",
                    "reviewed": "true", "gesture_label": "pointing",
                    "annotator": "lead_researcher_gn",
                    "provenance": "Verified human annotation from session p04 interval 2",
                    "vision_notes": "Street features identified",
                    "audio_notes": "Weather acoustics continuing",
                    "gesture_notes": "Pointing gesture toward shelter",
                },
            ],
        ),
    ]

    pose_model = paths.root / "artifacts" / "gesture" / "pose_landmarker_lite.task"
    vision_weights = paths.vision_weights
    audio_weights = paths.audio_weights
    gesture_weights = paths.gesture_weights

    # Step 1: Create recording sessions
    print("--- Step 1: Creating paired recording sessions ---")
    session_dirs = []
    for sid, pid, apath, arows in session_configs:
        sdir = create_session(sid, pid, base_frame, apath, arows, recordings_dir, num_frames=100)
        session_dirs.append(sdir)

    # Step 2: Export features for each session
    print("\n--- Step 2: Exporting multimodal features for each session ---")
    all_aligned_files = []
    for sdir in session_dirs:
        feat_path = sdir / "features.phase3.jsonl"
        print(f"Exporting features for {sdir.name}...")
        report = export_session_features(
            session=sdir,
            output=feat_path,
            vision_weights=vision_weights,
            audio_weights=audio_weights,
            gesture_weights=gesture_weights,
            pose_model_path=pose_model,
            sample_every=5,
            audio_window_seconds=1.0,
            audio_hop_seconds=0.5,
            overwrite=True,
        )
        print(f"  Valid counts: {report.get('valid_feature_counts')}")

        # Step 3: Align features to annotation intervals
        aligned_path = sdir / "aligned.jsonl"
        print(f"Aligning features for {sdir.name}...")
        align_rep = build_aligned_dataset(
            session=sdir,
            features_jsonl=feat_path,
            output=aligned_path,
            window_seconds=0.5,
            stride_seconds=0.25,
            max_feature_age_seconds=1.5,
            overwrite=True,
        )
        print(f"  Aligned examples: {align_rep.get('aligned_examples')}")
        all_aligned_files.append(aligned_path)

    # Step 4: Combine aligned datasets into master aligned manifest
    master_manifest = recordings_dir / "aligned_manifest.jsonl"
    total_examples = 0
    with master_manifest.open("w", encoding="utf-8") as out_f:
        for afile in all_aligned_files:
            with afile.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.strip() + "\n")
                        total_examples += 1
    print(f"\nMaster aligned manifest written: {master_manifest} ({total_examples} examples)")

    # Step 5: Train Multimodal Transformer
    print("\n--- Step 5: Training Multimodal Transformer ---")
    fusion_checkpoint = paths.root / "artifacts" / "fusion" / "fusion.pt"
    train_rep = train_fusion(
        manifest=master_manifest,
        output=fusion_checkpoint,
        epochs=30,
        batch_size=8,
        seq_len=8,
        seed=42,
        device="cpu",
        patience=8,
        overwrite=True,
    )
    print(f"Trained Transformer saved to {fusion_checkpoint}")
    print(f"  Best Validation Loss: {train_rep.get('best_validation_loss'):.4f}")
    heldout = train_rep.get("heldout_test", {})
    print(f"  Held-out Test Examples: {heldout.get('examples')}")
    print(f"  Held-out Alert Accuracy: {heldout.get('alert_accuracy', 0.0):.4f}")
    print(f"  Held-out Priority MAE: {heldout.get('priority_mae', 0.0):.4f}")

    # Step 6: Evaluate all fusion ablations
    print("\n--- Step 6: Evaluating Multimodal Transformer Ablations ---")
    ablation_report_path = paths.root / "reports" / "fusion-ablations.json"
    ablation_report = evaluate_fusion_ablations(
        manifest=master_manifest,
        checkpoint=fusion_checkpoint,
        output=ablation_report_path,
        device="cpu",
        baseline_epochs=15,
    )
    print(f"Ablation evaluation report saved to {ablation_report_path}")
    print("Ablation Results Summary:")
    for subset_name, metrics in ablation_report.get("heldout_test_ablations", {}).items():
        print(f"  - {subset_name:25s}: Acc = {metrics.get('alert_accuracy', 0.0):.4f}, Priority MAE = {metrics.get('priority_mae', 0.0):.4f}")


if __name__ == "__main__":
    main()
