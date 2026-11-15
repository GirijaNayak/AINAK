"""Package AI AINAK V2 demonstration assets into a standalone starter bundle.

Bundle includes:
- checkpoints/
  - gesture/best.pt (1.08 MB)
  - gesture/latest.pt (1.08 MB)
  - vision/best.pt (25.92 MB)
  - vision/yolo_world_classes.json
  - hazard/hazard_policy_v2.json
  - audio/downstream_mlp.pt (2.17 MB)
  - fusion/fusion.pt (2.40 MB)
  - media_pipe/pose_landmarker_lite.task (5.78 MB)
  - CHECKPOINT_HASHES.json
- datasets/
  - recordings/session-multimodal-01/ (video.mp4, audio.wav, recording.json, aligned.jsonl)
  - gesture-sessions-v2/gesture_manifest_v2.csv
  - gesture-sessions-v2/sequences/ (representative subject_p01 test sequences)

Note: Heavy public weights (e.g. Cnn14_mAP=0.431.pth, 327 MB) are auto-downloaded on demand.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUTPUTS_DIR / "ainak_v2_colab_starter_bundle.zip"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_v2_bundle():
    print(f"Packaging AINAK V2 Starter Bundle: {ZIP_PATH}")

    files_to_pack: list[tuple[Path, str]] = [
        # Checkpoints - V2
        (REPO_ROOT / "artifacts" / "v2" / "gesture" / "best.pt", "checkpoints/gesture/best.pt"),
        (REPO_ROOT / "artifacts" / "v2" / "gesture" / "latest.pt", "checkpoints/gesture/latest.pt"),
        (REPO_ROOT / "artifacts" / "v2" / "vision" / "best.pt", "checkpoints/vision/best.pt"),
        (REPO_ROOT / "artifacts" / "v2" / "vision" / "yolo_world_classes.json", "checkpoints/vision/yolo_world_classes.json"),
        (REPO_ROOT / "artifacts" / "v2" / "hazard" / "hazard_policy_v2.json", "checkpoints/hazard/hazard_policy_v2.json"),
        # Checkpoints - V1 Core (Audio, Fusion, MediaPipe)
        (REPO_ROOT / "artifacts" / "audio" / "downstream_mlp.pt", "checkpoints/audio/downstream_mlp.pt"),
        (REPO_ROOT / "artifacts" / "fusion" / "fusion.pt", "checkpoints/fusion/fusion.pt"),
        (REPO_ROOT / "artifacts" / "gesture" / "pose_landmarker_lite.task", "checkpoints/media_pipe/pose_landmarker_lite.task"),
        # Multimodal Demo Session 01
        (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "video.mp4", "datasets/recordings/session-multimodal-01/video.mp4"),
        (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "audio.wav", "datasets/recordings/session-multimodal-01/audio.wav"),
        (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "recording.json", "datasets/recordings/session-multimodal-01/recording.json"),
        (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "aligned.jsonl", "datasets/recordings/session-multimodal-01/aligned.jsonl"),
        # Gesture V2 manifest
        (REPO_ROOT / "datasets" / "gesture-sessions-v2" / "gesture_manifest_v2.csv", "datasets/gesture-sessions-v2/gesture_manifest_v2.csv"),
    ]

    # Select representative held-out test sequences (1 per class from subject_p01 = 25 sequences)
    manifest_path = REPO_ROOT / "datasets" / "gesture-sessions-v2" / "gesture_manifest_v2.csv"
    if manifest_path.exists():
        chosen_classes = set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row["label"]
                seq_rel = row["path"]
                subj = row.get("subject_id", "")
                if subj == "subject_p01" and cls not in chosen_classes:
                    seq_path = REPO_ROOT / "datasets" / "gesture-sessions-v2" / seq_rel
                    if seq_path.exists():
                        files_to_pack.append((seq_path, f"datasets/gesture-sessions-v2/{seq_rel}"))
                        chosen_classes.add(cls)

    # Generate hash manifest
    hashes = {}
    for src, arc in files_to_pack:
        if src.is_file() and "checkpoints" in arc:
            hashes[arc] = {
                "sha256": sha256_file(src),
                "size_bytes": src.stat().st_size,
            }

    hash_manifest_path = REPO_ROOT / "outputs" / "CHECKPOINT_HASHES.json"
    with open(hash_manifest_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    files_to_pack.append((hash_manifest_path, "checkpoints/CHECKPOINT_HASHES.json"))

    # Write zip archive
    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_pack:
            if src.is_file():
                sz_mb = src.stat().st_size / 1e6
                print(f"  Adding {arcname:<50} ({sz_mb:6.2f} MB)")
                zf.write(str(src), arcname=arcname)
            else:
                print(f"  [WARNING] Missing file: {src}")

    print("\nStarter bundle successfully created!")
    print(f"Total bundle size: {ZIP_PATH.stat().st_size / 1e6:.2f} MB")
    print(f"Total packed files: {len(files_to_pack)}")


if __name__ == "__main__":
    create_v2_bundle()
