"""Package AI AINAK project checkpoints and demonstration session into a lightweight starter zip.

Bundle includes:
- checkpoints/
  - downstream_mlp.pt (2.17 MB)
  - gesture_lstm.pt (1.07 MB)
  - fusion.pt (2.40 MB)
- datasets/recordings/session-multimodal-01/
  - video.mp4 (2.30 MB)
  - audio.wav (0.22 MB)
  - recording.json
  - aligned.jsonl

Note: Public checkpoints (yolov8n.pt, Cnn14_mAP=0.431.pth, pose_landmarker_lite.task)
can be downloaded automatically by the Colab notebooks from official URLs.
"""

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUTPUTS_DIR / "ainak_colab_starter_bundle.zip"

FILES_TO_PACK = [
    # Trained proprietary checkpoints
    (REPO_ROOT / "artifacts" / "audio" / "downstream_mlp.pt", "checkpoints/downstream_mlp.pt"),
    (REPO_ROOT / "artifacts" / "gesture" / "gesture_lstm.pt", "checkpoints/gesture_lstm.pt"),
    (REPO_ROOT / "artifacts" / "fusion" / "fusion.pt", "checkpoints/fusion.pt"),
    # Demonstration recording session (session-multimodal-01)
    (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "video.mp4", "datasets/recordings/session-multimodal-01/video.mp4"),
    (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "audio.wav", "datasets/recordings/session-multimodal-01/audio.wav"),
    (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "recording.json", "datasets/recordings/session-multimodal-01/recording.json"),
    (REPO_ROOT / "datasets" / "recordings" / "session-multimodal-01" / "aligned.jsonl", "datasets/recordings/session-multimodal-01/aligned.jsonl"),
]

def create_bundle():
    print(f"Creating starter bundle at: {ZIP_PATH}")
    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in FILES_TO_PACK:
            if src.is_file():
                print(f"  Adding {arcname} ({src.stat().st_size / 1e6:.2f} MB)")
                zf.write(str(src), arcname=arcname)
            else:
                print(f"  [WARNING] File not found: {src}")

    print(f"\nBundle created successfully!")
    print(f"Total zip size: {ZIP_PATH.stat().st_size / 1e6:.2f} MB")

if __name__ == "__main__":
    create_bundle()
