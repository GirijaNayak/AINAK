"""Model Audit Inspection Script for AI AINAK.
Inspects all 6 checkpoints, architectures, parameter counts, and metadata.
"""

import json
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

print("=" * 70)
print("PHASE 1: MODEL CHECKPOINT PARAMETERS & ARCHITECTURE AUDIT")
print("=" * 70)

# 1. Vision - YOLOv8n
print("\n--- 1. VISION (YOLOv8n) ---")
yolo_path = REPO_ROOT / "artifacts" / "vision" / "yolov8n.pt"
y = YOLO(str(yolo_path))
v_params = sum(p.numel() for p in y.model.parameters())
v_trainable = sum(p.numel() for p in y.model.parameters() if p.requires_grad)
print(f"Path: {yolo_path}")
print(f"File Size: {yolo_path.stat().st_size / 1e6:.2f} MB")
print(f"Total Parameters: {v_params:,}")
print(f"Trainable Parameters: {v_trainable:,}")
print(f"Supported Classes: {len(y.names)}")
print(f"Task: {y.task}")

# Source metadata if available
y_src = REPO_ROOT / "artifacts" / "vision" / "yolov8n.pt.source.json"
if y_src.is_file():
    print(f"Source JSON: {y_src.read_text(encoding='utf-8').strip()}")

# 2. Audio - PANNs CNN14 Backbone
print("\n--- 2. AUDIO BACKBONE (PANNs CNN14) ---")
c14_path = REPO_ROOT / "artifacts" / "audio" / "Cnn14_mAP=0.431.pth"
c14 = torch.load(str(c14_path), map_location="cpu", weights_only=False)
print(f"Path: {c14_path}")
print(f"File Size: {c14_path.stat().st_size / 1e6:.2f} MB")
if isinstance(c14, dict) and "model" in c14:
    c_p = sum(v.numel() for v in c14["model"].values())
    print(f"CNN14 Backbone Parameters: {c_p:,}")
    print(f"Dict Keys: {list(c14.keys())}")

# 3. Audio - Downstream AudioMLP
print("\n--- 3. AUDIO DOWNSTREAM CLASSIFIER (AudioMLP) ---")
mlp_path = REPO_ROOT / "artifacts" / "audio" / "downstream_mlp.pt"
mlp = torch.load(str(mlp_path), map_location="cpu", weights_only=False)
print(f"Path: {mlp_path}")
print(f"File Size: {mlp_path.stat().st_size / 1e6:.2f} MB")
print(f"Top-level Keys: {list(mlp.keys())}")
mlp_sd = mlp.get("state_dict", {})
mlp_p = sum(v.numel() for v in mlp_sd.values())
print(f"AudioMLP Parameters: {mlp_p:,}")
print(f"Classes ({len(mlp.get('classes', []))}): {mlp.get('classes', [])[:8]}...")
print("Layer Shapes:")
for k, v in mlp_sd.items():
    print(f"  {k}: {list(v.shape)}")

# 4. Gesture - MediaPipe Pose Landmarker Lite
print("\n--- 4. GESTURE LANDMARKER (MediaPipe Tasks Lite) ---")
pose_path = REPO_ROOT / "artifacts" / "gesture" / "pose_landmarker_lite.task"
print(f"Path: {pose_path}")
print(f"File Size: {pose_path.stat().st_size / 1e6:.2f} MB")
pose_src = REPO_ROOT / "artifacts" / "gesture" / "pose_landmarker_lite.task.source.json"
if pose_src.is_file():
    print(f"Source JSON: {pose_src.read_text(encoding='utf-8').strip()}")

# 5. Gesture - GestureLSTM
print("\n--- 5. GESTURE TEMPORAL CLASSIFIER (GestureLSTM) ---")
lstm_path = REPO_ROOT / "artifacts" / "gesture" / "gesture_lstm.pt"
lstm = torch.load(str(lstm_path), map_location="cpu", weights_only=False)
print(f"Path: {lstm_path}")
print(f"File Size: {lstm_path.stat().st_size / 1e6:.2f} MB")
print(f"Top-level Keys: {list(lstm.keys())}")
lstm_sd = lstm.get("state_dict", {})
lstm_p = sum(v.numel() for v in lstm_sd.values())
print(f"GestureLSTM Parameters: {lstm_p:,}")
print("Layer Shapes:")
for k, v in lstm_sd.items():
    print(f"  {k}: {list(v.shape)}")

# Check if unidirectional or bidirectional
# For nn.LSTM(input_size=132, hidden_size=128, num_layers=2):
# weight_ih_l0 is (512, 132) -> 4 * 128 = 512, meaning unidirectional!
# If bidirectional, weight_ih_l0_reverse would also exist.
has_reverse = any("reverse" in k for k in lstm_sd.keys())
print(f"Directionality: {'Bidirectional' if has_reverse else 'Unidirectional (Forward-only)'}")

# 6. Multimodal Fusion Transformer
print("\n--- 6. MULTIMODAL TRANSFORMER FUSION ---")
fusion_path = REPO_ROOT / "artifacts" / "fusion" / "fusion.pt"
fusion = torch.load(str(fusion_path), map_location="cpu", weights_only=False)
print(f"Path: {fusion_path}")
print(f"File Size: {fusion_path.stat().st_size / 1e6:.2f} MB")
print(f"Top-level Keys: {list(fusion.keys())}")
f_config = fusion.get("model_config", {})
print("Model Config:")
print(json.dumps(f_config, indent=2))
f_sd = fusion.get("state_dict", {})
f_p = sum(v.numel() for v in f_sd.values())
print(f"Transformer Parameters: {f_p:,}")
print("Layer Shapes:")
for k, v in f_sd.items():
    print(f"  {k}: {list(v.shape)}")

print("\n" + "=" * 70)
print("AUDIT INSPECTION SCRIPT FINISHED")
print("=" * 70)
