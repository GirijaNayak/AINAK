"""Kaggle GPU Training Script for AI AINAK.

Upload this file to a Kaggle GPU Notebook (Tesla T4 or P100, PyTorch 2.x, CUDA 12.x).
"""

# Cell 1: Environment Setup and Check
# !nvidia-smi
# !pip install ultralytics panns-inference mediapipe soundfile librosa scikit-learn

import sys
import torch
import platform
from pathlib import Path

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cell 2: Vision Validation on GPU
from ultralytics import YOLO

print("\n--- Running YOLOv8n Validation on GPU ---")
model = YOLO("yolov8n.pt")
metrics = model.val(data="coco128.yaml", device=0 if torch.cuda.is_available() else "cpu", imgsz=640)
print(f"GPU mAP@0.5: {metrics.box.map50:.4f}, mAP@0.5:0.95: {metrics.box.map:.4f}")

# Cell 3: Audio Downstream Training on GPU
print("\n--- Audio Downstream Training on GPU ---")
# When ESC-50 dataset is attached in /kaggle/input/esc50:
# from ainak.training.audio_downstream import extract_esc50_embeddings, train_downstream_models
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Cell 4: Checkpoint Export and Verification
print("\n--- Packaging Checkpoints for Download ---")
# !zip -r /kaggle/working/ainak_gpu_checkpoints.zip /kaggle/working/artifacts
print("Ready for local deployment into AINAK/artifacts/!")
