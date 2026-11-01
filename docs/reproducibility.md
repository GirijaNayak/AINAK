# AI AINAK — Complete End-to-End Reproducibility Guide

Follow these exact steps to reproduce the environment, model assets, training, evaluations, and demonstration on a fresh Windows system.

---

## 1. Prerequisites and Environment Setup

```powershell
# Clone repository and enter directory
cd C:\Users\girijanayak\OneDrive\Documents\ChatGPT\AINAK

# Verify Python version (3.10.x recommended)
python --version

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements / package in editable mode
pip install -e .
```

---

## 2. Pretrained Asset Acquisition

```powershell
# Acquire MediaPipe Lite Pose Task Model
python -m ainak.interfaces.cli acquire-pose --overwrite

# Acquire Vision Weights (YOLOv8n)
python -m ainak.interfaces.cli acquire vision

# Acquire Audio Weights (PANNs CNN14)
python -m ainak.interfaces.cli acquire audio --allow-large
```

---

## 3. Training Pipelines

```powershell
# 1. Vision Benchmark (COCO128 mAP validation)
python scripts/evaluate_vision_coco128.py

# 2. Audio Downstream Training (ESC-50 embeddings + MLP + noise robustness)
python scripts/train_audio_downstream.py

# 3. Gesture LSTM Training (Reviewed multi-subject sequences)
python scripts/prepare_and_train_gesture.py

# 4. Multimodal Fusion Transformer Training
python scripts/train_multimodal_fusion.py
```

---

## 4. Automated Testing and Verification

```powershell
# Run the complete test suite
python -m pytest -v
```

---

## 5. End-to-End Demonstration

```powershell
# Replay prerecorded multimodal session with full three-branch fusion
python -m ainak.interfaces.cli demo-session --session datasets/recordings/session-multimodal-01 --reuse-features

# Local speech output verification
python -m ainak.interfaces.cli speech-test --text "AI AINAK system online and ready."
```
