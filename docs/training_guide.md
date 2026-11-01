# AI AINAK — Training and Reproducibility Guide

## 1. Local Training Commands

### Vision Branch (YOLOv8n)
Pretrained weights are used for lightweight, real-time edge execution:
```powershell
# Evaluate COCO128 mAP and latency
python scripts/evaluate_vision_coco128.py

# Profile inference latency at 640px
python -m ainak.interfaces.cli profile-vision artifacts/vision/bus.jpg --imgsz 640
```

### Audio Branch (PANNs CNN14 & Downstream Classifiers)
Extract 2,048-D AudioSet embeddings from ESC-50 and train downstream classifiers:
```powershell
python scripts/train_audio_downstream.py
```
This trains:
1. An L2-regularized Logistic Regression baseline.
2. A PyTorch `AudioMLP` (2048 -> 256 -> 50) with BatchNorm, ReLU, and Dropout.
3. Tests noise robustness at 20 dB, 10 dB, 0 dB SNR, and silence rejection.
4. Outputs `artifacts/audio/downstream_mlp.pt` and `reports/audio-downstream-eval.json`.

### Gesture Branch (MediaPipe Pose + LSTM)
Extract reviewed landmark sequences and train the 2-layer temporal LSTM:
```powershell
python scripts/prepare_and_train_gesture.py
```
Or directly via the CLI:
```powershell
python -m ainak.interfaces.cli train-gesture datasets/gesture-sessions/gesture_manifest.csv --output artifacts/gesture/gesture_lstm.pt --epochs 30 --hidden-size 128 --layers 2 --device cpu
```

### Multimodal Transformer Fusion
Train the 2-layer, 4-head Transformer encoder on aligned multimodal session JSONL:
```powershell
python -m ainak.interfaces.cli train-fusion datasets/recordings/aligned_manifest.jsonl --output artifacts/fusion/fusion.pt --epochs 20 --batch-size 16 --device cpu
```
Evaluate modality ablations and late-fusion baseline:
```powershell
python -m ainak.interfaces.cli evaluate-fusion datasets/recordings/aligned_manifest.jsonl --weights artifacts/fusion/fusion.pt --output reports/fusion-ablations.json --device cpu
```

---

## 2. Remote Kaggle GPU Workflow

For larger experiments on NVIDIA Tesla T4 or P100 GPUs:
1. Open Kaggle and create a new notebook with GPU accelerator (Tesla T4 x2 or P100).
2. Upload or clone the repository.
3. Execute the cells provided in `notebooks/kaggle_gpu_training.py`.
4. Checkpoints are saved under `/kaggle/working/artifacts/` and can be downloaded directly into the local repository.
