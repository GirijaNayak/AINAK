# AI AINAK — Comprehensive Model Training & Architecture Audit

**Audit Date**: 2026-11-12  
**Auditor**: Senior ML Research Engineer & Deep Learning Auditor  
**Audit Scope**: All trained checkpoints, model architectures, training configurations, logs, and evaluation reports in `AI AINAK`.  
**Integrity Rule**: No models retrained; no weights modified; all numbers verified from disk or marked **NOT VERIFIED**.

---

## 1. Executive Summary

This audit provides a factual, evidence-backed evaluation of the four perception and decision modules in the AI AINAK assistive system. 

### Key Findings:
1. **Pretrained vs. Locally Trained Components**:
   - **Pretrained External Weights**: YOLOv8n (Vision, 3.16M params, Ultralytics release), PANNs CNN14 (Audio Backbone, 81.85M params, Zenodo/AudioSet), and MediaPipe Pose Landmarker Lite (Gesture Extractor, 5.78 MB, Google TFLite). These were **not** trained or fine-tuned from scratch within this project; they are utilized as frozen feature extractors and base detectors.
   - **Locally Trained Project Components**: Downstream AudioMLP (538K params, trained on ESC-50), GestureLSTM (267K params, trained on 128 local kinematic sequences), and MultimodalTransformer (595K params, trained on 24 aligned multimodal session windows).
2. **Architecture Correction (GestureLSTM)**:
   - Historical documentation previously cited GestureLSTM as a "2-layer Bidirectional LSTM". Direct inspection of `artifacts/gesture/gesture_lstm.pt` (`lstm.weight_ih_l0: [512, 132]`) and `ainak/models/gesture/lstm.py` confirms conclusively that the model is a **2-layer UNIDIRECTIONAL LSTM** (forward-only).
3. **Training Sample Scale Disparity**:
   - Pretrained backbones represent massive public datasets (AudioSet: 2.04M clips; COCO: 118K images).
   - Local project models represent small, controlled academic prototypes: AudioMLP (1,600 train / 400 test clips), GestureLSTM (64 train / 32 val / 32 test sequences), and Multimodal Transformer (12 train / 6 val / 6 test windows across 4 recording sessions).
4. **Transformer Overfitting & Generalization Reality**:
   - The Multimodal Transformer was trained on 12 windows from Sessions 01 and 04. Validation loss on Session 02 increased from 5.56 (Epoch 1) to 9.04 (Epoch 9), where early stopping terminated training. On held-out test Session 03 (6 windows, all ground-truth "urgent"), the model predicted "caution" for all 6 instances, achieving 0.0% alert classification accuracy. This authentic limitation demonstrates the need for expanded multi-subject training data before claiming production generalization.

---

## 2. Comprehensive 20-Point Model Specification Matrix

| Parameter / Criterion | 1. Vision Detector (YOLOv8n) | 2. Audio Backbone (PANNs CNN14) | 3. Audio Classifier (AudioMLP) | 4. Gesture Landmarker (MediaPipe Tasks) | 5. Gesture Classifier (GestureLSTM) | 6. Multimodal Fusion (Transformer) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Architecture** | YOLOv8n (CSPDarknet + PANet + Decoupled Head) | CNN14 (14-layer VGG-style CNN, global pooling) | 2-layer MLP (`Linear-BN-ReLU-Dropout-Linear`) | BlazePose 3D Detector + Tracker (TFLite) | 2-layer Unidirectional LSTM + LayerNorm + Linear | 2-layer 4-head Transformer Encoder + Multi-head Projections |
| **2. Checkpoint Path** | `artifacts/vision/yolov8n.pt` | `artifacts/audio/Cnn14_mAP=0.431.pth` | `artifacts/audio/downstream_mlp.pt` | `artifacts/gesture/pose_landmarker_lite.task` | `artifacts/gesture/gesture_lstm.pt` | `artifacts/fusion/fusion.pt` |
| **File Size (Disk)** | 6.55 MB (6,549,796 B) | 327.43 MB (327,428,481 B) | 2.17 MB (2,168,913 B) | 5.78 MB (5,777,746 B) | 1.07 MB (1,073,538 B) | 2.40 MB (2,402,683 B) |
| **3. Total Parameters** | 3,157,200 | 81,853,340 | 538,419 | **NOT VERIFIED** (Binary TFLite bundle) | 267,012 | 595,462 |
| **4. Trainable Params** | 0 (Frozen inference) | 0 (Frozen feature extractor) | 538,419 | 0 (Frozen inference) | 267,012 | 595,462 |
| **5. Pretrain vs Custom** | Pretrained (Ultralytics) | Pretrained (PANNs Zenodo) | Custom-trained in project | Pretrained (Google Research) | Custom-trained in project | Custom-trained in project |
| **6. Training Dataset** | MS COCO 2017 (`train2017`) | Google AudioSet | ESC-50 (Folds 1–4) | BlazePose proprietary corpus | Local Kinematic Dataset (`gesture_manifest.csv`) | Paired Session Manifest (`aligned_manifest.jsonl`) |
| **7. Training Samples** | 118,287 images | 2,041,789 audio clips | 1,600 audio clips (80%) | **NOT VERIFIED** (Proprietary Google data) | 64 sequences (Subjects p02, p04) | 12 temporal windows (Sessions 01 & 04) |
| **8. Validation Samples**| 128 images (COCO128 mini-split) | None recorded in project | Evaluated on Fold 5 | **NOT VERIFIED** | 32 sequences (Subject p03) | 6 temporal windows (Session 02) |
| **9. Test Samples** | 5,000 images (COCO `val2017`) | AudioSet Eval Split | 400 audio clips (Fold 5) | **NOT VERIFIED** | 32 sequences (Subject p01) | 6 temporal windows (Session 03) |
| **10. Training Epochs** | **NOT VERIFIED** (Pretrained) | **NOT VERIFIED** (Pretrained) | 40 epochs | **NOT VERIFIED** (Pretrained) | 30 epochs completed | 9 epochs completed (patience=8 stop) |
| **11. Batch Size** | **NOT VERIFIED** (Pretrained) | **NOT VERIFIED** (Pretrained) | 32 | **NOT VERIFIED** (Pretrained) | 8 | 8 |
| **12. Optimizer & LR** | SGD / AdamW (Pretrained) | Adam (Pretrained) | AdamW ($1 \times 10^{-3}$, decay $10^{-4}$) | **NOT VERIFIED** (Pretrained) | AdamW ($1 \times 10^{-3}$, clip norm 1.0) | AdamW ($1 \times 10^{-3}$, decay $10^{-4}$) |
| **13. Loss Function** | CIoU + DFL + BCE (Pretrained) | Binary CrossEntropy (Pretrained) | CrossEntropyLoss | **NOT VERIFIED** (Pretrained) | CrossEntropyLoss (weighted) | Multi-task: BCE(Priority) + BCE(Confidence) + CE(Alert) |
| **14. Best Epoch** | Pretrained checkpoint | Pretrained checkpoint | Epoch 17 (val loss: 0.2800) | Pretrained task | Epoch 30 (val loss: 0.00031) | Epoch 1 (val loss: 5.5607) |
| **15. Best Val Metric** | mAP@0.5: 0.6052 (COCO128) | mAP: 0.431 (AudioSet) | Accuracy: 90.00% (Epoch 1) | **NOT VERIFIED** | Val Loss: 0.000312 | Val Loss: 5.5607 (Epoch 1) |
| **16. Final Train Loss** | **NOT VERIFIED** (Pretrained) | **NOT VERIFIED** (Pretrained) | 0.00317 (Epoch 40) | **NOT VERIFIED** (Pretrained) | 0.000339 (Epoch 30) | 0.8402 (Epoch 9) |
| **17. Final Val Loss** | **NOT VERIFIED** (Pretrained) | **NOT VERIFIED** (Pretrained) | 0.3493 (Epoch 40) | **NOT VERIFIED** (Pretrained) | 0.000313 (Epoch 30) | 9.0401 (Epoch 9) |
| **18. Key Evaluation** | mAP50: 0.6052, mAP50-95: 0.4468 | AudioSet mAP: 0.431 | Accuracy: 89.75%, Macro-F1: 89.59% | 33 3D keypoints + visibility | Accuracy: 100.0%, Macro-F1: 1.0000 | Priority MAE: 0.4172, Alert Acc: 0.0% |
| **19. Inference Latency**| 49.23 ms / frame (CPU) | 29.30 ms / 2s chunk (CPU) | < 0.1 ms / vector | 12.5 ms / frame (CPU) | 1.01 ms / 24-frame window | 1.91 ms – 2.20 ms / step (CPU) |
| **20. Training Hardware**| Cloud TPU/GPU (Ultralytics) | Cloud GPU (Kong et al., PANNs) | Intel CPU (`torch 2.2.2+cpu`) | Cloud TPU (Google) | Intel CPU (`torch 2.2.2+cpu`, 3.52s) | Intel CPU (`torch 2.2.2+cpu`, 0.57s) |

---

## 3. Module-by-Module In-Depth Audit

### 3.1 Vision Branch: YOLOv8n
- **Origin**: Official Ultralytics pretrained release asset (`yolov8n.pt`, acquired from GitHub release `v8.3.0`).
- **Project Training**: Zero fine-tuning epochs were conducted in this project. The model weights are 100% stock COCO pretrained weights.
- **Evaluation Evidence**: Benchmarked on local CPU using the COCO128 mini-split (`reports/vision-coco128-val.json`).
  - Precision: 0.6395 (63.95%)
  - Recall: 0.5368 (53.68%)
  - mAP@0.5: 0.6052 (60.52%)
  - mAP@0.5:0.95: 0.4468 (44.68%)
- **Custom Post-Processing Extensions**:
  All spatial heuristics in `ainak/models/vision/` are rule-based algorithms rather than learned weights:
  - *Tracking*: ByteTrack-style greedy IoU matching across successive frames.
  - *Depth Proxy*: Perspective projection formula ($0.65 \times \text{bbox\_base\_y} + 0.35 \times \text{norm\_area}$).
  - *Looming Rate*: First derivative of bounding box area expansion ($dA/dt$).
  - *Assistive Risk Policy*: Weighted risk equation based on COCO class danger category and proximity.

### 3.2 Audio Branch: PANNs CNN14 + Downstream AudioMLP
- **Backbone**: PANNs CNN14 (Kong et al., 2020), pretrained on AudioSet. Kept completely frozen in AI AINAK. Extracts 2,048-D global pooling vectors.
- **Downstream Classifier**: `downstream_mlp.pt`, custom-trained in this project on ESC-50 (`scripts/train_audio_downstream.py`).
- **Data Splitting**: Folds 1, 2, 3, and 4 (1,600 clips) for training; Fold 5 (400 clips) strictly held out for testing.
- **Performance Comparison**:
  - *Baseline Regularized Logistic Regression*: Accuracy: 91.00%, Macro-F1: 90.85%.
  - *Trained PyTorch AudioMLP*: Accuracy: 89.75%, Macro-F1: 89.59%.
  - *Noise Robustness*: Accuracy remains steady under additive Gaussian noise:
    - Clean: 89.75%
    - 20 dB SNR: 90.00%
    - 10 dB SNR: 90.75%
    - 0 dB SNR: 89.50%
  - *Digital Silence Rejection*: Max softmax probability on digital silence is 0.215, safely below the 0.35 alert threshold.

### 3.3 Gesture Branch: MediaPipe Pose Tasks + GestureLSTM
- **Landmarker**: Google MediaPipe Pose Tasks Lite (`pose_landmarker_lite.task`), running 33 BlazePose 3D landmarks.
- **GestureLSTM**: Custom-trained in this project (`ainak/training/gesture.py`).
- **Architecture**: 2-layer forward-only Unidirectional LSTM ($d_{\text{hidden}} = 128$, input dimension = 132), followed by LayerNorm and a 4-class linear head.
- **Participant-Disjoint Splitting**:
  - Total sequences: 128 (24 frames each).
  - Train: `subject_p02`, `subject_p04` (64 sequences).
  - Validation: `subject_p03` (32 sequences).
  - Test: `subject_p01` (32 sequences).
- **Reported Metric Analysis**:
  - Test Accuracy: 100.0% (32/32 correct), Validation Loss: 0.00031.
  - *Critical Assessment*: While 100% accuracy on held-out subject `p01` is authentic, the evaluator must understand that $N=32$ test clips recorded under scripted laboratory conditions represents a constrained sample. It confirms robust kinematic learning of distinctive gestures (`pointing`, `waving`, `stop`, `neutral`), but does not prove unconstrained real-world robustness under severe clothing occlusions or variable camera angles.

### 3.4 Multimodal Transformer Fusion
- **Model Checkpoint**: `artifacts/fusion/fusion.pt` ($d_{\text{model}} = 128$, 2 encoder layers, 4 attention heads, $d_{\text{ff}} = 256$).
- **Multi-Head Structure**:
  - `priority_head`: $128 \to 64 \to 1$ (Sigmoid, predicting continuous priority $0.0$ to $1.0$).
  - `confidence_head`: $128 \to 64 \to 1$ (Sigmoid, predicting certainty $0.0$ to $1.0$).
  - `alert_head`: $128 \to 4$ (Linear, predicting logits for `none`, `informational`, `caution`, `urgent`).
- **Empirical Training Analysis**:
  - Dataset: 24 temporal windows across 4 paired recording sessions.
  - Split: Sessions 01 & 04 (12 windows) for training; Session 02 (6 windows) for validation; Session 03 (6 windows) for held-out testing.
  - Convergence: Training loss decreased from 2.42 to 0.84. However, validation loss increased monotonically from 5.56 to 9.04, triggering early stopping at Epoch 9.
  - Test Evaluation: On Session 03 (6 windows), Priority MAE was 0.4172, and Alert classification accuracy was 0.0% (model predicted "caution" for all windows where ground truth was "urgent").
  - *Auditor Conclusion*: The Transformer architecture, linear token projections, and self-attention masking mechanisms are fully implemented and functional. However, 12 training examples are statistically insufficient to train a 595,462-parameter Transformer without severe overfitting. The model functions as a research architectural proof-of-concept rather than a fully generalizable predictor.
