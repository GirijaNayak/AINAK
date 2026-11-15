# AI AINAK — Model Performance & Evaluation Comparison Report

**Audit Date**: 2026-11-12  
**Auditor**: Senior ML Research Engineer & Deep Learning Auditor  
**Scope**: Empirical benchmark analysis comparing model performance, baseline comparisons, noise robustness, confusion matrices, and processing latencies.

---

## 1. Overview of Evaluated Models

All metrics presented in this report were recovered directly from machine-generated evaluation artifacts stored on disk:
- `reports/vision-coco128-val.json`
- `reports/audio-downstream-eval.json`
- `artifacts/gesture/gesture_lstm.metrics.json`
- `artifacts/fusion/fusion.metrics.json`
- `reports/fusion-ablations.json`

---

## 2. Vision Model Benchmark (YOLOv8n)

### 2.1 Quantitative Detection Metrics (COCO128 Validation Split)
Evaluated at input resolution $640 \times 640$ on CPU (`intel-core-i5`):

| Metric | Measured Value | Standard Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Precision** | **0.6395 (63.95%)** | $\ge 0.60$ | Satisfied |
| **Recall** | **0.5368 (53.68%)** | $\ge 0.50$ | Satisfied |
| **mAP@0.5** | **0.6052 (60.52%)** | $\ge 0.55$ | Satisfied |
| **mAP@0.5:0.95** | **0.4468 (44.68%)** | $\ge 0.35$ | Satisfied |
| **Evaluation Duration** | 22.93 seconds | N/A | Completed across 128 images |

### 2.2 Inference Latency Profile (per frame)
- **Preprocessing (letterboxing & normalization)**: 0.77 ms
- **Neural Inference (forward pass)**: 49.23 ms
- **Loss Computation**: 0.0004 ms
- **Post-processing (NMS)**: 1.87 ms
- **Total Latency**: **51.87 ms per frame** ($\sim 19.3$ FPS on single-thread CPU)

### 2.3 Verification of Post-Processing Capabilities

| Capability | Implementation Type | Verification Status | Demonstrable in Presentation |
| :--- | :--- | :--- | :--- |
| **Object Detection** | Learned Weights (YOLOv8n) | Verified on COCO128 | Yes (`01_Vision_Module.ipynb`) |
| **Bounding Box Tracking** | Algorithmic (IoU Greedy Matcher) | Verified on `video.mp4` | Yes (Track IDs persist across frames) |
| **Spatial Cue (L/C/R)** | Algorithmic (Horizontal Bounding Center) | Verified in `hazard_scorer.py` | Yes (Lateral position cues displayed) |
| **Relative Depth Proxy** | Algorithmic ($0.65 y_{\text{base}} + 0.35 \text{area}$) | Verified in `depth.py` | Yes (Proximity scale $0.0$ to $1.0$) |
| **Optical Looming Hazard** | Algorithmic (Expansion rate $dA/dt$) | Verified in `hazard_scorer.py` | Yes (Approaching vehicles flagged) |
| **Temporal Consistency** | Algorithmic (Exponential Moving Average) | Verified in `detector.py` | Yes (Suppresses single-frame flickers) |

---

## 3. Audio Model Benchmark: Downstream Classifier Comparison

### 3.1 Downstream AudioMLP vs. Logistic Regression Baseline
Evaluated on ESC-50 held-out Fold 5 (400 clips, 50 balanced classes, 8 clips/class):

| Metric | Regularized Logistic Regression Baseline | Trained PyTorch AudioMLP | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **91.00% (364 / 400)** | **89.75% (359 / 400)** | $-1.25\%$ |
| **Macro Precision** | 91.42% | 90.15% | $-1.27\%$ |
| **Macro Recall** | 91.00% | 89.75% | $-1.25\%$ |
| **Macro F1-Score** | **90.85%** | **89.59%** | $-1.26\%$ |
| **Best Validation Loss** | N/A | **0.2800** | Epoch 17 |
| **Trainable Parameters** | 102,450 | 538,419 | $+435,969$ |

> **Auditor Note**: Regularized Logistic Regression slightly outperforms the 2-layer AudioMLP ($91.00\%$ vs. $89.75\%$) on Fold 5. This is common when classifying precomputed high-dimensional embeddings (2,048-D) with a small sample size ($N=1,600$), because linear models with $L_2$ regularization exhibit lower variance. However, AudioMLP offers non-linear expressiveness for end-to-end gradient fine-tuning.

### 3.2 Noise Robustness Benchmark
AudioMLP was evaluated against additive zero-mean Gaussian noise at varying Signal-to-Noise Ratios (SNR):

| Environmental Noise Condition | Accuracy | Macro F1-Score | Accuracy Degradation |
| :--- | :--- | :--- | :--- |
| **Clean Baseline** | **89.75%** | **89.59%** | $0.00\%$ |
| **+20 dB SNR (Mild Noise)** | 90.00% | 89.94% | $+0.25\%$ (Noise regularizing effect) |
| **+10 dB SNR (Moderate Noise)**| 90.75% | 90.68% | $+1.00\%$ |
| **0 dB SNR (Severe Noise)** | 89.50% | 89.52% | $-0.25\%$ |

### 3.3 Digital Silence Rejection
- When supplied an all-zero waveform (digital silence or disconnected mic), AudioMLP produces a maximum class probability of **0.215** across all 50 categories.
- Because the decision threshold is set to **0.350**, silent windows are safely rejected without false alarm triggering.

---

## 4. Gesture Model Benchmark (GestureLSTM)

### 4.1 Quantitative Results on Held-Out Participant (`subject_p01`)

| Metric | Pointing | Waving | Stop | Neutral | Overall Macro Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Precision** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000 (100.0%)** |
| **Recall** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000 (100.0%)** |
| **F1-Score** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000 (100.0%)** |
| **Support** | 8 | 8 | 8 | 8 | **32 sequences** |

### 4.2 Critical Audit of 100% Accuracy
1. **Participant Disjointness**: The test split strictly contains recordings from participant `subject_p01`, who was completely absent from training (`p02`, `p04`) and validation (`p03`).
2. **Sample Size Constraint**: $N=32$ test sequences is a small test set. In a 4-class problem, 32 correct classifications is statistically sound ($p < 10^{-19}$ under random guessing), but does not prove unconstrained real-world invariance.
3. **Controlled Scripting**: Gestures were performed facing the camera with minimal self-occlusion. Real-world users will exhibit partial occlusions and varied arm velocities.

---

## 5. End-to-End System Latency Comparison

| Pipeline Component | Hardware | Latency per Unit | Effective Throughput | Real-Time Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Vision (YOLOv8n)** | CPU | 49.23 ms / frame | ~20.3 FPS | Fits $\le 30$ FPS sampling |
| **Audio (CNN14 + MLP)** | CPU | 29.30 ms / 2s chunk | ~34.1 chunks/s | Fits 1.0s sliding hop |
| **Gesture (MediaPipe + LSTM)** | CPU | 1.01 ms / sequence | ~990 FPS | Fits real-time tracking |
| **Transformer Fusion** | CPU | 1.91 – 2.20 ms / step | ~450 steps/s | Fits 100 ms control loop |
| **Total Synchronous Cycle** | CPU | **~81.5 ms** | **~12.3 Hz** | **Fully real-time on CPU** |
