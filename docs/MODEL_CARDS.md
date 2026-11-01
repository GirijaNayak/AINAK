# AI AINAK Model Cards

## 1. YOLOv8n COCO Object Detector & Safety Policy

- **Status:** Fully operational pretrained checkpoint with relative depth estimation, temporal tracking, and hazard policy.
- **Model Checkpoint:** `artifacts/vision/yolov8n.pt` (SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`, 6,549,765 bytes).
- **Source:** Ultralytics YOLOv8n pretrained on COCO 80 classes.
- **Input:** 640×640 RGB/BGR video frames or images.
- **Output:** Bounding boxes, class predictions, object tracking IDs (ByteTrack-compatible temporal tracker), relative depth estimate $\hat{d}$, looming rate $\Delta d / \Delta t$, lateral position cue (left, center, right), and assistive hazard scores.
- **Feature Vector for Fusion:** 128-D normalized detection summary encoding class distribution, spatial location, looming velocity, and risk priority.
- **Quantitative Benchmark (COCO128 Validation Split):**
  - **mAP@0.5:** 0.6052
  - **mAP@0.5:0.95:** 0.4468
  - **Precision:** 0.6395
  - **Recall:** 0.5368
  - **Inference Latency:** 49.2 ms per frame on Intel Core i5 CPU (~20.3 FPS); 53.7 ms warmed average in real-session replay.
  - **Report File:** `reports/vision-coco128-val.json`.
- **Assistive Safety Policy:** Implemented in `ainak/models/vision/hazard_scorer.py`. Separates raw detection from hazard classification: computes base risk based on obstacle velocity, proximity, looming rate, and trajectory towards user walking corridor.
- **License:** Ultralytics AGPL-3.0 / Enterprise.

---

## 2. PANNs CNN14 Audio Classifier & Downstream Environmental Sound MLP

- **Status:** AudioSet-pretrained backbone + downstream environmental sound classifier verified on held-out fold and noise robustness benchmarks.
- **Model Checkpoint (Backbone):** `artifacts/audio/Cnn14_mAP=0.431.pth` (MD5: `541141fa2ee191a88f24a3219fff024e`, 327.4 MB).
- **Model Checkpoint (Downstream MLP):** `artifacts/audio/downstream_mlp.pt` (SHA-256 verified, 2-layer MLP with BatchNorm and Dropout).
- **Precomputed Embedding Cache:** `artifacts/audio/esc50_embeddings_cnn14.npz` (2,048-D embeddings for all 2,000 ESC-50 clips).
- **Input:** Mono 32 kHz waveform (fast polyphase resampling via `scipy.signal.resample_poly` from 44.1 kHz).
- **Output:** 527 AudioSet class probabilities, 2,048-D bottleneck acoustic embedding, and 50-class environmental sound predictions (traffic, horns, speech, footsteps, alarms).
- **Quantitative Benchmark (ESC-50 Held-Out Fold 5):**
  - **Downstream Logistic Regression:**
    - Accuracy: 91.00%
    - Macro-F1: 90.85%
  - **Downstream AudioMLP (2048 -> 256 -> 50):**
    - Accuracy: 89.75%
    - Macro-F1: 89.59%
  - **Zero-Shot AudioSet Transfer Baseline:**
    - Accuracy: 77.60%
    - Macro-F1: 74.83%
  - **Noise Robustness Evaluation:**
    - Clean: 91.00%
    - +20 dB SNR Gaussian noise: 90.00%
    - +10 dB SNR Gaussian noise: 90.75%
    - Silence Input Rejection: Max prediction confidence on digital silence is 0.215, cleanly below the 0.35 confidence threshold.
  - **Report File:** `reports/audio-downstream-eval.json`.
- **Inference Latency:** 29.3 ms per 2-second audio chunk on CPU.
- **Dataset License:** ESC-50 is Creative Commons BY-NC 3.0. PANNs inference code is MIT.

---

## 3. MediaPipe Pose Tracker & Temporal Gesture LSTM

- **Status:** MediaPipe Pose Tasks API landmarker installed + GestureLSTM trained on disjoint participant splits.
- **Pose Landmarker Model:** `artifacts/gesture/pose_landmarker_lite.task` (SHA-256: `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`, 5,777,746 bytes).
- **Gesture LSTM Checkpoint:** `artifacts/gesture/gesture_lstm.pt` (1,073,538 bytes, `ainak.gesture-checkpoint.v1` schema).
- **Training Dataset:** `datasets/gesture-sessions/gesture_manifest.csv` (128 multi-subject kinematic sequences across 4 independent subjects: `subject_p01` through `subject_p04`).
- **Input:** 24 temporal frames × 132 features (33 MediaPipe pose landmarks × normalized $(x, y, z, \text{visibility})$).
- **Target Classes:** `pointing`, `waving`, `stop`, `neutral`.
- **Architecture:** 2-layer Bidirectional LSTM, hidden dimension 128, dropout 0.2, linear projection to 4 gesture classes + 128-D bottleneck representation for fusion.
- **Quantitative Benchmark (Held-out Test Participant Split):**
  - **Held-out Test Sequences:** 32 (from independent participant)
  - **Accuracy:** 1.0000 (100.0%)
  - **Macro-F1:** 1.0000
  - **Validation Loss:** 0.0003
  - **Confusion Matrix:** Zero cross-class confusion between stop, waving, pointing, and neutral.
  - **Inference Latency:** 1.01 ms per 24-frame window on CPU.
  - **Metrics File:** `artifacts/gesture/gesture_lstm.metrics.json`.
- **Privacy & Safety:** No facial identification or biometric identifiers are stored; only anonymized normalized joint kinematics are extracted and processed.

---

## 4. Multimodal Sensor Fusion Transformer

- **Status:** 2-layer, 4-head Transformer encoder trained on aligned multi-subject session manifests with disjoint group splits.
- **Model Checkpoint:** `artifacts/fusion/fusion.pt` (`ainak.fusion-checkpoint.v1` schema, SHA-256: `49aead8f88b731f728d96ce73481c6aa6fc6f3afbf2d2544f7be4ef05c978adb`).
- **Architecture:**
  - Embedding dimension: 128
  - Attention heads: 4
  - Transformer encoder layers: 2
  - Feed-forward dimension: 256
  - Modality dropout: 0.10 (trains network to remain robust to missing sensor streams)
  - Projection layers: Linear projection from Vision (128-D), Audio (2,048-D), and Gesture (128-D) into shared 128-D latent space.
  - Heads: Scalar Priority Head (BCE), Scalar Confidence Head (BCE), 4-class Alert Classification Head (CrossEntropy: `none`, `informational`, `caution`, `urgent`).
- **Training Manifest:** `datasets/recordings/aligned_manifest.jsonl` (24 aligned temporal windows across 4 paired recording sessions: `session-multimodal-01` through `04`).
- **Disjoint Split Structure:**
  - Training: `session-multimodal-01`, `session-multimodal-04` (12 windows)
  - Validation: `session-multimodal-02` (6 windows)
  - Held-out Test: `session-multimodal-03` (6 windows)
- **Ablation Evaluation Report (`reports/fusion-ablations.json`):**
  - Evaluated on untouched held-out session across all 7 modality combinations and against a non-Transformer Late-Fusion MLP Baseline:
    - Vision only
    - Audio only
    - Gesture only
    - Vision + Audio
    - Vision + Gesture
    - Audio + Gesture
    - All Three Modalities (Vision + Audio + Gesture)
    - Late Fusion MLP Baseline
  - Demonstrates continuous temporal inference with zero runtime crashes even when 1 or 2 modalities are completely dropped.
- **Inference Latency:** 1.91 ms (p95: 2.36 ms) for fusion forward pass and decision logic on CPU.

---

## 5. Local Speech & Decision Engine

- **Status:** Production-ready local TTS with prioritized alert queue and deduplication.
- **Backends:** Primary Windows `pyttsx3` / SAPI, with automatic `.NET System.Speech` fallback.
- **Safety Policy:**
  - Priority queue: Preempts less urgent pending alerts when an urgent hazard is detected.
  - Cooldown suppression: Suppresses repetitive alerts for the same obstacle within configurable cooldown window (e.g. 3.0 seconds).
  - Text fallback: If audio hardware is missing or blocked, logs clear alert text to stdout without raising fatal exceptions.
