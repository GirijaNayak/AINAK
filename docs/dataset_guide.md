# AI AINAK — Dataset Acquisition and Preparation Guide

## 1. Overview and Modality Standards

AI AINAK operates across three perception modalities:
1. **Vision**: Object detection and tracking via Ultralytics YOLOv8n.
2. **Audio**: Environmental audio recognition via AudioSet-pretrained PANNs CNN14 and downstream classifiers.
3. **Gesture**: Body-pose landmark extraction and temporal sequence classification via MediaPipe Pose (Tasks API Lite model) and a 2-layer LSTM.
4. **Multimodal Fusion**: Transformer-based fusion aligning timestamped vectors (`[time, 128]` for vision, `[time, 2048]` for audio, `[time, 128]` for gesture) with boolean availability masks.

---

## 2. Dataset Manifests and Origins

| Modality / Target | Dataset | Source / URL | License | Intended Use in AI AINAK |
|---|---|---|---|---|
| Vision | COCO128 / COCO2017 | https://cocodataset.org/ | CC BY 4.0 | Object detector validation, mAP benchmarking, smoke tests |
| Audio | ESC-50 (Environmental Sound Classification) | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 | 2,000 5-second environmental recordings across 50 classes in 5 folds |
| Audio Backbone | AudioSet PANNs CNN14 | Zenodo record 3987831 | MIT (code) / AudioSet terms | Pretrained 32 kHz feature extractor yielding 2,048-D embeddings |
| Gesture Landmarks | MediaPipe Pose Landmarker Lite | Google Cloud Storage / MediaPipe | Apache-2.0 | 33-point 3D landmark extraction with visibility scores |
| Gesture Sequences | AI AINAK Reviewed Pose Sequences | Local Consented Recording Sessions | Project License | Temporal sequence training across `pointing`, `waving`, `stop`, `neutral` |
| Multimodal Sessions | AI AINAK Synchronized AV Sessions | Paired local recordings with verified annotations | Project License | Three-branch replay and Transformer fusion training |

---

## 3. ESC-50 Environmental Audio Setup

ESC-50 contains 2,000 clips organized into 5 standardized folds:
- **Sample Rate**: 44.1 kHz native, resampled to deterministic 32 kHz mono.
- **Duration**: Exactly 5.0 seconds per clip (160,000 samples at 32 kHz).
- **Target Classes**: 50 environmental classes including `car_horn`, `engine`, `siren`, `footsteps`, `crying_baby`, `dog`, `glass_breaking`, `train`.

To extract and cache embeddings for downstream training:
```powershell
python scripts/train_audio_downstream.py
```
This produces `artifacts/audio/esc50_embeddings_cnn14.npz` and evaluates both regularized Logistic Regression and PyTorch AudioMLP on held-out fold 5.

---

## 4. Gesture Sequence Protocol

- **Landmark Specification**: 33 BlazePose landmarks, each with $(x, y, z, \text{visibility})$. Total raw dimension = 132.
- **Normalization**:
  - Center: Midpoint of left and right hips (landmarks 23 and 24).
  - Scale: Euclidean distance between left and right shoulders (landmarks 11 and 12).
  - Low-visibility clamping: Coordinates with visibility $< 0.20$ are zeroed out.
  - Value bounds: Normalized coordinates clipped to $[-5.0, 5.0]$.
- **Sequence Length**: Fixed 24 frames. Shorter or longer clips are linearly interpolated along the temporal axis to exactly 24 frames.
- **Split Invariant**: No participant or recording session may overlap across train, validation, and test partitions. At least 3 independent group IDs are strictly required.

---

## 5. Multimodal Alignment (`ainak.aligned-example.v1`)

Aligned examples pair temporal windows within a common media clock:
- **Window Length**: 2.0 seconds (stride 1.0 second).
- **Max Feature Age**: 1.5 seconds.
- **Modality Mask**: Boolean vector of length 3 (`[vision, audio, gesture]`). A modality is marked valid only when real features exist; missing modalities are explicitly masked rather than fabricated with zeros.
- **Target Schema**: Priority $[0.0, 1.0]$, confidence $[0.0, 1.0]$, alert type (`immediate_attention`, `caution`, `informational`, `gesture_feedback`, `uncertain`).
