# AI AINAK — Dataset Provenance & Data Governance Report (Version 2)

**Date**: 2026-11-12  
**Auditor**: Senior Data Governance & ML Research Engineer  
**Scope**: Complete Provenance, Licensing, Class Distribution, and Leakage Audit for V1 and V2  

---

## 1. Master Dataset Provenance Matrix

| Modality Branch | Dataset Name | Version | Sample Count | Provenance & Source | License | Data Split Structure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision (Backbone V1)** | Microsoft COCO 2017 | V1 | 118,287 images | Microsoft Common Objects in Context | CC BY 4.0 | Pretrained official weights (`yolov8n.pt`) |
| **Vision (Validation V1)**| COCO128 Mini-split | V1 | 128 images | Ultralytics COCO subset | CC BY 4.0 | 128 validation images |
| **Vision (Backbone V2)** | Objects365 + GoldG + CC3M | V2 | 3M+ image-text pairs | Tencent AI / Ultralytics YOLO-World v2 | Apache 2.0 / CC | Pretrained open-vocabulary backbone (`yolov8s-worldv2.pt`) |
| **Vision (V2 Vocabulary)** | AINAK Accessibility 225 | V2 | 225 category prompts | Curated for blind navigation & safety | MIT / Project License | Zero-shot open-vocabulary prompt bindings |
| **Audio (Backbone V1/V2)**| Google AudioSet | V1 / V2 | 2,084,320 clips | Google Research AudioSet 527 classes | CC BY 4.0 | Pretrained frozen feature extractor (`Cnn14_mAP=0.431.pth`) |
| **Audio (Classifier V1/V2)**| ESC-50 Environmental Audio| V1 / V2 | 2,000 5-sec WAV clips | ESC: Dataset for Environmental Sound Classification | CC BY-NC 3.0 | Folds 1–4 Train (1,600 clips) / Fold 5 Test (400 clips) |
| **Gesture (Kinematic V1)** | AINAK Kinematic V1 | V1 | 128 sequences | MediaPipe BlazePose 3D (4 subjects) | Project License | `p02`,`p04` Train (64) / `p03` Val (32) / `p01` Test (32) |
| **Gesture (Kinematic V2)** | AINAK Kinematic V2 | V2 | **1,200 sequences** | MediaPipe BlazePose 3D (6 subjects) | Project License | `p02`,`p04`,`p05` Train (600) / `p03`,`p06` Val (400) / `p01` Test (200) |
| **Multimodal Replay** | Paired Sessions 01–04 | V1 / V2 | 4 synchronized sessions | Real corridor walking audiovisual sessions | Project License | Sessions 01 & 04 Train / Session 02 Val / Session 03 Test |

---

## 2. Gesture V2 Dataset Distribution & Leakage Analysis

### Subject-Disjoint Partitioning
A primary threat to validity in human action and gesture recognition is **subject leakage** (where the same person's body morphology appears in both training and test sets).

AINAK V2 strictly partitions subjects:
- **Training Set (600 sequences)**:
  - Participants: `subject_p02`, `subject_p04`, `subject_p05`
  - Balance: Exactly 24 sequences per class (8 per subject × 3 subjects)
- **Validation Set (400 sequences)**:
  - Participants: `subject_p03`, `subject_p06`
  - Balance: Exactly 16 sequences per class (8 per subject × 2 subjects)
- **Held-Out Test Set (200 sequences)**:
  - Participant: `subject_p01`
  - Balance: Exactly 8 sequences per class across all 25 classes
- **Leakage Verification**:
  $$\text{Subject}_{\text{train}} \cap \text{Subject}_{\text{val}} = \emptyset, \quad \text{Subject}_{\text{train}} \cap \text{Subject}_{\text{test}} = \emptyset, \quad \text{Subject}_{\text{val}} \cap \text{Subject}_{\text{test}} = \emptyset$$
  **Guaranteed 100% subject-independent generalization testing.**

### Augmentation Governance
Data augmentation is applied strictly at training time:
1. Joint spatial scaling ($[0.92, 1.08]$) simulating distance variations.
2. Additive Gaussian noise ($\mathcal{N}(0, 0.015^2)$) simulating sensor keypoint jitter.
3. Random cyclic roll ($\pm 1$ frame) simulating asynchronous sequence start.
4. Validation and test sets remain **100% unaugmented** to preserve ground truth integrity.

---

## 3. Data Integrity & Reproducibility Guarantees

1. **Manifest Immutability**:
   - V1 Manifest: `datasets/gesture-sessions/gesture_manifest.csv` (SHA-256: `1d18228...`)
   - V2 Manifest: `datasets/gesture-sessions-v2/gesture_manifest_v2.csv` (SHA-256 recorded in checkpoint metadata)
2. **Deterministic Resampling**:
   All sequence resampling to 24 frames uses piecewise linear interpolation over normalized keypoint coordinates.
3. **Storage Governance**:
   All sequences are saved in standard uncompressed IEEE 754 float32 `.npy` arrays with shape `(24, 132)` and validated for non-finite values (`NaN`, `Inf`).
