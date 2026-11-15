# AI AINAK V2 — Independent Gesture Capability Verification & Real-World Validation

**Audit Date**: 2026-11-13  
**Auditor**: Senior ML Research Engineer, Deep Learning Auditor & Kinematics Specialist  
**Evaluated Artifact**: `artifacts/v2/gesture/best.pt` (`GestureTemporalNet`, 269,721 parameters)  
**Dataset Inspected**: `datasets/gesture-sessions-v2/` ($N = 1,200$ sequences)  
**Audit Capability Verdict**: **PARTIALLY VERIFIED**

---

## 1. Executive Summary & Verdict Rationale

| Evaluation Dimension | Stated V2 Claim | Independent Audit Finding | Verification Status |
| :--- | :--- | :--- | :--- |
| **Vocabulary Scale** | 25 distinct gesture classes | 25 classes implemented & functionally evaluated | **Verified** |
| **Held-Out Test Accuracy** | 100.0% on subject `p01` ($N=200$) | Confirmed 100.0% accuracy & 1.0000 Macro-F1 on test set | **Verified (Synthetic)** |
| **Data Leakage Check** | Zero train-test leakage | Zero file/hash overlap; high mathematical correlation (99.24%) | **Partially Verified** |
| **Data Provenance** | Recorded human kinematics | Parametric simulation via `generate_base_kinematics()` | **Partially Verified** |
| **Unknown Gesture Rejection** | 100.0% unknown rejection | True for Gaussian noise; drops to 88.0% on unseen human gestures | **Partially Verified** |
| **Partial Occlusion Robustness** | Robust to occluded limbs | Hand occlusion collapses accuracy to 4.50% | **Not Verified** |
| **Camera Viewpoint Invariance** | Angle invariant | Accuracy drops to 36.0% at 45° yaw perspective | **Partially Verified** |
| **Real Video Extraction** | End-to-end real video pipeline | MediaPipe Tasks extractor extracts 33 landmarks from MP4 | **Verified** |

> [!IMPORTANT]
> **Audit Finding on Data Provenance**: The 1,200 sequences in `datasets/gesture-sessions-v2/` were synthetically generated through parameterized mathematical kinematics (`prepare_gesture_v2_dataset.py`), not physically recorded optical motion capture from 6 distinct human actors. While the train/val/test splits strictly partition subjects `p01` through `p06` with zero direct file or byte leakage, the mathematical correlation between test samples and training samples of the same class is **99.24%**. Real human in-the-wild video exhibits significantly higher variance, jitter, and occlusion.

---

## 2. Dataset Split & Cryptographic Leakage Audit

The dataset manifest at `datasets/gesture-sessions-v2/gesture_manifest_v2.csv` was audited across all 1,200 sequences:

```
Total Sequences: 1,200 (24 frames @ 30 FPS, 33 3D landmarks = 132 features/frame)
  ├── Training Split:   600 sequences (Subjects: subject_p02, subject_p04, subject_p05)
  ├── Validation Split: 400 sequences (Subjects: subject_p03, subject_p06)
  └── Held-Out Test:    200 sequences (Subject:  subject_p01)
```

### Cryptographic Leakage Verification
1. **File Overlap**: Checked pairwise filenames between `train`, `val`, and `test`. Overlap = `0` files.
2. **Byte-Level Hash Overlap**: SHA-256 digests computed across all `.npy` landmark arrays. Shared content hashes = `0`.
3. **Kinematic Correlation**: Computed pairwise normalized dot-product cosine similarity between test sequences (`subject_p01`) and training sequences (`subject_p02`, `p04`, `p05`) for identical gesture classes.
   $$\text{Mean Kinematic Correlation} = 0.9924 \quad (99.24\%)$$
   **Conclusion**: There is no direct data leakage, but test sequences are mathematically derived from the same underlying trajectory generator.

---

## 3. Individual Evaluation of All 25 Gesture Classes

Every class was audited individually on the 200 held-out test sequences (`subject_p01`):

| Class | Train ($N$) | Val ($N$) | Test ($N$) | Precision | Recall | F1-Score | FPR | Mean Confidence | Confidence Range |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `idle` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 96.0% | [96.0%, 96.0%] |
| `wave` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 96.0% | [95.9%, 96.1%] |
| `stop` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.5% | [95.4%, 95.5%] |
| `come_here` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.7% | [95.6%, 95.7%] |
| `go_away` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.8% | [95.7%, 95.9%] |
| `thumbs_up` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.8% | [95.7%, 95.9%] |
| `thumbs_down` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 96.0% | [95.9%, 96.0%] |
| `ok_sign` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.9% | [95.8%, 96.1%] |
| `peace_sign` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 96.2% | [96.2%, 96.3%] |
| `open_palm` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.7% | [95.6%, 95.8%] |
| `fist` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.8% | [95.7%, 95.9%] |
| `pointing_left` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.7% | [95.7%, 95.8%] |
| `pointing_right` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.3% | [95.3%, 95.4%] |
| `pointing_forward` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.9% | [95.9%, 96.0%] |
| `pointing_upward` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.6% | [95.6%, 95.6%] |
| `pointing_downward`| 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.9% | [95.9%, 95.9%] |
| `help` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.7% | [95.7%, 95.7%] |
| `emergency` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 96.3% | [96.2%, 96.3%] |
| `call_someone` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.8% | [95.8%, 95.9%] |
| `follow_me` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.6% | [95.6%, 95.6%] |
| `wait` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.8% | [95.8%, 95.8%] |
| `turn_left` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.9% | [95.8%, 95.9%] |
| `turn_right` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 96.0% | [95.9%, 96.0%] |
| `slow_down` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.7% | [95.6%, 95.7%] |
| `thank_you` | 24 | 16 | 8 | 1.000 | 1.000 | 1.000 | 0.0000 | 95.9% | [95.9%, 96.0%] |
| **Macro Average** | **600** | **400** | **200** | **1.000** | **1.000** | **1.000** | **0.0000** | **95.8%** | **[95.3%, 96.3%]** |

### Confusion Matrix
On the 200 held-out test sequences, the confusion matrix is strictly diagonal ($25 \times 25$ identity block with 8 samples per diagonal entry; 0 off-diagonal false classifications).

---

## 4. Unknown Gesture Rejection: Synthetic Noise vs Real Unseen Movements

The previous report claimed a **100.0% unknown gesture rejection rate**. The independent audit tested this claim against two distinct categories of unknown inputs:

```
                                 UNKNOWN INPUT DISTRIBUTION
                                  ┌───────────────────────┐
                                  │   Evaluation Modality │
                                  └───────────┬───────────┘
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
          Synthetic Gaussian Noise                      Structured Unseen Human Gestures
       - Pure random coordinates                      - Scratching head, Arms crossed
       - No anatomical skeleton                       - Hands in pockets, Clapping, Drinking
       - Mean Confidence: 25.9%                       - Mean Confidence: 62.0%
       - Rejection @ 0.70: 100.0%                     - Rejection @ 0.70: 88.0% (12.0% False Alarms)
       - Rejection @ 0.50: 95.0%                      - Rejection @ 0.50: 0.0%  (100.0% False Alarms)
```

### Empirical Rejection Breakdown
1. **Synthetic Gaussian Noise**:
   - Rejection Rate at $\tau = 0.70$: **100.0%**
   - Mean Top-Class Confidence: **0.2590** (25.9%)
2. **Structured Unseen Human Movements** (8 movements outside the 25 classes: head scratching, folded arms, hands in pockets, clapping, drinking, checking wristwatch, conversational gesticulation, phone to ear):
   - Rejection Rate at $\tau = 0.70$: **88.0%** (**12.0% false alarms** trigger gesture events)
   - Rejection Rate at $\tau = 0.50$: **0.0%** (100.0% of unfamiliar gestures accepted)
   - Mean Top-Class Confidence: **0.6201** (62.0%)
   - **Primary False Alarm Victim**: Unfamiliar movements frequently activate `thumbs_down` due to resting arm postures.

> [!WARNING]
> **Audit Finding**: The 100% unknown rejection claim does NOT generalize to real unfamiliar human gestures. When a user engages in natural conversational arm movements or non-vocabulary gestures, the model exhibits a **12.0% false-positive rate** at the standard 0.70 threshold.

---

## 5. Adversarial Robustness & Kinematic Invariance Audit

The model was subjected to 5 distinct physical perturbations:

### 1. Partial Hand & Wrist Occlusion
In assistive environments, users frequently walk with hands partially off-camera or occluded by clothing/bags. Hand landmarks (MediaPipe indices 15–22) were zeroed out:
- **Baseline Accuracy**: 100.00%
- **Accuracy with Hands Occluded**: **4.50%**
- **Performance Drop**: **-95.50%**
- **Finding**: The model relies almost exclusively on wrist and finger coordinates relative to hips; it does not degrade gracefully under hand occlusion.

### 2. Camera Viewpoint Yaw Rotations
Body pose observed from oblique perspective angles:
- **0° (Frontal)**: 100.00%
- **±15° Yaw**: 77.00% – 82.00%
- **±30° Yaw**: 52.00% – 61.00%
- **±45° Yaw**: 36.00% – 52.50%
- **Finding**: Yaw departures beyond 20° cause rapid degradation because the 3D depth normalization does not rotate coordinates to a canonical body frame.

### 3. Distance & Scale Variations
Testing distance variations from 0.5m to 3.5m:
- **0.50x Scale (Far Distance)**: 53.50%
- **0.75x Scale (Medium-Far)**: 90.50%
- **1.00x Scale (Nominal)**: 100.00%
- **1.25x Scale (Medium-Close)**: 95.00%
- **2.00x Scale (Very Close-up)**: 73.50%

### 4. Sensor Noise & Poor Lighting Simulations
Gaussian coordinate jitter added to simulate low-light camera sensor noise ($\sigma$ from 0.01 to 0.20):
- $\sigma = 0.01$: 100.00%
- $\sigma = 0.05$: 100.00%
- $\sigma = 0.10$: 100.00%
- $\sigma = 0.20$: 99.00%
- **Finding**: Highly resilient to high-frequency sensor noise.

---

## 6. Real Recorded Video Verification (`session-multimodal-01`)

The actual MediaPipe Tasks extractor (`artifacts/gesture/pose_landmarker_lite.task`) was run over 100 real frames of a human walking down an indoor corridor:
- **Frames Inspected**: 100 frames @ 30 FPS
- **Frames with Extracted Landmarks**: 100% (33 keypoints extracted per frame)
- **Sliding Sequences Evaluated**: 13 windows (24 frames each)
- **Observed Predictions**:
  - Model predicted `idle` or low-confidence intermediate states.
  - Confidences ranged between 0.35 and 0.65 (below the 0.70 threshold).
  - **Zero spurious emergency alerts emitted**.

---

## 7. Final Capability Verdict: Partially Verified

```
Capability Classification: PARTIALLY VERIFIED
- Mathematical/Synthetic Architecture: VERIFIED (100% accuracy, clean LayerNorm, 128-D interface)
- Rejection on Gaussian Noise:          VERIFIED (100.0%)
- Rejection on Real Human Gestures:     PARTIALLY VERIFIED (88.0% @ 0.70; 12.0% false alarms)
- Real Human Video Extraction:         VERIFIED (TasksPoseAdapter operational)
- Occlusion Robustness:                 NOT VERIFIED (Collapses to 4.5% on occluded hands)
- Multi-Angle Invariance:              PARTIALLY VERIFIED (Degrades to 36% at 45 deg yaw)
```

**Recommendation**: Retain V1 as default in production; before deploying V2 in the field, fine-tune `GestureTemporalNet` with random joint dropout (simulating occlusions) and 3D rotational augmentation.
