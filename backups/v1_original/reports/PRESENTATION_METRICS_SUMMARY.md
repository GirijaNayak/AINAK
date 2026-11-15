# AI AINAK — Presentation Metrics & Evaluator Summary

**Date**: 2026-11-12  
**Auditor**: Senior ML Research Engineer & Technical Evaluator  
**Purpose**: Concise, evidence-backed presentation guide for the college project defense and Google Colab multimodal demonstration.

---

## 1. Executive Summary Table

| Module | Dataset Size (Train / Val / Test) | Training Epochs | Best Evaluation Metric | Main Technical Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Vision (YOLOv8n)** | Pretrained on COCO (~118k images). Evaluated on 128 images. | Pretrained (0 local epochs) | **mAP@0.5: 0.6052**<br>Precision: 63.95%, Recall: 53.68% | Not fine-tuned on custom assistive datasets; relies on rule-based heuristics for depth and looming. |
| **Audio (PANNs + AudioMLP)** | 1,600 train / 400 test clips (ESC-50, Folds 1–4 vs. 5). | 40 epochs (AdamW) | **Accuracy: 89.75%**<br>Macro-F1: 89.59% (Noise: 89.5–90.7%) | Backbone is frozen; classifier evaluates 5-second clean audio clips rather than continuous speech/audio streams. |
| **Gesture (Pose + LSTM)** | 64 train / 32 val / 32 test sequences across 4 subjects. | 30 epochs (AdamW) | **Test Accuracy: 100.0%**<br>Macro-F1: 1.0000 on held-out subject | Small participant pool ($N=4$ subjects, $N=32$ test clips); recorded in controlled indoor environments. |
| **Multimodal Transformer** | 12 train / 6 val / 6 test windows across 4 sessions. | 9 epochs (early stop on divergence) | **Inference Latency: 2.20 ms**<br>Ablations: Stable under all 7 combos | Overfitting due to small sample size ($N=12$ training windows); 0.0% alert accuracy on held-out test session. |

---

## 2. Presentation Figures Catalog (`reports/presentation_assets/`)

All 15 figures below have been generated from real evidence and saved in `reports/presentation_assets/` for direct embedding into presentation slides:

| Figure File | Description | Underlying Evidence / Source | Recommended Slide Placement |
| :--- | :--- | :--- | :--- |
| `01_vision_pr_curve_yolov8n.png` | Precision-Recall Curve for YOLOv8n detector | `runs/detect/val/PR_curve.png` & `reports/vision-coco128-val.json` | Slide: Computer Vision Performance |
| `02_vision_confusion_matrix.png` | 80-class detection confusion matrix | `runs/detect/val/confusion_matrix.png` | Slide: Computer Vision Analysis |
| `03_vision_session_detection_sample.jpg` | Real video detection sample with bounding boxes & hazards | `datasets/recordings/session-multimodal-01/video.mp4` | Slide: Real-World Vision Demonstration |
| `04_audio_confusion_matrix_esc50.png` | 50-class confusion matrix for AudioMLP on held-out fold 5 | `reports/audio-downstream-eval.json` (400 clips) | Slide: Acoustic Event Recognition |
| `05_audio_top_class_performance.png` | Top-12 environmental sound classes by F1-score | `reports/audio-downstream-eval.json` | Slide: Acoustic Classifier Performance |
| `06_audio_noise_robustness.png` | Accuracy curve under Clean, 20 dB, 10 dB, and 0 dB SNR noise | `reports/audio-downstream-eval.json` | Slide: Environmental Noise Robustness |
| `07_gesture_confusion_matrix.png` | Confusion matrix on held-out subject `p01` (100% accuracy) | `artifacts/gesture/gesture_lstm.metrics.json` | Slide: Kinematic Gesture Recognition |
| `08_gesture_loss_curves.png` | Historical training & validation loss curves across 30 epochs | `artifacts/gesture/gesture_lstm.metrics.json` | Slide: Gesture Model Convergence |
| `09_gesture_skeleton_sample.jpg` | 33-point BlazePose skeletal keypoint overlay on video | `datasets/recordings/session-multimodal-01/video.mp4` | Slide: Pose Extraction Demonstration |
| `10_transformer_loss_curves.png` | Training vs. validation loss curve showing overfitting | `artifacts/fusion/fusion.metrics.json` (Epochs 1–9) | Slide: Transformer Fusion Training Analysis |
| `11_transformer_ablation_comparison.png` | Priority and Confidence MAE across all 7 modality combinations | `reports/fusion-ablations.json` | Slide: Modality Ablation & Robustness |
| `12_transformer_modality_contributions.png` | Attention contribution trajectory for Vision, Audio, Gesture | `outputs/fusion_outputs.npz` | Slide: Cross-Modal Attention Weights |
| `13_inference_latency_comparison.png` | CPU latency (ms) and effective throughput (FPS) per module | Profiling logs across models | Slide: Real-Time Execution Feasibility |
| `14_dataset_size_comparison.png` | Log-scale comparison of training samples across models | Manifest inspection | Slide: Dataset Scale & Provenance |
| `15_complete_ainak_pipeline_diagram.png` | High-level system architecture flowchart | Architectural specification | Slide: System Overview / Architecture |

---

## 3. Evaluator Defense Strategy

### What are the Defensible Strengths of AI AINAK?
1. **Fully Operational End-to-End Pipeline**: The complete data path from video and audio ingestion through 3 sensory branches, cross-modal Transformer attention, rule-based decision filtering, and speech alerts is fully working and demonstrable without mock data.
2. **Participant-Independent Gesture Splitting**: GestureLSTM was trained and tested on strictly disjoint participants (trained on `p02`/`p04`, validated on `p03`, tested on `p01`).
3. **Rigorous Acoustic Benchmark**: AudioMLP was evaluated against standardized ESC-50 held-out fold 5, benchmarked against a regularized Logistic Regression baseline, and validated for noise robustness down to 0 dB SNR and digital silence rejection.
4. **Missing-Modality Transformer Robustness**: The Transformer encoder uses key-padding masks and demonstrates zero runtime failures when any 1 or 2 sensory branches drop out.
5. **Real-Time Efficiency**: Complete multi-stage inference executes in **~81.5 ms** on commodity CPU hardware (~12.3 Hz), meeting real-time assistive requirements.

### What are the Known Weaknesses & How to Present Them?
1. **YOLOv8n Pretraining**: Be honest that YOLOv8n was not fine-tuned on custom data. Frame it as leveraging industry-standard transfer learning from Microsoft COCO, while the custom novelty lies in the downstream depth proxy, looming calculation, and multimodal fusion.
2. **Small Multimodal Training Corpus**: Acknowledge that the Transformer was trained on 12 windows across 4 recording sessions, which led to validation loss divergence after epoch 1. Frame this as an **architectural prototype demonstrating cross-modal attention mechanics and missing-modality robustness**, with future work focused on scaling up collection to 50+ sessions.
3. **Audio Windowing Latency**: Audio processing operates on 2.0-second sliding windows with 1.0-second hops, meaning acoustic hazards have up to 1.0-second detection latency compared to frame-by-frame vision.
