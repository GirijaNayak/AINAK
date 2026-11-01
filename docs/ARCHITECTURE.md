# Architecture and data contract

## Modules

- `models/vision/detector.py`: YOLOv8 inference and detection-derived 128-value feature summary.
- `models/audio/classifier.py`: `panns_inference.AudioTagging` on the 32 kHz CNN14 AudioSet checkpoint; emits exact checkpoint labels and the 2048-value embedding.
- `models/gesture/lstm.py`: 33-point MediaPipe pose normalization, bounded temporal sequence and trainable two-layer LSTM. Checkpoint labels are strictly checked.
- `models/gesture/pose.py`: uses the MediaPipe legacy pose API when present and provides a Tasks API adapter requiring an explicitly supplied local `.task` bundle on MediaPipe 0.10.31+.
- `models/fusion/transformer.py`: modality-specific projections and embeddings, two pre-norm Transformer encoder blocks, four attention heads, GELU feed-forward size 256, dropout, relative-time projection, padding masks, then context/priority/confidence/alert heads.
- `data/features.py`: versioned `ainak.feature.v1` JSONL contract with time intervals, modality dimensions, validity, predictions, model hashes, preprocessing IDs, and latency validation.
- `data/feature_export.py`: bounded-window audio and sampled video export using actual YOLO/CNN14 weights; gesture embeddings are emitted only when a compatible LSTM checkpoint is loaded.
- `data/alignment.py`: disk-backed feature indexing and timestamp/interval alignment to explicitly reviewed labels; emits `ainak.aligned-example.v1` and a dataset quality summary.
- `runtime/synchronization.py`: bounded per-modality event queues, timestamp order checks, stale masking, and temporal history anchors.
- `runtime/prerecorded_demo.py`: local paired-media validation, streamed feature replay, trained-checkpoint loading, per-step trace, latency reporting, and explicit blocker states.
- `runtime/checkpoint_validation.py`: validates trainer-written checkpoint schema, dataset hash, positive split counts, and disjoint participant/session group IDs before serving.
- `runtime/pipeline.py`: validates event times and feature shapes, applies per-modality masks, distinguishes attempted from successful Transformer calls, and records failed forward diagnostics.
- `decision/alert_engine.py`: deterministic fallback priorities, confidence gate, staleness, persistence and cooldown. It only accepts learned outputs when caller marks the model as trained.
- `audio_output/speech.py`: nonblocking priority queue, duplicate suppression, pyttsx3 first and local Windows System.Speech fallback; a missing speech backend is nonfatal.
- `interfaces/desktop_app.py`: live vision preview and local prerecorded session preview/replay with branch predictions, Transformer status, warnings, recent alerts, and latency.

## Tensor and manifest contract

Transformer input is `features[modality]` shaped `[batch,time,feature_dim]`; modality dimensions are vision 128, audio 2048, gesture 128. `mask` is boolean `[batch,time,3]`, true means an observation exists. `relative_times` is `[batch,time]` in seconds relative to the window's first event. All-empty examples yield a zero context and finite outputs. Fusion JSONL labels describe priority and alert type. Split by `recording_id`; never split windows from a single recording across validation.

The synchronizer uses a session-relative media clock. `record-session` writes audio/video and per-block/frame monotonic clock indexes; `import-session` stages paired media from another device with user-declared file offsets and an explicit same-session assertion. The importer checks media overlap, but pairing provenance remains user supplied. `record-gesture` captures timestamped normalized landmarks only (no raw video) and provides a human-reviewed interval annotation template; `extract-gesture` emits sequences only for reviewed intervals with provenance. The feature exporter writes actual branch outputs and `align-features` joins them to reviewed targets without index-based timing assumptions. Camera pixel position is not physical bearing, distance or collision time.

Feature dimensions are 128 for the vision detection summary, 2048 for the actual PANNs embedding, and 128 for the LSTM representation. The vision vector is a hand-engineered summary of YOLO detections, not an internal learned backbone embedding. Invalid/silent windows are represented as missing with reasons; they do not receive fabricated zero vectors. The fusion checkpoint loader refuses missing or insufficiently provenanced weights. Synthetic unit tests use an explicit private test-only switch and do not create trained checkpoint evidence. The current workspace has no reviewed gesture training sequences or aligned real fusion labels, so a genuine three-branch learned inference run remains blocked.

The project pins MediaPipe below 0.10.31 for the legacy `solutions.pose` path. The currently installed 0.10.31 package removed that API; the Tasks adapter requires the user to supply a compatible Pose Landmarker model bundle. `pose-status` distinguishes dependency readiness from frame-level verification. No such bundle was found locally, and an install attempt for the compatible pinned wheel could not reach PyPI because DNS/network access was unavailable, so pose extraction from phone videos remains blocked in this exact environment.

`evaluate-audio` runs zero-shot PANNs CNN14 on ESC-50. The mapping is an explicit dictionary from each ESC-50 category to one or more AudioSet checkpoint labels; aliases are max-reduced and the highest mapped score is selected. Three classes use declared nearest-label proxies due to missing exact AudioSet labels. Reports include archive and metadata hashes, checkpoint hash, fold IDs and predictions.

## Dataset audit

No dataset in the repository provides verified aligned labels for all three target branches and the desired safety-context output. Ego4D is a partial candidate because portions pair first-person audio/video and the collection has interaction annotations, but the specific stop/pointing/waving plus hazard-priority labels were not verified and access requires a license agreement. HaGRIDv2 is image-centric, approximately 1.5 TB, and does not supply the needed temporal pose sequences/verified waving label. Neither dataset was downloaded or used. The workspace capture tool provides synchronized raw recordings for future consented collection.

## Inference states

Vision and audio use pretrained external weights when acquired. Gesture and fusion weights only exist after data-backed training. The installed code contains no initialized checkpoint fallback; absent files produce clear errors. Synthetic vectors in unit tests test integration shape only.
