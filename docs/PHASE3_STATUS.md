# AI AINAK Phase 3 status

Updated 2026-10-12. Phase 3 implementation is **partially complete**. The data, feature-export, alignment, replay, training, and evaluation workflows are implemented and exercised where local inputs permit. Real gesture and fusion training, their held-out evaluation, and a trained three-branch demonstration remain blocked because this workspace has no consented reviewed gesture sequences or human-labeled aligned multimodal sessions.

## Evidence from this workspace

| Area | Status | Evidence |
|---|---|---|
| Vision branch | Pretrained model available; no ground-truth evaluation | YOLOv8n checkpoint present. In the prerecorded smoke, 3 sampled frames yielded 3 features. Cold first inference: 10,389.63 ms; warmed mean: 62.45 ms (2 samples). |
| Audio branch | Pretrained model available; zero-shot benchmark completed | PANNs CNN14 checkpoint present. ESC-50 ontology-mapped evaluation: 2,000 clips, accuracy 0.7760, macro precision 0.7716, recall 0.7760, F1 0.7483. This is not fine-tuning or a safety evaluation. |
| Gesture collection and extraction | Workflow implemented; local extraction conditionally blocked | No reviewed participant sequence or LSTM checkpoint. Installed MediaPipe is 0.10.31, which removed the legacy `mp.solutions` API. Use the project's pinned version below 0.10.31 or supply a compatible local Pose Landmarker `.task` file with `--pose-model`. |
| Timestamp alignment | Implemented; synthetic checks passed | Feature export emits `ainak.feature.v1`; alignment emits reviewed `ainak.aligned-example.v1` rows and rejects unreviewed or conflicting labels. |
| Fusion training and ablations | Implemented; not trained or evaluated | No aligned, reviewed real sessions. Trainers require at least three independent participant/session groups and preserve a held-out test split. |
| Full prerecorded AV replay | Partial execution demonstrated | Smoke session `phase3-smoke-20261012`: 3 vision + 5 audio features, 8 timeline steps and trace rows, 0 gesture features, 0 Transformer invocations. Status `partial_blocked`; no same-invocation three-branch result. Export took 16.606 s; replay took 0.090 s. |
| Desktop UI | Starts and shuts down in offscreen smoke | Desktop remains a vision-only shell; camera 0 was unavailable during the earlier audit. |
| Spoken alerts | Text fallback available; speech output unverified | Earlier Windows SAPI test failed with COM error `-2147200967`. |
| Local tests | Passed, with expected integration skip | `24 passed, 1 skipped`; skip is the unavailable real-data gesture/fusion integration. `compileall`, CLI help, `git diff --check`, and the offscreen desktop startup/shutdown also passed. |

The replay report and trace are in the ignored local session directory `datasets/recordings/phase3-smoke-20261012/`. The inputs are a synthetic video/audio plumbing fixture. They validate execution only and do not support model accuracy, alert-quality, safety, or participant-performance claims. The machine is Python 3.10.6, PyTorch 2.2.2 CPU-only, CUDA unavailable; YOLO and CNN14 weights are present, gesture/fusion weights are absent.

## Next data-dependent steps

1. Install the declared compatible MediaPipe dependency or provide a licensed local Pose Landmarker bundle, then collect consented gestures with anonymous participant codes. Review and label neutral, pointing, waving, and stop intervals.
2. Import genuinely paired audio/video recordings with verified offsets and recording IDs. Have a reviewer label time intervals, priority, and alert category; mark provenance and review status explicitly.
3. Export branch features, validate them, and align only reviewed intervals. Inspect split IDs to ensure participant/session groups do not leak across train, validation, and test.
4. Train the gesture LSTM and fusion Transformer, run held-out reports and modality ablations, then replay a session containing all three valid branches. Report metrics, uncertainty, failures, and actual device timings.
5. Resolve the SAPI issue if spoken output is needed. No real-time or safety claim should be made from the current partial replay.

## Licensing and scope

Commercial and investor distribution are not cleared. Review the applicable Ultralytics YOLOv8 AGPL-3.0/Enterprise terms, PANNs implementation and checkpoint terms, MediaPipe/Pose Landmarker model terms, and any dataset terms before distribution. ESC-50 is CC BY-NC 3.0, which restricts commercial use of that data. Keep these licensing checks distinct from model-quality evidence.

This remains a research prototype, not a certified mobility aid. It must not replace a cane, guide dog, or human judgment for unsupervised navigation.
