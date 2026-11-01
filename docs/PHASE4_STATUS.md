# AI AINAK — Phase 4 evidence register

Updated 2026-10-12. Phase 4 is **engineering-complete where independent of missing real training data, but product/model completion is blocked**. The workspace has functioning local vision/audio inference, tested synthetic fusion control flow, a prerecorded replay workflow, a responsive desktop shell, and verified local spoken output. It does not have gesture/fusion training data or checkpoints, a ready MediaPipe pose model, or evidence of real learned three-branch fusion.

## Evidence status

| Requirement | Status | Evidence and limit |
|---|---|---|
| Repository and artifact audit | **Inspected** | Workspace initially had source/docs/tests but no gesture/fusion checkpoints or reviewed gesture/aligned recording datasets. YOLOv8n and PANNs CNN14 checkpoints are present and load. Python 3.10.6, PyTorch 2.2.2 CPU-only; CUDA unavailable. |
| Zero-Transformer-call diagnosis | **Diagnosed; guard hardened** | The replay correctly gates invocation on a provenance-validated trained fusion checkpoint. That checkpoint is absent, so calls are not attempted. The earlier synthetic test path could mark a randomly initialized model as `trained`; the pipeline now requires verified trainer metadata in production, while synthetic tests must opt into a private test-only path. |
| Fusion contract and masks | **Unit-tested** | Regression tests cover all three inputs, one/two missing modalities, stale data, bad dimensions/timestamps, missing/untrained checkpoint rejection, successful forward counting, and failed-forward diagnostics. Synthetic forward success verifies software wiring only. |
| Real trained three-branch replay | **Blocked** | Actual prerecorded smoke replay generated 3 vision and 5 audio features, 0 gesture features, and 8 trace steps. It recorded 0 attempts / 0 successful Transformer calls because the gesture and trained fusion checkpoints are absent. `all_three_branches_same_invocation=false`; report status is `partial_blocked`. The trace is a synthetic plumbing fixture, not accuracy evidence. |
| MediaPipe readiness | **Implemented; extraction unverified** | Diagnostic reports MediaPipe 0.10.31, Tasks API selected, missing `.task` bundle, and `extraction_ready=false`. The compatible 0.10.30 package install and the official pose-bundle download both failed with DNS resolution errors. No real pose extraction was claimed. |
| Gesture collection and training | **Workflow implemented; training blocked** | Consent-based landmark capture, local phone-video import, human-reviewed annotations, fixed 24-frame sequences, overlap/malformed rejection, grouped splits, provenance-aware checkpoints, and confirmed session deletion are implemented. Unit fixtures pass. No participant sequences exist; no LSTM training or held-out metrics were produced. |
| Real synchronized training data | **Blocked** | No reviewed gesture clips and no aligned human-labeled multimodal examples are available. The trainers reject synthetic or unreviewed examples and require at least three independent participant/session groups. Fusion dataset size: 0 eligible examples. |
| Fusion model and ablation workflow | **Implemented; training/evaluation blocked** | The 128-D, four-head, two-layer Transformer, checkpoint provenance checks, split validation, training/evaluation and modality-ablation paths are present. No real training, held-out evaluation, or MLP-baseline comparison was possible. |
| Local desktop | **Implemented; offscreen lifecycle verified** | Live preview is vision-only. Prerecorded session preview/replay runs available model branches on workers and exposes branch/fusion states, warnings, latency, recent alerts, and speech. Offscreen startup/shutdown exited successfully. Camera hardware remains unverified/unavailable here. |
| Local speech | **Verified via fallback** | `speech-test` spoke one alert via Windows `.NET System.Speech`; `spoken_count=1`, worker stopped, no backend error. pyttsx3 still reports COM error `-2147200967`; fallback is active. |
| Vision evaluation | **Inference/profiled; accuracy unverified** | YOLOv8n loads and detects objects in bundled sample images; there are no local ground-truth labels or mAP metrics. CPU wall means were 29.4/23.0 ms at 320 px and 52.0/48.0 ms at 640 px on two sample images (one warmup, two measured repeats each). |
| Audio evaluation | **Zero-shot evaluated** | 2,000 ESC-50 clips, five folds, ontology mapping to AudioSet labels: accuracy 0.7760, macro precision 0.7716, recall 0.7760, F1 0.7483. Three target categories use declared proxies. This is not fine-tuning, calibrated hazard probability, or safety evaluation. |
| Performance evidence | **Measured, limited** | Synthetic paired replay: YOLO cold first 10,558.57 ms, warmed mean 55.04 ms (n=2); CNN14 cold 49.95 ms, warmed mean 38.30 ms (n=4); fusion/decision replay mean 0.039 ms, p95 0.082 ms (8 steps; no Transformer call); feature export 16.321 s; replay 0.073 s. This is not a full learned pipeline or real-time guarantee. No gesture latency or memory benchmark is available. |
| Licensing | **Key terms inventoried; commercial clearance unresolved** | See the register below and generated [dependency metadata inventory](DEPENDENCY_LICENSES.json). Metadata inventory covers 229 distributions visible to this system-site-packages venv; it is not a complete native/transitive-asset audit or legal review. No commercial-use clearance is claimed. |

## Transformer call: cause and safeguards

The old run's zero call count came from the intended missing/untrained-checkpoint gate, not a failing Transformer forward pass: the gesture LSTM and fusion checkpoint did not exist. The earlier synthetic test setup also had a software-level trust gap because it could pass random model weights with `trained=True`. That gap is closed by validating trainer-written checkpoint schema, real-data/provenance markers, manifest hash, positive epoch/example counts, and disjoint train/validation/test groups before enabling production calls. Synthetic tests use an explicit test-only opt-in and never establish trained-model status.

When a verified trained checkpoint is available, one or two valid modalities may reach the Transformer with explicit missing-modality masks; an all-empty or stale window is rejected to deterministic fallback. A rejected or failed call is not counted as successful. A “complete trained three-branch” replay requires at least one successful Transformer forward in which vision, audio, and gesture are all valid in the same selected window. The demo report enforces this invariant.

The unit suite proves that a real `MultimodalTransformer` forward can consume synthetic vectors from all three branches and increments the successful-call counter. **That is software integration evidence, not learned fusion evidence.** No available local checkpoint or data permits a real three-branch replay.

## Reproducible checks

Executed in the workspace `.venv` on Python 3.10.6 / Windows x64:

- Baseline before Phase 4 edits: **24 passed, 1 skipped**.
- Final `python -m pytest -q`: **44 passed, 1 skipped**. The skip is `test_real_three_branch_replay_invokes_one_trained_transformer`, with concrete blockers: no trained gesture checkpoint, no trained Transformer checkpoint, MediaPipe pose model unavailable, and no paired reviewed session. The Python process also printed a pyreadline teardown `WinError 0xc0000008` after reporting a successful test exit; no test failed.
- `python -m compileall -q ainak tests scripts`: **passed**.
- CLI `--help`: **passed**; `doctor`: **passed**; `pose-status`: **passed**, reported not ready.
- `demo-session --session datasets/recordings/phase3-smoke-20261012 --reuse-features ...`: **ran actual YOLO/CNN14 replay, returned `partial_blocked` and nonzero as designed**. The report and trace are `reports/phase4-replay-check.json` and `reports/phase4-replay-check.trace.jsonl`.
- Offscreen PySide6 desktop construction, display, and close: **passed**.
- `speech-test`: **passed with audible local speech through System.Speech**; pyttsx3 failure was captured as a backend warning.
- `acquire-pose`: **blocked by network DNS** (`getaddrinfo failed`). No pose bundle was written. The earlier compatible MediaPipe wheel installation was blocked the same way.
- Existing evaluation/profile artifacts were inspected; no expensive repeat of the 130-second ESC-50 benchmark was necessary.

Use the following commands after supplying suitable local inputs/checkpoints:

```powershell
python -m ainak.interfaces.cli doctor
python -m ainak.interfaces.cli pose-status --pose-model artifacts\gesture\pose_landmarker_lite.task
python -m ainak.interfaces.cli demo-session --session datasets\recordings\<paired-session-id>
python -m ainak.interfaces.cli speech-test
python -m pytest -q
python scripts\export_dependency_licenses.py
```

The smoke report is also preserved in `datasets/recordings/phase3-smoke-20261012/phase3-demo.json`; it hashes its source video/audio and states the partial status. It is not participant footage and does not support safety claims.

## Dataset and training record

- **ESC-50:** downloaded and used only for the zero-shot PANNs evaluation described above. Its archive SHA-256 is `805afc618aff80eff0641d51311647871fead5807da68bb18e43a27db47a79ce`; dataset terms are CC BY-NC 3.0. It is not cleared for commercial use.
- **Gesture recordings/sequences:** none collected or used. HaGRIDv2 and Ego4D were not downloaded. No external gesture dataset was silently mapped to temporal pointing/waving/stop sequences.
- **Fusion examples:** zero eligible aligned, reviewed real examples. No LSTM/Transformer training, checkpoint reload, ablation, held-out metrics, or learned fusion claims.
- **Synthetic inputs:** used only for shape/control-flow tests and the explicitly synthetic plumbing replay.

To unblock training, collect/import consented recordings from at least three independent participant/session groups, review pointing, waving, stop, and neutral intervals, and create genuinely paired AV sessions with reviewed contextual targets, timestamps, annotator and provenance. More groups are needed for a credible held-out evaluation. Then extract pose/features, validate alignments and splits, train and evaluate the LSTM, re-export gesture features, and train/evaluate the fusion model on the same grouped data.

## Licensing and distribution register

| Item | Observed terms/provenance | Attribution/restrictions and unresolved review |
|---|---|---|
| Ultralytics YOLOv8n code/weights | Installed Ultralytics 8.3.116 identifies AGPL-3.0; official project offers Enterprise terms. Weights are official `yolov8n.pt` release assets. | Evaluate AGPL obligations or obtain appropriate commercial terms. Review pretrained weight and COCO training-data terms. No distribution clearance. [Ultralytics license/model docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md). |
| PANNs inference package | Installed `panns-inference` 0.1.1 metadata identifies MIT. CNN14 checkpoint came from Zenodo record 3987831; its content license is not resolved by this workspace's checksum verification. | Preserve MIT notice for code; independently review the checkpoint and AudioSet source-data terms before distribution. [PANNs inference](https://github.com/qiuqiangkong/panns_inference), [checkpoint record](https://zenodo.org/records/3987831). |
| ESC-50 | Official repository states CC BY-NC 3.0; archive SHA above. | Non-commercial restriction is incompatible with assuming investor/commercial product use is permitted. Review attribution and exact dataset terms. [ESC-50 license](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE). |
| MediaPipe Python package | Installed 0.10.31 package metadata identifies Apache-2.0. | The Tasks model bundle was not obtained; review the exact model-bundle terms separately before use/distribution. [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python). |
| Gesture dataset | None present or used. | Future participant data requires consent, purpose limitation, retention/deletion controls, and reviewed provenance. No external gesture dataset is cleared. |
| Direct Python dependencies | See inventory for exact versions and reported metadata: NumPy/Torch/scikit-learn/SoundFile report BSD-family terms; OpenCV and MediaPipe Apache-2.0; librosa ISC; PANNs MIT; PySide6 offers LGPL/GPL or commercial Qt terms; pyttsx3 MPL-2.0; Ultralytics AGPL-3.0. | The inventory lists 229 visible distributions, 3 with incomplete license metadata. It does not inspect all native binaries or bundled assets. Review Qt linking/distribution obligations, OpenCV third-party licenses, all transitive package notices, and the final shipped dependency lock. |
| Other assets | No HaGRID/Ego4D data or MediaPipe task bundle acquired. The workspace test recordings/reports are local ignored artifacts. | Do not ship personal recordings or restricted evaluation data without separate review. |

The relevant official sources include [MediaPipe's Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python), the [Ultralytics YOLOv8 licensing documentation](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md), and the [ESC-50 license](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE). This register is evidence triage, not legal advice or commercial clearance.

## Remaining blockers and safety scope

1. Restore DNS/network access or provide a locally obtained compatible MediaPipe 0.10.30 wheel / licensed Pose Landmarker `.task` model; process a suitable local person video and verify landmark output.
2. Collect consented reviewed temporal gesture sequences and genuinely paired AV recordings with reviewed labels. No synthetic substitute is acceptable for training.
3. Train and reload the gesture LSTM and fusion Transformer on grouped real data; report held-out metrics and all seven modality ablations plus the MLP baseline.
4. Run a same-window three-branch replay with a provenance-validated trained fusion checkpoint and verify successful invocation in the trace.
5. Resolve model, checkpoint, dataset, Qt, and transitive dependency terms before any commercial/investor distribution.

AI AINAK remains a research prototype. It is not certified and must not be used or marketed as safe for unsupervised navigation or as a replacement for a cane, guide dog, or human judgment. No collision prevention, physical distance/bearing, calibrated risk, or spatial-audio claim is supported.
