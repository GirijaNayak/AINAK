# AI AINAK — Troubleshooting and Diagnostic Guide

## 1. System Diagnostics

Run the integrated system check:
```powershell
python -m ainak.interfaces.cli doctor
```
This inspects:
- Python version, architecture (x64), and thread allocation.
- PyTorch version and CUDA availability.
- Presence of pretrained checkpoints (`yolov8n.pt`, `Cnn14_mAP=0.431.pth`).
- Availability of optional packages (`panns_inference`, `PySide6`, `pyttsx3`).

---

## 2. MediaPipe Pose Landmarker on MediaPipe 0.10.31+

### Symptom:
`pose-status` reports `extraction_ready: false` or `selected_api: tasks_pose_landmarker` with missing model.

### Resolution:
MediaPipe 0.10.31 deprecates `mp.solutions.pose` in favor of the Tasks Vision API, which requires an official `.task` bundle. Download the verified Google-hosted Lite model:
```powershell
python -m ainak.interfaces.cli acquire-pose --overwrite
```
This saves `artifacts/gesture/pose_landmarker_lite.task` and validates its SHA-256 digest (`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`).

---

## 3. Speech Backend and COM Errors on Windows

### Symptom:
`pyttsx3` raises Windows COM error `-2147200967`.

### Resolution:
AI AINAK implements an automatic fallback mechanism. If `pyttsx3` encounters COM initialization failures or driver locks, the system automatically falls back to the native Windows `.NET System.Speech.Synthesis.SpeechSynthesizer` backend. Run the speech diagnostic:
```powershell
python -m ainak.interfaces.cli speech-test
```
This will speak an audible test phrase through the fallback backend and report `spoken_count=1`.

---

## 4. Audio Sample Rate and PANNs CNN14 Preprocessing

### Symptom:
`ValueError: PANNs CNN14 expects mono 32 kHz waveform; resample explicitly before calling`.

### Resolution:
ESC-50 and standard microphone audio are typically sampled at 44.1 kHz or 48 kHz. AI AINAK requires explicit resampling before passing waveforms to `prepare_audio_window`:
- For 44.1 kHz: use deterministic rational resampling `scipy.signal.resample_poly(waveform, 320, 441)` which runs in under 5 ms per 5-second clip.
- For other sample rates: use `librosa.resample(waveform, orig_sr=sr, target_sr=32000)`.
