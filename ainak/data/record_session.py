from __future__ import annotations
import csv, json, time, uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np


def record_synchronized_session(output_dir: Path, duration_seconds: int=60, camera_index: int=0,
                                sample_rate: int=32000, max_duration_seconds: int=300) -> Path:
    """Locally capture bounded webcam video and mono microphone audio with host-clock indexes.

    This records source media only. It does not extract model features or invent annotations.
    """
    if duration_seconds<1 or duration_seconds>max_duration_seconds:raise ValueError(f"duration must be 1..{max_duration_seconds} seconds")
    import cv2
    import sounddevice as sd
    import soundfile as sf
    session_id=f"session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    folder=output_dir/session_id;folder.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(camera_index)
    if not cap.isOpened():raise RuntimeError(f"could not open camera index {camera_index}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640;height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps=float(cap.get(cv2.CAP_PROP_FPS));fps=fps if 1<=fps<=120 else 15.
    video_path=folder/"video.mp4";audio_path=folder/"audio.wav"
    writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():cap.release();raise RuntimeError("could not initialize local MP4 writer")
    # Preallocated, time-bounded buffer: mono float32 is about 128 KB/second at 32 kHz.
    audio=np.zeros((duration_seconds*sample_rate,1),dtype=np.float32);cursor=0;audio_blocks=[];overflow=False
    start_wall=datetime.now(timezone.utc);start=time.perf_counter()
    def callback(indata,frames,time_info,status):
        nonlocal cursor,overflow
        if status:audio_blocks.append({"status":str(status),"sample_start":cursor,"frames":0,"monotonic_seconds":time.perf_counter()-start})
        count=min(frames,len(audio)-cursor)
        if count<=0:overflow=True;return
        audio[cursor:cursor+count]=indata[:count,:1];cursor+=count
        audio_blocks.append({"sample_start":cursor-count,"frames":count,"monotonic_seconds":time.perf_counter()-start})
    frames=[]
    try:
        with sd.InputStream(samplerate=sample_rate,channels=1,dtype="float32",callback=callback,blocksize=1024):
            while time.perf_counter()-start<duration_seconds:
                ok,frame=cap.read()
                if not ok:continue
                relative=time.perf_counter()-start;writer.write(frame);frames.append((relative,frame.shape[1],frame.shape[0]))
                # Yield lightly to keep the audio callback and UI/capture stack responsive.
                time.sleep(.001)
    finally:
        cap.release();writer.release()
    sf.write(audio_path,audio[:cursor,0],sample_rate)
    with (folder/"video_frames.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["frame_index","monotonic_seconds","width","height"])
        for i,(ts,wid,hei) in enumerate(frames):w.writerow([i,f"{ts:.6f}",wid,hei])
    with (folder/"audio_blocks.csv").open("w",encoding="utf-8",newline="") as f:
        keys=("sample_start","frames","monotonic_seconds","status");w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(audio_blocks)
    annotation_fields=("start_seconds","end_seconds","event_label","priority","alert_type","annotation_confidence","reviewed",
                       "gesture_label","annotator","provenance","vision_notes","audio_notes","gesture_notes")
    with (folder/"annotation_template.csv").open("w",encoding="utf-8",newline="") as f:
        csv.DictWriter(f,fieldnames=annotation_fields).writeheader()
    metadata={"recording_id":session_id,"created_utc":start_wall.isoformat(),"duration_seconds":duration_seconds,
        "audio_sample_rate":sample_rate,"audio_samples":cursor,"video_frames":len(frames),"video_fps_header":fps,
        "video_file":video_path.name,"audio_file":audio_path.name,"annotation_template":"annotation_template.csv",
        "clock":"local time.perf_counter relative to session start","audio_overflow":overflow,
        "annotations":"unlabeled review template; no event labels are inferred by the recorder"}
    (folder/"recording.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return folder
