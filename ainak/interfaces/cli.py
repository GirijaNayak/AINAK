from __future__ import annotations
import argparse, hashlib, json, logging, os, sys, time
import importlib.util
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import torch
from ..config import RuntimeConfig, Paths
from ..utils.logging_config import configure_logging

LOG=logging.getLogger("ainak")
SOURCES={
    "vision":("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", "artifacts/vision/yolov8n.pt"),
    "audio":("https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1", "artifacts/audio/Cnn14_mAP=0.431.pth"),
}
EXPECTED_MD5={"audio":"541141fa2ee191a88f24a3219fff024e"}
PANN_LABELS_URL="https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def ensure_audio_labels(root: Path) -> Path:
    target=root/"artifacts"/"audio"/"panns-home"/"panns_data"/"class_labels_indices.csv"
    if target.exists(): return target
    target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix(".csv.part")
    try:
        with urlopen(Request(PANN_LABELS_URL,headers={"User-Agent":"AI-AINAK/0.1"}),timeout=30) as response,partial.open("wb") as out:
            while chunk:=response.read(65536):out.write(chunk)
        import csv
        with partial.open(encoding="utf-8",newline="") as handle: rows=list(csv.reader(handle))
        if len(rows)!=528 or rows[0][:3]!=["index","mid","display_name"]:
            raise RuntimeError(f"AudioSet label file unexpected schema or count: {len(rows)} rows")
        partial.replace(target)
        return target
    except Exception:
        if partial.exists():partial.unlink()
        raise


def acquire(name: str, root: Path, allow_large: bool=False) -> Path:
    if name not in SOURCES: raise ValueError("supported pretrained downloads: vision, audio")
    url, relative=SOURCES[name]; destination=root/relative; destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if name in EXPECTED_MD5:
            existing=hashlib.md5()
            with destination.open("rb") as downloaded:
                for chunk in iter(lambda:downloaded.read(1024*1024),b""): existing.update(chunk)
            if existing.hexdigest()!=EXPECTED_MD5[name]: raise RuntimeError(f"existing {name} checkpoint has an invalid MD5; preserved at {destination}")
        if name=="audio":ensure_audio_labels(root)
        print(f"Already present and verified: {destination} ({destination.stat().st_size:,} bytes)"); return destination
    if name=="audio" and not allow_large:
        print("PANNs CNN14 32 kHz checkpoint is about 327.4 MB. Confirm available storage and run again with --allow-large.")
        raise SystemExit(2)
    partial=destination.with_suffix(destination.suffix+".part")
    offset=partial.stat().st_size if partial.exists() else 0
    headers={"User-Agent":"AI-AINAK/0.1 local-model-acquisition"}
    if offset: headers["Range"]=f"bytes={offset}-"
    request=Request(url,headers=headers)
    try:
        with urlopen(request,timeout=60) as response:
            if offset and response.status != 206: offset=0
            mode="ab" if offset else "wb"
            with partial.open(mode) as f:
                while chunk:=response.read(1024*1024): f.write(chunk)
    except Exception:
        print(f"Download interrupted; partial file retained at {partial}. Re-run the same command to resume if the server supports byte ranges.",file=sys.stderr)
        raise
    hasher=hashlib.sha256()
    with partial.open("rb") as downloaded:
        for chunk in iter(lambda:downloaded.read(1024*1024),b""): hasher.update(chunk)
    digest=hasher.hexdigest()
    if name in EXPECTED_MD5:
        md5=hashlib.md5()
        with partial.open("rb") as downloaded:
            for chunk in iter(lambda:downloaded.read(1024*1024),b""): md5.update(chunk)
        if md5.hexdigest()!=EXPECTED_MD5[name]:
            corrupt=partial.with_suffix(partial.suffix+".checksum-failed")
            partial.replace(corrupt)
            raise RuntimeError(f"upstream MD5 mismatch for {name}; expected {EXPECTED_MD5[name]}, got {md5.hexdigest()}. Preserved at {corrupt}")
    if partial.stat().st_size < 1_000_000:
        raise RuntimeError(f"download suspiciously small ({partial.stat().st_size} bytes); retained at {partial}")
    partial.replace(destination)
    destination.with_suffix(destination.suffix+".source.json").write_text(json.dumps({"url":url,"sha256":digest,"md5":EXPECTED_MD5.get(name),"bytes":destination.stat().st_size,"acquired_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},indent=2),encoding="utf-8")
    if name=="audio":ensure_audio_labels(root)
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes), SHA-256 {digest}")
    return destination


def vision_mode(args) -> None:
    import cv2
    from ..models.vision.detector import VisionDetector
    if args.sample_every<1 or args.max_frames<1: raise ValueError("--sample-every and --max-frames must be positive")
    detector=VisionDetector(args.weights, args.device, args.conf, args.iou)
    source=Path(args.source)
    if not source.exists(): raise FileNotFoundError(f"vision input not found: {source}")
    image=cv2.imread(str(source))
    if image is not None:
        detections,event,_=detector.infer(image)
        print(json.dumps({"input_type":"image","timestamp":event.timestamp,"latency_ms":event.latency_ms,"detections":[d.__dict__ for d in detections],"feature_dim":len(event.feature)},default=list,indent=2)); return
    capture=cv2.VideoCapture(str(source))
    if not capture.isOpened(): raise ValueError("input is neither a readable image nor an openable local video")
    sampled=[]; frame_index=0; fps=float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps<=0: fps=30.
    try:
        while len(sampled)<args.max_frames:
            ok,frame=capture.read()
            if not ok: break
            if frame_index%args.sample_every==0:
                media_seconds=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000.
                if not np.isfinite(media_seconds) or media_seconds<0: media_seconds=frame_index/fps
                detections,event,_=detector.infer(frame,timestamp=media_seconds)
                sampled.append({"frame_index":frame_index,"latency_ms":event.latency_ms,"detections":[d.__dict__ for d in detections]})
            frame_index+=1
    finally: capture.release()
    if not sampled: raise ValueError("video contained no decodable frames")
    print(json.dumps({"input_type":"video","source":str(source),"timestamp_basis":"media seconds from start","fps_header":fps,"decoded_frames":frame_index,"sample_every":args.sample_every,"sampled_frames":len(sampled),"mean_latency_ms":sum(x['latency_ms'] for x in sampled)/len(sampled),"frames":sampled},default=list,indent=2))


def audio_mode(args) -> None:
    numba_cache=Path(args.weights).resolve().parent/"numba-cache";numba_cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR",str(numba_cache))
    panns_home=Path(args.weights).resolve().parent/"panns-home"
    os.environ["USERPROFILE"]=str(panns_home)
    import librosa
    from ..models.audio.classifier import PannsCNN14
    audio,sr=librosa.load(args.source,sr=32000,mono=True)
    model=PannsCNN14(args.weights,args.device); seconds=args.window_seconds; size=int(seconds*sr); hop=int(size*(1-args.overlap))
    windows=[]
    for start in range(0,max(len(audio),1),hop):
        segment=audio[start:start+size]
        if len(segment)<sr//4: continue
        predictions,event=model.infer(segment,sr,timestamp=time.time()+start/sr)
        windows.append({"timestamp":event.timestamp,"latency_ms":event.latency_ms,"top_classes":predictions[:10]})
        if start+size>=len(audio): break
    print(json.dumps({"window_count":len(windows),"embedding_dim":2048,"windows":windows},indent=2))


def gesture_mode(args) -> None:
    import cv2
    from ..models.gesture.lstm import GestureRecognizer,extract_pose_landmarks
    from ..models.gesture.pose import create_pose_tracker
    rec=GestureRecognizer(args.weights,args.sequence_length,args.device,args.threshold)
    capture=cv2.VideoCapture(args.source if not str(args.source).isdigit() else int(args.source))
    if not capture.isOpened(): raise RuntimeError(f"cannot open video source {args.source}")
    out=[]
    try:
        with create_pose_tracker(model_path=args.pose_model,static_image_mode=False,model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5) as pose:
            fps=float(capture.get(cv2.CAP_PROP_FPS));fps=fps if np.isfinite(fps) and fps>0 else 30.
            frame_index=0
            while True:
                ok,frame=capture.read()
                if not ok: break
                timestamp=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000.
                if not np.isfinite(timestamp) or timestamp<0:timestamp=frame_index/fps
                landmarks=extract_pose_landmarks(frame,pose,int(timestamp*1000))
                if landmarks is not None:
                    event=rec.append(landmarks,timestamp)
                    if event and event.labels: out.append({"timestamp":event.timestamp,"labels":event.labels,"confidence":event.confidence})
                frame_index+=1
    finally: capture.release()
    print(json.dumps({"frames_with_accepted_gestures":out,"count":len(out)},indent=2))


def fusion_mode(args) -> None:
    from ..data.validation import validate_jsonl
    from ..training.fusion import FusionManifestDataset,collate
    from ..models.fusion.transformer import ALERT_TYPES
    from ..runtime.prerecorded_demo import _fusion_model
    manifest=Path(args.source); count,errors=validate_jsonl(manifest)
    if errors: raise ValueError("manifest invalid: " + "; ".join(errors[:8]))
    model,status=_fusion_model(Path(args.weights),args.device)
    if model is None: raise ValueError(status.get("reason","fusion checkpoint unavailable"))
    ds=FusionManifestDataset(manifest,seq_len=args.sequence_length)
    batch=collate([ds[i] for i in range(min(len(ds),args.batch_size))]); f,m,t,_,_,_=batch
    with torch.no_grad(): output=model({k:v.to(args.device) for k,v in f.items()},m.to(args.device),t.to(args.device))
    print(json.dumps({"examples_in_manifest":count,"context_shape":list(output.context.shape),"priority":output.priority.cpu().tolist(),
        "confidence":output.confidence.cpu().tolist(),"alert_type":[ALERT_TYPES[i] for i in output.alert_logits.argmax(-1).cpu().tolist()],
        "contribution_fractions":output.modality_contributions.cpu().tolist()},indent=2))


def live_mode(args) -> None:
    """Local camera demonstration; visible fallback status if detector/model is unavailable."""
    import cv2
    from ..models.vision.detector import VisionDetector
    detector=VisionDetector(args.weights,args.device,args.conf,args.iou)
    capture=cv2.VideoCapture(args.camera)
    if not capture.isOpened(): raise RuntimeError("could not open webcam")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,args.width); capture.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    try:
        while True:
            ok,frame=capture.read()
            if not ok: break
            _,event,annotated=detector.infer(frame)
            cv2.putText(annotated,f"Vision active | {event.latency_ms:.0f} ms | research prototype",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
            cv2.imshow("AI AINAK — vision-only local demo (Q to quit)",annotated)
            if cv2.waitKey(1)&0xff==ord('q'): break
    finally: capture.release(); cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="python -m ainak.interfaces.cli",description="AI AINAK local research prototype; incomplete models fail explicitly")
    parser.add_argument("--log-level",default="INFO"); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="show local Python, torch, CUDA, and artifact readiness")
    pose_status=sub.add_parser("pose-status",help="diagnose MediaPipe API, model bundle, and pose extraction readiness")
    pose_status.add_argument("--pose-model",type=Path,default=Path("artifacts/gesture/pose_landmarker_lite.task"))
    pose_acquire=sub.add_parser("acquire-pose",help="download Google's documented local Pose Landmarker Lite bundle")
    pose_acquire.add_argument("--output",type=Path,default=Path("artifacts/gesture/pose_landmarker_lite.task"))
    pose_acquire.add_argument("--overwrite",action="store_true")
    acquire_p=sub.add_parser("acquire",help="download one pretrained model (no datasets are downloaded)"); acquire_p.add_argument("model",choices=SOURCES); acquire_p.add_argument("--root",type=Path,default=Path(".")); acquire_p.add_argument("--allow-large",action="store_true")
    p=sub.add_parser("vision",help="run YOLOv8n on a local image or sampled local video"); p.add_argument("source"); p.add_argument("--weights",default="artifacts/vision/yolov8n.pt"); p.add_argument("--device",default="cpu"); p.add_argument("--conf",type=float,default=.35); p.add_argument("--iou",type=float,default=.45); p.add_argument("--sample-every",type=int,default=10); p.add_argument("--max-frames",type=int,default=200)
    p=sub.add_parser("profile-vision",help="profile YOLOv8 on one or more actual local images after warm-up"); p.add_argument("images",nargs="+",type=Path); p.add_argument("--weights",type=Path,default=Path("artifacts/vision/yolov8n.pt")); p.add_argument("--device",default="cpu"); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--warmup",type=int,default=1); p.add_argument("--repeats",type=int,default=3); p.add_argument("--output",type=Path,default=Path("reports/vision-profile.json"))
    p=sub.add_parser("audio",help="run PANNs CNN14 on local audio with overlapping windows"); p.add_argument("source"); p.add_argument("--weights",default="artifacts/audio/Cnn14_mAP=0.431.pth"); p.add_argument("--device",default="cpu"); p.add_argument("--window-seconds",type=float,default=2.0); p.add_argument("--overlap",type=float,default=.5)
    p=sub.add_parser("evaluate-audio",help="zero-shot evaluate PANNs CNN14 on an extracted ESC-50 dataset"); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--weights",default="artifacts/audio/Cnn14_mAP=0.431.pth"); p.add_argument("--output",type=Path,default=Path("reports/audio-esc50.json")); p.add_argument("--device",default="cpu"); p.add_argument("--batch-size",type=int,default=8)
    p=sub.add_parser("gesture",help="run a trained gesture LSTM on a local video or camera index"); p.add_argument("source"); p.add_argument("--weights",default="artifacts/gesture/gesture_lstm.pt"); p.add_argument("--device",default="cpu"); p.add_argument("--sequence-length",type=int,default=24); p.add_argument("--threshold",type=float,default=.75); p.add_argument("--pose-model",type=Path,help="local MediaPipe Pose Landmarker .task model when using MediaPipe 0.10.31+")
    p=sub.add_parser("fusion",help="run a trained Transformer over a validated feature JSONL manifest"); p.add_argument("source"); p.add_argument("--weights",default="artifacts/fusion/fusion.pt"); p.add_argument("--device",default="cpu"); p.add_argument("--sequence-length",type=int,default=8); p.add_argument("--batch-size",type=int,default=8)
    p=sub.add_parser("live",help="local webcam vision-only demo; requires pretrained YOLOv8n"); p.add_argument("--camera",type=int,default=0); p.add_argument("--width",type=int,default=640); p.add_argument("--height",type=int,default=480); p.add_argument("--weights",default="artifacts/vision/yolov8n.pt"); p.add_argument("--device",default="cpu"); p.add_argument("--conf",type=float,default=.35); p.add_argument("--iou",type=float,default=.45)
    p=sub.add_parser("train-fusion",help="train Transformer on reviewed real aligned JSONL with held-out test split"); p.add_argument("manifest",type=Path); p.add_argument("--output",type=Path,default=Path("artifacts/fusion/fusion.pt")); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--sequence-length",type=int,default=8); p.add_argument("--seed",type=int,default=17); p.add_argument("--device",default="cpu"); p.add_argument("--patience",type=int,default=5); p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("evaluate-fusion",help="evaluate all modality ablations against a late-fusion MLP baseline"); p.add_argument("manifest",type=Path); p.add_argument("--weights",default="artifacts/fusion/fusion.pt"); p.add_argument("--output",type=Path,default=Path("reports/fusion-ablations.json")); p.add_argument("--device",default="cpu"); p.add_argument("--baseline-epochs",type=int,default=10)
    p=sub.add_parser("train-gesture",help="train LSTM on reviewed human-labeled landmark sequences"); p.add_argument("manifest",type=Path); p.add_argument("--output",type=Path,default=Path("artifacts/gesture/gesture_lstm.pt")); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--seed",type=int,default=17); p.add_argument("--device",default="cpu"); p.add_argument("--patience",type=int,default=5); p.add_argument("--hidden-size",type=int,default=128); p.add_argument("--layers",type=int,default=2); p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("validate-manifest",help="validate timestamped fusion JSONL"); p.add_argument("path",type=Path)
    p=sub.add_parser("validate-recording",help="validate a synchronized local recording and reviewed annotation intervals"); p.add_argument("session",type=Path)
    p=sub.add_parser("record-session",help="capture a bounded local camera+microphone session with timestamp indexes"); p.add_argument("--output",type=Path,default=Path("datasets/recordings")); p.add_argument("--seconds",type=int,default=60); p.add_argument("--camera",type=int,default=0); p.add_argument("--sample-rate",type=int,default=32000)
    p=sub.add_parser("record-gesture",help="consent-based pose-only gesture data capture; writes no raw video"); p.add_argument("--output",type=Path,default=Path("datasets/gesture-sessions")); p.add_argument("--seconds",type=int,default=60); p.add_argument("--camera",type=int,default=0); p.add_argument("--participant-code",required=True); p.add_argument("--consent-confirmed",action="store_true"); p.add_argument("--pose-model",type=Path,help="local MediaPipe Pose Landmarker .task model when using MediaPipe 0.10.31+")
    p=sub.add_parser("extract-gesture",help="extract fixed-length sequences from reviewed gesture intervals"); p.add_argument("session",type=Path); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--min-frames",type=int,default=20); p.add_argument("--min-visibility",type=float,default=.2); p.add_argument("--sequence-length",type=int,default=24)
    p=sub.add_parser("delete-gesture-session",help="delete a confirmed local gesture session and optional manifest rows"); p.add_argument("--session",type=Path,required=True); p.add_argument("--confirm-session-id",required=True); p.add_argument("--manifest",type=Path,action="append",default=[],help="manifest to scrub; may be supplied multiple times")
    p=sub.add_parser("speech-test",help="exercise local speech output and report the text fallback if unavailable"); p.add_argument("--text",default="AI AINAK local speech adapter test.")
    p=sub.add_parser("export-features",help="export versioned, provenance-tagged branch features from a paired recording session"); p.add_argument("--session",type=Path,required=True); p.add_argument("--output",type=Path); p.add_argument("--vision-weights",type=Path,default=Path("artifacts/vision/yolov8n.pt")); p.add_argument("--audio-weights",type=Path,default=Path("artifacts/audio/Cnn14_mAP=0.431.pth")); p.add_argument("--gesture-weights",type=Path,default=Path("artifacts/gesture/gesture_lstm.pt")); p.add_argument("--pose-model",type=Path,help="local MediaPipe Pose Landmarker .task model for MediaPipe 0.10.31+"); p.add_argument("--sample-every",type=int,default=5); p.add_argument("--max-frames",type=int,default=10000); p.add_argument("--audio-window-seconds",type=float,default=2.); p.add_argument("--audio-hop-seconds",type=float,default=1.); p.add_argument("--device",default="cpu"); p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("validate-features",help="validate versioned model-feature JSONL without loading it all into memory"); p.add_argument("path",type=Path)
    p=sub.add_parser("align-features",help="align exported branch features to explicitly reviewed annotation intervals"); p.add_argument("--session",type=Path,required=True); p.add_argument("--features",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--window-seconds",type=float,default=2.); p.add_argument("--stride-seconds",type=float,default=1.); p.add_argument("--max-feature-age-seconds",type=float,default=1.5); p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("demo-session",help="run all available local branches on a paired prerecorded video/audio session; reports blockers honestly"); p.add_argument("--session",type=Path,required=True); p.add_argument("--features",type=Path); p.add_argument("--report",type=Path); p.add_argument("--vision-weights",type=Path,default=Path("artifacts/vision/yolov8n.pt")); p.add_argument("--audio-weights",type=Path,default=Path("artifacts/audio/Cnn14_mAP=0.431.pth")); p.add_argument("--gesture-weights",type=Path,default=Path("artifacts/gesture/gesture_lstm.pt")); p.add_argument("--fusion-weights",type=Path,default=Path("artifacts/fusion/fusion.pt")); p.add_argument("--pose-model",type=Path,help="local MediaPipe Pose Landmarker .task model for MediaPipe 0.10.31+"); p.add_argument("--sample-every",type=int,default=5); p.add_argument("--max-frames",type=int,default=10000); p.add_argument("--audio-window-seconds",type=float,default=2.); p.add_argument("--audio-hop-seconds",type=float,default=1.); p.add_argument("--sequence-length",type=int,default=8); p.add_argument("--lookback-seconds",type=float,default=8.); p.add_argument("--device",default="cpu"); p.add_argument("--reuse-features",action="store_true"); p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("import-session",help="copy an explicitly paired prerecorded video/audio set into a local session folder"); p.add_argument("--video",type=Path,required=True); p.add_argument("--audio",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("datasets/recordings")); p.add_argument("--recording-id",required=True,help="user-provided ID asserting the media belong to one session"); p.add_argument("--video-offset-seconds",type=float,default=0.); p.add_argument("--audio-offset-seconds",type=float,default=0.); p.add_argument("--annotations",type=Path); p.add_argument("--participant-code")
    p=sub.add_parser("import-gesture-video",help="copy an explicitly consented phone video and extract timestamped MediaPipe pose landmarks"); p.add_argument("--video",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("datasets/gesture-sessions")); p.add_argument("--participant-code",required=True); p.add_argument("--pose-model",type=Path,help="local MediaPipe Pose Landmarker .task model for MediaPipe 0.10.31+"); p.add_argument("--consent-confirmed",action="store_true")
    p=sub.add_parser("dashboard",help="launch AI AINAK interactive multimodal intelligence console web application"); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8000); p.add_argument("--no-open",action="store_true")
    return parser


def main(argv: list[str] | None=None) -> int:
    args=build_parser().parse_args(argv); configure_logging(args.log_level)
    try:
        if args.command=="doctor":
            paths=Paths(); report={"python":sys.version,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),
                "vision_weights":paths.vision_weights.exists(),"audio_weights":paths.audio_weights.exists(),"gesture_checkpoint":paths.gesture_weights.exists(),"fusion_checkpoint":paths.fusion_weights.exists(),
                "cpu_threads":torch.get_num_threads(),"optional_packages":{"panns_inference":importlib.util.find_spec("panns_inference") is not None,
                "PySide6":importlib.util.find_spec("PySide6") is not None,"pyttsx3":importlib.util.find_spec("pyttsx3") is not None}}
            print(json.dumps(report,indent=2)); return 0
        if args.command=="pose-status":
            from ..models.gesture.pose import pose_readiness
            report=pose_readiness(args.pose_model)
            print(json.dumps(report,indent=2)); return 0
        if args.command=="acquire-pose":
            from ..models.gesture.pose import acquire_pose_model
            print(json.dumps(acquire_pose_model(args.output,overwrite=args.overwrite),indent=2)); return 0
        if args.command=="acquire": acquire(args.model,args.root,args.allow_large); return 0
        if args.command=="vision": vision_mode(args)
        elif args.command=="profile-vision":
            from ..training.vision_profile import profile_vision_images
            report=profile_vision_images(args.images,args.weights,args.device,args.imgsz,args.warmup,args.repeats)
            args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(report,indent=2))
        elif args.command=="audio": audio_mode(args)
        elif args.command=="evaluate-audio":
            from ..training.audio_evaluation import evaluate_esc50
            print(json.dumps(evaluate_esc50(args.dataset,Path(args.weights),args.output,args.device,args.batch_size),indent=2))
        elif args.command=="gesture": gesture_mode(args)
        elif args.command=="fusion": fusion_mode(args)
        elif args.command=="live": live_mode(args)
        elif args.command=="train-fusion":
            from ..training.fusion import train_fusion
            print(json.dumps(train_fusion(args.manifest,args.output,args.epochs,args.batch_size,args.sequence_length,args.seed,args.device,args.patience,args.overwrite),indent=2))
        elif args.command=="evaluate-fusion":
            from ..training.evaluation import evaluate_fusion_ablations
            args.output.parent.mkdir(parents=True,exist_ok=True)
            print(json.dumps(evaluate_fusion_ablations(Path(args.manifest),Path(args.weights),args.output,args.device,args.baseline_epochs),indent=2))
        elif args.command=="train-gesture":
            from ..training.gesture import train_gesture
            print(json.dumps(train_gesture(args.manifest,args.output,args.epochs,args.batch_size,args.seed,args.device,args.patience,args.hidden_size,args.layers,args.overwrite),indent=2))
        elif args.command=="validate-manifest":
            from ..data.validation import validate_jsonl
            count,errors=validate_jsonl(args.path); print(json.dumps({"valid_examples":count,"errors":errors},indent=2)); return 1 if errors else 0
        elif args.command=="validate-recording":
            from ..data.validation import validate_recording_session
            errors=validate_recording_session(args.session); print(json.dumps({"session":str(args.session),"valid":not errors,"errors":errors},indent=2)); return 1 if errors else 0
        elif args.command=="record-session":
            from ..data.record_session import record_synchronized_session
            print(record_synchronized_session(args.output,args.seconds,args.camera,args.sample_rate))
        elif args.command=="record-gesture":
            from ..data.gesture_collection import record_gesture_landmarks
            print(record_gesture_landmarks(args.output,args.participant_code,args.seconds,args.camera,args.consent_confirmed,args.pose_model))
        elif args.command=="extract-gesture":
            from ..data.gesture_collection import extract_reviewed_gesture_sequences
            print(json.dumps(extract_reviewed_gesture_sequences(args.session,args.manifest,args.min_frames,args.min_visibility,args.sequence_length),indent=2))
        elif args.command=="delete-gesture-session":
            from ..data.gesture_collection import delete_gesture_session
            print(json.dumps(delete_gesture_session(args.session,args.confirm_session_id,args.manifest),indent=2))
        elif args.command=="speech-test":
            from ..audio_output.speech import LocalSpeech
            speaker=LocalSpeech(); queued=speaker.say(args.text); speaker.close(); diagnostics=speaker.diagnostics()
            print(json.dumps({"queued":queued,**diagnostics,"text_fallback":None if diagnostics["spoken_count"] else args.text},indent=2))
            if not diagnostics["spoken_count"]: print(f"TEXT FALLBACK: {args.text}")
        elif args.command=="export-features":
            from ..data.feature_export import export_session_features
            output=args.output or (args.session/"features.phase3.jsonl")
            report=export_session_features(args.session,output,vision_weights=args.vision_weights,
                audio_weights=args.audio_weights,gesture_weights=args.gesture_weights,pose_model_path=args.pose_model,sample_every=args.sample_every,
                max_frames=args.max_frames,audio_window_seconds=args.audio_window_seconds,
                audio_hop_seconds=args.audio_hop_seconds,device=args.device,overwrite=args.overwrite)
            print(json.dumps(report,indent=2))
        elif args.command=="validate-features":
            from ..data.features import validate_feature_jsonl
            report,errors=validate_feature_jsonl(args.path); print(json.dumps({**report,"errors":errors},indent=2)); return 1 if errors else 0
        elif args.command=="align-features":
            from ..data.alignment import build_aligned_dataset
            print(json.dumps(build_aligned_dataset(args.session,args.features,args.output,
                window_seconds=args.window_seconds,stride_seconds=args.stride_seconds,
                max_feature_age_seconds=args.max_feature_age_seconds,overwrite=args.overwrite),indent=2))
        elif args.command=="demo-session":
            from ..data.feature_export import export_session_features
            from ..runtime.prerecorded_demo import run_prerecorded_demo
            features=args.features or (args.session/"features.phase3.jsonl")
            report_path=args.report or (args.session/"phase3-demo.json")
            if not args.reuse_features:
                export_session_features(args.session,features,vision_weights=args.vision_weights,
                    audio_weights=args.audio_weights,gesture_weights=args.gesture_weights,pose_model_path=args.pose_model,sample_every=args.sample_every,
                    max_frames=args.max_frames,audio_window_seconds=args.audio_window_seconds,
                    audio_hop_seconds=args.audio_hop_seconds,device=args.device,overwrite=args.overwrite)
            report=run_prerecorded_demo(args.session,features,report_path,fusion_weights=args.fusion_weights,
                device=args.device,sequence_length=args.sequence_length,lookback_seconds=args.lookback_seconds,
                overwrite=args.overwrite)
            print(json.dumps(report,indent=2)); return 0 if report["full_trained_multimodal_demo"] else 3
        elif args.command=="import-session":
            from ..data.import_session import import_prerecorded_session
            session=import_prerecorded_session(args.video,args.audio,args.output,args.recording_id,
                video_offset_seconds=args.video_offset_seconds,audio_offset_seconds=args.audio_offset_seconds,
                annotations=args.annotations,participant_code=args.participant_code)
            print(json.dumps({"session":str(session),"recording_id":args.recording_id,
                "annotations_imported":bool(args.annotations),"status":"source media copied; human review required"},indent=2))
        elif args.command=="import-gesture-video":
            from ..data.gesture_collection import import_gesture_video
            session=import_gesture_video(args.video,args.output,args.participant_code,consent_confirmed=args.consent_confirmed,pose_model_path=args.pose_model)
            print(json.dumps({"session":str(session),"status":"pose landmarks extracted; review annotation_template.csv",
                "raw_video_saved":True,"participant_code":args.participant_code},indent=2))
        elif args.command=="dashboard":
            from .dashboard.__main__ import main as run_dashboard
            sys.argv = [sys.argv[0], "--host", args.host, "--port", str(args.port)] + (["--no-open"] if args.no_open else [])
            run_dashboard()
            return 0
        return 0
    except Exception as exc:
        LOG.error("%s: %s",type(exc).__name__,exc); return 2


if __name__=="__main__": raise SystemExit(main())
