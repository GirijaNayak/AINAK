from __future__ import annotations
import time
import os
from pathlib import Path
import numpy as np
from ...runtime.events import ModelEvent


def prepare_audio_window(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate mono CNN14 input; sample-rate conversion is explicit at the caller."""
    if sample_rate != 32000: raise ValueError("PANNs CNN14 expects mono 32 kHz waveform; resample explicitly before calling")
    x=np.asarray(waveform,dtype=np.float32).reshape(-1)
    if x.size < sample_rate//4: raise ValueError("audio window is too short (minimum 250 ms)")
    if not np.isfinite(x).all(): raise ValueError("audio contains NaN or infinite samples")
    x=np.clip(x,-1.,1.)
    if float(np.sqrt(np.mean(x*x))) < 1e-5: raise ValueError("audio window is silence; no prediction emitted")
    return x


class PannsCNN14:
    """Adapter for the AudioSet CNN14 (32 kHz) PANNs release and canonical preprocessing."""
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.path = Path(checkpoint)
        if not self.path.is_file(): raise FileNotFoundError(f"PANNs CNN14 checkpoint missing: {self.path}. Run `python -m ainak.interfaces.cli acquire audio`.")
        try:
            # panns-inference 0.1.1 shells out to Unix wget for labels. Point its standard cache at
            # our already acquired local official AudioSet CSV and suppress that broken Windows fallback.
            panns_home=self.path.parent/"panns-home"
            labels_file=panns_home/"panns_data"/"class_labels_indices.csv"
            if not labels_file.is_file():raise FileNotFoundError(f"AudioSet class labels missing: {labels_file}. Re-run `python -m ainak.interfaces.cli acquire audio --allow-large`.")
            numba_cache=self.path.parent/"numba-cache";numba_cache.mkdir(parents=True,exist_ok=True)
            os.environ.setdefault("NUMBA_CACHE_DIR",str(numba_cache.resolve()))
            old_profile=os.environ.get("USERPROFILE");os.environ["USERPROFILE"]=str(panns_home)
            try:
                from panns_inference import AudioTagging, labels as panns_labels
            finally:
                if old_profile is None:os.environ.pop("USERPROFILE",None)
                else:os.environ["USERPROFILE"]=old_profile
        except ImportError as exc: raise RuntimeError("Install panns-inference to use CNN14") from exc
        try:
            self.model = AudioTagging(checkpoint_path=str(self.path), device=device)
        except Exception as exc:
            raise RuntimeError(f"CNN14 checkpoint failed panns_inference compatibility validation: {exc}") from exc
        self.labels = list(panns_labels)

    def infer(self, waveform: np.ndarray, sample_rate: int = 32000, timestamp: float | None = None) -> tuple[list[tuple[str, float]], ModelEvent]:
        now = time.time() if timestamp is None else timestamp
        x=prepare_audio_window(waveform,sample_rate)
        start = time.perf_counter()
        clipwise, embedding = self.model.inference(x[None, :])
        probs = np.asarray(clipwise).reshape(-1); emb = np.asarray(embedding).reshape(-1).astype(np.float32)
        if len(probs) != len(self.labels): raise RuntimeError(f"class-map mismatch: model returned {len(probs)} scores, labels contain {len(self.labels)}")
        ranked = sorted(((str(self.labels[i]), float(probs[i])) for i in range(len(probs))), key=lambda pair: pair[1], reverse=True)
        latency = (time.perf_counter()-start)*1000
        return ranked, ModelEvent("audio", now, emb, ranked[0][1] if ranked else 0., [{"class": n, "confidence": s} for n,s in ranked[:10]], True, latency)
