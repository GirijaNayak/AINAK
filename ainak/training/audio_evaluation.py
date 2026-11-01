from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from ..models.audio.classifier import PannsCNN14


# Each ESC-50 class is mapped to one or more semantically corresponding
# AudioSet labels. Scores for aliases are reduced with max before choosing an
# ESC-50 prediction. This is an explicit zero-shot ontology mapping, not a
# claim that the taxonomies are identical.
ESC50_TO_AUDIOSET: dict[str, tuple[str, ...]] = {
    "dog": ("Dog",), "rooster": ("Crowing, cock-a-doodle-doo",),
    "pig": ("Pig",), "cow": ("Cattle, bovinae",), "frog": ("Frog",), "cat": ("Cat",),
    "hen": ("Chicken, rooster",), "insects": ("Insect",), "sheep": ("Sheep",), "crow": ("Crow",),
    "rain": ("Rain", "Rain on surface"), "sea_waves": ("Waves, surf",),
    "crackling_fire": ("Fire", "Crackle"), "crickets": ("Cricket",),
    "chirping_birds": ("Bird vocalization, bird call, bird song", "Chirp, tweet"),
    "water_drops": ("Raindrop",), "wind": ("Wind",),
    "pouring_water": ("Pour",), "toilet_flush": ("Toilet flush",),
    "thunderstorm": ("Thunderstorm", "Thunder"), "crying_baby": ("Baby cry, infant cry",),
    "sneezing": ("Sneeze",), "clapping": ("Clapping",), "breathing": ("Breathing",),
    "coughing": ("Cough",), "footsteps": ("Walk, footsteps",),
    "laughing": ("Laughter", "Belly laugh"), "brushing_teeth": ("Toothbrush", "Electric toothbrush"),
    # AudioSet has no dedicated drinking/sipping, can-opening, or washing-
    # machine classes; those entries use explicit nearest-label proxies.
    "snoring": ("Snoring",), "drinking_sipping": ("Liquid",),
    "door_wood_knock": ("Knock",), "mouse_click": ("Clicking",),
    "keyboard_typing": ("Typing",), "door_wood_creaks": ("Creak",),
    "can_opening": ("Crack",), "washing_machine": ("Mechanisms",),
    "vacuum_cleaner": ("Vacuum cleaner",), "clock_alarm": ("Alarm clock",),
    "clock_tick": ("Tick", "Tick-tock"), "glass_breaking": ("Glass",),
    "helicopter": ("Helicopter",), "chainsaw": ("Chainsaw",), "siren": ("Siren",),
    "car_horn": ("Vehicle horn, car horn, honking",), "engine": ("Engine", "Medium engine (mid frequency)"),
    "train": ("Train",), "church_bells": ("Church bell",),
    "airplane": ("Fixed-wing aircraft, airplane",), "fireworks": ("Fireworks",),
    "hand_saw": ("Sawing",),
}


def _load_metadata(dataset: Path) -> tuple[Path, list[dict[str, str]]]:
    metadata = dataset / "meta" / "esc50.csv"
    if not metadata.is_file():
        # Official GitHub ZIP extracts to ESC-50-master/.
        candidates = list(dataset.glob("ESC-50-*/meta/esc50.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"ESC-50 metadata not found at {metadata} or in one extracted ESC-50-* folder")
        metadata = candidates[0]
        audio_dir = metadata.parent.parent / "audio"
    else:
        audio_dir = dataset / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"ESC-50 audio directory missing: {audio_dir}")
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "fold", "target", "category"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid ESC-50 metadata; required columns: {sorted(required)}")
    if len(rows) != 2000 or len({row["filename"] for row in rows}) != len(rows):
        raise ValueError(f"expected 2000 unique ESC-50 examples, got {len(rows)}")
    for row in rows:
        if not (audio_dir / row["filename"]).is_file():
            raise FileNotFoundError(f"ESC-50 WAV missing: {audio_dir / row['filename']}")
    return audio_dir, rows


def validate_esc50_mapping(categories: set[str], audioset_labels: list[str]) -> dict[str, list[int]]:
    missing_classes = sorted(categories - set(ESC50_TO_AUDIOSET))
    extra_classes = sorted(set(ESC50_TO_AUDIOSET) - categories)
    if missing_classes or extra_classes:
        raise ValueError(f"ESC-50 mapping category mismatch; missing={missing_classes}, extra={extra_classes}")
    index_by_name = {name: i for i, name in enumerate(audioset_labels)}
    mapping: dict[str, list[int]] = {}
    absent = {}
    for category in sorted(categories):
        indexes = [index_by_name[label] for label in ESC50_TO_AUDIOSET[category] if label in index_by_name]
        if not indexes:
            absent[category] = list(ESC50_TO_AUDIOSET[category])
        mapping[category] = indexes
    if absent:
        raise ValueError(f"mapped AudioSet labels absent from checkpoint label map: {absent}")
    label_owners: dict[int, list[str]] = defaultdict(list)
    for category, indexes in mapping.items():
        for index in indexes:
            label_owners[index].append(category)
    collisions = {audioset_labels[index]: owners for index, owners in label_owners.items() if len(owners) > 1}
    if collisions:
        raise ValueError(f"ambiguous AudioSet labels map to multiple ESC-50 classes: {collisions}")
    return mapping


def evaluate_esc50(dataset: Path, checkpoint: Path, output: Path, device: str = "cpu", batch_size: int = 8) -> dict:
    """Run zero-shot CNN14 evaluation against explicit ESC-50/AudioSet labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    audio_dir, rows = _load_metadata(dataset)
    model = PannsCNN14(checkpoint, device)
    categories = sorted({row["category"] for row in rows})
    mapped_indexes = validate_esc50_mapping(set(categories), model.labels)
    category_index = {category: i for i, category in enumerate(categories)}

    # Keep only one mini-batch of waveforms in RAM. Source files are 5-second
    # ESC-50 mono clips and are resampled to CNN14's expected 32 kHz here.
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict] = []
    latency_ms: list[float] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        current = rows[offset : offset + batch_size]
        waveforms = [librosa.load(audio_dir / row["filename"], sr=32000, mono=True)[0].astype(np.float32) for row in current]
        # AudioTagging supports a batch with a shared padded length. ESC-50
        # files all share duration, but handle any variation deterministically.
        max_length = max(map(len, waveforms))
        batch = np.zeros((len(waveforms), max_length), dtype=np.float32)
        for i, waveform in enumerate(waveforms):
            batch[i, : len(waveform)] = waveform
        infer_start = time.perf_counter()
        clipwise, _ = model.model.inference(batch)
        latency_ms.append((time.perf_counter() - infer_start) * 1000 / len(current))
        scores = np.asarray(clipwise, dtype=np.float32)
        if scores.shape != (len(current), len(model.labels)):
            raise RuntimeError(f"PANNs returned unexpected score shape {scores.shape}")
        for row, row_scores in zip(current, scores):
            class_scores = np.array([max(row_scores[indexes]) for category, indexes in ((c, mapped_indexes[c]) for c in categories)])
            pred_category = categories[int(class_scores.argmax())]
            truth = row["category"]
            y_true.append(category_index[truth]); y_pred.append(category_index[pred_category])
            predictions.append({"filename": row["filename"], "fold": int(row["fold"]), "reference": truth,
                                "prediction": pred_category, "mapped_score": float(class_scores.max())})

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(categories))), zero_division=0)
    report = classification_report(y_true, y_pred, labels=list(range(len(categories))), target_names=categories, output_dict=True, zero_division=0)
    dataset_hash = hashlib.sha256((dataset / "meta" / "esc50.csv").read_bytes()).hexdigest() if (dataset / "meta" / "esc50.csv").is_file() else hashlib.sha256((audio_dir.parent / "meta" / "esc50.csv").read_bytes()).hexdigest()
    archive_hash_path = dataset / "archive.sha256"
    archive_sha256 = archive_hash_path.read_text(encoding="utf-8").strip() if archive_hash_path.is_file() else None
    checkpoint_hash = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_hash.update(chunk)
    result = {
        "evaluation_type": "zero_shot_pretrained_checkpoint",
        "training_or_finetuning": False,
        "dataset": "ESC-50",
        "dataset_version": "official karolpiczak/ESC-50 GitHub master archive snapshot; see archive_sha256 and provenance",
        "dataset_metadata_sha256": dataset_hash,
        "archive_sha256": archive_sha256,
        "dataset_source": "https://github.com/karolpiczak/ESC-50 (dataset README links to the downloadable archive)",
        "dataset_license": "CC BY-NC 3.0; evaluation copy only, review terms before redistribution or commercial use",
        "sample_count": len(rows), "categories": categories, "folds": sorted({int(row["fold"]) for row in rows}),
        "sample_rate_hz": 32000, "waveform_preprocessing": "librosa mono decode/resample to 32 kHz; PANNs package preprocessing",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "class_mapping": {category: list(ESC50_TO_AUDIOSET[category]) for category in categories},
        "proxy_mappings": {"drinking_sipping": "Water/Liquid proxy; no dedicated AudioSet class",
                           "can_opening": "Crack/Clicking proxies; no dedicated AudioSet class",
                           "washing_machine": "Mechanisms proxy; no dedicated AudioSet class"},
        "mapping_method": "For each ESC class, use the maximum CNN14 AudioSet clipwise score over its explicitly listed aliases; predict the ESC class with the largest mapped score.",
        "accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)), "macro_f1": float(np.mean(f1)),
        "per_class": {category: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, category in enumerate(categories)},
        "classification_report": report, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(categories)))).tolist(),
        "predictions": predictions, "batch_size": batch_size, "device": device,
        "runtime_seconds": time.perf_counter() - started, "mean_model_latency_ms_per_clip": float(np.mean(latency_ms)),
        "python": platform.python_version(), "torch": torch.__version__,
        "caveat": "This zero-shot ontology-mapped benchmark measures transfer on ESC-50 only. It does not validate all safety-critical sounds or local deployment conditions.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
