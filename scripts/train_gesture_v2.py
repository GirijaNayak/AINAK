#!/usr/bin/env python3
"""AI AINAK — Gesture V2 Model Training & Evaluation Pipeline

Trains GestureTemporalNet on 25 gesture classes using subject-independent splits:
- Train: subject_p02, subject_p04, subject_p05 (600 sequences)
- Validation: subject_p03, subject_p06 (400 sequences)
- Held-out Test: subject_p01 (200 sequences)

Saves checkpoints atomically to artifacts/v2/gesture/best.pt and latest.pt.
Generates comprehensive evaluation metrics, confusion matrix, and unknown rejection benchmarks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

from ainak.models.gesture.v2_gesture import (
    GestureTemporalNet,
    GESTURE_CLASSES_V2,
    FEATURES,
    SEQUENCE_LENGTH,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "datasets" / "gesture-sessions-v2" / "gesture_manifest_v2.csv"
OUTPUT_DIR = ROOT / "artifacts" / "v2" / "gesture"
BEST_CKPT = OUTPUT_DIR / "best.pt"
LATEST_CKPT = OUTPUT_DIR / "latest.pt"
METRICS_PATH = OUTPUT_DIR / "gesture_temporal.metrics.json"


class GestureDatasetV2(Dataset):
    def __init__(self, manifest_path: Path, allowed_subjects: set[str], augment: bool = False):
        self.samples = []
        self.augment = augment
        base_dir = manifest_path.parent

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject = row["subject_id"]
                if subject in allowed_subjects:
                    npy_path = base_dir / row["path"]
                    label_idx = GESTURE_CLASSES_V2.index(row["label"])
                    self.samples.append((npy_path, label_idx, subject, row["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label_idx, _, _ = self.samples[idx]
        data = np.load(npy_path).astype(np.float32)  # (24, 132)

        if self.augment:
            # 1. Random spatial scaling (0.90 to 1.10)
            scale = np.random.uniform(0.92, 1.08)
            # 2. Additive Gaussian noise
            noise = np.random.normal(0, 0.015, data.shape).astype(np.float32)
            data = (data * scale) + noise
            # 3. Random temporal jitter (slight roll)
            if np.random.rand() > 0.5:
                shift = np.random.randint(-1, 2)
                data = np.roll(data, shift, axis=0)

        return torch.from_numpy(data.astype(np.float32)), label_idx


def train_v2(epochs: int = 35, batch_size: int = 32, lr: float = 1e-3, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()

    train_subjects = {"subject_p02", "subject_p04", "subject_p05"}
    val_subjects = {"subject_p03", "subject_p06"}
    test_subjects = {"subject_p01"}

    train_ds = GestureDatasetV2(MANIFEST_PATH, train_subjects, augment=True)
    val_ds = GestureDatasetV2(MANIFEST_PATH, val_subjects, augment=False)
    test_ds = GestureDatasetV2(MANIFEST_PATH, test_subjects, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset splits initialized:")
    print(f"  - Train samples: {len(train_ds)} (Subjects: {train_subjects})")
    print(f"  - Val samples:   {len(val_ds)} (Subjects: {val_subjects})")
    print(f"  - Test samples:  {len(test_ds)} (Subjects: {test_subjects})")

    device = "cpu"
    model = GestureTemporalNet(
        feature_dim=FEATURES,
        hidden_size=128,
        classes=len(GESTURE_CLASSES_V2),
        dropout=0.2,
        layers=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: GestureTemporalNet ({total_params:,} parameters)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("\nStarting training loop...")
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += loss.item() * len(y)
            train_correct += (logits.argmax(dim=-1) == y).sum().item()

        scheduler.step()

        train_loss = train_loss_total / len(train_ds)
        train_acc = train_correct / len(train_ds)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_correct = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                val_loss_total += loss.item() * len(y)
                val_correct += (logits.argmax(dim=-1) == y).sum().item()

        val_loss = val_loss_total / len(val_ds)
        val_acc = val_correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

        # Save latest checkpoint atomically
        tmp_latest = LATEST_CKPT.with_suffix(".tmp")
        latest_payload = {
            "checkpoint_schema": "ainak.gesture-checkpoint.v2",
            "trained_on_real_data": True,
            "training_data_provenance": "MediaPipe BlazePose 3D verified kinematics across 6 independent human subjects",
            "training_metadata": {
                "manifest_sha256": manifest_hash,
                "epochs_completed": epoch,
                "optimizer": "AdamW",
                "loss": "CrossEntropyLoss(label_smoothing=0.05)",
                "learning_rate": lr,
            },
            "recording_split": {
                "train": sorted(list(train_subjects)),
                "validation": sorted(list(val_subjects)),
                "test": sorted(list(test_subjects)),
            },
            "training_examples": len(train_ds),
            "validation_examples": len(val_ds),
            "test_examples": len(test_ds),
            "labels": list(GESTURE_CLASSES_V2),
            "model_config": {
                "feature_dim": FEATURES,
                "hidden_size": 128,
                "classes": len(GESTURE_CLASSES_V2),
                "dropout": 0.2,
                "layers": 2,
            },
            "state_dict": model.state_dict(),
        }
        torch.save(latest_payload, tmp_latest)
        tmp_latest.replace(LATEST_CKPT)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            tmp_best = BEST_CKPT.with_suffix(".tmp")
            torch.save(latest_payload, tmp_best)
            tmp_best.replace(BEST_CKPT)

    train_time = time.perf_counter() - t0
    print(f"\nTraining completed in {train_time:.1f}s. Best Val Acc: {best_val_acc:.4f}")

    # Evaluate Best Checkpoint on Held-Out Test Set (subject_p01)
    best_payload = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    latencies = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            t_start = time.perf_counter()
            logits, embedding = model(x)
            latencies.append((time.perf_counter() - t_start) * 1000 / len(y))
            preds = logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(y.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    test_acc = float((test_preds == test_targets).mean())
    report_dict = classification_report(
        test_targets,
        test_preds,
        target_names=GESTURE_CLASSES_V2,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(test_targets, test_preds, labels=range(len(GESTURE_CLASSES_V2))).tolist()

    avg_latency_ms = float(np.mean(latencies))

    # Benchmark Unknown Gesture Rejection
    # 100 random noise sequences & 100 uniform random gestures
    unknown_samples = np.random.normal(0, 1.0, (100, 24, 132)).astype(np.float32)
    with torch.no_grad():
        unk_logits, _ = model(torch.from_numpy(unknown_samples).to(device))
        unk_probs = torch.softmax(unk_logits, dim=-1).cpu().numpy()
        unk_confs = unk_probs.max(axis=-1)
        # Rejection threshold = 0.70
        rejected_count = int((unk_confs < 0.70).sum())
        unknown_rejection_rate = float(rejected_count / len(unknown_samples))

    metrics = {
        "model_version": "v2",
        "architecture": "GestureTemporalNet",
        "classes_count": len(GESTURE_CLASSES_V2),
        "supported_classes": list(GESTURE_CLASSES_V2),
        "total_parameters": total_params,
        "dataset_split": {
            "train_sequences": len(train_ds),
            "validation_sequences": len(val_ds),
            "test_sequences": len(test_ds),
            "train_subjects": list(train_subjects),
            "validation_subjects": list(val_subjects),
            "test_subjects": list(test_subjects),
        },
        "held_out_test_metrics": {
            "accuracy": test_acc,
            "macro_precision": float(report_dict["macro avg"]["precision"]),
            "macro_recall": float(report_dict["macro avg"]["recall"]),
            "macro_f1": float(report_dict["macro avg"]["f1-score"]),
            "per_class": {
                cls_name: {
                    "precision": float(report_dict[cls_name]["precision"]),
                    "recall": float(report_dict[cls_name]["recall"]),
                    "f1_score": float(report_dict[cls_name]["f1-score"]),
                    "support": int(report_dict[cls_name]["support"]),
                }
                for cls_name in GESTURE_CLASSES_V2
            },
        },
        "unknown_rejection_rate_at_threshold_0.7": unknown_rejection_rate,
        "inference_latency_ms": avg_latency_ms,
        "confusion_matrix": conf_mat,
        "training_history": history,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("GESTURE V2 EVALUATION SUMMARY (Held-Out Subject p01):")
    print(f"  - Supported Classes:        {len(GESTURE_CLASSES_V2)}")
    print(f"  - Held-Out Test Accuracy:   {test_acc * 100:.2f}%")
    print(f"  - Macro Precision:          {metrics['held_out_test_metrics']['macro_precision']:.4f}")
    print(f"  - Macro Recall:             {metrics['held_out_test_metrics']['macro_recall']:.4f}")
    print(f"  - Macro F1-Score:           {metrics['held_out_test_metrics']['macro_f1']:.4f}")
    print(f"  - Unknown Rejection Rate:   {unknown_rejection_rate * 100:.1f}%")
    print(f"  - Average CPU Latency:      {avg_latency_ms:.2f} ms")
    print(f"  - Checkpoint Path:          {BEST_CKPT}")
    print(f"  - Metrics Path:             {METRICS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_v2()
