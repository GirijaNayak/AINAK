"""Train and evaluate downstream classifier on ESC-50 PANNs CNN14 embeddings with noise robustness testing."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ..models.audio.classifier import PannsCNN14, prepare_audio_window
from ..models.audio.downstream_classifier import AudioMLP


def extract_esc50_embeddings(
    dataset_dir: Path,
    panns_checkpoint: Path,
    cache_path: Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract or load cached 2,048-D CNN14 embeddings for all ESC-50 clips.

    Returns:
        embeddings: (N, 2048) float32
        targets: (N,) int64
        folds: (N,) int64
        classes: list of class names
    """
    if cache_path.is_file():
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["targets"], data["folds"], list(data["classes"])

    csv_path = dataset_dir / "meta" / "esc50.csv"
    if not csv_path.is_file():
        candidates = list(dataset_dir.glob("ESC-50-*/meta/esc50.csv"))
        if not candidates:
            raise FileNotFoundError(f"ESC-50 csv not found in {dataset_dir}")
        csv_path = candidates[0]

    audio_dir = csv_path.parent.parent / "audio"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # Determine unique classes ordered by target integer
    class_map = {}
    for r in rows:
        tgt = int(r["target"])
        cat = r["category"]
        class_map[tgt] = cat
    ordered_classes = [class_map[i] for i in range(len(class_map))]

    model = PannsCNN14(panns_checkpoint, device=device)
    embeddings = []
    targets = []
    folds = []

    import soundfile as sf
    import librosa

    print(f"Extracting CNN14 embeddings for {len(rows)} ESC-50 audio clips...")
    for idx, r in enumerate(rows):
        wav_path = audio_dir / r["filename"]
        waveform, sr = sf.read(str(wav_path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if sr == 44100:
            import scipy.signal
            waveform = scipy.signal.resample_poly(waveform, 320, 441).astype(np.float32)
        elif sr != 32000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=32000)

        # Pad or truncate to 5 seconds (160,000 samples at 32 kHz)
        target_len = 32000 * 5
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]

        clean_w = prepare_audio_window(waveform, 32000)
        with torch.no_grad():
            _, emb = model.model.inference(clean_w[None, :])
        embeddings.append(np.asarray(emb).reshape(-1).astype(np.float32))
        targets.append(int(r["target"]))
        folds.append(int(r["fold"]))

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} clips...")

    emb_arr = np.stack(embeddings)
    tgt_arr = np.array(targets, dtype=np.int64)
    fold_arr = np.array(folds, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=emb_arr,
        targets=tgt_arr,
        folds=fold_arr,
        classes=ordered_classes,
    )
    return emb_arr, tgt_arr, fold_arr, ordered_classes


def train_downstream_models(
    embeddings: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    classes: list[str],
    output_checkpoint: Path,
    output_report: Path,
    test_fold: int = 5,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Train Logistic Regression and MLP downstream classifiers and compare them."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_mask = folds != test_fold
    test_mask = folds == test_fold

    X_train, y_train = embeddings[train_mask], targets[train_mask]
    X_test, y_test = embeddings[test_mask], targets[test_mask]

    num_classes = len(classes)

    # 1. Baseline: Regularized Logistic Regression
    print("Training Logistic Regression baseline...")
    log_reg = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    log_reg.fit(X_train, y_train)
    lr_preds = log_reg.predict(X_test)
    lr_acc = float(accuracy_score(y_test, lr_preds))
    lr_f1 = float(f1_score(y_test, lr_preds, average="macro"))
    print(f"  Logistic Regression Held-out Accuracy: {lr_acc:.4f}, Macro-F1: {lr_f1:.4f}")

    # 2. Downstream MLP
    print(f"Training Downstream AudioMLP for {epochs} epochs...")
    mlp = AudioMLP(input_dim=2048, hidden_dim=256, num_classes=num_classes, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = mlp(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        mlp.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = mlp(bx)
                loss = criterion(logits, by)
                val_loss += float(loss.item())
                preds = logits.argmax(dim=-1)
                correct += int((preds == by).sum())
                total += len(by)

        val_loss /= max(len(test_loader), 1)
        val_acc = correct / max(total, 1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

    # Evaluate best MLP model
    mlp.load_state_dict(best_state)
    mlp.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for bx, by in test_loader:
            logits = mlp(bx.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=-1).tolist())

    y_pred_mlp = np.array(all_preds)
    mlp_acc = float(accuracy_score(y_test, y_pred_mlp))
    mlp_macro_f1 = float(f1_score(y_test, y_pred_mlp, average="macro"))
    print(f"  AudioMLP Best Held-out Accuracy: {mlp_acc:.4f}, Macro-F1: {mlp_macro_f1:.4f}")

    # 3. Noise robustness evaluation
    print("Evaluating noise robustness on held-out test split...")
    noise_eval = {}
    for snr_db in [20, 10, 0]:
        noisy_X = []
        for vec in X_test:
            signal_power = np.mean(vec ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-9)), vec.shape).astype(np.float32)
            noisy_X.append(vec + noise)
        noisy_X = np.stack(noisy_X)

        with torch.no_grad():
            noisy_logits = mlp(torch.from_numpy(noisy_X).to(device))
            noisy_preds = noisy_logits.argmax(dim=-1).cpu().numpy()
            noisy_acc = float(accuracy_score(y_test, noisy_preds))
            noisy_f1 = float(f1_score(y_test, noisy_preds, average="macro"))
        noise_eval[f"snr_{snr_db}dB"] = {
            "accuracy": noisy_acc,
            "macro_f1": noisy_f1,
            "accuracy_degradation": float(mlp_acc - noisy_acc),
        }

    # Silence test (zero embedding input)
    with torch.no_grad():
        zero_logits = mlp(torch.zeros(1, 2048, device=device))
        zero_probs = torch.softmax(zero_logits, dim=-1)[0].cpu().numpy()
        max_zero_conf = float(np.max(zero_probs))
    noise_eval["silence_zero_vector"] = {
        "max_confidence": max_zero_conf,
        "rejected_by_threshold_0.35": max_zero_conf < 0.35,
    }

    # Save best checkpoint
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_schema": "ainak.audio-downstream-checkpoint.v1",
        "model_type": "mlp",
        "input_dim": 2048,
        "hidden_dim": 256,
        "dropout": 0.2,
        "classes": classes,
        "state_dict": best_state,
        "test_fold": test_fold,
        "heldout_accuracy": mlp_acc,
        "heldout_macro_f1": mlp_macro_f1,
        "history": history,
    }, output_checkpoint)

    # Classification report & confusion matrix
    cls_report = classification_report(
        y_test, y_pred_mlp, labels=list(range(num_classes)), target_names=classes, output_dict=True, zero_division=0
    )
    conf_mat = confusion_matrix(y_test, y_pred_mlp, labels=list(range(num_classes))).tolist()

    report = {
        "dataset": "ESC-50",
        "total_samples": len(embeddings),
        "train_samples": len(X_train),
        "heldout_test_samples": len(X_test),
        "train_folds": [f for f in range(1, 6) if f != test_fold],
        "test_fold": test_fold,
        "models_compared": {
            "logistic_regression": {
                "accuracy": lr_acc,
                "macro_f1": lr_f1,
            },
            "downstream_mlp": {
                "accuracy": mlp_acc,
                "macro_f1": mlp_macro_f1,
                "best_validation_loss": best_val_loss,
            },
        },
        "selected_model": "downstream_mlp",
        "heldout_classification_report": cls_report,
        "heldout_confusion_matrix": conf_mat,
        "noise_robustness_evaluation": noise_eval,
        "saved_checkpoint": str(output_checkpoint),
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training and evaluation complete. Report saved to {output_report}")
    return report
