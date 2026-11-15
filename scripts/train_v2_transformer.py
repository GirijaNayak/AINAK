"""AI AINAK Version 2 — Multimodal Fusion Transformer Training & Evaluation Pipeline

Trains the new V2MultimodalTransformer on genuine aligned V2 features:
- Vision: YOLO-World v2 Small (256-D)
- Audio: PANNs CNN14 (2,048-D)
- Gesture: GestureTemporalNet (128-D)

Saves:
- artifacts/v2/fusion/best.pt
- artifacts/v2/fusion/latest.pt
- artifacts/v2/fusion/training_history.csv
- artifacts/v2/fusion/training_config.json
- artifacts/v2/fusion/checkpoint_manifest.json
"""
import os
import sys
import json
import time
import random
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

from ainak.models.fusion.v2_transformer import (
    V2MultimodalTransformer,
    MODALITIES_V2,
    DEFAULT_FEATURE_DIMS_V2,
    HAZARD_SCENARIOS,
    ALERT_TIERS,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V2FusionDataset(Dataset):
    """Dataset for temporal sequences of aligned V2 multimodal features."""

    def __init__(self, manifest_items: list[dict], feat_dir: str, seq_len: int = 4):
        self.items = manifest_items
        self.feat_dir = feat_dir
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        feat_path = os.path.join(self.feat_dir, item["feature_file"])
        data = np.load(feat_path)

        v_feat = torch.from_numpy(data["vision"].astype(np.float32))   # [256]
        a_feat = torch.from_numpy(data["audio"].astype(np.float32))    # [2048]
        g_feat = torch.from_numpy(data["gesture"].astype(np.float32))  # [128]
        mask_vec = torch.from_numpy(data["mask"].astype(bool))         # [3]
        base_t = float(data["timestamp"])

        # Repeat across temporal sequence length T=seq_len
        v_seq = v_feat.unsqueeze(0).repeat(self.seq_len, 1)            # [T, 256]
        a_seq = a_feat.unsqueeze(0).repeat(self.seq_len, 1)            # [T, 2048]
        g_seq = g_feat.unsqueeze(0).repeat(self.seq_len, 1)            # [T, 128]
        mask_seq = mask_vec.unsqueeze(0).repeat(self.seq_len, 1)       # [T, 3]
        time_seq = torch.linspace(base_t, base_t + 0.5 * (self.seq_len - 1), self.seq_len)

        features = {
            "vision": v_seq,
            "audio": a_seq,
            "gesture": g_seq,
        }

        scenario_idx = torch.tensor(item["scenario_idx"], dtype=torch.long)
        alert_idx = torch.tensor(item["alert_idx"], dtype=torch.long)
        priority = torch.tensor(item["priority"], dtype=torch.float32)

        return features, mask_seq, time_seq, scenario_idx, alert_idx, priority


def collate_fn(batch):
    fs, ms, ts, scens, alerts, pris = zip(*batch)
    batch_features = {
        m: torch.stack([f[m] for f in fs]) for m in MODALITIES_V2
    }
    batch_mask = torch.stack(ms)
    batch_times = torch.stack(ts)
    batch_scens = torch.stack(scens)
    batch_alerts = torch.stack(alerts)
    batch_pris = torch.stack(pris)
    return batch_features, batch_mask, batch_times, batch_scens, batch_alerts, batch_pris


def evaluate_model(model, loader, device, forced_mask=None):
    """Evaluate model on a dataset loader with optional forced modality mask for ablations."""
    model.eval()
    all_scen_preds = []
    all_scen_targets = []
    all_alert_preds = []
    all_alert_targets = []
    all_pri_preds = []
    all_pri_targets = []
    total_loss = 0.0
    count = 0

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for features, mask, times, scens, alerts, pris in loader:
            b = mask.shape[0]
            features = {k: v.to(device) for k, v in features.items()}
            times = times.to(device)
            scens = scens.to(device)
            alerts = alerts.to(device)
            pris = pris.to(device)

            if forced_mask is not None:
                # Override mask for ablation testing: [B, T, 3]
                mask = torch.tensor(forced_mask, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0).repeat(b, mask.shape[1], 1)
                # Zero out unobserved features
                for m_idx, m_name in enumerate(MODALITIES_V2):
                    if not forced_mask[m_idx]:
                        features[m_name] = torch.zeros_like(features[m_name])
            else:
                mask = mask.to(device)

            out = model(features, mask, times)
            l_scen = ce_loss(out.scenario_logits, scens)
            l_alert = ce_loss(out.alert_logits, alerts)
            l_pri = mse_loss(out.priority, pris)
            loss = l_scen + l_alert + 2.0 * l_pri

            total_loss += loss.item() * b
            count += b

            scen_pred = out.scenario_logits.argmax(dim=-1).cpu().numpy()
            alert_pred = out.alert_logits.argmax(dim=-1).cpu().numpy()
            pri_pred = out.priority.cpu().numpy()

            all_scen_preds.extend(scen_pred.tolist())
            all_scen_targets.extend(scens.cpu().numpy().tolist())
            all_alert_preds.extend(alert_pred.tolist())
            all_alert_targets.extend(alerts.cpu().numpy().tolist())
            all_pri_preds.extend(pri_pred.tolist())
            all_pri_targets.extend(pris.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, count)
    scen_acc = accuracy_score(all_scen_targets, all_scen_preds)
    scen_f1 = f1_score(all_scen_targets, all_scen_preds, average="macro", zero_division=0)
    alert_acc = accuracy_score(all_alert_targets, all_alert_preds)
    alert_f1 = f1_score(all_alert_targets, all_alert_preds, average="macro", zero_division=0)
    pri_mae = float(np.mean(np.abs(np.array(all_pri_preds) - np.array(all_pri_targets))))

    return {
        "loss": avg_loss,
        "scenario_acc": scen_acc,
        "scenario_f1": scen_f1,
        "alert_acc": alert_acc,
        "alert_f1": alert_f1,
        "priority_mae": pri_mae,
        "scen_preds": all_scen_preds,
        "scen_targets": all_scen_targets,
        "alert_preds": all_alert_preds,
        "alert_targets": all_alert_targets,
        "pri_preds": all_pri_preds,
        "pri_targets": all_pri_targets,
    }


def main():
    print("=== STARTING AINAK V2 FUSION TRANSFORMER TRAINING ===", flush=True)
    start_time = time.time()
    set_seed(42)

    device = "cpu"
    output_dir = "artifacts/v2/fusion"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Manifest & Splits
    manifest_path = "datasets/v2_fusion/manifests/aligned_manifest_v2.jsonl"
    split_path = "datasets/v2_fusion/splits/train_val_test_split.json"
    feat_dir = "datasets/v2_fusion/features"

    with open(manifest_path, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]
    record_map = {r["example_id"]: r for r in all_records}

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_records = [record_map[eid] for eid in splits["train"] if eid in record_map]
    val_records = [record_map[eid] for eid in splits["validation"] if eid in record_map]
    test_records = [record_map[eid] for eid in splits["test"] if eid in record_map]

    print(f"Data Splits: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}", flush=True)

    train_ds = V2FusionDataset(train_records, feat_dir, seq_len=4)
    val_ds = V2FusionDataset(val_records, feat_dir, seq_len=4)
    test_ds = V2FusionDataset(test_records, feat_dir, seq_len=4)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 2. Initialize Model
    model = V2MultimodalTransformer(
        feature_dims=DEFAULT_FEATURE_DIMS_V2,
        dim=128,
        heads=4,
        layers=2,
        ff_dim=256,
        dropout=0.1,
        modality_dropout=0.1,
        max_steps=32,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized: {total_params:,} parameters", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    epochs = 25
    best_val_loss = float("inf")
    best_state = None
    history = []

    print(f"\n--- Beginning Training Run: {epochs} Epochs ---", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for features, mask, times, scens, alerts, pris in train_loader:
            b = mask.shape[0]
            features = {k: v.to(device) for k, v in features.items()}
            mask = mask.to(device)
            times = times.to(device)
            scens = scens.to(device)
            alerts = alerts.to(device)
            pris = pris.to(device)

            optimizer.zero_grad()
            out = model(features, mask, times)

            l_scen = ce_loss(out.scenario_logits, scens)
            l_alert = ce_loss(out.alert_logits, alerts)
            l_pri = mse_loss(out.priority, pris)
            loss = l_scen + l_alert + 2.0 * l_pri

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * b
            train_count += b

        scheduler.step()
        avg_train_loss = train_loss / max(1, train_count)

        # Validation
        val_res = evaluate_model(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_res["loss"], 4),
            "val_scenario_acc": round(val_res["scenario_acc"], 4),
            "val_scenario_f1": round(val_res["scenario_f1"], 4),
            "val_alert_acc": round(val_res["alert_acc"], 4),
            "val_alert_f1": round(val_res["alert_f1"], 4),
            "val_priority_mae": round(val_res["priority_mae"], 4),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_best_mark = " [*BEST]"
        else:
            save_best_mark = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} | "
            f"Val Scen F1: {val_res['scenario_f1']:.3f} | "
            f"Val Alert F1: {val_res['alert_f1']:.3f} | "
            f"Val Pri MAE: {val_res['priority_mae']:.3f}{save_best_mark}",
            flush=True,
        )

    # 3. Save Checkpoints
    print("\n--- Saving Model Checkpoints ---", flush=True)
    best_path = os.path.join(output_dir, "best.pt")
    latest_path = os.path.join(output_dir, "latest.pt")

    # Common payload metadata
    payload_metadata = {
        "checkpoint_schema": "ainak.v2_fusion_transformer.v1",
        "trained_on_real_data": True,
        "feature_dimensions": DEFAULT_FEATURE_DIMS_V2,
        "scenarios": list(HAZARD_SCENARIOS),
        "alert_tiers": list(ALERT_TIERS),
        "sequence_length": 4,
        "seed": 42,
        "epochs_completed": epochs,
        "best_val_loss": best_val_loss,
        "splits": splits["counts"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save Best
    torch.save(
        {
            "state_dict": best_state,
            "metrics": {"best_val_loss": best_val_loss},
            **payload_metadata,
        },
        best_path,
    )

    # Save Latest
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metrics": {"final_val_loss": val_res["loss"]},
            **payload_metadata,
        },
        latest_path,
    )

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved latest checkpoint: {latest_path}")

    # 4. Save History CSV
    history_csv_path = os.path.join(output_dir, "training_history.csv")
    import csv
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training history: {history_csv_path}")

    # 5. Evaluate on Held-Out Test Set (Unseen Session 04 + Subject p01)
    print("\n--- Held-Out Test Set Evaluation ---", flush=True)
    model.load_state_dict(best_state)
    test_res = evaluate_model(model, test_loader, device)

    print(f"Test Loss: {test_res['loss']:.4f}")
    print(f"Test Scenario Accuracy: {test_res['scenario_acc']*100:.2f}% (Macro-F1: {test_res['scenario_f1']:.4f})")
    print(f"Test Alert Accuracy:    {test_res['alert_acc']*100:.2f}% (Macro-F1: {test_res['alert_f1']:.4f})")
    print(f"Test Priority MAE:      {test_res['priority_mae']:.4f}")

    # Per-class scenario report
    scen_report = classification_report(
        test_res["scen_targets"],
        test_res["scen_preds"],
        target_names=[HAZARD_SCENARIOS[i] for i in sorted(list(set(test_res["scen_targets"])))],
        output_dict=True,
        zero_division=0,
    )
    alert_report = classification_report(
        test_res["alert_targets"],
        test_res["alert_preds"],
        target_names=[ALERT_TIERS[i] for i in sorted(list(set(test_res["alert_targets"])))],
        output_dict=True,
        zero_division=0,
    )

    # 6. Comprehensive Modality Ablation Study
    print("\n--- Running Modality Ablation Study ---", flush=True)
    ablations = [
        ("Vision Only", [True, False, False]),
        ("Audio Only", [False, True, False]),
        ("Gesture Only", [False, False, True]),
        ("Vision + Audio", [True, True, False]),
        ("Vision + Gesture", [True, False, True]),
        ("Audio + Gesture", [False, True, True]),
        ("Full Multimodal (V+A+G)", [True, True, True]),
    ]

    ablation_results = []
    for ab_name, ab_mask in ablations:
        ab_eval = evaluate_model(model, test_loader, device, forced_mask=ab_mask)
        res_row = {
            "ablation": ab_name,
            "mask": ab_mask,
            "scenario_f1": round(ab_eval["scenario_f1"], 4),
            "scenario_acc": round(ab_eval["scenario_acc"], 4),
            "alert_f1": round(ab_eval["alert_f1"], 4),
            "alert_acc": round(ab_eval["alert_acc"], 4),
            "priority_mae": round(ab_eval["priority_mae"], 4),
            "test_loss": round(ab_eval["loss"], 4),
        }
        ablation_results.append(res_row)
        print(
            f"  {ab_name:24s} | Scen F1: {res_row['scenario_f1']:.3f} | "
            f"Alert F1: {res_row['alert_f1']:.3f} | Pri MAE: {res_row['priority_mae']:.3f}"
        )

    # Save training configuration
    train_config = {
        "model": "V2MultimodalTransformer",
        "parameters": total_params,
        "seed": 42,
        "optimizer": "AdamW",
        "initial_lr": 1e-3,
        "min_lr": 1e-5,
        "scheduler": "CosineAnnealingLR(T_max=25)",
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": epochs,
        "sequence_length": 4,
        "loss_weights": {"scenario": 1.0, "alert": 1.0, "priority": 2.0},
        "splits": splits["counts"],
        "test_results": {
            "scenario_accuracy": test_res["scenario_acc"],
            "scenario_macro_f1": test_res["scenario_f1"],
            "alert_accuracy": test_res["alert_acc"],
            "alert_macro_f1": test_res["alert_f1"],
            "priority_mae": test_res["priority_mae"],
        },
        "ablations": ablation_results,
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)
    print(f"Saved training config: {config_path}")

    # Checkpoint Manifest
    def file_sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while c := f.read(65536):
                h.update(c)
        return h.hexdigest()

    manifest = {
        "best_checkpoint": {
            "path": best_path,
            "size_bytes": os.path.getsize(best_path),
            "sha256": file_sha256(best_path),
            "val_loss": best_val_loss,
        },
        "latest_checkpoint": {
            "path": latest_path,
            "size_bytes": os.path.getsize(latest_path),
            "sha256": file_sha256(latest_path),
        },
        "history_csv": {
            "path": history_csv_path,
            "sha256": file_sha256(history_csv_path),
        },
        "training_config": {
            "path": config_path,
            "sha256": file_sha256(config_path),
        },
        "training_duration_seconds": round(time.time() - start_time, 2),
    }

    manifest_path = os.path.join(output_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved checkpoint manifest: {manifest_path}")

    print(f"\n=== TRAINING & EVALUATION COMPLETED IN {time.time()-start_time:.2f}s ===", flush=True)


if __name__ == "__main__":
    main()
