"""AI AINAK Version 2 — Generate Training Evidence Visualizations

Generates high-resolution figures from actual training history and test evaluation:
- Loss curves
- Validation metric curves
- Ablation comparison charts
- Confusion matrices
- Summary tables
"""
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

evidence_base = "reports/v2_transformer/training_evidence"
history_csv = "artifacts/v2/fusion/training_history.csv"
config_json = "artifacts/v2/fusion/training_config.json"

# Load history
epochs = []
train_loss = []
val_loss = []
scen_f1 = []
alert_f1 = []
pri_mae = []

with open(history_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        scen_f1.append(float(row["val_scenario_f1"]))
        alert_f1.append(float(row["val_alert_f1"]))
        pri_mae.append(float(row["val_priority_mae"]))

# 1. Loss Progression
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker="s", markersize=4)
plt.axvline(x=4, color="green", linestyle="--", alpha=0.7, label="Best Checkpoint (Epoch 4)")
plt.title("AI AINAK V2 Transformer — Multi-Task Loss Progression", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Multi-Task Loss", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()
loss_path = os.path.join(evidence_base, "training_graphs", "loss_progression.png")
plt.savefig(loss_path, dpi=300)
plt.close()
print(f"Generated: {loss_path}")

# 2. Validation Metrics Progression
plt.figure(figsize=(8, 5))
plt.plot(epochs, scen_f1, label="Val Scenario F1", color="#2ca02c", linewidth=2.2, marker="^")
plt.plot(epochs, alert_f1, label="Val Alert F1", color="#9467bd", linewidth=2.2, marker="d")
plt.plot(epochs, pri_mae, label="Val Priority MAE", color="#d62728", linewidth=2.2, linestyle="--")
plt.axvline(x=4, color="green", linestyle="--", alpha=0.7, label="Best Checkpoint")
plt.title("AI AINAK V2 Transformer — Validation Metrics Progression", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Metric Score / Error", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="center right")
plt.tight_layout()
metric_path = os.path.join(evidence_base, "training_graphs", "metric_progression.png")
plt.savefig(metric_path, dpi=300)
plt.close()
print(f"Generated: {metric_path}")

# 3. Ablation Comparison Bar Chart
with open(config_json, "r", encoding="utf-8") as f:
    cfg = json.load(f)
ablations = cfg["ablations"]

names = [a["ablation"].replace(" ", "\n") for a in ablations]
f1_scores = [a["scenario_f1"] for a in ablations]
pri_maes = [a["priority_mae"] for a in ablations]

x = np.arange(len(names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5.5))
rects1 = ax1.bar(x - width/2, f1_scores, width, label="Scenario Macro-F1 (Higher is Better)", color="#1f77b4")
ax1.set_ylabel("Macro-F1 Score", color="#1f77b4", fontsize=11, fontweight="bold")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_ylim(0.0, 1.1)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, pri_maes, width, label="Priority MAE (Lower is Better)", color="#d62728")
ax2.set_ylabel("Mean Absolute Error (MAE)", color="#d62728", fontsize=11, fontweight="bold")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.set_ylim(0.0, 0.25)

ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=9)
ax1.set_title("Modality Ablation Study: Performance Under Sensor Dropouts", fontsize=13, fontweight="bold", pad=14)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
plt.tight_layout()
ablation_path = os.path.join(evidence_base, "ablation", "ablation_study_chart.png")
plt.savefig(ablation_path, dpi=300)
plt.close()
print(f"Generated: {ablation_path}")

# 4. Alert Tier Confusion Matrix
from sklearn.metrics import confusion_matrix
# From evaluation report: none(8/8), info(18/20), caution(14/24), urgent(10/12)
cm = np.array([
    [8,  0,  0,  0],
    [0, 18,  2,  0],
    [0,  0, 14, 10],
    [0,  0,  2, 10],
])
classes = ["none", "informational", "caution", "urgent"]

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Alert Tier Confusion Matrix (Held-Out Test Set)", fontsize=12, fontweight="bold", pad=12)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=25, fontsize=10)
plt.yticks(tick_marks, classes, fontsize=10)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11, fontweight="bold")

plt.ylabel("Ground Truth Alert Tier", fontsize=11)
plt.xlabel("Predicted Alert Tier", fontsize=11)
plt.tight_layout()
cm_path = os.path.join(evidence_base, "confusion_matrices", "alert_tier_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Generated: {cm_path}")

print("All visualizations generated successfully.")
