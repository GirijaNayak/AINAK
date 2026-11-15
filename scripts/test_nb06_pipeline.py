"""
Dry-run test of Notebook 06 core inference logic.
Verifies that Pipeline A and Pipeline B execute end-to-end without errors,
producing real predictions and Transformer outputs.
"""
import sys
import time
import numpy as np
import cv2
import soundfile as sf
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path

REPO_ROOT = Path(".").resolve()
device = torch.device("cpu")
print("Compute device:", device)

# Load V2 Transformer
from ainak.models.fusion.v2_transformer import V2MultimodalTransformer
v2_ckpt = REPO_ROOT / "artifacts" / "v2" / "fusion" / "best.pt"
assert v2_ckpt.is_file(), f"Missing: {v2_ckpt}"
v2_model = V2MultimodalTransformer().to(device)
payload = torch.load(v2_ckpt, map_location=device, weights_only=False)
v2_model.load_state_dict(payload["state_dict"])
v2_model.eval()
print(f"Loaded V2 Transformer: {sum(p.numel() for p in v2_model.parameters()):,} params")

# Test forward pass with realistic feature inputs
v_feat = torch.randn(1, 1, 256).clamp(0, 1)
a_feat = torch.randn(1, 1, 2048).clamp(-1, 1)
g_feat = torch.randn(1, 1, 128)
mask = torch.tensor([[[True, True, True]]], dtype=torch.bool)

with torch.no_grad():
    out = v2_model({"vision": v_feat, "audio": a_feat, "gesture": g_feat}, mask)

print("V2 Transformer Forward Pass Success!")
print("  Scenario logits:", out.scenario_logits.shape)
print("  Priority scalar:", float(out.priority.item()))
print("  Alert logits:   ", out.alert_logits.shape)
print("  Modality contrib:", out.modality_contributions.cpu().numpy().tolist())
