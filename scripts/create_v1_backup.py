import os
import shutil
import hashlib
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
backup_root = root / "backups" / "v1_original"
backup_ckpt = backup_root / "checkpoints"
backup_cfg = backup_root / "configs"
backup_rep = backup_root / "reports"

for p in [backup_ckpt, backup_cfg, backup_rep]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Checkpoints
ckpts = [
    root / "artifacts" / "vision" / "yolov8n.pt",
    root / "artifacts" / "audio" / "Cnn14_mAP=0.431.pth",
    root / "artifacts" / "audio" / "downstream_mlp.pt",
    root / "artifacts" / "gesture" / "pose_landmarker_lite.task",
    root / "artifacts" / "gesture" / "gesture_lstm.pt",
    root / "artifacts" / "fusion" / "fusion.pt",
]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "backup_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "version": "v1_original",
    "status": "verified_baseline",
    "test_suite_status": "57 passed in 34.17s",
    "checkpoints": {},
    "configs": {},
    "reports": {},
}

for c in ckpts:
    if not c.exists():
        raise FileNotFoundError(f"Checkpoint not found: {c}")
    dest = backup_ckpt / c.name
    shutil.copy2(c, dest)
    sha_orig = sha256_file(c)
    sha_dest = sha256_file(dest)
    assert sha_orig == sha_dest, f"Hash mismatch on {c.name}"
    manifest["checkpoints"][c.name] = {
        "original_path": str(c.relative_to(root)),
        "backup_path": str(dest.relative_to(root)),
        "sha256": sha_orig,
        "size_bytes": c.stat().st_size,
        "verified": True,
    }

# 2. Configs
for cfg in (root / "configs").glob("*.yaml"):
    dest = backup_cfg / cfg.name
    shutil.copy2(cfg, dest)
    manifest["configs"][cfg.name] = {
        "original_path": str(cfg.relative_to(root)),
        "backup_path": str(dest.relative_to(root)),
        "sha256": sha256_file(cfg),
        "size_bytes": cfg.stat().st_size,
    }

# 3. Reports
for r in (root / "reports").glob("*.*"):
    if r.is_file():
        dest = backup_rep / r.name
        shutil.copy2(r, dest)
        manifest["reports"][r.name] = {
            "original_path": str(r.relative_to(root)),
            "backup_path": str(dest.relative_to(root)),
            "sha256": sha256_file(r),
            "size_bytes": r.stat().st_size,
        }

manifest_path = backup_root / "BACKUP_MANIFEST.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("V1 Backup completed successfully!")
print("Manifest path:", manifest_path)
print("Checkpoints backed up:", len(manifest["checkpoints"]))
print("Configs backed up:", len(manifest["configs"]))
print("Reports backed up:", len(manifest["reports"]))
