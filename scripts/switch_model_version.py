#!/usr/bin/env python3
"""AI AINAK — Model Version Switcher & Immediate Rollback Controller

Supports switching active versions between v1 and v2 across:
- vision
- gesture
- hazard
- or all modules simultaneously.

Guarantees 100% reversible rollback to v1 baseline without retraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "version_config.yaml"
REGISTRY_PATH = ROOT / "configs" / "model_registry.yaml"
MANIFEST_PATH = ROOT / "backups" / "v1_original" / "BACKUP_MANIFEST.json"


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_version_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"models": {"vision": "v1", "gesture": "v1", "hazard": "v1"}}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"models": {}}


def save_version_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry config missing: {REGISTRY_PATH}")
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_v1_baseline() -> bool:
    """Verify that all original V1 checkpoints match their recorded SHA-256 hashes."""
    if not MANIFEST_PATH.exists():
        print(f"[WARN] V1 backup manifest not found at {MANIFEST_PATH}")
        return False
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    all_ok = True
    print("[INFO] Verifying V1 checkpoint integrity against baseline manifest...")
    for name, item in manifest.get("checkpoints", {}).items():
        orig_p = ROOT / item["original_path"]
        if not orig_p.exists():
            print(f"  [FAIL] {name}: Missing at {orig_p}")
            all_ok = False
            continue
        cur_hash = get_sha256(orig_p)
        expected_hash = item["sha256"]
        if cur_hash != expected_hash:
            print(f"  [FAIL] {name}: Hash mismatch!\n    Current:  {cur_hash}\n    Expected: {expected_hash}")
            all_ok = False
        else:
            print(f"  [PASS] {name} verified ({item['size_bytes']} bytes)")
    return all_ok


def switch_version(target_version: str, module: str | None = None) -> None:
    target_version = target_version.lower()
    if target_version not in {"v1", "v2"}:
        raise ValueError(f"Invalid version: {target_version}. Must be 'v1' or 'v2'.")

    cfg = load_version_config()
    models = cfg.setdefault("models", {})

    modules_to_update = [module] if module else ["vision", "gesture", "hazard"]

    for m in modules_to_update:
        if m not in {"vision", "gesture", "hazard"}:
            raise ValueError(f"Unknown module '{m}'. Must be one of vision, gesture, hazard.")
        models[m] = target_version

    save_version_config(cfg)
    print(f"\n[SUCCESS] Active configuration updated to {target_version.upper()}:")
    for k, v in models.items():
        print(f"  - {k}: {v}")

    # If rolling back to v1, verify baseline checkpoints
    if target_version == "v1":
        ok = verify_v1_baseline()
        if ok:
            print("[SUCCESS] V1 baseline verification PASSED. System fully restored to V1.")
        else:
            print("[ERROR] V1 baseline verification failed! Inspect above warnings.")


def status() -> None:
    cfg = load_version_config()
    print("AI AINAK Current Active Model Versions:")
    for k, v in cfg.get("models", {}).items():
        print(f"  {k:10s} -> {v}")
    verify_v1_baseline()


def main():
    parser = argparse.ArgumentParser(description="Switch AI AINAK model versions between v1 and v2")
    parser.add_argument("--version", choices=["v1", "v2"], help="Target version to activate ('v1' or 'v2')")
    parser.add_argument("--module", choices=["vision", "gesture", "hazard"], default=None,
                        help="Specific module to switch (default: all modules)")
    parser.add_argument("--status", action="store_true", help="Print current active version status")
    parser.add_argument("--verify-baseline", action="store_true", help="Verify V1 baseline hashes")

    args = parser.parse_args()

    if args.status:
        status()
    elif args.verify_baseline:
        ok = verify_v1_baseline()
        sys.exit(0 if ok else 1)
    elif args.version:
        switch_version(args.version, args.module)
    else:
        status()


if __name__ == "__main__":
    main()
