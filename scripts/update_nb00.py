import json

nb_path = "notebooks/v2_colab/final/00_V2_Setup_and_Preflight_FINAL.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "ASSET_MANIFEST = [" in src and "V2 Fusion Transformer" not in src:
            v2_entry = """    {
        "name": "V2 Fusion Transformer (best.pt)",
        "role": "Trained V2 Multimodal Transformer",
        "candidates": [
            PROJECT_ROOT / "artifacts" / "v2" / "fusion" / "best.pt",
            PROJECT_ROOT / "checkpoints" / "fusion" / "v2_transformer.pt",
        ],
        "expected_sha256": "2c1beaca8b05b6f11038f23d34be8f03ae609fd8c4d6e45a9318f3fda964a88b",
        "min_size_mb": 2.0,
    },
"""
            idx = src.find("ASSET_MANIFEST = [\n") + len("ASSET_MANIFEST = [\n")
            new_src = src[:idx] + v2_entry + src[idx:]
            cell["source"] = [line + "\n" for line in new_src.split("\n")]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 00_V2_Setup_and_Preflight_FINAL.ipynb with V2 Fusion Transformer")
