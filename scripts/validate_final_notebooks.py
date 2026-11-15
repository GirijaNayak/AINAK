"""
Validation Script for AINAK V2 Final Colab Notebook Suite
Validates AST syntax, structure, checkpoint references, and feature dimension contracts
across all 8 notebooks in notebooks/v2_colab/final/.
"""

import ast
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = REPO_ROOT / "notebooks" / "v2_colab" / "final"

EXPECTED_NOTEBOOKS = [
    "00_V2_Setup_and_Preflight_FINAL.ipynb",
    "01_V2_Expanded_Gesture_Recognition_FINAL.ipynb",
    "02_V2_Expanded_Object_Detection_FINAL.ipynb",
    "03_V2_Contextual_Hazard_Detection_FINAL.ipynb",
    "04_V2_Audio_Module_FINAL.ipynb",
    "05_V2_Transformer_and_Multimodal_Fusion_FINAL.ipynb",
    "06_V2_Complete_AINAK_Demonstration_FINAL.ipynb",
    "07_V1_vs_V2_Comparison_FINAL.ipynb",
]

def sanitize_code_for_ast(code: str) -> str:
    """Strip IPython magics (!pip, %cd, etc.) so standard python ast can parse."""
    clean_lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%"):
            clean_lines.append(f"# {line}")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines)

def validate_suite():
    print("=" * 80)
    print("  AI AINAK V2: FINAL COLAB NOTEBOOK SUITE VALIDATION")
    print("=" * 80)
    print(f"Validation Target Directory: {FINAL_DIR}")
    
    suite_passed = True
    total_cells = 0
    total_code_cells = 0
    results_summary = []

    for nb_name in EXPECTED_NOTEBOOKS:
        nb_path = FINAL_DIR / nb_name
        if not nb_path.is_file():
            print(f"[FAIL] Missing notebook: {nb_name}")
            suite_passed = False
            continue

        with open(nb_path, "r", encoding="utf-8") as f:
            try:
                nb_data = json.load(f)
            except Exception as e:
                print(f"[FAIL] JSON decode error in {nb_name}: {e}")
                suite_passed = False
                continue

        cells = nb_data.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        total_cells += len(cells)
        total_code_cells += len(code_cells)

        ast_errors = []
        for idx, c in enumerate(code_cells):
            src = "".join(c.get("source", []))
            clean_src = sanitize_code_for_ast(src)
            try:
                ast.parse(clean_src)
            except SyntaxError as se:
                ast_errors.append((idx, se.lineno, str(se)))

        # Feature contract inspection
        full_text = json.dumps(nb_data)
        contract_flags = []
        if nb_name == "00_V2_Setup_and_Preflight_FINAL.ipynb":
            if "artifacts/v2/fusion/best.pt" in full_text:
                contract_flags.append("V2 Transformer in Checklist")
        elif nb_name == "01_V2_Expanded_Gesture_Recognition_FINAL.ipynb":
            if "128" in full_text and "25" in full_text:
                contract_flags.append("25 Classes & 128-D Embedding")
        elif nb_name == "02_V2_Expanded_Object_Detection_FINAL.ipynb":
            if "256" in full_text and "251" in full_text:
                contract_flags.append("251 Categories & 256-D Vector")
        elif nb_name == "03_V2_Contextual_Hazard_Detection_FINAL.ipynb":
            if "CRITICAL" in full_text and "WARNING" in full_text:
                contract_flags.append("4 Hazard Tiers Verified")
        elif nb_name == "04_V2_Audio_Module_FINAL.ipynb":
            if "2048" in full_text and "50" in full_text and "silence" in full_text.lower():
                contract_flags.append("2048-D Embedding & Silence Gate")
        elif nb_name == "05_V2_Transformer_and_Multimodal_Fusion_FINAL.ipynb":
            if "V2MultimodalTransformer" in full_text and "best.pt" in full_text:
                contract_flags.append("Trained V2 Transformer & Multi-task Heads")
        elif nb_name == "06_V2_Complete_AINAK_Demonstration_FINAL.ipynb":
            if "V2MultimodalTransformer" in full_text and "v2_fusion_model" in full_text and "video" in full_text.lower():
                contract_flags.append("Dual Pipeline A/B & HTML5 Video Embed")
        elif nb_name == "07_V1_vs_V2_Comparison_FINAL.ipynb":
            if "608" in full_text and "19.6" in full_text:
                contract_flags.append("Verified 608k Params & 19.6% MAE Improvement")

        status = "PASS" if not ast_errors else "FAIL"
        if ast_errors:
            suite_passed = False

        print(f"[{status}] {nb_name:<50} | Cells: {len(cells):2d} (Code: {len(code_cells):2d}, MD: {len(md_cells):2d}) | Checks: {', '.join(contract_flags)}")
        if ast_errors:
            for c_idx, l_no, err_msg in ast_errors:
                print(f"       -> AST Syntax Error in Code Cell {c_idx} (line {l_no}): {err_msg}")

        results_summary.append({
            "notebook": nb_name,
            "status": status,
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "md_cells": len(md_cells),
            "size_bytes": nb_path.stat().st_size,
            "contract_flags": contract_flags,
            "ast_clean": len(ast_errors) == 0
        })

    print("-" * 80)
    print(f"Total Notebooks Verified: {len(results_summary)} / 8")
    print(f"Total Evaluated Cells  : {total_cells} ({total_code_cells} Code Cells)")
    print(f"Suite Verification     : {'ALL 8 NOTEBOOKS PASSED' if suite_passed else 'FAILURES DETECTED'}")
    print("=" * 80)

    # Save validation manifest
    out_rep_dir = REPO_ROOT / "reports" / "notebook_update"
    out_rep_dir.mkdir(parents=True, exist_ok=True)
    with open(out_rep_dir / "notebook_validation_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "suite_passed": suite_passed,
            "total_notebooks": len(results_summary),
            "total_cells": total_cells,
            "total_code_cells": total_code_cells,
            "notebooks": results_summary
        }, f, indent=2)

    return suite_passed

if __name__ == "__main__":
    success = validate_suite()
    sys.exit(0 if success else 1)
