"""Export observed Python distribution license metadata for audit triage.

This is an inventory aid, not a license scanner or legal opinion. It reads the
active Python environment and never copies license text into the report.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError as exc:  # pragma: no cover - packaging is a project dependency
    raise SystemExit("Install packaging to export normalized dependency names") from exc


def _requirements(project_file: Path) -> tuple[set[str], set[str]]:
    text = project_file.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current = ""
    for line in text.splitlines():
        header = re.fullmatch(r"\[([^]]+)\]", line.strip())
        if header:
            current = header.group(1)
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"

    def parse(section: str, key: str) -> set[str]:
        match = re.search(rf"(?ms)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*\[(.*?)\]", sections.get(section, ""))
        if not match:
            return set()
        return {canonicalize_name(Requirement(value).name)
                for value in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))}

    direct = parse("project", "dependencies")
    development = parse("project.optional-dependencies", "dev")
    return direct, development


def export_inventory(project_file: Path) -> dict:
    direct, development = _requirements(project_file)
    packages = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        normalized = canonicalize_name(name)
        classifiers = sorted({item.split(" :: ", 1)[1]
                              for item in dist.metadata.get_all("Classifier", [])
                              if item.startswith("License :: ")})
        license_files = sorted({Path(str(item)).name for item in (dist.files or [])
                                if Path(str(item)).name.lower().startswith(("license", "copying", "notice"))})
        expression = dist.metadata.get("License-Expression")
        metadata_license = dist.metadata.get("License") or None
        if metadata_license and ("\n" in metadata_license or len(metadata_license) > 180):
            metadata_license = "present (full text omitted; inspect installed license file)"
        packages.append({
            "name": name,
            "normalized_name": normalized,
            "version": dist.version,
            "dependency_scope": ("direct" if normalized in direct else
                                 "development" if normalized in development else "transitive_or_environment"),
            "license_expression": expression,
            "license_metadata": metadata_license,
            "license_classifiers": classifiers,
            "license_file_names": license_files,
            "metadata_incomplete": not bool(expression or metadata_license or classifiers or license_files),
        })
    packages.sort(key=lambda item: (item["dependency_scope"], item["normalized_name"], item["version"]))
    return {
        "schema": "ainak.python-license-metadata-inventory.v1",
        "generated_by": "scripts/export_dependency_licenses.py",
        "python": sys.version,
        "platform": platform.platform(),
        "project_file": str(project_file.resolve()),
        "inventory_scope": "All distributions visible to the active Python interpreter; this workspace venv uses system-site-packages.",
        "limitations": [
            "Distribution metadata is self-reported and may be absent, stale, or incomplete.",
            "This does not inspect every native library, bundled asset, model weight, dataset, or runtime download.",
            "This is not legal review or clearance for commercial use or redistribution.",
        ],
        "packages": packages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--output", type=Path, default=Path("docs/DEPENDENCY_LICENSES.json"))
    args = parser.parse_args()
    report = export_inventory(args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "packages": len(report["packages"]),
                      "metadata_incomplete": sum(item["metadata_incomplete"] for item in report["packages"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
