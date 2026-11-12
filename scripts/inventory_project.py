import os
import json
from pathlib import Path

root = Path(".")
categories = {
    "python_sources": [],
    "datasets_and_data": [],
    "features": [],
    "models_and_pipelines": [],
    "calibration_and_uncertainty": [],
    "routing_and_cost": [],
    "explainability": [],
    "evaluation_and_metrics": [],
    "artifacts_csv_json": [],
    "figures": [],
    "unit_tests": [],
    "documentation_reports": [],
    "archives_and_zips": [],
    "configs_and_envs": [],
}

for p in sorted(root.rglob("*")):
    if p.is_file():
        parts = p.parts
        if ".git" in parts or ".venv" in parts or "__pycache__" in parts or ".matplotlib" in parts:
            continue
        rel = p.as_posix()
        ext = p.suffix.lower()
        if ext == ".py":
            if "tests" in parts:
                categories["unit_tests"].append(rel)
            else:
                categories["python_sources"].append(rel)
        elif ext in (".csv", ".json", ".parquet"):
            if "datasets" in parts:
                categories["datasets_and_data"].append(rel)
            elif "features" in parts:
                categories["features"].append(rel)
            else:
                categories["artifacts_csv_json"].append(rel)
        elif ext in (".png", ".jpg", ".svg", ".pdf"):
            categories["figures"].append(rel)
        elif ext in (".md", ".txt", ".tex"):
            categories["documentation_reports"].append(rel)
        elif ext == ".zip":
            categories["archives_and_zips"].append(rel)
        elif p.name in ("requirements.txt", "pyproject.toml", ".gitignore", "LICENSE"):
            categories["configs_and_envs"].append(rel)

print("=" * 60)
print("PROJECT INVENTORY SUMMARY")
print("=" * 60)
total_files = sum(len(items) for items in categories.values())
for cat, items in categories.items():
    print(f"{cat:30s}: {len(items):4d} files")
print("-" * 60)
print(f"{'TOTAL TRACKED / INVENTORIED':30s}: {total_files:4d} files")
print("=" * 60)

with open("scripts/inventory.json", "w", encoding="utf-8") as f:
    json.dump(categories, f, indent=2)
print("Saved detailed inventory to scripts/inventory.json")
