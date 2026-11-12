"""Phase 1: read-only repository inventory for the research-analysis phase.

Walks the repository (excluding .git and caches), categorizes files by type and
role, and writes research_analysis/analysis_scripts/repository_inventory.json.
Does NOT modify, move, or delete any existing project file.
"""
import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/user/ml")
OUT = Path("/home/user/ml/research_analysis/analysis_scripts/repository_inventory.json")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


CATEGORY_MAP = {
    "source_code": {".py"},
    "notebooks": {".ipynb"},
    "data_csv": {".csv"},
    "data_parquet": {".parquet"},
    "documentation": {".md", ".txt", ".pdf"},
    "figures": {".png", ".jpg", ".svg"},
    "structured": {".json", ".toml", ".bib"},
    "archives": {".zip"},
    "other": set(),
}


def categorize(p: Path) -> str:
    ext = p.suffix.lower()
    for cat, exts in CATEGORY_MAP.items():
        if ext in exts:
            return cat
    return "other"


def main() -> None:
    inventory = {"root": str(ROOT), "files": [], "by_category": defaultdict(int),
                 "by_top_level_dir": defaultdict(int), "total_bytes": 0}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = p.relative_to(ROOT)
            size = p.stat().st_size
            cat = categorize(p)
            inventory["by_category"][cat] += 1
            top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            inventory["by_top_level_dir"][top] += 1
            inventory["total_bytes"] += size
            inventory["files"].append({
                "path": str(rel), "bytes": size, "category": cat,
                "sha256_12": sha256(p)[:12],
            })
    inventory["total_files"] = len(inventory["files"])
    inventory["by_category"] = dict(inventory["by_category"])
    inventory["by_top_level_dir"] = dict(inventory["by_top_level_dir"])
    OUT.write_text(json.dumps(inventory, indent=2))
    print(f"files={inventory['total_files']} bytes={inventory['total_bytes']}")
    print("by_category:", json.dumps(inventory["by_category"], indent=1))
    print("by_top_level_dir:", json.dumps(inventory["by_top_level_dir"], indent=1))


if __name__ == "__main__":
    main()
