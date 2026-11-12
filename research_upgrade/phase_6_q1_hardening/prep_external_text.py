"""Phase 6 hardening — freeze external text corpora into evaluation parquets.

Sources (purely for held-only evaluation, never for fitting):
  1. Clough–Stevenson corpus: 95 student answers graded g0..g4 over 5 tasks,
     paired with the 5 Wikipedia source passages (orig_task{a..e}.txt).
       labels: g0 -> 0 (non-plagiarised); g1..g4 -> 1 (plagiarised)
       group  = task x source-passage index (5 tasks x 5 passages = 25 groups)
       severity = obfuscation grade g1..g4 (kept for slicing)
  2. MRPC test split: 1,725 sentence paraphrase pairs (label col 0).
       group folding: 50 contiguous blocks of ~35 pairs (practice grouping).

Outputs research_upgrade/phase_6_q1_hardening/artifacts/external_text/{dataset}.parquet
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/ml")
EXT = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "external_data"
OUT = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "artifacts" / "external_text"
OUT.mkdir(parents=True, exist_ok=True)


def load_clough() -> pd.DataFrame:
    corpus = next(EXT.glob("Plagiarism-detection-*-HEAD")) / "corpus"
    origs = {t: (corpus / f"orig_task{t}.txt").read_text(
        encoding="utf-8", errors="replace").strip() for t in "abcde"}
    rows = []
    # Canonical C-S protocol: each student answer judged ONLY against its own
    # task's Wikipedia source passage. g0 = genuinely non-plagiarised.
    pat = re.compile(r"g(\d)p([A-Z])_task([a-e])\.txt")
    for f in sorted(corpus.glob("g*_task*.txt")):
        m = pat.match(f.name)
        if not m:
            continue
        g, person, task = int(m.group(1)), m.group(2), m.group(3)
        txt = f.read_text(encoding="utf-8", errors="replace").strip()
        rows.append({
            "pair_id": f"CS_{f.stem}",
            "reference_text": origs[task],
            "suspicious_text": txt,
            "ground_truth_label": 0 if g == 0 else 1,
            "group_id": f"task{task}",
            "severity_grade": f"g{g}",
            "answer_id": f.stem,
        })
    df = pd.DataFrame(rows)
    assert set(df.severity_grade) == {"g0", "g1", "g2", "g3", "g4"}
    return df


def load_clough_crosstopic() -> pd.DataFrame:
    """Secondary distractor protocol: every answer x ALL 5 source passages.
    Cross-topic pairs are trivially separable negatives — reported separately
    with an honest caveat, never pooled with the canonical protocol."""
    corpus = next(EXT.glob("Plagiarism-detection-*-HEAD")) / "corpus"
    origs = {t: (corpus / f"orig_task{t}.txt").read_text(
        encoding="utf-8", errors="replace").strip() for t in "abcde"}
    pat = re.compile(r"g(\d)p([A-Z])_task([a-e])\.txt")
    rows = []
    for f in sorted(corpus.glob("g*_task*.txt")):
        m = pat.match(f.name)
        if not m:
            continue
        g, task = int(m.group(1)), m.group(3)
        txt = f.read_text(encoding="utf-8", errors="replace").strip()
        for ot, ref in origs.items():
            rows.append({
                "pair_id": f"CSX_{f.stem}_vs_{ot}",
                "reference_text": ref,
                "suspicious_text": txt,
                "ground_truth_label": 1 if (ot == task and g >= 1) else 0,
                "group_id": f"task{task}_src{ot}",
                "severity_grade": f"g{g}" if ot == task else "cross",
                "answer_id": f.stem,
            })
    return pd.DataFrame(rows)


def load_mrpc() -> pd.DataFrame:
    f = EXT / "msr_paraphrase_test.txt"
    lines = f.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    header, body = lines[0], lines[1:]
    assert header.lower().startswith("quality")
    rows = []
    for i, ln in enumerate(body):
        if not ln.strip():
            continue
        parts = ln.split("\t")
        q, s1, s2 = parts[0], parts[3], parts[4]
        rows.append({
            "pair_id": f"MRPC_{i:05d}",
            "reference_text": s1.strip(),
            "suspicious_text": s2.strip(),
            "ground_truth_label": int(q),
            "group_id": f"blk{i // 35:03d}",
            "severity_grade": "nr",
            "answer_id": f"mrpc{i}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cs = load_clough()
    cs.to_parquet(OUT / "clough_pairs.parquet", index=False)
    print(f"Clough–Stevenson: {len(cs)} pairs "
          f"({int((cs.ground_truth_label == 1).sum())} positive), "
          f"groups={cs.group_id.nunique()}")
    print(cs.groupby(["severity_grade", "ground_truth_label"]).size())

    mr = load_mrpc()
    mr.to_parquet(OUT / "mrpc_pairs.parquet", index=False)
    print(f"MRPC test: {len(mr)} pairs "
          f"({int((mr.ground_truth_label == 1).sum())} positive), "
          f"groups={mr.group_id.nunique()}")

    csx = load_clough_crosstopic()
    csx.to_parquet(OUT / "clough_crosstopic_pairs.parquet", index=False)
    print(f"C-S cross-topic distractor: {len(csx)} pairs "
          f"({int((csx.ground_truth_label == 1).sum())} positive)")
