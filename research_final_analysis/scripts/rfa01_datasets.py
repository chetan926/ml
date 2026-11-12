"""Phase RFA-2/3: genuine dataset profiling for both datasets (recomputed)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from rfa_common import ROOT, RFA, load_code_raw, load_text_raw, wc, wj, full_metrics, wilson

OUT = RFA / "01_DATASET_ANALYSIS"


def text_analysis():
    raw = load_text_raw()
    tr, va, te = raw["train"], raw["val"], raw["test"]
    rows = []
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        y = df["is_plagiarized"].values.astype(int)
        rows.append({
            "partition": name, "rows": len(df),
            "premise_groups": int(df["reference_text"].nunique()),
            "positives": int(y.sum()),
            "positives_pct": round(100.0 * y.mean(), 2),
            "negatives": int((y == 0).sum()),
            "missing_reference": int(df["reference_text"].isna().sum()),
            "missing_suspicious": int(df["suspicious_text"].isna().sum()),
            "duplicate_pairs_exact": int(df.duplicated(
                subset=["reference_text", "suspicious_text"]).sum()),
        })
    # Cross-partition premise leakage check (must be zero by construction)
    sets = {n: set(d["reference_text"]) for n, d in (("train", tr), ("val", va), ("test", te))}
    leaks = {
        "train_val_premise_overlap": len(sets["train"] & sets["val"]),
        "train_test_premise_overlap": len(sets["train"] & sets["test"]),
        "val_test_premise_overlap": len(sets["val"] & sets["test"]),
    }
    # Cross-split exact duplicate pair check
    pairkey = lambda d: (d["reference_text"] + "||" + d["suspicious_text"]).map(
        lambda x: hashlib.sha256(x.encode()).hexdigest())
    k_tr, k_va, k_te = map(lambda d: set(pairkey(d)), (tr, va, te))
    leaks["train_val_pair_overlap"] = len(k_tr & k_va)
    leaks["train_test_pair_overlap"] = len(k_tr & k_te)
    leaks["val_test_pair_overlap"] = len(k_va & k_te)
    df = pd.DataFrame(rows)
    wc(df, OUT / "DATASET1_ANALYSIS.csv")
    return df, leaks


def code_analysis():
    raw = load_code_raw()
    rows = []
    for pol, df in raw.items():
        y = df["is_plagiarized"].values.astype(int)
        sev = df["obfuscation_level"].astype(object)
        rows.append({
            "policy": pol, "rows": len(df),
            "cases": int(df["case"].nunique()),
            "positives": int(y.sum()), "positives_pct": round(100.0 * y.mean(), 2),
            "negatives": int((y == 0).sum()),
            "missing_reference": int(df["reference_code"].isna().sum()),
            "missing_suspicious": int(df["suspicious_code"].isna().sum()),
            **{f"sev_{lv}": int((sev == lv).sum()) for lv in ["L1", "L2", "L3", "L4", "L5", "L6"]},
            "sev_unlabelled_negatives": int((sev != sev).sum()),
        })
    df = pd.DataFrame(rows)
    wc(df, OUT / "DATASET2_ANALYSIS.csv")
    # Physical java files on disk
    java_root = ROOT / "datasets" / "raw" / "ir_plag" / "extracted" / "IR-Plag-Dataset"
    n_java = len(list(java_root.rglob("*.java")))
    return df, {"physical_java_files": n_java}


def main():
    d1, text_leaks = text_analysis()
    d2, code_info = code_analysis()
    # Comparison table (§3)
    comp = pd.DataFrame([
        ["Dataset name", "SNLI/Pereira 'MIT Plagiarism Detection Dataset' (Kaggle re-export; provenance: SNLI val/test)", "IR-Plag Java corpus (Karnalim, 2019)"],
        ["Modality", "Text (semantic equivalence / text reuse)", "Java source code plagiarism"],
        ["Number of samples/files", "19,658 clean rows (8 purged); partitions below", f"{code_info['physical_java_files']} physical .java files; pairs below"],
        ["Number of evaluated pairs", f"train={int(d1.iloc[0]['rows'])}, val={int(d1.iloc[1]['rows'])}, test={int(d1.iloc[2]['rows'])}",
         f"Policy A={int(d2.iloc[0]['rows'])}, Policy B={int(d2.iloc[1]['rows'])}"],
        ["Positive samples", f"test={int(d1.iloc[2]['positives'])}", f"A={int(d2.iloc[0]['positives'])}, B={int(d2.iloc[1]['positives'])}"],
        ["Negative samples", f"test={int(d1.iloc[2]['negatives'])}", f"A={int(d2.iloc[0]['negatives'])}, B={int(d2.iloc[1]['negatives'])}"],
        ["Class distribution", "positives ~33-35% per partition", f"positives {d2.iloc[0]['positives_pct']}% (A)"],
        ["Missing values", f"test missing ref/sus: {int(d1.iloc[2]['missing_reference'])}/{int(d1.iloc[2]['missing_suspicious'])}", "obfuscation_level NaN = legitimate negatives (structural)"],
        ["Duplicate records", f"test exact-pair duplicates: {int(d1.iloc[2]['duplicate_pairs_exact'])}", "11 duplicate collision groups handled by policy A/B (audited upstream)"],
        ["Extracted features", "16-dim phase-3 text feature store (lexical/character/semantic/structural)", "13-dim phase-3 code feature store (token/structural/AST-derived)"],
        ["Preprocessing", "Whitespace normalization; premise-group purge; SHA-256 premise allocation", "Comment/whitespace stripping; javalang AST parsing"],
        ["Training samples", int(d1.iloc[0]["rows"]), "LOGO folds over 7 cases (nested)"],
        ["Validation samples", int(d1.iloc[1]["rows"]), "inner LOGO folds"],
        ["Test samples", int(d1.iloc[2]["rows"]), "outer held-out case per fold"],
        ["Limitations", "SNLI-derived: premise/hypothesis style; paraphrase-reuse proxy for plagiarism", "Single host language (Java); severity L6 slice is all-positive"],
    ], columns=["Parameter", "Dataset 1", "Dataset 2"])
    wc(comp, OUT / "DATASET_COMPARISON.csv")

    quality = {
        "text": {"partitions": d1.to_dict("records"), "leakage_checks": text_leaks,
                 "leakage_verdict": ("PASS: zero premise/pair overlap" if all(
                     v == 0 for v in text_leaks.values()) else "FAIL")},
        "code": {"policies": d2.to_dict("records"), **code_info},
        "generated_by": "rfa01_datasets.py on repository bytes (loaders + raw files)",
    }
    wj(quality, OUT / "DATASET_QUALITY_REPORT.json")
    print("datasets done:", quality["text"]["leakage_verdict"])


if __name__ == "__main__":
    main()
