"""Research-analysis quantitative engine (READ-ONLY w.r.t. project files).

Reproduces, without modifying anything, the statistical analyses required by
Phases 4, 6, 7 and 15 of the research-analysis prompt:

  DATASET_1_STATISTICS.csv / DATASET_1_CLASS_DISTRIBUTION.csv / DATASET_1_METRICS.csv
  DATASET_2_STATISTICS.csv / DATASET_2_CLASS_DISTRIBUTION.csv / DATASET_2_METRICS.csv
  DATASET_COMPARISON_STATISTICS.csv
  PROPOSED_SOLUTION_METRICS.csv / CALIBRATION.csv / ROUTING.csv / ABLATION.csv

All numbers are computed from the existing dataset CSVs and the existing frozen
per-pair prediction artifacts (phase_4_1 = canonical audited framework results,
phase_5 = Q1 upgrade). No model training, no pipeline modification.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/ml")
OUT = ROOT / "research_analysis"
A41 = ROOT / "research_upgrade/phase_4_1_corrections/artifacts"
A5 = ROOT / "research_upgrade/phase_5_q1_upgrade/artifacts"


def sha(p: str) -> int:
    return int(hashlib.sha256(str(p).encode("utf-8")).hexdigest(), 16)


def allocate(pool, counts, target):
    alloc, cur, i = [], 0, 0
    pool = list(pool)
    while i < len(pool) and cur < target:
        c = counts[pool[i]]
        if cur + c <= target:
            cur += c
            alloc.append(pool.pop(i))
        else:
            i += 1
    return alloc, cur, pool


def desc_stats(x: np.ndarray, name: str) -> dict:
    from scipy import stats as sps
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "feature": name, "n": len(x), "mean": float(np.mean(x)),
        "median": float(np.median(x)), "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)), "max": float(np.max(x)),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "skewness": float(sps.skew(x)), "kurtosis_excess": float(sps.kurtosis(x)),
    }


def binary_metrics(y, yhat, score=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score)
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    m = {
        "n_samples": int(len(y)), "n_positive": int(tp + fn), "n_negative": int(tn + fp),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, yhat, average="weighted", zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "mcc": matthews_corrcoef(y, yhat),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if score is not None:
        m["roc_auc"] = float(roc_auc_score(y, score))
        m["pr_auc"] = float(average_precision_score(y, score))
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}


def wilson_ci(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - r) / d, 6), round((c + r) / d, 6)


# ------------------------------------------------------------------ DATASET 1
def analyze_text():
    raw = pd.read_csv(ROOT / "datasets/processed/text_plagiarism_benchmark.csv")
    rep = {}

    val_prems = set(raw.loc[raw.split == "val", "reference_text"].unique())
    test_prems = set(raw.loc[raw.split == "test", "reference_text"].unique())
    crossing = val_prems & test_prems
    crossing_ids = set(raw.loc[raw.reference_text.isin(crossing), "id"])
    dupe_ids = set(raw.loc[raw.duplicated(["reference_text", "suspicious_text"], keep="first"), "id"])
    purged = crossing_ids | dupe_ids
    clean = raw[~raw.id.isin(purged)].copy()
    rep["raw_rows"] = len(raw)
    rep["purged_rows"] = len(purged)
    rep["purged_crossing"] = len(crossing_ids)
    rep["purged_redundant_dupes"] = len(dupe_ids)
    rep["clean_rows"] = len(clean)
    rep["crossing_premises"] = len(crossing)
    rep["unique_premises_val"] = len(val_prems)
    rep["unique_premises_test"] = len(test_prems)
    rep["null_cells"] = int(raw.isna().sum().sum())
    rep["duplicate_full_rows"] = int(raw.duplicated(
        ["reference_text", "suspicious_text", "label", "is_plagiarized", "split"]).sum())
    rep["exact_pair_dupes_drop_first"] = len(dupe_ids)

    clean["ref_chars"] = clean.reference_text.str.len()
    clean["sus_chars"] = clean.suspicious_text.str.len()
    clean["ref_words"] = clean.reference_text.str.split().str.len()
    clean["sus_words"] = clean.suspicious_text.str.split().str.len()

    dev = clean[clean.split == "val"].copy()
    tst = clean[clean.split == "test"].copy()
    rep["dev_pool_rows"] = len(dev)
    rep["test_pool_rows"] = len(tst)

    dev_counts = dev.reference_text.value_counts().to_dict()
    dev_prems = sorted(dev_counts, key=sha)
    tr_p, tr_n, rem = allocate(dev_prems, dev_counts, 4000)
    va_p, va_n, rem = allocate(rem, dev_counts, 1500)
    cp_p, cp_n, rem = allocate(rem, dev_counts, 1000)
    cc_p, cc_n, rem = allocate(rem, dev_counts, 1000)
    tst_counts = tst.reference_text.value_counts().to_dict()
    tst_prems = sorted(tst_counts, key=sha)
    te_p, te_n, te_rem = allocate(tst_prems, tst_counts, 3000)

    def mk(df, prems, name):
        sub = df[df.reference_text.isin(prems)]
        return {"partition": name, "rows": len(sub), "premise_groups": sub.reference_text.nunique(),
                "positives": int(sub.is_plagiarized.sum()),
                "positives_pct": round(100 * sub.is_plagiarized.mean(), 2)}
    parts = [mk(dev, tr_p, "train"), mk(dev, va_p, "validation"),
             mk(dev, cp_p, "calibration_prob"), mk(dev, cc_p, "calibration_conformal"),
             {"partition": "dev_reserve", "rows": len(rem), "premise_groups": None,
              "positives": None, "positives_pct": None},
             mk(tst, te_p, "test"),
             {"partition": "test_reserve", "rows": len(te_rem), "premise_groups": None,
              "positives": None, "positives_pct": None}]
    pd.DataFrame(parts).to_csv(OUT / "DATASET_1_PARTITIONS.csv", index=False)
    rep["partitions"] = parts

    stat_rows = [desc_stats(getattr(clean, f).values, f) for f in
                 ("ref_chars", "sus_chars", "ref_words", "sus_words")]
    stats_df = pd.DataFrame(stat_rows)
    stats_df["n_unique_values"] = [clean.reference_text.nunique(), clean.suspicious_text.nunique(),
                                   clean.ref_words.nunique(), clean.sus_words.nunique()]
    stats_df["missing_values"] = 0
    stats_df.to_csv(OUT / "DATASET_1_STATISTICS.csv", index=False)

    cls = clean.is_plagiarized.value_counts().sort_index()
    pd.DataFrame([
        {"scope": "clean_pool", "class": "positive(plagiarized proxy=entailment)",
         "count": int(cls[1]), "pct": round(100 * cls[1] / len(clean), 3)},
        {"scope": "clean_pool", "class": "negative(non-plagiarized)",
         "count": int(cls[0]), "pct": round(100 * cls[0] / len(clean), 3)},
        {"scope": "clean_pool", "class": "POS:NEG ratio",
         "count": round(cls[1] / cls[0], 4), "pct": ""},
        {"scope": "raw_pool", "class": "label=0 entailment(->positive)",
         "count": int((raw.label == 0).sum()), "pct": round(100 * (raw.label == 0).mean(), 3)},
        {"scope": "raw_pool", "class": "label=1 neutral", "count": int((raw.label == 1).sum()),
         "pct": round(100 * (raw.label == 1).mean(), 3)},
        {"scope": "raw_pool", "class": "label=2 contradiction",
         "count": int((raw.label == 2).sum()), "pct": round(100 * (raw.label == 2).mean(), 3)},
    ]).to_csv(OUT / "DATASET_1_CLASS_DISTRIBUTION.csv", index=False)

    from scipy import stats as sps
    tr_sub = dev[dev.reference_text.isin(set(tr_p))]
    te_sub = tst[tst.reference_text.isin(set(te_p))]
    rep["train_len_mean"] = float(tr_sub.sus_chars.mean())
    rep["test_len_mean"] = float(te_sub.sus_chars.mean())
    rep["train_posrate"] = float(tr_sub.is_plagiarized.mean())
    rep["test_posrate"] = float(te_sub.is_plagiarized.mean())
    rep["ks_len_p"] = float(sps.ks_2samp(tr_sub.sus_chars, te_sub.sus_chars).pvalue)
    rep["ks_len_stat"] = float(sps.ks_2samp(tr_sub.sus_chars, te_sub.sus_chars).statistic)

    met = pd.read_csv(A41 / "PHASE_4_1_PREDICTIONS_TEXT_TEST.csv")
    base = pd.read_csv(A41 / "PHASE_4_1_BASELINE_PREDICTIONS_TEXT_TEST.csv")
    rows = []
    rows.append({"system": "PROPOSED framework (phase 4.1 frozen)", "eval_splits": "text test (n=3000)",
                 **binary_metrics(met.ground_truth_label, met.final_prediction, met.calibrated_prob)})
    rows.append({"system": "B1 TF-IDF lexical baseline (frozen)", "eval_splits": "text test (n=3000)",
                 **binary_metrics(base.ground_truth_label, base.tfidf_prediction, base.tfidf_score)})
    rows.append({"system": "B2 Semantic MiniLM baseline (frozen)", "eval_splits": "text test (n=3000)",
                 **binary_metrics(base.ground_truth_label, base.semantic_prediction, base.semantic_score)})
    p5 = pd.read_csv(A5 / "PHASE_5_PREDICTIONS_TEXT_TEST.csv")
    rows.append({"system": "PROPOSED heavy-S2 framework (phase 5 upgrade)", "eval_splits": "text test (n=3000)",
                 **binary_metrics(p5.ground_truth_label, p5.final_prediction, p5.calibrated_prob)})
    pd.DataFrame(rows).to_csv(OUT / "DATASET_1_METRICS.csv", index=False)
    rep["metrics_systems"] = [r["system"] for r in rows]

    # subgroup: suspicious-text length terciles on the frozen phase-4.1 framework
    lookup = dict(zip(te_sub.id, te_sub.sus_chars))
    qs = np.quantile(te_sub.sus_chars.values, [1 / 3, 2 / 3])
    def terc(c):
        return "short" if c < qs[0] else ("medium" if c < qs[1] else "long")
    met = met.copy()
    met["tercile"] = met.pair_id.map(lambda i: terc(lookup.get(i, np.nan)))
    grp_rows = []
    for g in ("short", "medium", "long"):
        gdf = met[met.tercile == g]
        if len(gdf):
            b = binary_metrics(gdf.ground_truth_label, gdf.final_prediction)
            grp_rows.append({"subgroup": f"suspicious_text_len={g}", "n": len(gdf),
                             "recall": b["recall"], "precision": b["precision"], "f1": b["f1"]})
    pd.DataFrame(grp_rows).to_csv(OUT / "DATASET_1_SUBGROUP_METRICS.csv", index=False)
    return rep


# ------------------------------------------------------------------ DATASET 2
def analyze_code():
    raw = pd.read_csv(ROOT / "datasets/processed/code_plagiarism_benchmark.csv")
    dupe_first = raw.duplicated(["reference_code", "suspicious_code"], keep="first")
    dupe_any = raw.duplicated(["reference_code", "suspicious_code"], keep=False)
    pa = raw[~dupe_first].copy()
    pb = raw[~dupe_any].copy()
    rep = {"raw_rows": len(raw), "duplicate_rows_keep_first": int(dupe_first.sum()),
           "duplicate_rows_in_groups": int(dupe_any.sum()),
           "duplicate_groups": int(raw.loc[dupe_any].groupby(
               ["reference_code", "suspicious_code"]).ngroups),
           "policy_a_rows": len(pa), "policy_b_rows": len(pb),
           "null_cells_total": int(raw.isna().sum().sum()),
           "obf_nulls": int(raw.obfuscation_level.isna().sum())}

    pc = []
    for name, df in [("raw_733", raw), ("policy_A_720", pa), ("policy_B_709", pb)]:
        n = len(df)
        pos = int(df.is_plagiarized.sum())
        pc.append({"scope": name, "class": "positive(plagiarized)", "count": pos,
                   "pct": round(100 * pos / n, 3)})
        pc.append({"scope": name, "class": "negative(non-plagiarized)", "count": n - pos,
                   "pct": round(100 * (n - pos) / n, 3)})
        pc.append({"scope": name, "class": "POS:NEG ratio", "count": round(pos / (n - pos), 4), "pct": ""})
    pd.DataFrame(pc).to_csv(OUT / "DATASET_2_CLASS_DISTRIBUTION.csv", index=False)

    raw["ref_chars"] = raw.reference_code.str.len()
    raw["sus_chars"] = raw.suspicious_code.str.len()
    raw["ref_lines"] = raw.reference_code.str.count("\n") + 1
    raw["sus_lines"] = raw.suspicious_code.str.count("\n") + 1
    raw["ref_tokens_ws"] = raw.reference_code.str.split().str.len()
    raw["sus_tokens_ws"] = raw.suspicious_code.str.split().str.len()
    stats = [desc_stats(getattr(raw, f).values, f) for f in
             ("ref_chars", "sus_chars", "ref_lines", "sus_lines", "ref_tokens_ws", "sus_tokens_ws")]
    sdf = pd.DataFrame(stats)
    sdf["n_unique_values"] = [raw.reference_code.nunique(), raw.suspicious_code.nunique(),
                              raw.ref_lines.nunique(), raw.sus_lines.nunique(),
                              raw.ref_tokens_ws.nunique(), raw.sus_tokens_ws.nunique()]
    sdf["missing_values"] = 0
    sdf.to_csv(OUT / "DATASET_2_STATISTICS.csv", index=False)

    per_case = raw.groupby("case").agg(
        pairs=("id", "count"), positives=("is_plagiarized", "sum"))
    per_case["pos_pct"] = (100 * per_case.positives / per_case.pairs).round(2)
    per_case.reset_index().to_csv(OUT / "DATASET_2_CASE_DISTRIBUTION.csv", index=False)
    obf = raw.dropna(subset=["obfuscation_level"]).groupby("obfuscation_level")\
        .agg(pairs=("id", "count")).reset_index()
    obf.to_csv(OUT / "DATASET_2_OBFUSCATION_DISTRIBUTION.csv", index=False)

    met = pd.read_csv(A41 / "PHASE_4_1_PREDICTIONS_CODE_OOF.csv")
    base = pd.read_csv(A41 / "PHASE_4_1_BASELINE_PREDICTIONS_CODE_OOF.csv")
    rows = [{"system": "PROPOSED framework Policy A (phase 4.1 frozen, nested LOGO OOF)",
             "eval_splits": "code Policy A OOF (n=720)",
             **binary_metrics(met.ground_truth_label, met.final_prediction, met.calibrated_prob)},
            {"system": "B2 Token Jaccard baseline (frozen)", "eval_splits": "code Policy A OOF (n=720)",
             **binary_metrics(base.ground_truth_label, base.token_jaccard_prediction, base.token_jaccard_score)}]
    p5 = pd.read_csv(A5 / "PHASE_5_PREDICTIONS_CODE_OOF.csv")
    rows.append({"system": "PROPOSED heavy-S2 framework Policy A (phase 5 upgrade, nested LOGO OOF)",
                 "eval_splits": "code Policy A OOF (n=720)",
                 **binary_metrics(p5.ground_truth_label, p5.final_prediction, p5.calibrated_prob)})
    pd.DataFrame(rows).to_csv(OUT / "DATASET_2_METRICS.csv", index=False)

    grp = []
    for case, gdf in met.groupby("case"):
        b = binary_metrics(gdf.ground_truth_label, gdf.final_prediction)
        grp.append({"subgroup": case, "n": len(gdf), "precision": b["precision"],
                    "recall": b["recall"], "f1": b["f1"]})
    pd.DataFrame(grp).to_csv(OUT / "DATASET_2_SUBGROUP_METRICS.csv", index=False)
    rep["per_case_metric_range"] = {"f1_min": float(min(g["f1"] for g in grp)),
                                    "f1_max": float(max(g["f1"] for g in grp))}
    return rep


# ------------------------------------------------- PROPOSED SOLUTION (PHASE 15)
def proposed_solution():
    res41 = json.loads((A41 / "PHASE_4_1_RESEARCH_RESULTS.json").read_text())
    res5 = json.loads((A5 / "PHASE_5_RESEARCH_RESULTS.json").read_text())

    tm = pd.read_csv(A41 / "PHASE_4_1_PREDICTIONS_TEXT_TEST.csv")
    cm = pd.read_csv(A41 / "PHASE_4_1_PREDICTIONS_CODE_OOF.csv")
    t5 = pd.read_csv(A5 / "PHASE_5_PREDICTIONS_TEXT_TEST.csv")
    c5 = pd.read_csv(A5 / "PHASE_5_PREDICTIONS_CODE_OOF.csv")
    rows = [{"domain": "Text", "generation": "phase_4_1_frozen",
             **binary_metrics(tm.ground_truth_label, tm.final_prediction, tm.calibrated_prob)},
            {"domain": "Code(PolicyA)", "generation": "phase_4_1_frozen",
             **binary_metrics(cm.ground_truth_label, cm.final_prediction, cm.calibrated_prob)},
            {"domain": "Text", "generation": "phase_5_upgrade",
             **binary_metrics(t5.ground_truth_label, t5.final_prediction, t5.calibrated_prob)},
            {"domain": "Code(PolicyA)", "generation": "phase_5_upgrade",
             **binary_metrics(c5.ground_truth_label, c5.final_prediction, c5.calibrated_prob)}]
    pd.DataFrame(rows).to_csv(OUT / "PROPOSED_SOLUTION_METRICS.csv", index=False)

    pd.read_csv(A41 / "PHASE_4_1_CALIBRATION_METRICS.csv").to_csv(
        OUT / "PROPOSED_SOLUTION_CALIBRATION.csv", index=False)

    cost41 = pd.read_csv(A41 / "PHASE_4_1_COST_ANALYSIS.csv")
    cost5 = pd.read_csv(A5 / "PHASE_5_COST_ANALYSIS.csv")
    r41 = []
    for _, r in cost41.iterrows():
        r41.append({"generation": "phase_4_1_frozen", "domain": r["Domain"],
                    "stage1_ms_per_pair": r["stage1_inference_ms_per_pair"],
                    "stage2_ms_per_pair": r["stage2_verifier_ms_per_pair"],
                    "escalated_pairs": r["escalated_to_stage2_pairs"],
                    "stage2_avoided_pct": r["stage2_calls_avoided_percent"],
                    "peak_memory_mb": r["peak_memory_mb"]})
    esc41_t = tm.routing_decision_state.value_counts().to_dict()
    esc41_c = cm.decision_state.value_counts().to_dict()
    esc5_t = t5.routing_decision_state.value_counts().to_dict()
    esc5_c = c5.decision_state.value_counts().to_dict()
    routing = {"phase_4_1_cost": r41, "phase_5_cost": cost5.to_dict("records"),
               "text_decision_states_41": esc41_t, "code_decision_states_41": esc41_c,
               "text_decision_states_5": esc5_t, "code_decision_states_5": esc5_c}
    (OUT / "PROPOSED_SOLUTION_ROUTING_DETAIL.json").write_text(json.dumps(routing, indent=2))

    rt_rows = [{"generation": "phase_4_1_frozen", "domain": d, **v}
               for d, v in [("text", esc41_t), ("code", esc41_c)]]
    rt_rows += [{"generation": "phase_5_upgrade", "domain": d, **v}
                for d, v in [("text", esc5_t), ("code", esc5_c)]]
    long = []
    for row in rt_rows:
        gen, dom = row.pop("generation"), row.pop("domain")
        for state, n in row.items():
            long.append({"generation": gen, "domain": dom, "decision_state": state, "pairs": n})
    pd.DataFrame(long).to_csv(OUT / "PROPOSED_SOLUTION_ROUTING.csv", index=False)

    pd.read_csv(A41 / "PHASE_4_1_ABLATION_RESULTS.csv").to_csv(
        OUT / "PROPOSED_SOLUTION_ABLATION.csv", index=False)

    conf41 = res41.get("text_conformal_summary") or res41.get("conformal") or {}
    conf5 = (res5.get("text", {}).get("proposed", {}).get("group_conformal_summary", {}))
    return {"conformal_41_keys": list(res41.keys())[:12], "conformal_5": bool(conf5)}


# ----------------------------------------------------------------- COMPARISON
def compare(rep_t, rep_c):
    rows = [
        {"dimension": "raw_rows", "dataset_1_text": rep_t["raw_rows"], "dataset_2_code": rep_c["raw_rows"]},
        {"dimension": "duplicates_removed_rows", "dataset_1_text": rep_t["purged_rows"],
         "dataset_2_code": rep_c["duplicate_rows_keep_first"]},
        {"dimension": "modality", "dataset_1_text": "natural-language sentence pairs (SNLI-derived)",
         "dataset_2_code": "Java source-code pairs (IR-Plag)"},
        {"dimension": "label_semantics", "dataset_1_text": "binary: plagiarism proxy = entailment",
         "dataset_2_code": "binary: is_plagiarized with obfuscation level L1-L6 on positives"},
        {"dimension": "positive_share_clean_pool_%",
         "dataset_1_text": round(100 * 6591 / (19658 * 1.0), 2) if False else None,
         "dataset_2_code": round(100 * 355 / 733, 2)},
        {"dimension": "grouping_unit", "dataset_1_text": "premise group",
         "dataset_2_code": "assignment case (7 groups)"},
        {"dimension": "evaluation_protocol", "dataset_1_text": "premise-group-disjoint holdout",
         "dataset_2_code": "nested Leave-One-Case-Out (LOGO)"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "DATASET_COMPARISON_STATISTICS.csv", index=False)


if __name__ == "__main__":
    rp_t = analyze_text()
    rp_c = analyze_code()
    prop = proposed_solution()
    compare(rp_t, rp_c)
    (OUT / "analysis_scripts/analysis_summary.json").write_text(
        json.dumps({"dataset1": rp_t, "dataset2": rp_c, "proposed": prop}, indent=2, default=str))
    print(json.dumps({"dataset1": {k: v for k, v in rp_t.items() if k != "partitions"},
                      "dataset2": rp_c, "proposed": prop}, indent=2, default=str)[:1800])
