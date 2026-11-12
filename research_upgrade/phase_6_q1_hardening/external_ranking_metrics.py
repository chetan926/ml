"""Threshold-free external-validity analysis.

The val-tuned operating point is corpus-specific Platt-calibrated on the
ORIGINAL validation split; external corpora (CS, MRPC) undergo distribution
shift, so fixed-threshold metrics penalize calibration drift. For honest Q1
external validity, we additionally report ranking metrics that do NOT depend on
any corpus-specific threshold/operating point:

  * ROC-AUC and Average Precision on calibrated stage-1 probs and stage-2
    scores, computed directly from the saved PHASE_6_EXTERNAL_*_PREDICTIONS
    parquets (no refitting).
  * Clough–Stevenson per-grade analysis: AUC of grade g1..g4 (plagiarised)
    vs g0 (genuine) using stage-2 scores; Spearman monotone check between
    obfuscation grade and median stage-2 score.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/home/user/ml")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "artifacts"
from scipy.stats import spearmanr


def rank_block(df, score_col):
    y = df["ground_truth_label"].values.astype(int)
    s = df[score_col].values.astype(float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": "NR (single class)", "average_precision": "NR (single class)"}
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4),
            "average_precision": round(float(average_precision_score(y, s)), 4)}


def main():
    ext_text = ROOT / ("research_upgrade/phase_6_q1_hardening/artifacts"
                       "/external_text")
    out = {}

    for name, label_file in [
        ("MRPC_TEST", "mrpc_pairs.parquet"),
        ("CLOUGH_STEVENSON", "clough_pairs.parquet"),
        ("CLOUGH_STEVENSON_CROSSTOPIC", "clough_crosstopic_pairs.parquet"),
    ]:
        preds = pd.read_parquet(OUT / f"PHASE_6_EXTERNAL_{name}_PREDICTIONS.parquet")
        y = preds["ground_truth_label"].values.astype(int)
        out[name] = {
            "n": int(len(preds)),
            "stage1_calibrated_prob": rank_block(preds, "calibrated_prob"),
            "stage2_score": rank_block(preds, "stage2_score"),
        }
        print(name, json.dumps(out[name], indent=1))

    # C-S per-grade ranking vs g0
    cs = pd.read_parquet(OUT / "PHASE_6_EXTERNAL_CLOUGH_STEVENSON_PREDICTIONS.parquet")
    meta = pd.read_parquet(ext_text / "clough_pairs.parquet")[["pair_id", "severity_grade"]]
    cs = cs.merge(meta, on="pair_id")
    g0 = cs[cs.severity_grade == "g0"]["stage2_score"].values
    grades = {}
    for g in ["g1", "g2", "g3", "g4"]:
        gs = cs[cs.severity_grade == g]["stage2_score"].values
        yy = np.r_[np.zeros(len(g0), int), np.ones(len(gs), int)]
        ss = np.r_[g0, gs]
        grades[g] = {
            "n_plag": int(len(gs)),
            "auc_vs_g0": round(float(roc_auc_score(yy, ss)), 4),
            "median_s2_plag": round(float(np.median(gs)), 4),
            "median_s2_g0": round(float(np.median(g0)), 4),
        }
        print(g, grades[g])
    med_by_grade = [np.median(cs[cs.severity_grade == g]["stage2_score"].values)
                    for g in ["g0", "g1", "g2", "g3", "g4"]]
    rho, pv = spearmanr([0, 1, 2, 3, 4], med_by_grade)
    grades["monotonicity"] = {"median_s2_g0_g1_g2_g3_g4": [round(float(v), 4) for v in med_by_grade],
                              "spearman_grade_vs_medianscore": round(float(rho), 4),
                              "p_value": round(float(pv), 6),
                              "note": "Spearman over per-grade medians (5 points); "
                                      "interpret descriptively."}
    out["CLOUGH_STEVENSON_grade_ranking"] = grades

    json.dump(out, open(OUT / "PHASE_6_EXTERNAL_RANKING_METRICS.json", "w"), indent=2)
    pd.DataFrame([
        {"dataset": k, "s1_auc": v["stage1_calibrated_prob"].get("roc_auc"),
         "s2_auc": v["stage2_score"].get("roc_auc"),
         "s2_ap": v["stage2_score"].get("average_precision")}
        for k, v in out.items() if k != "CLOUGH_STEVENSON_grade_ranking"
    ]).to_csv(OUT / "PHASE_6_EXTERNAL_RANKING_METRICS.csv", index=False)
    print("saved PHASE_6_EXTERNAL_RANKING_METRICS.{json,csv}")


if __name__ == "__main__":
    main()
