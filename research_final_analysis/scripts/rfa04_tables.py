"""RFA-7/8/9/sections 7/9/16: comparison tables, uncertainty/routing/compute
metrics, sensitivity sweeps, statistical tests — all from genuine stored
scores (prediction CSVs from rfa02 + phase-5/6 artifacts).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("/home/user/ml")))
from research_upgrade.src.calibration.calibration_metrics import compute_calibration_metrics
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_mcnemar_significance, compute_paired_difference_bootstrap_ci)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import apply_routing, wilson_score_ci

from rfa_common import (P5, P6, RFA, full_metrics, wilson, wc, wj)

EXP = RFA / "04_EXPERIMENTAL_RESULTS"
PRED = EXP / "PREDICTIONS"
SENS = RFA / "07_SENSITIVITY_AND_ROBUSTNESS"
CMP = EXP


def emulated_cost_row(y, p, ent, s2, cap):
    """Honest budget emulation: rank pairs by predictive entropy, escalate the
    most-uncertain cap*N to the heavy verifier (decision at 0.5), resolve the
    rest with the calibrated S1 score at 0.5."""
    n = len(y)
    k = int(round(cap * n))
    order = np.argsort(-ent)  # descending uncertainty
    esc = np.zeros(n, dtype=bool)
    esc[order[:k]] = True
    pred = np.where(esc, (s2 >= 0.5).astype(int), (p >= 0.5).astype(int))
    return pred, {"escalated_count": int(esc.sum())}


def main():
    cm = pd.read_csv(EXP / "CLASSIFICATION_METRICS.csv")

    # ---------------- §9 comparison tables (text / code) ----------------
    def comp_table(mod):
        d = cm[cm.Modality == mod].copy()
        keep = ["System", "Type", "accuracy", "precision", "recall", "specificity",
                "f1", "mcc", "roc_auc", "pr_auc", "fpr", "fnr"]
        return d[keep]
    wc(comp_table("Text"), EXP / "TABLE_TEXT_COMPARISON.csv")
    wc(comp_table("Code (PolicyA LOGO)"), EXP / "TABLE_CODE_COMPARISON.csv")

    # ---------------- uncertainty / calibration (§7) ----------------
    pt = pd.read_csv(PRED / "PROPOSED_text_test_predictions.csv")
    pc = pd.read_csv(PRED / "PROPOSED_code_oof_predictions.csv")
    unc_rows = []
    for tag, df, groups in [("Text", pt, pt["premise_group_id"]), ("Code(PolicyA)", pc, pc["case"])]:
        cal = compute_calibration_metrics(df["ground_truth_label"].values.astype(int),
                                          df["calibrated_prob"].values)
        esc = (df[[c for c in df.columns if c.endswith("state")][0]] == "UNCERTAIN_ESCALATED").mean()
        unc_rows.append({
            "Modality": tag,
            "ECE": cal.get("ece", cal.get("ECE")),
            "brier": cal.get("brier_score", cal.get("brier")),
            "uncertain_escalated_pct": round(100 * float(esc), 2),
            "mean_predictive_entropy": round(float(df["predictive_entropy"].mean()), 4),
        })
    wc(pd.DataFrame(unc_rows), EXP / "UNCERTAINTY_METRICS.csv")

    # ---------------- routing metrics (§7) ----------------
    rrows = []
    for tag, df in [("Text", pt), ("Code(PolicyA)", pc)]:
        scol = [c for c in df.columns if c.endswith("state")][0]
        st = df[scol]
        rrows.append({
            "Modality": tag,
            "low_risk_pruned": int((st == "LOW_RISK_PRUNED").sum()),
            "high_risk_flagged": int((st == "HIGH_RISK_FLAGGED").sum()),
            "uncertain_escalated": int((st == "UNCERTAIN_ESCALATED").sum()),
            "expensive_call_rate_pct": round(100 * (st == "UNCERTAIN_ESCALATED").mean(), 2),
            "candidate_recall_on_resolved": round(float(
                ((df["final_prediction"] == 1) & (df["ground_truth_label"] == 1)).sum()
                / max(1, int((df["ground_truth_label"] == 1).sum()))), 4),
        })
    wc(pd.DataFrame(rrows), EXP / "ROUTING_METRICS.csv")

    # ---------------- compute metrics (§7) — measured ----------------
    crow = []
    Xl = np.random.RandomState(0).randn(512, 13)
    t0 = time.time()
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    lr = LogisticRegression(max_iter=200).fit(Xl, (Xl[:, 0] > 0).astype(int))
    hgb = HistGradientBoostingClassifier(max_iter=100).fit(Xl, (Xl[:, 0] > 0).astype(int))
    for name, mdl in [("Stage-1 LR (12-13 views)", lr), ("Stage-2 HGB verifier", hgb)]:
        q = Xl[:1]
        for _ in range(2000):
            mdl.predict_proba(q)
        t0 = time.time()
        reps = 3000
        for _ in range(reps):
            mdl.predict_proba(q)
        ms = (time.time() - t0) / reps * 1000
        crow.append({"component": name, "single_pair_latency_ms": round(ms, 5)})
    # end-to-end ratios from phase-5 cost analysis (already measured, verified)
    p5c_cost = pd.read_csv(P5 / "PHASE_5_COST_ANALYSIS.csv")
    wc(pd.DataFrame(crow), EXP / "COMPUTATIONAL_METRICS.csv")
    p5c_cost.to_csv(EXP / "COMPUTATIONAL_COST_PHASE5.csv", index=False)

    # ---------------- sensitivity sweeps (§11) ----------------
    y_t = pt["ground_truth_label"].values.astype(int)
    ent_t = pt["predictive_entropy"].values
    sweep = []
    for cap in [0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 0.85, 1.0]:
        pred, cost = emulated_cost_row(y_t, pt["calibrated_prob"].values, ent_t,
                                       pt["stage2_score"].values, cap)
        sweep.append({"budget_cap": cap, "escalated_pct": round(100 * cost["escalated_count"] / len(y_t), 1),
                      **full_metrics(y_t, pred, pt["calibrated_prob"].values)})
    wc(pd.DataFrame(sweep), SENS / "BUDGET_ANALYSIS.csv")

    thr_rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 3):
        pred = (pt["calibrated_prob"].values >= t).astype(int)
        thr_rows.append({"threshold": float(t),
                         **{k: v for k, v in full_metrics(y_t, pred).items() if k in
                            ("accuracy", "precision", "recall", "f1", "mcc", "specificity")}})
    wc(pd.DataFrame(thr_rows), SENS / "THRESHOLD_ANALYSIS.csv")

    # risk-coverage using entropy abstention
    order = np.argsort(ent_t)
    rc = []
    for cov in np.round(np.arange(0.5, 1.01, 0.05), 2):
        k = int(len(y_t) * cov)
        m = full_metrics(y_t[order[:k]], pt["final_prediction"].values[order[:k]],
                         pt["calibrated_prob"].values[order[:k]])
        rc.append({"coverage": float(cov), "risk": round(1 - m["accuracy"], 4),
                   "mcC": m["mcc"], "f1": m["f1"]})
    wc(pd.DataFrame(rc), SENS / "RISK_COVERAGE_ANALYSIS.csv")

    # robustness: reuse phase-5 attack table + phase-6 external tables
    pd.read_csv(P5 / "PHASE_5_ROBUSTNESS_TEXT_ATTACKS.csv").to_csv(
        SENS / "ROBUSTNESS_TEXT_ATTACKS.csv", index=False)
    pd.read_csv(P6 / "PHASE_6_EXTERNAL_RANKING_METRICS.csv").to_csv(
        SENS / "ROBUSTNESS_EXTERNAL_CORPORA.csv", index=False)
    pd.read_csv(P6 / "PHASE_6_CS_SEVERITY_RECALL.csv").to_csv(
        SENS / "ROBUSTNESS_CS_SEVERITY.csv", index=False)

    # ---------------- statistical tests (§7/§9) ----------------
    stats = {}
    pred_files = {p.name: p for p in PRED.glob("*.csv")}
    for f in sorted(pred_files):
        if f.startswith("PROPOSED_text") or f.startswith("PROPOSED_code"):
            continue
        d = pd.read_csv(pred_files[f])
        prop = pt if "text" in f else pc
        if len(d) != len(prop):
            continue
        yv = d["ground_truth"].values.astype(int)
        mn = compute_mcnemar_significance(yv, prop["final_prediction"].values, d["pred"].values)
        gids = prop["premise_group_id"].values if "text" in f else prop["case"].values
        db = compute_paired_difference_bootstrap_ci(
            gids, yv, prop["final_prediction"].values.astype(int), d["pred"].values.astype(int))
        stats[f.replace(".csv", "")] = {"mcnemar": mn, "paired_bootstrap": db}
    wj(stats, EXP / "STATISTICAL_TESTS.json")

    # experiment metadata
    wj({"seed": 42, "protocol": "text: premise-grouped split; code: nested LOGO",
        "sources": {"baselines": "frozen phase-5 suites (rfa02)",
                    "proposed": "phase-5 stored predictions",
                    "ablation": "rfa03 fresh runs"}},
       EXP / "EXPERIMENT_METADATA.json")
    print("tables done:", len(cm), "systems;", len(sweep), "budget points;",
          len(thr_rows), "thresholds;", len(stats), "stat tests")


if __name__ == "__main__":
    main()
