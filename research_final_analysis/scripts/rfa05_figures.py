"""RFA-8/§12: publication figures (PNG + PDF + underlying CSV/JSON data).

Every plotted point comes from machine-readable artifacts generated in this run
(CLASSIFICATION_METRICS.csv, BUDGET_ANALYSIS.csv, THRESHOLD_ANALYSIS.csv,
ABLATION_RESULTS.csv, prediction CSVs, dataset analyses). No invented values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path("/home/user/ml")))
from rfa_common import RFA, wc, wj

V = RFA / "08_VISUALIZATIONS"
EXP = RFA / "04_EXPERIMENTAL_RESULTS"
SENS = RFA / "07_SENSITIVITY_AND_ROBUSTNESS"
ABL = RFA / "06_ABLATION_STUDY"
PRED = EXP / "PREDICTIONS"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})


def save(fig, subdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(V / subdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def f1_architecture():
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")
    stages = ["Dataset Input", "Modality\nIdentification", "Multi-View\nDetection (S1)",
              "Uncertainty\nEstimation (C2)", "Decision\nFiltering", "Adaptive\nEscalation (C3)",
              "Final Decision +\nAudit Dossier (C4)"]
    x = np.linspace(0.03, 0.97, len(stages))
    for i, (xi, s) in enumerate(zip(x, stages)):
        ax.annotate(s, xy=(xi, 0.5), ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#dbeafe" if i % 2 == 0 else "#fef3c7"))
        if i:
            ax.annotate("", xy=(xi - 0.095, 0.5), xytext=(x[i - 1] + 0.095, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Figure 1 — Framework architecture: modality-aware multi-view pipeline with "
                 "uncertainty gating and budget-constrained escalation")
    save(fig, "ARCHITECTURE", "figure_1_architecture")


def f2_distributions():
    d1 = pd.read_csv(RFA / "01_DATASET_ANALYSIS" / "DATASET1_ANALYSIS.csv")
    d2 = pd.read_csv(RFA / "01_DATASET_ANALYSIS" / "DATASET2_ANALYSIS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax = axes[0]
    x = np.arange(len(d1)); w = 0.35
    ax.bar(x - w / 2, d1["positives"], w, label="Positive", color="#ef4444")
    ax.bar(x + w / 2, d1["negatives"], w, label="Negative", color="#3b82f6")
    ax.set_xticks(x); ax.set_xticklabels(d1["partition"])
    ax.set_title("Dataset 1 (text): class distribution by partition")
    ax.set_ylabel("pairs"); ax.legend()
    ax = axes[1]
    x = np.arange(len(d2))
    ax.bar(x - w / 2, d2["positives"], w, label="Positive", color="#ef4444")
    ax.bar(x + w / 2, d2["negatives"], w, label="Negative", color="#3b82f6")
    ax.set_xticks(x); ax.set_xticklabels(d2["policy"])
    ax.set_title("Dataset 2 (IR-Plag Java): class distribution by policy")
    ax.legend()
    fig.suptitle("Figure 2 — Dataset distributions (computed from repository bytes)")
    save(fig, "DATASET_DISTRIBUTION", "figure_2_distributions")


def f3_performance():
    cm = pd.read_csv(EXP / "CLASSIFICATION_METRICS.csv")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for mod, sub in [("Text", "text"), ("Code (PolicyA LOGO)", "code")]:
        d = cm[cm.Modality == mod].copy()
        d = d[d.System.str.contains("beta=1.0") == False].tail(10)
        for mt in metrics:
            d[mt] = pd.to_numeric(d[mt], errors="coerce")
        labels = [s.split("(")[0].strip()[:26] for s in d["System"]]
        fig, ax = plt.subplots(figsize=(8.2, 3.4))
        x = np.arange(len(d)); w = 1 / (len(metrics) + 1)
        colors = plt.cm.tab10.colors
        for i, mt in enumerate(metrics):
            ax.bar(x + i * w, d[mt], w, label=mt.upper(), color=colors[i])
        ax.set_xticks(x + w * 2.5); ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylim(0, 1.0); ax.set_ylabel("score")
        ax.set_title(f"Figure 3 ({sub}) — performance comparison across systems")
        ax.legend(ncol=3, fontsize=7)
        save(fig, "PERFORMANCE_COMPARISON", f"figure_3_{sub}_performance")
        wc(d[["System"] + metrics], V / "PERFORMANCE_COMPARISON" / f"figure_3_{sub}_data.csv")


def f4_lines():
    bud = pd.read_csv(SENS / "BUDGET_ANALYSIS.csv")
    thr = pd.read_csv(SENS / "THRESHOLD_ANALYSIS.csv")
    abl = pd.read_csv(ABL / "ABLATION_RESULTS.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    ax = axes[0, 0]
    ax.plot(bud["budget_cap"], bud["f1"], "o-", color="#16a34a")
    ax.set_title("F1 vs budget level (emulated from stored scores)")
    ax.set_xlabel("S2 budget (fraction escalated)"); ax.set_ylabel("F1")
    ax = axes[0, 1]
    ax.plot(bud["budget_cap"], bud["escalated_pct"], "o-", color="#9333ea")
    ax.set_title("Heavy-detector calls vs budget")
    ax.set_xlabel("budget cap"); ax.set_ylabel("escalated %")
    ax = axes[0, 2]
    ax.plot(thr["threshold"], thr["precision"], "s-", label="precision", color="#dc2626")
    ax.plot(thr["threshold"], thr["recall"], "o-", label="recall", color="#2563eb")
    ax.set_title("Precision/recall vs decision threshold (calibrated S1)")
    ax.set_xlabel("threshold"); ax.legend()
    ax = axes[1, 0]
    ax.plot(bud["budget_cap"], 1 - bud["budget_cap"], "o-", color="#ea580c")
    ax.set_title("Pruning ratio vs budget")
    ax.set_xlabel("budget cap"); ax.set_ylabel("1 - budget")
    ax = axes[1, 1]
    tt = abl[abl.Configuration.str.startswith("Text")].sort_values("Configuration")
    ax.plot(tt["Configuration"], tt["mcc"], "D-", color="#0891b2")
    ax.set_title("MCC across ablation configurations (text)")
    ax.set_ylabel("MCC")
    ax = axes[1, 2]
    tc = abl[abl.Configuration.str.startswith("Code")].sort_values("Configuration")
    ax.plot(tc["Configuration"], tc["mcc"], "D-", color="#65a30d")
    ax.set_title("MCC across ablation configurations (code)")
    fig.suptitle("Figure 4 — implementation performance line graphs (all points measured)")
    fig.tight_layout()
    save(fig, "LINE_GRAPHS", "figure_4_line_graphs")


def f5_ablation():
    abl = pd.read_csv(ABL / "ABLATION_RESULTS.csv").sort_values("Configuration")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for ax, pfx, ttl in [(axes[0], "Text", "text ablation"), (axes[1], "Code", "code ablation (LOGO)")]:
        d = abl[abl.Configuration.str.startswith(pfx)]
        x = np.arange(len(d))
        ax.bar(x, d["mcc"], color="#7c3aed", label="MCC")
        ax.plot(x, d["escalation_rate_pct"] / 100, "ko--", label="escalation rate")
        ax.set_xticks(x); ax.set_xticklabels([c.split("-")[1] for c in d["Configuration"]])
        ax.set_title(f"Ablation — {ttl}"); ax.set_ylabel("MCC"); ax.legend()
    fig.suptitle("Figure 5 — contribution-wise ablation (A0..A6, genuine runs)")
    save(fig, "ABLATION", "figure_5_ablation")


def f6_efficiency():
    comp = pd.read_csv(EXP / "COMPUTATIONAL_METRICS.csv")
    p5 = pd.read_csv(EXP / "COMPUTATIONAL_COST_PHASE5.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ax = axes[0]
    ax.bar(comp["component"], comp["single_pair_latency_ms"], color=["#2563eb", "#dc2626"])
    ax.set_yscale("log"); ax.set_ylabel("ms per pair (log scale)")
    ax.set_title("Measured single-pair latency by component")
    ax.tick_params(axis="x", rotation=15)
    ax = axes[1]
    cols = [c for c in p5.columns]
    ax.axis("off")
    txt = "\n".join([f"{c}: {p5[c].iloc[0]}" for c in cols[:8]])
    ax.text(0.02, 0.95, "Phase-5 measured cost ratios:\n" + txt, va="top", fontsize=8,
            family="monospace")
    ax.set_title("Verifier call economics (phase-5 measured)")
    fig.suptitle("Figure 6 — computational efficiency")
    fig.tight_layout()
    save(fig, "COST_QUALITY", "figure_6_efficiency")


def f7_confusion():
    cm = pd.read_csv(EXP / "CLASSIFICATION_METRICS.csv")
    import json, ast
    systems = ["B-T1 TF-IDF Lexical (Sajid 2025)", "B-T3 Flat Multi-View Hybrid LR",
               "PROPOSED (C1-C4, heavy S2, tuned OP)", "B-C3 Gandhi RBF-SVM (2024)",
               "B-C4 Mehsen RandomForest (2024)", "PROPOSED (C1-C4, heavy S2, tuned OP)"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    sel = [(0, systems[0], cm[(cm.Modality == "Text") & (cm.System == systems[0])]),
           (1, systems[1], cm[(cm.Modality == "Text") & (cm.System == systems[1])]),
           (2, systems[2], cm[(cm.Modality == "Text") & (cm.System == systems[2])]),
           (3, "B-C3 SVM", cm[cm.System == "B-C3 Gandhi RBF-SVM (2024)"]),
           (4, "B-C4 RF", cm[cm.System == "B-C4 Mehsen RandomForest (2024)"]),
           (5, "Proposed (code)", cm[(cm.Modality == "Code (PolicyA LOGO)") & (cm.System == systems[2])])]
    for idx, name, row in sel:
        ax = axes[idx // 3, idx % 3]
        conf = ast.literal_eval(row.iloc[0]["confusion"]) if isinstance(row.iloc[0]["confusion"], str) else row.iloc[0]["confusion"]
        M = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
        ax.imshow(M, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["true-neg", "true-pos"])
        ax.set_title(name, fontsize=8)
    fig.suptitle("Figure 7 — confusion matrices (genuine stored predictions)")
    fig.tight_layout()
    save(fig, "CONFUSION_MATRICES", "figure_7_confusion")


def f8_roc_pr():
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7))
    combos = [(("Text", axes[0, 0]), ("Text", axes[1, 0]), "text"),
              (("Code", axes[0, 1]), ("Code", axes[1, 1]), "code")]
    for f in sorted(PRED.glob("*.csv")):
        if f.stat().st_size > 3_000_000:
            continue
        d = pd.read_csv(f)
        if "score" not in d or d.empty:
            continue
        y = d["ground_truth"].values.astype(int)
        s = d["score"].values.astype(float)
        if len(np.unique(s)) < 2:
            continue
        is_text = "text" in f.name
        rax = axes[0, 0 if is_text else 1]
        pax = axes[1, 0 if is_text else 1]
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        lbl = f.name.replace("_predictions.csv", "").replace("_text", "").replace("_code", "")[:24]
        rax.plot(fpr, tpr, lw=1, label=lbl)
        pax.plot(rec, prec, lw=1, label=lbl)
    for ax, tt in [(axes[0, 0], "ROC — text"), (axes[0, 1], "ROC — code"),
                   (axes[1, 0], "PR — text"), (axes[1, 1], "PR — code")]:
        ax.plot([0, 1], [0, 1], "k:", lw=0.6) if "ROC" in tt else None
        ax.set_title(tt); ax.legend(fontsize=6, loc="best"); ax.grid(alpha=0.3)
    fig.suptitle("Figure 8 — ROC and precision-recall curves from stored score vectors")
    fig.tight_layout()
    save(fig, "ROC_PR_CURVES", "figure_8_roc_pr")


def f9_calibration():
    pt = pd.read_csv(PRED / "PROPOSED_text_test_predictions.csv")
    y = pt["ground_truth_label"].values.astype(int)
    p = pt["calibrated_prob"].values
    ent = pt["predictive_entropy"].values
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    ax.hist(p[y == 1], bins=30, alpha=0.6, label="pos", color="#dc2626", density=True)
    ax.hist(p[y == 0], bins=30, alpha=0.6, label="neg", color="#2563eb", density=True)
    ax.set_title("Confidence distribution"); ax.legend()
    ax = axes[1]
    frac, mp = calibration_curve(y, p, n_bins=12, strategy="quantile")
    ax.plot(mp, frac, "o-", label="empirical", color="#16a34a")
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    ax.set_title("Reliability diagram (calibrated S1)"); ax.legend()
    ax = axes[2]
    rc = pd.read_csv(SENS / "RISK_COVERAGE_ANALYSIS.csv")
    ax.plot(rc["coverage"], rc["risk"], "o-", color="#9333ea")
    ax.set_title("Risk-coverage (entropy abstention)")
    ax.set_xlabel("coverage"); ax.set_ylabel("risk (1-acc)")
    fig.suptitle("Figure 9 — uncertainty and calibration analysis")
    fig.tight_layout()
    save(fig, "CALIBRATION", "figure_9_calibration")
    wc(pd.DataFrame({"mean_pred": mp, "empirical_frac": frac}),
       V / "CALIBRATION" / "figure_9_reliability_data.csv")


def f10_cost_quality():
    bud = pd.read_csv(SENS / "BUDGET_ANALYSIS.csv")
    abl = pd.read_csv(ABL / "ABLATION_RESULTS.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(bud["escalated_pct"] / 100, bud["mcc"], "o-", color="#0d9488", label="budget sweep (text)")
    for _, r in abl[abl.Configuration.str.startswith("Text")].iterrows():
        ax.plot(r["escalation_rate_pct"] / 100, r["mcc"], "s", ms=7, color="#7c3aed")
        ax.annotate(r["Configuration"].split("-")[1],
                    (r["escalation_rate_pct"] / 100, r["mcc"]), fontsize=7)
    ax.set_xlabel("relative verifier cost (fraction of pairs escalated)")
    ax.set_ylabel("MCC")
    ax.set_title("Figure 10 — cost-quality trade-off (measured points)")
    ax.legend()
    save(fig, "COST_QUALITY", "figure_10_cost_quality")


def main():
    f1_architecture(); f2_distributions(); f3_performance(); f4_lines()
    f5_ablation(); f6_efficiency(); f7_confusion(); f8_roc_pr()
    f9_calibration(); f10_cost_quality()
    wj({"figures": [
        {"no": 1, "title": "Framework architecture", "dir": "ARCHITECTURE"},
        {"no": 2, "title": "Dataset distributions", "dir": "DATASET_DISTRIBUTION"},
        {"no": 3, "title": "Performance comparison (text & code)", "dir": "PERFORMANCE_COMPARISON"},
        {"no": 4, "title": "Line graphs (budgets, thresholds, configs)", "dir": "LINE_GRAPHS"},
        {"no": 5, "title": "Ablation visualization", "dir": "ABLATION"},
        {"no": 6, "title": "Computational efficiency", "dir": "COST_QUALITY"},
        {"no": 7, "title": "Confusion matrices", "dir": "CONFUSION_MATRICES"},
        {"no": 8, "title": "ROC and PR curves", "dir": "ROC_PR_CURVES"},
        {"no": 9, "title": "Uncertainty and calibration", "dir": "CALIBRATION"},
        {"no": 10, "title": "Cost-quality trade-off", "dir": "COST_QUALITY"}],
        "formats": ["png", "pdf"], "source_data": "saved as CSV next to figures where applicable"},
       V / "FIGURE_INDEX.json")
    print("figures done")


if __name__ == "__main__":
    main()
