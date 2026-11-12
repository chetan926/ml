"""Phase 22: publication-quality analysis figures from ACTUAL measured data.

Every figure is reproducible: re-run ./.venv/bin/python
research_analysis/analysis_scripts/run_figures.py. No fabricated values.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

ROOT = Path("/home/user/ml")
RA = ROOT / "research_analysis"
FIG = RA / "figures"
A41 = ROOT / "research_upgrade/phase_4_1_corrections/artifacts"
A5 = ROOT / "research_upgrade/phase_5_q1_upgrade/artifacts"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

text_pool = pd.read_csv(ROOT / "datasets/processed/text_plagiarism_benchmark.csv")
code_pool = pd.read_csv(ROOT / "datasets/processed/code_plagiarism_benchmark.csv")
tm41 = pd.read_csv(A41 / "PHASE_4_1_PREDICTIONS_TEXT_TEST.csv")
cm41 = pd.read_csv(A41 / "PHASE_4_1_PREDICTIONS_CODE_OOF.csv")
tm5 = pd.read_csv(A5 / "PHASE_5_PREDICTIONS_TEXT_TEST.csv")
cm5 = pd.read_csv(A5 / "PHASE_5_PREDICTIONS_CODE_OOF.csv")
tb41 = pd.read_csv(A41 / "PHASE_4_1_BASELINE_PREDICTIONS_TEXT_TEST.csv")
cb41 = pd.read_csv(A41 / "PHASE_4_1_BASELINE_PREDICTIONS_CODE_OOF.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


# 1. Class distribution comparison ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
tc = text_pool.is_plagiarized.value_counts().sort_index()
axes[0].bar(["negative", "positive"], [tc[0], tc[1]], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Dataset 1 (text, raw pool n=19,666)\nproxy positives (entailment) vs negatives")
axes[0].set_ylabel("pairs")
for i, v in enumerate([tc[0], tc[1]]):
    axes[0].text(i, v + 120, f"{v} ({100*v/tc.sum():.1f}%)", ha="center", fontsize=8)
cc = code_pool.is_plagiarized.value_counts().sort_index()
axes[1].bar(["negative", "positive"], [cc[0], cc[1]], color=["#4C72B0", "#DD8452"])
axes[1].set_title("Dataset 2 (code, raw n=733)\nplagiarized vs non-plagiarized")
for i, v in enumerate([cc[0], cc[1]]):
    axes[1].text(i, v + 6, f"{v} ({100*v/cc.sum():.1f}%)", ha="center", fontsize=8)
save(fig, "fig01_class_distribution_comparison.png")

# 2. Feature (length) distributions -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
axes[0].hist(text_pool.suspicious_text.str.len(), bins=50, color="#4C72B0")
axes[0].set_title("Dataset 1: suspicious-text length (chars)\nmean=37.8, median=34, n=19,666")
axes[0].set_xlabel("characters"); axes[0].set_ylabel("pairs")
axes[1].hist(code_pool.suspicious_code.str.count("\n") + 1, bins=40, color="#DD8452")
axes[1].set_title("Dataset 2: suspicious-code length (lines)\nmean=28.7, median=28, n=733")
axes[1].set_xlabel("source lines"); axes[1].set_ylabel("pairs")
save(fig, "fig02_feature_length_distributions.png")

# 3. Missing-value analysis ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.0))
miss = pd.Series({
    "text: any column": int(text_pool.isna().sum().sum()),
    "code: obfuscation_level\n(structural, negatives)": int(code_pool.obfuscation_level.isna().sum()),
    "code: any other column": int(code_pool.drop(columns=["obfuscation_level"]).isna().sum().sum()),
})
miss.plot.bar(ax=ax, color=["#55A868", "#C44E52", "#55A868"])
ax.set_title("Missing-value analysis (structural vs accidental)\nDataset 1: 0 missing; Dataset 2: 378 structural NaNs only")
ax.set_ylabel("missing cells")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
save(fig, "fig03_missing_value_analysis.png")

# 4. Dataset statistical comparison ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.2))
dims = ["rows", "positive share (%)", "median sus. length", "IQR sus. length"]
d1 = [19658, 33.53, 34.0, 18.0]
d2 = [720, 47.5, 28.0, 12.0]
x = np.arange(len(dims)); w = 0.38
ax.bar(x - w / 2, d1, w, label="Dataset 1 text (clean pool / Policy n/a)", color="#4C72B0")
ax.bar(x + w / 2, d2, w, label="Dataset 2 code (Policy A, lines for length)", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(dims, fontsize=8)
ax.set_yscale("log")
ax.set_title("Dataset statistical comparison (log scale; lengths: chars vs code lines)")
ax.legend(fontsize=7)
save(fig, "fig04_dataset_statistical_comparison.png")

# 5. Three-paper methodology coverage matrix -------------------------------------------
papers = ["Setu et al. 2025\n(text, SBERT hybrid)", "Gandhi et al. 2024\n(code, RBF-SVM)",
          "Mehsen & Joshi 2024\n(code, Random Forest)"]
dims5 = ["multi-view\nfeatures", "uncertainty /\ncalibration", "conformal\nprediction",
         "budget-aware\nrouting", "explainable\ndossiers", "leak-free\neval. (LOGO)"]
# 0 = not addressed, 1 = partial, 2 = implemented (derived from repo docs)
M = np.array([[2, 0, 0, 1, 0, 0],
              [2, 0, 0, 0, 0, 0],
              [2, 0, 0, 0, 0, 0]])
fig, ax = plt.subplots(figsize=(6.8, 2.8))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
ax.set_xticks(range(len(dims5))); ax.set_xticklabels(dims5, fontsize=7)
ax.set_yticks(range(3)); ax.set_yticklabels(papers, fontsize=7)
lab = {0: "not addressed", 1: "partial (cost reported)", 2: "implemented"}
for i in range(3):
    for j in range(6):
        ax.text(j, i, {0: "—", 1: "~", 2: "✓"}[M[i, j]], ha="center", va="center", fontsize=10)
ax.set_title("Three repository papers: capability coverage vs framework dimensions")
save(fig, "fig05_three_paper_methodology_matrix.png")

# 6. Repository-audited performance of the three paper implementations ------------------
fig, ax = plt.subplots(figsize=(6.6, 3.2))
names = ["Setu'25 MiniLM\n(text test)", "Gandhi'24 SVM\n(code case-06/07)", "Mehsen'24 RF\n(code case-06/07)"]
acc = [0.7200, 0.7000, 0.7333]; f1 = [0.6844, 0.7549, 0.7686]
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, acc, w, label="accuracy", color="#4C72B0")
ax.bar(x + w / 2, f1, w, label="F1", color="#DD8452")
ax.axhline(0.5, ls="--", lw=0.8, color="gray")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0, 1.0); ax.set_ylabel("score")
ax.set_title("Repository-audited reproduction results of the three paper implementations\n"
             "(values from research_audit/final_literature_verification.md — NOT the papers' own results)")
ax.legend()
save(fig, "fig06_paper_implementation_performance.png")

# 7/8. Proposed performance and baseline comparison --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
mets = ["accuracy", "precision", "recall", "f1", "mcc"]
p41t = [0.789333, 0.723356, 0.621832, 0.668763, 0.518771]
p5t = [0.808000, 0.707182, 0.748538, 0.727273, 0.579866]
x = np.arange(len(mets)); w = 0.38
axes[0].bar(x - w / 2, p41t, w, label="phase-4.1 frozen", color="#8172B3")
axes[0].bar(x + w / 2, p5t, w, label="phase-5 upgrade", color="#55A868")
axes[0].set_xticks(x); axes[0].set_xticklabels(mets, fontsize=8)
axes[0].set_title("Proposed framework — text modality\n(n=3,000 held-out pairs)")
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=7)
p41c = [0.848611, 0.815718, 0.880117, 0.846695, 0.699566]
p5c = [0.830556, 0.797297, 0.862573, 0.828652, 0.663586]
axes[1].bar(x - w / 2, p41c, w, label="phase-4.1 frozen", color="#8172B3")
axes[1].bar(x + w / 2, p5c, w, label="phase-5 upgrade (heavy S2)", color="#55A868")
axes[1].set_xticks(x); axes[1].set_xticklabels(mets, fontsize=8)
axes[1].set_title("Proposed framework — code modality\n(nested LOGO OOF, n=720)")
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7)
save(fig, "fig07_proposed_framework_performance.png")

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
bt = [(0.668763, "Proposed 4.1"), (0.547517, "TF-IDF baseline"), (0.662435, "MiniLM baseline"),
      (0.727273, "Proposed 5.0")]
axes[0].barh([n for _, n in bt], [v for v, _ in bt], color="#4C72B0")
axes[0].set_title("Text F1 comparison (held-out test)"); axes[0].set_xlim(0, 1)
bc = [(0.846695, "Proposed 4.1"), (0.644068, "Jaccard baseline\n(degenerate)"),
      (0.8363, "B-C3 SVM'24 (P5)"), (0.7859, "B-C4 RF'24 (P5)"),
      (0.828652, "Proposed 5.0"), (0.8481, "Proposed 5.0 β=1")]
axes[1].barh([n for _, n in bc], [v for v, _ in bc], color="#DD8452")
axes[1].set_title("Code F1 comparison (LOGO OOF)"); axes[1].set_xlim(0, 1)
save(fig, "fig08_baseline_comparison.png")

# 9/10. Calibration / reliability ---------------------------------------------------------
def reliability(ax, df, label, color):
    p, y = df.calibrated_prob.values, df.ground_truth_label.values
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="uniform")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(mp, fr, "o-", label=label, color=color)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")

fig, ax = plt.subplots(figsize=(4.4, 3.6))
reliability(ax, tm41, "text proposed (4.1)", "#8172B3")
reliability(ax, tm5, "text proposed (5.0)", "#55A868")
ax.set_title("Calibration curve, text modality (n=3,000)\nECE 4.1: 0.0145 uncal / phase-5: 0.031")
ax.legend(fontsize=7)
save(fig, "fig09_calibration_curves.png")

fig, ax = plt.subplots(figsize=(4.4, 3.6))
reliability(ax, cm41, "code proposed (4.1)", "#C44E52")
reliability(ax, cm5, "code proposed (5.0)", "#DD8452")
ax.set_title("Reliability diagram, code modality (LOGO OOF, n=720)")
ax.legend(fontsize=7)
save(fig, "fig10_reliability_diagrams.png")

# 11. Confusion matrices -------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
for ax, (df, ttl) in zip(axes, [(tm41, "text 4.1"), (tm5, "text 5.0"),
                                (cm41, "code 4.1"), (cm5, "code 5.0")]):
    M = confusion_matrix(df.ground_truth_label, df.final_prediction, labels=[0, 1])
    im = ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(ttl, fontsize=9)
fig.suptitle("Confusion matrices of the proposed framework (stored predictions)", y=1.02)
save(fig, "fig11_confusion_matrices.png")

# 12/13. PR and ROC curves ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.8))
for df, ttl, c in [(tm41, "text 4.1", "#8172B3"), (tm5, "text 5.0", "#55A868"),
                   (cm41, "code 4.1", "#C44E52"), (cm5, "code 5.0", "#DD8452")]:
    pr, rc, _ = precision_recall_curve(df.ground_truth_label, df.calibrated_prob)
    ax.plot(rc, pr, label=ttl, color=c, lw=1.2)
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision–recall curves (stored calibrated probabilities)")
ax.legend(fontsize=7)
save(fig, "fig12_precision_recall_curves.png")

fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
for df, ttl, c in [(tm41, "text 4.1 (AUC .863)", "#8172B3"), (tm5, "text 5.0 (AUC .864)", "#55A868"),
                   (cm41, "code 4.1 (AUC .832)", "#C44E52"), (cm5, "code 5.0 (AUC .758)", "#DD8452")]:
    fr, tr, _ = roc_curve(df.ground_truth_label, df.calibrated_prob)
    ax.plot(fr, tr, label=ttl, lw=1.2, color=c)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("ROC curves (stored calibrated probabilities)")
ax.legend(fontsize=7)
save(fig, "fig13_roc_curves.png")

# 14. Routing distribution -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.2))
gens = pd.read_csv(RA / "PROPOSED_SOLUTION_ROUTING.csv")
states = ["LOW_RISK_PRUNED", "UNCERTAIN_ESCALATED", "HIGH_RISK_FLAGGED"]
colors = {"LOW_RISK_PRUNED": "#55A868", "UNCERTAIN_ESCALATED": "#DD8452", "HIGH_RISK_FLAGGED": "#C44E52"}
labels = [("phase_4_1_frozen", "text", "text 4.1 (n=3000)"), ("phase_5_upgrade", "text", "text 5.0 (n=3000)"),
          ("phase_4_1_frozen", "code", "code 4.1 (n=720)"), ("phase_5_upgrade", "code", "code 5.0 (n=720)")]
left = np.zeros(4)
for s in states:
    vals = []
    for g, d, _ in labels:
        r = gens[(gens.generation == g) & (gens.domain == d) & (gens.decision_state == s)]
        vals.append(int(r.pairs.iloc[0]) if len(r) else 0)
    vals = np.array(vals, dtype=float)
    ax.barh([l[2] for l in labels], vals, left=left, label=s, color=colors[s])
    left += vals
ax.set_xlabel("pairs")
ax.set_title("Routing / decision-state distribution by generation and modality")
ax.legend(fontsize=7)
save(fig, "fig14_routing_distribution.png")

# 15. Accuracy vs computational cost ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.6))
pts = [("text 4.1 (47.2% saved)", 0.6688, 47.23, "#8172B3", (4, -12)),
       ("text 5.0 (61.4% saved)", 0.7273, 61.37, "#55A868", (-4, 8)),
       ("code 4.1 (15.1% saved)", 0.8467, 15.14, "#C44E52", (8, -2)),
       ("code 5.0 (27.4% saved)", 0.8287, 27.36, "#DD8452", (6, -12)),
       ("code 5.0 beta=1 (0% saved)", 0.8481, 0.0, "#937860", (8, 6))]
for name, f1, s, c, off in pts:
    ax.scatter(s, f1, s=70, color=c)
    ax.annotate(name, (s, f1), fontsize=6.5, xytext=off, textcoords="offset points")
ax.set_xlabel("Stage-2 verifier calls avoided (%)")
ax.set_ylabel("F1")
ax.set_title("Accuracy (F1) versus computational-cost reduction")
save(fig, "fig15_accuracy_vs_cost.png")

# 16. Ablation comparison ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
abl = pd.read_csv(A41 / "PHASE_4_1_ABLATION_RESULTS.csv")
abl_t = abl[abl.Domain == "Text"]
ax.barh(abl_t.Configuration, abl_t.mcc, color="#4C72B0")
ax.set_xlabel("MCC"); ax.set_xlim(-0.05, 0.65)
ax.set_title("Contribution-wise ablation, text modality (phase-4.1 frozen)")
save(fig, "fig16_ablation_comparison.png")

# 17. Contribution research-gap matrix ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 3.4))
rows_lab = ["Paper 1 (Setu'25)", "Paper 2 (Gandhi'24)", "Paper 3 (Mehsen'24)",
            "Proposed framework (4.1+5)"]
cols_lab = ["C1 multi-view", "C2 uncertainty+\nconformal", "C3 budget\nrouting", "C4 explainable\ndossiers"]
Gm = np.array([[1, 0, 1, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [2, 2, 2, 2]])
ax.imshow(Gm, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(cols_lab, fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(rows_lab, fontsize=7.5)
for i in range(4):
    for j in range(4):
        ax.text(j, i, {0: "✗", 1: "~", 2: "✓"}[Gm[i, j]], ha="center", va="center", fontsize=11)
ax.set_title("Four-contribution gap matrix (✓ implemented · ~ partial · ✗ absent)")
save(fig, "fig17_contribution_gap_matrix.png")

print("ALL FIGURES DONE")
