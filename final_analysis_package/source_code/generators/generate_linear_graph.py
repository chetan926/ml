"""
Generate linear graph for the four contributions (C1-C4) across both datasets.
Line charts showing MCC and Accuracy for C1, C2, C3, C4 on Dataset 1 and Dataset 2.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

BASE = "/home/user/ml"
TABLE_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "tables")
CHART_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Load the four contributions results
df = pd.read_csv(os.path.join(TABLE_DIR, "four_contributions_results.csv"))

# Filter to contributions only
contribs = df[df["Type"] == "contribution"].copy()
contrib_order = ["C1_EntropyAwareAdaptive", "C2_MultiViewFusion",
                 "C3_CostSensitiveCascade", "C4_ConformalCalibrated"]
contrib_labels = ["C1\nEntropy-Aware\nAdaptive", "C2\nMulti-View\nFusion",
                  "C3\nCost-Sensitive\nCascade", "C4\nConformal\nCalibrated"]

datasets = ["Dataset1_repo_text_benchmark", "Dataset2_quora_question_pairs"]
dataset_labels = ["Dataset 1\n(MSRP-style Benchmark)", "Dataset 2\n(Quora QP)"]

metrics = ["MCC", "Accuracy", "F1-Score"]
metric_labels = ["MCC", "Accuracy", "F1-Score"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Linear Graph — Four Contributions (C1–C4) Across Datasets",
             fontsize=13, fontweight="bold", y=1.02)

for di, dname in enumerate(datasets):
    sub = contribs[contribs["Dataset"] == dname].set_index("Method").reindex(contrib_order)
    for mi, metric in enumerate(metrics):
        ax = axes[di, mi]
        vals = sub[metric].values
        x = np.arange(len(contrib_order))
        ax.plot(x, vals, marker="o", linewidth=2.5, markersize=8,
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][di % 4])
        ax.fill_between(x, vals, alpha=0.25,
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][di % 4])
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.4f}", (x[i], v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(contrib_labels, fontsize=7)
        ax.set_ylabel(metric_labels[mi], fontsize=9)
        ax.set_title(f"{dataset_labels[di]}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(1.0, float(vals.max()) * 1.25))
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "q1_linear_graph_four_contributions.png"),
            bbox_inches="tight", dpi=150)
plt.close()
print("Saved q1_linear_graph_four_contributions.png")

# Also generate a combined MCC line chart
fig, ax = plt.subplots(figsize=(10, 5))
for ci, cname in enumerate(contrib_order):
    mcc_vals = []
    for dname in datasets:
        sub = contribs[(contribs["Dataset"] == dname) & (contribs["Method"] == cname)]
        mcc_vals.append(float(sub["MCC"].values[0]))
    ax.plot(datasets, mcc_vals, marker="o", linewidth=2.5, markersize=8,
            label=cname.replace("_", " "), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][ci])
    for di, v in enumerate(mcc_vals):
        ax.annotate(f"{v:.4f}", (di, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(dataset_labels, fontsize=9)
ax.set_ylabel("MCC", fontsize=10)
ax.set_title("Linear Graph — C1–C4 MCC Across Both Datasets", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax.legend(fontsize=9, loc="best")
all_mcc = []
for ci, cname in enumerate(contrib_order):
    for dname in datasets:
        sub = contribs[(contribs["Dataset"] == dname) & (contribs["Method"] == cname)]
        all_mcc.append(float(sub["MCC"].values[0]))
ax.set_ylim(0, max(all_mcc) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "q1_linear_graph_c1c4_mcc.png"),
            bbox_inches="tight", dpi=150)
plt.close()
print("Saved q1_linear_graph_c1c4_mcc.png")
