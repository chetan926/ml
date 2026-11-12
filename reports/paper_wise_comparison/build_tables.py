"""Generate the 12 comparison CSV tables from verified/reproduced evidence."""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# Sign convention (applied uniformly):
#   higher-is-better metrics:  improvement = proposed - paper
#   lower-is-better metrics:   improvement = paper - proposed
#   Relative improvement (%)   = improvement / paper * 100  (when paper != 0 and meaningful)
#   Percentage-point metrics:  difference reported in percentage points; relative % not meaningful.
# Positive improvement  => proposed better; negative => paper better; zero => tie.

HIGHER = "higher"
LOWER = "lower"

METRICS_ORDER = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "Specificity",
    "ROC-AUC",
    "PR-AUC",
    "MCC",
    "Candidate Recall",
    "Pruning Ratio (%)",
    "Expensive Call Rate (%)",
    "Average Latency (ms)",
    "Training Time (s)",
    "Memory Consumption (MB)",
]

PCT_METRICS = {"Pruning Ratio (%)", "Expensive Call Rate (%)"}


def fmt_val(v):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if abs(v) < 1e-9:
        return "0.0000"
    return f"{v:.4f}"


def fmt_diff(v):
    if v is None:
        return "N/A"
    return f"{v:+.4f}"


def fmt_rel(v):
    if v is None:
        return "N/A"
    return f"{v:+.2f}%"


def build_rows(paper, prop, higher_metrics, pct_metrics=PCT_METRICS, latency_note=None):
    """paper, prop: dict metric -> value or None. higher_metrics: set of higher-is-better metrics."""
    rows = []
    for m in METRICS_ORDER:
        p = paper.get(m)
        q = prop.get(m)
        higher = m in higher_metrics
        is_pct = m in pct_metrics
        p_s = fmt_val(p)
        q_s = fmt_val(q)
        if p is None or q is None:
            diff_s = "N/A"
            rel_s = "N/A"
            better = "N/A"
        else:
            improvement = (q - p) if higher else (p - q)
            if is_pct:
                diff_s = f"{improvement:+.4f} pp"
                rel_s = "N/A (percentage-point metric)"
            else:
                diff_s = f"{improvement:+.4f}"
                rel_s = fmt_rel(improvement / p * 100.0) if p != 0 else "N/A (zero denominator)"
            if abs(improvement) < 1e-12:
                better = "Tie"
            else:
                better = "Proposed" if improvement > 0 else "Paper baseline"
        rows.append([m, p_s, q_s, diff_s, rel_s, better])
    return rows


def write_csv(path, title, caption, protocol, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TABLE", title])
        w.writerow(["CAPTION", caption])
        for k, v in protocol.items():
            w.writerow([k.upper(), v])
        w.writerow([])
        w.writerow(["Evaluation Metric", "Paper Implementation", "Proposed Solution",
                    "Absolute Difference (Improvement)", "Relative Improvement (%)",
                    "Better Method"])
        for r in rows:
            w.writerow(r)
    print("wrote", path)


# ----------------------------------------------------------------------------
# SOURCE / PROVENANCE LABELS (used in the manuscript)
# ----------------------------------------------------------------------------
PROV = {
    "text_p1_acc": "Archived verified record (experiments/results/text_results_detailed.json; res_sbert; N=3,000)",
}

# ----------------------------------------------------------------------------
# TABLE 1 — TEXT: Paper 1 (Setu et al. 2025, SBERT Condition A) vs Proposed (Condition C)
# ----------------------------------------------------------------------------
paper1_text = {
    "Accuracy": 0.7200,
    "Precision": 0.5697,
    "Recall": 0.8570,
    "F1-Score": 0.6844,
    "Specificity": 0.6448,
    "ROC-AUC": 0.8198,
    "PR-AUC": 0.6729,
    "MCC": 0.4811,          # computed from archived confusion matrix [[1249,688],[152,911]]
    "Candidate Recall": 1.0000,
    "Pruning Ratio (%)": 0.00,
    "Expensive Call Rate (%)": 100.00,
    "Average Latency (ms)": 4.143,
    "Training Time (s)": None,   # pretrained encoder; threshold tuning only
    "Memory Consumption (MB)": None,
}
prop_text = {
    "Accuracy": 0.7180,
    "Precision": 0.5751,
    "Recall": 0.7817,
    "F1-Score": 0.6627,
    "Specificity": 0.6830,
    "ROC-AUC": None,   # tri-band pipeline emits discrete decisions; score-level artifact not archived
    "PR-AUC": None,
    "MCC": 0.4449,     # computed from archived confusion matrix [[1323,614],[232,831]]
    "Candidate Recall": 0.8598,
    "Pruning Ratio (%)": 26.53,
    "Expensive Call Rate (%)": 70.03,
    "Average Latency (ms)": 3.167,
    "Training Time (s)": None,   # threshold tuning only; no model training
    "Memory Consumption (MB)": None,
}
text_higher = {"Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "ROC-AUC", "PR-AUC", "MCC", "Candidate Recall", "Pruning Ratio (%)"}
rows_t1 = build_rows(paper1_text, prop_text, text_higher)
write_csv(OUT / "table_01_paper1_setu_vs_proposed_text_quantitative.csv",
          "Table 1: Quantitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Text Plagiarism Detection",
          "Side-by-side comparison on the same held-out text test partition (N=3,000 pairs). Paper 1 = repository-audited reproduction of Setu et al. (2025), Condition A (full SBERT analysis, max(TF-IDF, SBERT), threshold 0.5000). Proposed = Condition C typed-decision tri-band pipeline. Differences follow the signed improvement convention (positive => proposed better).",
          {
            "Modality": "Natural language text",
            "Dataset": "SNLI/Pereira text-reuse proxy (grouped premise benchmark)",
            "Split": "Premise-disjoint train / validation / test; test partition = 3,000 pairs across 1,015 premise groups (zero premise overlap with training/validation)",
            "Evaluated test samples": "3,000 pairs (1,063 plagiarized / 1,937 non-plagiarized)",
            "Paper 1 model configuration": "Sentence-BERT all-MiniLM-L6-v2 (384-d dense embeddings, cosine similarity) + sublinear TF-IDF; hybrid score max(TF-IDF, SBERT); decision threshold tau=0.5000 (F1-tuned on validation); exhaustive scoring of all pairs",
            "Proposed model configuration": "Typed-decision tri-band pipeline: cheap lexical Stage-1 view; calibrated SBERT Stage-2 verifier invoked only on uncertain band; validation-tuned dual thresholds (theta_low, theta_high); direct high-risk flags; Stage-2 avoided for 26.53% of pairs",
            "Evidence status": "Paper 1: reproduced implementation result (repository-audited, verified authentic). Proposed: proposed framework result (repository-audited, verified authentic). ROC-AUC/PR-AUC for the proposed condition: N/A (discrete band decisions; no score-level artifact archived). Training time / memory: N/R (not instrumented in the audited protocol).",
          },
          rows_t1)

# ----------------------------------------------------------------------------
# TABLE 9 (numbered per sequence below as Table 9) — JAVA: Paper 2 (Gandhi SVM) vs Proposed
# ----------------------------------------------------------------------------
paper2_java = {
    "Accuracy": 0.7000,
    "Precision": 0.6258,
    "Recall": 0.9510,
    "F1-Score": 0.7549,
    "Specificity": 0.4630,
    "ROC-AUC": 0.7786,
    "PR-AUC": 0.7199,
    "MCC": 0.4705,          # computed from archived confusion matrix [[50,58],[5,97]]
    "Candidate Recall": 1.0000,
    "Pruning Ratio (%)": 0.00,
    "Expensive Call Rate (%)": 100.00,
    "Average Latency (ms)": 0.013,   # archived: SVM classifier-only inference (feature extraction +2.050 ms/pair unbudgeted in paper)
    "Training Time (s)": 0.0173,     # measured: SVM fit on cases 01-04 (416 pairs)
    "Memory Consumption (MB)": None,
}
prop_java = {
    "Accuracy": 0.6762,
    "Precision": 0.6049,
    "Recall": 0.9608,
    "F1-Score": 0.7424,
    "Specificity": 0.4074,
    "ROC-AUC": None,
    "PR-AUC": None,
    "MCC": 0.4382,          # computed from reproduced confusion matrix [[44,64],[4,98]]
    "Candidate Recall": 1.0000,
    "Pruning Ratio (%)": 0.48,
    "Expensive Call Rate (%)": 63.33,
    "Average Latency (ms)": 0.058,  # archived: end-to-end tri-band pipeline (heavy SVM on 63.33% of pairs only)
    "Training Time (s)": 0.0173,    # heavy SVM verifier fit on cases 01-04; tri-band thresholds tuned on case-05
    "Memory Consumption (MB)": None,
}
code_higher = {"Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "MCC", "Candidate Recall", "Pruning Ratio (%)"}
rows_t9 = build_rows(paper2_java, prop_java, code_higher)
write_csv(OUT / "table_09_paper2_gandhi_vs_proposed_java_quantitative.csv",
          "Table 9: Quantitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Java Source-Code Plagiarism Detection",
          "Side-by-side comparison on the same held-out Java test partition (N=210 pairs, IR-Plag cases 06-07). Paper 2 = repository-audited reproduction of Gandhi et al. (2024) adapted to Java (7-D syntactic/token features + RBF-SVM, threshold 0.3700). Proposed = Condition C typed-decision tri-band pipeline with SVM heavy verifier. Differences follow the signed improvement convention.",
          {
            "Modality": "Java source code",
            "Dataset": "IR-Plag Java Corpus, Policy A (720 pairs across 7 programming assignments; zero program sharing across cases, SHA-256 verified)",
            "Split": "Case-disjoint holdout: cases 01-04 training (416 pairs), case-05 validation (107 pairs), cases 06-07 test (210 pairs; 102 plagiarized / 108 non-plagiarized)",
            "Evaluated test samples": "210 pairs",
            "Paper 2 model configuration": "SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42) over 7-D features (raw/anonymized token Jaccard, keyword-distribution cosine, operator Jaccard, line-count ratio, cyclomatic-complexity differential, character 3-grams); StandardScaler; decision threshold 0.3700 (F1-tuned on case-05); exhaustive scoring of all pairs",
            "Proposed model configuration": "Token-Jaccard cheap pre-filter; tri-band routing theta_low=0.61 / theta_high=0.87 (case-05 tuned, target candidate recall >= 0.98); SVM heavy verifier (same configuration as Paper 2) invoked only on uncertain band; direct high-risk flags bypass the verifier",
            "Evidence status": "Both sides: reproduced implementation results, exactly matching the archived verified record (experiments/results/code_results_detailed.json; final_comparison_table.csv). ROC-AUC/PR-AUC for the proposed condition: N/A (discrete band decisions). Memory: N/R (not instrumented). Latency rows use non-comparable accounting (see notes): paper = classifier-only inference; proposed = end-to-end tri-band pipeline.",
          },
          rows_t9)

# ----------------------------------------------------------------------------
# TABLE 11 — JAVA: Paper 3 (Mehsen RF) vs Proposed
# ----------------------------------------------------------------------------
paper3_java = {
    "Accuracy": 0.7333,
    "Precision": 0.6643,
    "Recall": 0.9118,
    "F1-Score": 0.7686,
    "Specificity": 0.5648,
    "ROC-AUC": 0.8031,
    "PR-AUC": 0.7738,
    "MCC": 0.5053,          # computed from reproduced confusion matrix [[61,47],[9,93]]
    "Candidate Recall": 1.0000,
    "Pruning Ratio (%)": 0.00,
    "Expensive Call Rate (%)": 100.00,
    "Average Latency (ms)": 0.025,   # archived: RF classifier-only inference
    "Training Time (s)": 0.1505,     # measured: RF fit on cases 01-04 (416 pairs)
    "Memory Consumption (MB)": None,
}
rows_t11 = build_rows(paper3_java, prop_java, code_higher)
write_csv(OUT / "table_11_paper3_mehsen_vs_proposed_java_quantitative.csv",
          "Table 11: Quantitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Java Source-Code Plagiarism Detection",
          "Side-by-side comparison on the same held-out Java test partition (N=210 pairs, IR-Plag cases 06-07). Paper 3 = repository-audited reproduction of Mehsen & Joshi (2024) (Random Forest, 100 trees, max depth 6, threshold 0.4500). Proposed = Condition C typed-decision tri-band pipeline with SVM heavy verifier.",
          {
            "Modality": "Java source code",
            "Dataset": "IR-Plag Java Corpus, Policy A",
            "Split": "Case-disjoint holdout: cases 01-04 training, case-05 validation, cases 06-07 test (210 pairs; 102 plagiarized / 108 non-plagiarized)",
            "Evaluated test samples": "210 pairs",
            "Paper 3 model configuration": "RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42) over the same 7-D syntactic/token features; decision threshold 0.4500 (F1-tuned on case-05); exhaustive scoring of all pairs",
            "Proposed model configuration": "Same as Table 9 (token-Jaccard pre-filter + tri-band routing + SVM heavy verifier)",
            "Evidence status": "Paper 3: reproduced implementation result matching the archived verified record (experiments/results/code_experiments_summary.csv). Proposed: reproduced result matching final_comparison_table.csv Condition C. ROC-AUC/PR-AUC for the proposed condition: N/A. Memory: N/R. Latency accounting note as Table 9.",
          },
          rows_t11)

print("\nQuantitative tables written.")