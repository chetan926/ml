# Phase 10 — Paper 1 Quantitative Analysis

**Setu et al. (2025)** — all numbers below are from the repository's **audited reproduction record**
(`research_audit/final_literature_verification.md` §2.1; `literature/selected_papers.csv`;
`experiments/results/setu_hybrid_*.csv`), **not** from the paper PDF (absent). Values that the
paper itself would report but no repository document records are marked **NR**.

| Measure | Value | Context / Evidence |
|---|---|---|
| Reproduction dataset size (test) | 3,000 pairs (SNLI/Pereira; validation sample N=3,000 for tuning) | audit §2.1 |
| Accuracy | **0.7200** | τ=0.5000, out-of-sample test (Method 1 MiniLM reproduction) |
| Precision | **0.5697** | 〃 |
| Recall | **0.8570** | 〃 |
| F1-score | **0.6844** | 〃 |
| ROC-AUC | **0.8198** | 〃 |
| Specificity / FPR / PR-AUC | NR in audit for paper-reproduction context | (phase-4.1 frozen baseline row computes: specificity 0.6125, FPR 0.3875, PR-AUC 0.6494 — see `DATASET_1_METRICS.csv`) |
| Training time | NR | encoder is pretrained; only threshold tuned |
| Inference time | **4.143 ms/pair** (dense SBERT scorer) | `literature/methodology_comparison.csv` |
| E1 full-hybrid latency | **4.163 ms/pair** | 〃 |
| E2 staged-hybrid latency | **2.875 ms/pair** (~30% workload reduction) | 〃 |
| Memory consumption | NR | — |
| Computational complexity | Embedding ≈ transformer forward per sentence; hybrid cost stated per-pair above | recorded qualitatively |
| Statistical significance / CIs | NR (paper-context), **available for reproduction** via phase-4.1/5 clustered bootstrap comparisons vs proposed framework | `PHASE_4_1_STATISTICAL_ANALYSIS.json` |
| Validation-tuned threshold | τ = 0.5000 (val F1 = 0.6690) | audit §2.1; documentation-reconciliation note recorded there (earlier docs confused 0.7200 accuracy with τ) |
| Confidence intervals | NR in paper context | — |

**Original experimental context:** the paper evaluated full-length academic documents/thesis
chapters; the repository adaptation is sentence-level and explicitly labeled as such. Page/section
references into the paper PDF are unavailable (document absence disclosed in Phase 8); repository
evidence locations are cited in every row above.
