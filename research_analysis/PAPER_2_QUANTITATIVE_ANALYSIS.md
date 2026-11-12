# Phase 11 — Paper 2 Quantitative Analysis

**Gandhi et al. (2024).** All values are repository-audited *reproduction* numbers unless marked NR;
the paper's own published tables are not in the repository (Phase 8).

| Measure | Value | Evidence |
|---|---|---|
| Reproduction corpus size | IR-Plag Java; Policy A = 720 pairs (7 cases) | `datasets/processed/code_plagiarism_benchmark.csv` |
| Accuracy | 0.7000 (adaptation test on case-06/07) | audit §2.2 |
| Precision | 0.6258 | 〃 |
| Recall | 0.9510 | 〃 |
| F1-score | **0.7549** | 〃 |
| F1 / MCC under corrected nested LOGO (Policy A) | **0.8363 / 0.6754** | `PHASE_5_HEADLINE_COMPARISON.csv` (B-C3) |
| ROC-AUC / PR-AUC | NR (paper context) | — |
| Specificity / FPR | NR (paper context); project frame: part of phase-5 headline row B-C3 | — |
| Training time | NR | — |
| Inference time | **0.013 ms/pair** (classifier only) | `literature/methodology_comparison.csv` |
| Upstream feature-extraction cost | **2.050 ms/pair** (AST extraction, measured separately) | `research_upgrade/novelty_and_literature_analysis.md` |
| Memory consumption | NR | — |
| Computational complexity | SVM RBF over 7-D; linear in support vectors per pair | recorded qualitatively |
| Threshold | τ = 0.3700 (selected on validation case-05) | audit §2.2 |
| Statistical significance vs proposed | ΔF1 = −0.0083 (CI [−0.046,+0.033]) at proposed OP; ΔF1 = +0.0117 (CI [−0.035,+0.068]) at proposed β=1 → **not significant either way; statistically indistinguishable** | `PHASE_5_STATISTICAL_TESTS.json` |
| Confidence intervals (own results) | NR in paper-context; project-level CIs listed above | 〃 |
| Systematic-renaming resilience | positive recall **0.2778** under extra-benchmark attack | `PHASE_5_CODE_RENAMING_ATTACK.csv` |
