# Phase 12 — Paper 3 Quantitative Analysis

**Mehsen & Joshi (2024).** Repository-audited reproduction numbers; paper's own published values NR.

| Measure | Value | Evidence |
|---|---|---|
| Reproduction corpus | IR-Plag Java; Policy A = 720 pairs | processed benchmark CSV |
| Accuracy | 0.7333 (adaptation test case-06/07) | audit §2.3 |
| Precision | 0.6643 | 〃 |
| Recall | 0.9118 | 〃 |
| F1-score | **0.7686** | 〃 |
| F1 / MCC under corrected nested LOGO | **0.7859 / 0.5798** | phase-5 headline (B-C4) |
| Obfuscation resilience (L6 compound) | **83.33% recall** (adaptation record); in phase-5 runs: 0.5085 pos-recall at L6 (LOGO OOF policy) | audit §2.3; `PHASE_5_CODE_OBFUSCATION_LEVELS.csv` |
| Statistical significance vs proposed (β=1) | ΔF1 = **+0.0625**, 95% CI [+0.028, +0.102] — significant proposed win | phase-5 statistical tests |
| vs proposed (cost-min OP) | ΔF1 = +0.0426, CI [+0.017, +0.065] — significant | 〃 |
| Inference time | **0.025 ms/pair** | `methodology_comparison.csv` |
| Model parameters (adaptation) | 100 trees, max_depth=6 (audit-corrected), seed 42 | audit §2.3 |
| ROC-AUC / PR-AUC / specificity | NR (paper context) | — |
| Training time / memory | NR | — |
| Computational complexity | RF ensemble: 100 trees × depth ≤ 6 | recorded structurally |
| Systematic-renaming resilience | pos-recall **0.3216** | `PHASE_5_CODE_RENAMING_ATTACK.csv` |
