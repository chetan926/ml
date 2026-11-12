# Phase 6 — Dataset 2 Quantitative Analysis
*Machine outputs: `DATASET_2_STATISTICS.csv`, `DATASET_2_CLASS_DISTRIBUTION.csv`,
`DATASET_2_METRICS.csv`, `DATASET_2_CASE_DISTRIBUTION.csv`,
`DATASET_2_OBFUSCATION_DISTRIBUTION.csv`, `DATASET_2_SUBGROUP_METRICS.csv`.*

## Dataset statistics

| Measure | Value |
|---|---|
| Raw pairs | 733 |
| Duplicate rows (keep-first) | 13 | 
| Rows inside duplicate collision groups | 24 (11 groups) |
| Policy A pairs | **720** (342 pos / 378 neg) |
| Policy B pairs | **709** (subset without collision groups) |
| Missing cells | 378 total = 100% in `obfuscation_level` and all on negatives (structural) |
| Numerical analysis features | 6: `ref_chars, sus_chars, ref_lines, sus_lines, ref_tokens_ws, sus_tokens_ws` |
| Unique reference programs | 105; unique suspicious programs: 447 |

Descriptive statistics (raw pool, n = 733):

| Feature | Mean | Median | Std | Min | Max | Q1 | Q3 | IQR | Skew | Kurt. |
|---|---|---|---|---|---|---|---|---|---|---|
| ref_chars | 548.6 | 501 | 242.4 | 161 | 1,382 | 376 | 723 | 347 | +0.88 | 0.040 |
| sus_chars | 709.4 | 678 | 276.4 | 161 | 1,587 | 493 | 890 | 397 | +0.49 | −0.098 |
| ref_lines | 24.0 | 21 | 8.18 | 11 | 61 | 20 | 29 | 9 | +0.77 | 0.747 |
| sus_lines | 28.7 | 28 | 8.70 | 10 | 61 | 22 | 34 | 12 | +0.48 | 0.402 |
| ref_tokens_ws | 69.2 | 65 | 29.6 | 18 | 151 | 48 | 92 | 44 | +0.42 | −0.576 |
| sus_tokens_ws | 82.3 | 79 | 33.8 | 18 | 188 | 58 | 105 | 47 | +0.43 | −0.308 |

## Class distribution & imbalance

| Scope | Pos | Neg | Pos % | POS:NEG |
|---|---|---|---|---|
| Raw (733) | 355 | 378 | 48.43% | 0.939 |
| Policy A (720) | 342 | 378 | 47.50% | 0.905 |
| Policy B (709) | 342 | 367 | 48.24% | 0.932 |

Per-case: pairs 94–108; positive share 42.55%–50.00% (see `DATASET_2_CASE_DISTRIBUTION.csv`).
Obfuscation strata on positives: L1 60, L2 56, L3 57, L4 60, L5 59, L6 63.
Class balance ≈ 1:1 — no imbalance handling required.

## Existing evaluation metrics (from stored OOF predictions — nested LOGO; no retraining)

Policy A, n = 720 (342 pos / 378 neg):

| System | Acc | Prec | Rec | Spec | F1 | macroF1 | wtdF1 | ROC-AUC | PR-AUC | FPR | FNR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Proposed framework (4.1 frozen, nested LOGO)** | 0.8486 | 0.8157 | 0.8801 | 0.8201 | 0.8467 | 0.8486 | 0.8487 | 0.8322 | 0.7492 | 0.1799 | 0.1199 | **0.6996** |
| Token-Jaccard baseline (frozen) | 0.4750 | 0.4750 | 1.0000 | **0.0000** | 0.6441 | 0.3220 | 0.3059 | 0.6825 | 0.6092 | 1.0000 | 0.0000 | **0.0000** |
| **Proposed heavy-S2 (5 upgrade)** | 0.8306 | 0.7973 | 0.8626 | 0.8016 | 0.8287 | 0.8305 | 0.8306 | 0.7582 | 0.7055 | 0.1984 | 0.1374 | 0.6636 |

Confusion matrices — 4.1 frozen: TP 301, FP 68, TN 310, FN 41. Phase-5: TP 295, FP 75, TN 303, FN 47.
The Jaccard baseline is an exact always-positive classifier at its frozen threshold (FPR = 1.0,
specificity = 0, MCC = 0) — a measured degeneracy, not an assumption.

## Additional analysis

- **Subgroup (case-level) performance, frozen framework:** F1 ranges **0.494 (worst case) → 0.982
  (best case)** (`DATASET_2_SUBGROUP_METRICS.csv`) — high cross-case variance, the expected
  signature of a 7-case corpus with one singular case (05).
- **Distribution differences:** suspicious programs are systematically longer than references
  (+131 chars / +4.6 lines median offset), consistent with transformation additions.
- **Class-specific errors:** frozen framework FN = 41 (recall 0.88) vs FP = 68 (precision 0.82);
  errors concentrate in L5–L6 and the singular Case-05.
- **Potential leakage:** duplicate-pair collisions (11 groups) handled by Policies A/B; task-level
  leakage handled by nested LOGO; both quantified above.
- **Error distributions:** L6 recall is the principal error mass; per the upgrade attack study,
  systematic identifier-renaming drops the proposed OP positive recall 0.8626 → 0.7456 while the
  β=1 heavy configuration holds 0.8918 (vs SVM'24 0.2778, RF'24 0.3216).
