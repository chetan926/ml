# Phase 4 — Dataset 1 Quantitative Analysis
*All values computed in this phase from `datasets/processed/text_plagiarism_benchmark.csv` and the
frozen prediction artifacts. Machine outputs: `DATASET_1_STATISTICS.csv`,
`DATASET_1_CLASS_DISTRIBUTION.csv`, `DATASET_1_METRICS.csv`, `DATASET_1_PARTITIONS.csv`,
`DATASET_1_SUBGROUP_METRICS.csv`; script: `analysis_scripts/run_quantitative_analysis.py`.*

## 4.1 Dataset statistics

| Measure | Value |
|---|---|
| Raw instances | 19,666 (val 9,842 + test 9,824) |
| Purged rows | 8 (6 cross-split premise contamination from 1 premise + 2 redundant dupes) |
| Clean instances | 19,658 |
| Columns | 6 (`id, reference_text, suspicious_text, label, is_plagiarized, split`) |
| Numerical features (derived for analysis) | 4: `ref_chars, sus_chars, ref_words, sus_words` |
| Categorical/id features | `id`, `reference_text`, `suspicious_text`, `split`, `label` |
| Missing values | **0 cells (0.0%)** across all columns |
| Duplicate rows | 2 ordered redundant duplicates (purged); 2 full-identity duplicates |
| Unique reference texts | 6,640 (clean pool); unique suspicious texts: 19,106 |

Descriptive statistics (clean pool, n = 19,658):

| Feature | Mean | Median | Std | Min | Max | Q1 | Q3 | IQR | Skew | Kurt. (excess) |
|---|---|---|---|---|---|---|---|---|---|---|
| ref_chars | 72.10 | 64 | 32.52 | 9 | 300 | 48 | 93 | 45 | +1.254 | 2.740 |
| sus_chars | 37.80 | 34 | 16.23 | 3 | 232 | 27 | 45 | 18 | +1.625 | 5.472 |
| ref_words | 13.93 | 13 | 6.24 | 2 | 54 | 9 | 18 | 9 | +1.199 | 2.489 |
| sus_words | 7.50 | 7 | 3.11 | 1 | 48 | 5 | 9 | 4 | +1.593 | 5.631 |

All four length distributions are right-skewed (skew ≥ 1.2); suspicious texts are heavier-tailed
(excess kurtosis 5.5–5.6).

## 4.2 Class distribution

| Scope | Positive (plagiarism proxy) | Negative | POS:NEG |
|---|---|---|---|
| Clean pool (19,658) | 6,589 (33.52%) [label=0] | 13,069 (66.48%) | 0.504 |
| Raw 3-way | label 0: 6,697 (34.05%); label 1: 6,454 (32.82%); label 2: 6,515 (33.13%) | — | — |
| Train (4,000) | 1,347 (33.67%) | 2,653 | 0.508 |
| Validation (1,500) | 524 (34.93%) | 976 | 0.537 |
| Test (3,000) | 1,026 (34.20%) | 1,974 | 0.520 |

Imbalance ratio (majority:minority): ≈ **1.92:1** — mild imbalance; no resampling warranted.

## 4.3 Descriptive statistics

Given in §4.1 (mean, median, std, min, max, quartiles, IQR, skewness, kurtosis).

## 4.4 Existing evaluation metrics (from stored predictions — no retraining)

Binary suite on the held-out **test set (n = 3,000, 1,026 positives / 1,974 negatives)**:

| System | Acc | Prec | Rec | Spec | F1 | macroF1 | wtdF1 | ROC-AUC | PR-AUC | FPR | FNR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Proposed framework (phase-4.1 frozen)** | 0.7893 | 0.7234 | 0.6218 | 0.8764 | 0.6688 | 0.7572 | 0.7851 | 0.8634 | 0.7506 | 0.1236 | 0.3782 | 0.5188 |
| TF-IDF lexical baseline (frozen) | 0.6143 | 0.4572 | 0.6823 | 0.5790 | 0.5475 | 0.6057 | 0.6241 | 0.6650 | 0.4786 | 0.4210 | 0.3177 | 0.2480 |
| MiniLM semantic baseline (frozen) | 0.6987 | 0.5369 | 0.8645 | 0.6125 | 0.6624 | 0.6952 | 0.7055 | 0.8150 | 0.6494 | 0.3875 | 0.1355 | 0.4549 |
| **Proposed heavy-S2 framework (phase-5 upgrade)** | **0.8080** | 0.7072 | 0.7485 | 0.8389 | **0.7273** | **0.7896** | **0.8092** | **0.8636** | **0.7522** | 0.1611 | 0.2515 | **0.5799** |

Confusion matrix (phase-4.1 frozen): TP 638, FP 244, TN 1,730, FN 388.
Confusion matrix (phase-5 upgrade): TP 768, FP 318, TN 1,656, FN 258.

## 4.5 Additional analysis

- **Subgroup performance (length terciles of suspicious text, frozen framework test):**
  short (n=945): F1 = 0.717; medium (n=973): F1 = 0.700; long (n=1,082): **F1 = 0.563** —
  performance degrades systematically on longer suspicious texts (error pattern: recall 0.567,
  precision 0.560 on long vs 0.641/0.813 on short).
- **Class-specific performance:** recall (0.62) < specificity (0.88) under the frozen precision-first
  operating point; the upgrade rebalances (0.75/0.84) while raising F1 and MCC.
- **Distribution shift, train↔test:** KS test on suspicious-text char length: statistic 0.0267,
  **p = 0.172** (no detectable length shift); class-rate 33.67% → 34.20% (Δ = +0.53pp). The
  premise-grouped protocol leaves train/test statistically near-identical on these marginals.
- **Potential leakage:** 1 premise crossed the raw val/test boundary (purged; documented);
  allocator-verified disjointness holds for all downstream partitions.
- **Error patterns:** under the frozen OP, false negatives (388) dominate false positives (244);
  in the upgrade the targeted recall floor shifts this balance (FN 258 < FP 318) at higher F1.
