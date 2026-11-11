# Experimental Benchmarking and Verification Protocol (Version 3.5)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.4 (Final Independent Certification and Phase 3 Readiness Gate)  
**Document Version:** Version 3.5  
**Date:** November 10, 2026  
**Status:** INDEPENDENTLY AUDITED & CERTIFIED (Reproducible Artifacts Verified)

---

## 1. Experimental Objectives and Benchmarking Scope

The experimental protocol evaluates the four core research contributions across academic text and source code modalities:
1. **Candidate Recall vs. Pruning Efficiency:** Evaluating whether Stage-1 multi-view screening achieves $\ge 98\%$ candidate recall while pruning $\ge 80\%$ of unplagiarized pairs.
2. **Zero-Overlap & Transformation Robustness:** Specifically evaluating recovery on the 119 zero-overlap positive text pairs and L1–L6 obfuscated Java code pairs.
3. **Probability Calibration & Conformal Coverage:** Assessing empirical coverage ($1 - \epsilon = 0.95$), ECE, and Brier score on escalated test pairs.
4. **End-to-End Latency & Cost Reductions:** Verifying system execution under global computational budget constraints ($C_{\text{total}} \le B$).

---

## 2. Dataset Accounting and Partition Allocation Protocol

### 2.1 Academic Text Benchmark (`datasets/processed/text_plagiarism_benchmark.csv`)
- **Total Raw Rows:** Exactly **19,666 pairs** (9,842 in `snli_val`, 9,824 in `snli_test`).
- **Class Balance:** 6,554 plagiarized (33.33%) and 13,112 non-plagiarized (66.67%).
- **Exclusion Filters:**
  - 6 rows purged due to cross-split premise overlap (`"A dog catching a Frisbee."`: rows `TXT_001181`–`83` and `TXT_016302`–`04`).
  - 2 duplicate rows purged (`TXT_002505`, `TXT_003942`).
- **Partition Allocation (Deterministic Group Premise Hashing):**

| Partition | Source Slice | Pair Count ($N$) | Unique Premises ($G$) | Positives ($y=1$) | Negatives ($y=0$) | Operational Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **$D_{\text{train}}$** | `snli_val` | 4,000 | 1,349 | 1,333 | 2,667 | Base model and feature extractor training |
| **$D_{\text{val}}$** | `snli_val` | 1,500 | 508 | 500 | 1,000 | Threshold optimization $(\theta_{\text{low}}, \theta_{\text{high}})$ |
| **$D_{\text{cal-prob}}$** | `snli_val` | 1,000 | 336 | 333 | 667 | Probability calibrator fitting $(A^*, B^*)$ |
| **$D_{\text{cal-conf}}$** | `snli_val` | 1,000 | 334 | 334 | 666 | Conformal cutoff calibration $\hat{q}_{1-\epsilon}$ |
| **$D_{\text{test}}$** | `snli_test` | 3,001 | 1,015 | 1,000 | 2,001 | Final untouched statistical evaluation |
| **Active Total** | Both | **10,501** | **3,542** | **3,500** | **7,001** | Primary experimental evaluation |
| **Dev Reserve** | `snli_val` | 2,337 | 791 | 778 | 1,559 | Preserved development reserve |
| **Test Reserve** | `snli_test` | 6,820 | 2,307 | 2,274 | 4,546 | Preserved out-of-distribution reserve |

*Reconciliation Arithmetic:* $10,501 + 6 \text{ (purged)} + 2 \text{ (purged)} + 2,337 + 6,820 = 19,666$ rows.

### 2.2 Complete Partition Overlap Matrix (Premise Independence)
The deterministic premise hash splitter guarantees zero leakage across all subsets:

```
┌────────────────┬────────┬───────┬──────────┬──────────┬───────┬─────────────┬──────────────┐
│ Partition      │ Train  │ Val   │ Cal-Prob │ Cal-Conf │ Test  │ Dev Reserve │ Test Reserve │
├────────────────┼────────┼───────┼──────────┼──────────┼───────┼─────────────┼──────────────┤
│ Train          │ 1,349  │   0   │    0     │    0     │   0   │      0      │      0       │
│ Val            │   0    │  508  │    0     │    0     │   0   │      0      │      0       │
│ Cal-Prob       │   0    │   0   │   336    │    0     │   0   │      0      │      0       │
│ Cal-Conf       │   0    │   0   │    0     │   334    │   0   │      0      │      0       │
│ Test           │   0    │   0   │    0     │    0     │ 1,015 │      0      │      0       │
│ Dev Reserve    │   0    │   0   │    0     │    0     │   0   │     791     │      0       │
│ Test Reserve   │   0    │   0   │    0     │    0     │   0   │      0      │    2,307     │
└────────────────┴────────┴───────┴──────────┴──────────┴───────┴─────────────┴──────────────┘
```

---

## 3. Academic Source Code Benchmark Protocol (IR-Plag)

### 3.1 Resolving the Denominator Issue Across Tasks
The IR-Plag corpus contains 467 physical `.java` files and 733 comparison pairs in `datasets/processed/code_plagiarism_benchmark.csv`. Due to duplicate submissions across levels, the evaluation protocol establishes exact denominators under both raw and deduplicated conditions:

| Task ID | Raw Pairs | Duplicate Rows | Policy A (Keep 1st) | Policy A Pos/Neg | Policy B (Drop All) | Policy B Pos/Neg | Physical Files |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`case-01`** | 94 | 7 | 90 | 36 / 54 | 87 | 33 / 54 | 56 |
| **`case-02`** | 108 | 7 | 104 | 50 / 54 | 101 | 47 / 54 | 70 |
| **`case-03`** | 106 | 6 | 103 | 49 / 54 | 100 | 46 / 54 | 68 |
| **`case-04`** | 108 | 0 | 108 | 54 / 54 | 108 | 54 / 54 | 70 |
| **`case-05`** | 107 | 0 | 107 | 53 / 54 | 107 | 53 / 54 | 69 |
| **`case-06`** | 105 | 4 | **103** | 49 / 54 | **101** | 47 / 54 | 67 |
| **`case-07`** | 105 | 0 | **105** | 51 / 54 | **105** | 51 / 54 | 67 |
| **Total** | **733** | **24** | **720** | **342 / 378** | **709** | **331 / 378** | **467** |

**Clarification of Denominators:**
- The historical baseline in `src/evaluation.py` evaluated on raw pairs ($N = 105$ for `case-06`, total $N = 210$ for `case-06` + `case-07`).
- When deduplication keeping the first instance is enforced (Policy A), `case-06` contains **103 unique evaluation pairs**.
- When aggressive collision exclusion is enforced (Policy B), `case-06` contains **101 non-collision pairs**.
- Both denominators are published in `code_task_accounting.csv` to ensure reproducibility.

### 3.2 Task-Disjoint Protocol and Limitations
The 7 tasks represent independent programming problems:
- **`case-01` to `case-05`:** Used for exploratory feature tuning and method design.
- **`case-06` and `case-07`:** Held out as independent test assignments.
- **Strict Prohibition on Pooling:** Because tasks involve completely distinct algorithms and data structures, performance on `case-06` and `case-07` **must be reported separately**. Pooled statistics are descriptive only.

### 3.3 Degeneracy Proof on `case-05`
In `case-05`, all 53 plagiarized pairs originate from `original/Main.java`. Holding out this file in a 5-fold group split creates a training fold with $N_{\text{pos}}^{\text{train}} = 0$, causing Platt scaling solvers to diverge to $-\infty$. Code pairs are therefore evaluated via exploratory raw margin thresholding $z(\mathbf{x})$, with conformal guarantees formally disclaimed.

---

## 4. Statistical Significance and Hypothesis Testing Protocol

### 4.1 Paired Binary Classifier Comparisons
To compare the proposed framework against baseline models (TF-IDF, SBERT, BM25, Jaccard):
1. **Contingency Matrix Construction:** Construct $2 \times 2$ table of discordant pairs $(b, c)$.
2. **Exact Binomial McNemar Test:** Compute two-sided exact p-value:
   $$p = \min\left(1, \, 2 \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} 0.5^{b + c}\right)$$
3. **Clustering Limitation Notice:** Because pairs sharing a premise exhibit intracluster correlation, McNemar p-values on text are reported with the caveat that they reflect sample discordance rather than cluster-robust statistical hypothesis testing.

### 4.2 Group-Level Bootstrap Confidence Intervals
All reported metrics (Accuracy, Precision, Recall, F1, Candidate Recall, ECE) must include 95% bootstrap confidence intervals computed over 1,000 resamples **clustered at the premise group level** ($G = 1,015$ in $D_{\text{test}}$).

---

## 5. Comprehensive Systematic Ablation Suite (A01–A12)

| Ablation ID | Description | Experimental Variation | Target Hypothesis |
| :--- | :--- | :--- | :--- |
| **A01** | Single-View vs. Multi-View Text | Char TF-IDF vs. Word vs. SBERT vs. Fused | Multi-view recovers zero-overlap positive pairs |
| **A02** | Single-View vs. Multi-View Code | Token Jaccard vs. AST Diff vs. Metrics vs. Fused | Structural views resist L1–L6 obfuscations |
| **A03** | Threshold Tuning Granularity | Grid sweep $\theta_{\text{low}} \in [0.1, 0.5]$ on $D_{\text{val}}$ | Establishes Pareto frontier (Recall vs. Cost) |
| **A04** | Direct Bypass Impact | With vs. Without $\theta_{\text{high}}$ bypass | Bypass conserves heavy transformer calls without recall loss |
| **A05** | Calibration Method Comparison | Platt Scaling vs. Isotonic Regression | Parametric Platt scaling yields superior ECE on small slices |
| **A06** | Escalation Slice Calibration vs. Full | Calibrating on $\mathcal{E}$ vs. Calibrating on all pairs | Eliminates distribution mismatch between cal and test |
| **A07** | Conformal Non-Conformity Formulation | Worst-case $\alpha_g$ vs. Mean $\alpha_g$ across cluster | Worst-case guarantees conservative coverage |
| **A08** | Conformal Coverage vs. Set Size | Varying $\epsilon \in \{0.01, 0.05, 0.10, 0.20\}$ | Quantifies trade-off between coverage and singleton sets |
| **A09** | Three-Band Routing vs. Binary Cutoff | 4-State Schema vs. Fixed Threshold | Ambiguous routing reduces false positives in human review |
| **A10** | Document Caching vs. Naive Recomputation | Cached representations vs. Pairwise extraction | Confirms $O(M)$ scaling over cohort pairs $\binom{M}{2}$ |
| **A11** | Cold-Start vs. Warm-Cache Latency | First batch execution vs. Warm-cache inference | Establishes production deployment latency bounds |
| **A12** | Evidence Masking Faithfulness | Masking aligned character spans and re-evaluating | Confirms attribution features are faithful to predictions |

