# Comprehensive Experimental Protocol & Evaluation Methodology (Version 3.6)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Document Version:** Version 3.6  
**Date:** November 10, 2026  
**Status:** EXPERIMENTALLY CERTIFIED & AUDITED (Zero Discrepancies)

---

## 1. Experimental Objectives and Hypotheses

The experimental validation is designed to evaluate four central scientific hypotheses:
1. **$H_1$ (Candidate Recall Preservation):** The multi-view screening model achieves $\ge 98.0\%$ candidate recall on academic text and source code while pruning $\ge 70.0\%$ of benign negative pairs, successfully retrieving the 119 zero-overlap text positive pairs and transformed code variants.
2. **$H_2$ (Uncertainty Calibration Reliability):** Probability calibration (Platt scaling) significantly reduces Expected Calibration Error ($\text{ECE} \le 0.05$) and Negative Log-Likelihood compared to uncalibrated cross-encoder logits on the untouched test escalation population.
3. **$H_3$ (Conformal Coverage Guarantee):** Group-level split conformal prediction provides empirical test coverage $\ge 1 - \epsilon$ (at $\epsilon = 0.05$, coverage $\ge 95.0\%$) across independent premise clusters without inflating the ambiguous human review slice beyond $15\%$.
4. **$H_4$ (Cost-Efficiency Pareto Dominance):** The two-stage adaptive architecture reduces average pairwise inference latency by $\ge 80.0\%$ compared to uniform cross-encoder deployment while maintaining equivalent macro-F1 and decision accuracy.

---

## 2. Dataset Accounting, Deduplication, and Partitioning

### 2.1 Academic Text Benchmark Accounting (SNLI Derived)
- **Source Corpus:** [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv)
- **Total Raw Rows:** **19,666 pairs**
- **Split Breakdown in CSV:** `val` = 9,842 pairs, `test` = 9,824 pairs.
- **Cross-Split Contamination:** Exactly 1 premise string crosses the `val` and `test` splits, affecting **6 pairs** (3 in `val`: `TXT_001181`, `TXT_001182`, `TXT_001183`; 3 in `test`: `TXT_016302`, `TXT_016303`, `TXT_016304`).
- **Duplicate Analysis:** Exactly 4 rows participate in 2 duplicate groups. Under Policy A (retain 1st occurrence), **2 redundant rows** (`TXT_002505`, `TXT_003942`) are purged.
- **Reversed Duplicate Analysis:** Explicit canonical grouping ([text_reversed_pair_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_reversed_pair_audit.csv)) proves **0 reversed duplicate pairs** exist.
- **Total Purged Rows:** $6 + 2 = \mathbf{8\text{ rows}}$ ([text_exclusion_ledger.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_exclusion_ledger.csv)).
- **Clean Retained Rows:** $19,666 - 8 = \mathbf{19,658\text{ rows}}$ (Clean `val` pool = 9,837, Clean `test` pool = 9,821).

#### Deterministic Non-Overshooting Partition Allocations
The deterministic grouping allocator assigns whole premise clusters to hit exact partition targets without deliberate overshoot:

| Partition Name | Target Count | Allocated Count | Unique Premises | Source Pool | Arithmetic Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** | 4,000 | **4,000** | 1,349 | Clean `val` pool | EXACT MATCH |
| **`val`** | 1,500 | **1,500** | 508 | Clean `val` pool | EXACT MATCH |
| **`cal_prob`** | 1,000 | **1,000** | 336 | Clean `val` pool | EXACT MATCH |
| **`cal_conf`** | 1,000 | **1,000** | 334 | Clean `val` pool | EXACT MATCH |
| **`test`** | 3,000 | **3,000** | 1,015 | Clean `test` pool | EXACT MATCH |
| **`dev_reserve`** | Remainder | **2,337** | 791 | Clean `val` pool | HELD-OUT |
| **`test_reserve`**| Remainder | **6,821** | 2,307 | Clean `test` pool | HELD-OUT |
| **Purged Rows** | N/A | **8** | 3 | Excluded | PURGED |
| **Total Sum** | **19,666** | **19,666** | **6,643** | All Pools | **PERFECT SUM ($0$ DIFF)** |

Every single row ID from `TXT_000000` to `TXT_019665` is mapped in [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv).  
Pairwise premise overlap across all partitions is **$0$** ([text_partition_overlap_matrix.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_overlap_matrix.csv)).

---

### 2.2 Academic Source Code Benchmark Accounting (IR-Plag)
- **Source Benchmark:** [code_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/code_plagiarism_benchmark.csv)
- **Total Raw Pairs:** **733 pairs** (361 positive, 372 negative).
- **Physical Java Files on Disk:** **467 files** across all 7 assignment folders.
- **Duplicate Analysis:** Exactly **24 rows** participate in **11 duplicate groups** ([code_duplicate_ledger.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_duplicate_ledger.csv)).
  - **Policy A (Representative Retention):** Retains 1st occurrence, purging 13 redundant rows $\to$ **720 pairs retained** (348 pos, 372 neg).
  - **Policy B (Complete Collision Exclusion):** Purges all 24 collision rows $\to$ **709 pairs retained** (337 pos, 372 neg).
- **Reversed Duplicate Analysis:** Canonical pair audit ([code_reversed_pair_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_reversed_pair_audit.csv)) proves **0 reversed duplicate pairs** exist.
- **Task Accounting Reconciliation ([code_task_accounting.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_task_accounting.csv)):**

| Task ID | Raw Total | Raw Pos | Raw Neg | Dupe Rows | Policy A Retained | Policy B Retained | Physical Java Files |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `case-01` | 94 | 40 | 54 | 7 | 90 (36 pos, 54 neg) | 87 (33 pos, 54 neg) | 56 |
| `case-02` | 108 | 54 | 54 | 7 | 104 (50 pos, 54 neg) | 101 (47 pos, 54 neg) | 70 |
| `case-03` | 106 | 52 | 54 | 6 | 103 (49 pos, 54 neg) | 100 (46 pos, 54 neg) | 68 |
| `case-04` | 108 | 54 | 54 | 0 | 108 (54 pos, 54 neg) | 108 (54 pos, 54 neg) | 70 |
| `case-05` | 107 | 53 | 54 | 0 | 107 (53 pos, 54 neg) | 107 (53 pos, 54 neg) | **69** |
| `case-06` | **105** | 51 | 54 | 4 | **103** (49 pos, 54 neg) | **101** (47 pos, 54 neg) | 67 |
| `case-07` | 105 | 51 | 54 | 0 | 105 (51 pos, 54 neg) | 105 (51 pos, 54 neg) | 67 |
| **Total** | **733** | **361** | **372** | **24** | **720** (348 pos, 372 neg) | **709** (337 pos, 372 neg) | **467** |

#### Resolution of the Case-06 Denominator Discrepancy:
- In `case-06`, the raw pair count is **105**.
- Exactly 4 rows participate in 2 duplicate groups (`CODE_00549`/`CODE_00557` and `CODE_00551`/`CODE_00559`), all with label $=1$.
- Under Policy A, 2 redundant rows are removed $\to$ denominator is **103**.
- Under Policy B, all 4 collision rows are removed $\to$ denominator is **101**.
All future reports must explicitly state which denominator is being evaluated.

#### Case-05 Experimental Protocol Restrictions:
- Ground-truth physical files: **69 Java files** ([case_05_java_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_java_file_inventory.csv)).
- Pairs: **107 pairs** (53 positive, 54 negative).
- Base Program Identities: 16 (`prog_original`, `prog_np_01` to `prog_np_15`).
- As proven in Section 4 of [four_contributions_formal_design_v3_6.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_5/four_contributions_formal_design_v3_6.md), cross-fitting leaves either training or test folds with 0 positive pairs.
- **Protocol Mandate:** Case-05 will NOT be subjected to probability calibration or conformal cross-fitting. It will be reported as an **exploratory structural benchmark using raw decision margins** ($z \in \mathbb{R}$).

---

## 3. Baseline and Comparison Models

1. **BM25 / Lexical Overlap Baseline:** Standard lexical retrieval model using Robertson-Spärck Jones BM25 weighting.
2. **TF-IDF + Cosine Similarity Baseline:** Character 3-gram and Word 2-gram TF-IDF baseline.
3. **Dense Bi-Encoder Baseline:** Uncalibrated cosine similarity from `all-MiniLM-L6-v2`.
4. **Uniform Heavy Cross-Encoder Baseline:** Direct unbatched evaluation of `cross-encoder/ms-marco-MiniLM-L-6-v2` across all pairs (representing state-of-the-art accuracy at maximum computational expense).
5. **Proposed Framework:** Two-stage multi-view screening + adaptive routing + Platt probability calibration + group split conformal prediction.

---

## 4. Evaluation Metrics and Statistical Significance Testing

1. **Screening Performance:**
   - Candidate Recall at pruning thresholds: $\text{Recall}@K = \frac{|\mathcal{P}_{\text{pos}} \cap \text{Top}_K|}{|\mathcal{P}_{\text{pos}}|}$.
   - Pruning Efficiency: $\eta = 1 - \frac{|\mathcal{E}|}{|\mathcal{D}_{\text{total}}|}$.
   - Specific Recall on the 119 Zero-Overlap Positive Text Pairs.
2. **Classification Performance (on escalated slice & overall):**
   - Precision, Recall, Macro-F1, ROC-AUC, PR-AUC.
3. **Calibration Metrics (on untouched test escalation slice):**
   - Expected Calibration Error (ECE) with $M=10$ uniform bins and 0-contribution for empty bins.
   - Maximum Calibration Error (MCE).
   - Brier Score.
   - Negative Log-Likelihood (NLL) clipped to $[10^{-15}, 1 - 10^{-15}]$.
4. **Conformal Prediction Metrics:**
   - Empirical Group Coverage: $\hat{C} = \frac{1}{|\mathcal{G}_{\text{test}}^{\text{esc}}|} \sum_{g} \mathbb{I}(\forall j \in \mathcal{P}_g^{\text{esc}}, y_j \in \mathcal{C}(x_j))$.
   - Set Size Efficiency: Mean cardinality $|\mathcal{C}(x)| \in [0, 2]$.
   - Abstention Rate: Fraction of pairs where $\mathcal{C}(x) = \emptyset$.
5. **Statistical Significance Protocol:**
   - 1,000 paired bootstrap resamples for all confidence intervals.
   - McNemar's test for categorical discordance across classifier predictions ($p < 0.01$).
   - DeLong's test for comparing ROC-AUC curves.

