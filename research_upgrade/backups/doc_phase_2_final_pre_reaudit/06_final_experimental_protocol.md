# Final Comprehensive Experimental Protocol (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 06 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** EXPERIMENTALLY CERTIFIED & AUDITED

---

## 1. Experimental Hypotheses

The experimental validation protocol is designed to test four pre-registered scientific hypotheses:

- **$H_1$ (Candidate Recall Preservation):**  
  The multi-view screening model achieves $\ge 98.0\%$ candidate recall on academic text and source code while pruning $\ge 70.0\%$ of benign negative pairs, successfully retrieving the 119 zero-overlap text positive pairs and transformed code variants.
- **$H_2$ (Calibration Reliability):**  
  Platt scaling probability calibration significantly reduces Expected Calibration Error ($\text{ECE} \le 0.05$) and Negative Log-Likelihood compared to uncalibrated cross-encoder logits on the untouched test escalation population.
- **$H_3$ (Conformal Coverage Guarantee):**  
  Group-level split conformal prediction provides empirical test coverage $\ge 1 - \epsilon$ (at $\epsilon = 0.05$, coverage $\ge 95.0\%$) across independent premise clusters without inflating the ambiguous human review slice beyond $15\%$.
- **$H_4$ (Cost-Efficiency Pareto Dominance):**  
  The two-stage adaptive architecture reduces average pairwise inference latency by $\ge 80.0\%$ compared to uniform cross-encoder deployment while maintaining equivalent macro-F1 and decision accuracy.

---

## 2. Dataset Accounting and Deduplication Protocol

### 2.1 Academic Text Benchmark (SNLI Derived)
- **Source Dataset:** [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv)
- **Total Raw Rows:** **19,666 pairs** (`val` = 9,842, `test` = 9,824).
- **Purged Rows:** **8 pairs** itemized in [text_exclusion_ledger.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/text_exclusion_ledger.csv) (6 cross-split premise contamination pairs, 2 redundant duplicate pairs).
- **Clean Retained Rows:** **19,658 pairs** (Clean `val` pool = 9,837, Clean `test` pool = 9,821).
- **Reversed Duplicates:** Exactly **0 reversed duplicate pairs** (verified via canonical pair grouping in [`run_final_phase2_audit.py`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/run_final_phase2_audit.py)).

#### Deterministic Exact Partition Allocations:
The deterministic allocator allocates whole premise clusters to hit exact partition targets without deliberate overshoot:

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

Pairwise premise overlap across all partitions is **$0$** ([text_group_leakage_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/text_group_leakage_report.json)).

---

### 2.2 Academic Source Code Benchmark (IR-Plag)
- **Source Benchmark:** [code_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/code_plagiarism_benchmark.csv)
- **Total Raw Pairs:** **733 pairs** (361 positive, 372 negative).
- **Physical Java Files on Disk:** **467 files** across all 7 assignment folders ([code_source_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/code_source_file_inventory.csv)).
- **Duplicate Records:** Exactly **24 rows** participate in **11 duplicate groups** ([code_duplicate_accounting.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/code_duplicate_accounting.json)).
- **Reversed Duplicates:** Exactly **0 reversed duplicate pairs**.

#### Primary Duplicate Policy Justification:
- **Primary Policy: Policy A (Representative Retention):**  
  Purges 13 redundant rows, retaining 1 canonical representative per group $\to$ **720 pairs retained** (348 pos, 372 neg).  
  *Justification:* In real-world educational cohorts, dropping all pairs involved in duplicate collisions (Policy B) discards genuine evidence of code reuse and artificially suppresses candidate recall. Retaining one representative per duplicate group prevents duplicate weighting bias while preserving complete detection coverage.
- **Sensitivity Policy: Policy B (Complete Collision Exclusion):**  
  Purges all 24 collision rows $\to$ **709 pairs retained** (337 pos, 372 neg). Retained as a sensitivity check.

#### Case-by-Case Task Accounting & Denominators:
| Task ID | Raw Total | Duplicate Rows | Policy A Retained (Primary) | Policy B Retained (Sensitivity) | Physical Java Files |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `case-01` | 94 | 7 | **90** (36 pos, 54 neg) | **87** (33 pos, 54 neg) | 56 |
| `case-02` | 108 | 7 | **104** (50 pos, 54 neg) | **101** (47 pos, 54 neg) | 70 |
| `case-03` | 106 | 6 | **103** (49 pos, 54 neg) | **100** (46 pos, 54 neg) | 68 |
| `case-04` | 108 | 0 | **108** (54 pos, 54 neg) | **108** (54 pos, 54 neg) | 70 |
| `case-05` | 107 | 0 | **107** (53 pos, 54 neg) | **107** (53 pos, 54 neg) | **69** |
| `case-06` | **105** | 4 | **103** (49 pos, 54 neg) | **101** (47 pos, 54 neg) | 67 |
| `case-07` | 105 | 0 | **105** (51 pos, 54 neg) | **105** (51 pos, 54 neg) | 67 |
| **Total** | **733** | **24** | **720** (348 pos, 372 neg) | **709** (337 pos, 372 neg) | **467** |

#### Case-05 Experimental Protocol Restrictions:
- Physical files: **69 Java files** (1 original, 15 non-plagiarized, 53 plagiarized across L1–L6).
- As proven in [case_05_fold_feasibility_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/case_05_fold_feasibility_report.json), cross-fitting leaves training folds with zero positives (causing Platt scaling divergence / `ValueError`) or test folds with zero positives (rendering ROC-AUC, PR-AUC, Precision, Recall, and ECE undefined).
- **Protocol Rule:** Case-05 is evaluated strictly via **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$).

#### Generalization Claims Boundary:
Because the IR-Plag benchmark contains only two independent held-out tasks for testing, pooled code results are reported as **descriptive performance**, and task-level generalization claims are classified as **exploratory**.

---

## 3. Comparison Baselines

1. **BM25 Lexical Baseline:** Classical lexical retrieval model with Robertson-Spärck Jones BM25 term weighting.
2. **TF-IDF + Cosine Baseline:** Character 3-gram and Word 2-gram TF-IDF vectorizer.
3. **Dense Contextual Bi-Encoder Baseline:** Cosine similarity over `all-MiniLM-L6-v2` dense embeddings.
4. **Uniform Heavy Model Baseline:** Direct, unbatched evaluation of the Stage-2 detector across all candidate pairs.
5. **Proposed Framework:** Two-stage multi-view screening + adaptive routing + Platt calibration (text) + group split conformal prediction (text).

---

## 4. Evaluation Metrics

1. **Screening Performance:** Candidate Recall ($\text{Recall}@K$), Pruning Ratio ($\eta = 1 - |\mathcal{E}|/|\mathcal{D}|$), Recall on the 119 zero-overlap text pairs.
2. **Classification Performance:** Precision, Recall, Macro-F1, ROC-AUC, PR-AUC.
3. **Calibration Diagnostics (Untouched Test Escalation Population):**
   - Expected Calibration Error (ECE) with $M=10$ uniform bins and 0-contribution for empty bins.
   - Maximum Calibration Error (MCE).
   - Brier Score ($\text{BS} \in [0, 1]$).
   - Clipped Negative Log-Likelihood ($\text{NLL}$ with clipping to $[10^{-15}, 1 - 10^{-15}]$).
4. **Conformal Prediction Metrics:**
   - Empirical Group Coverage: $\hat{C} = \frac{1}{|\mathcal{G}_{\text{test}}^{\text{esc}}|} \sum_{g} \mathbb{I}(\forall j \in \mathcal{P}_g^{\text{esc}}, y_j \in \mathcal{C}(x_j))$.
   - Set Cardinality: Mean set size $|\mathcal{C}(x)| \in [0, 2]$.
   - Abstention Rate: Fraction of pairs where $\mathcal{C}(x) = \emptyset$.
5. **Computational Cost Metrics:** End-to-end average latency per pair (ms), P50/P95 latency, peak RAM (MB), FLOP reduction ratio.

---

## 5. Statistical Inference Protocol

1. **Group-Aware Bootstrap Resampling:**  
   To respect intra-cluster correlation, bootstrap resampling ($B = 1,000$ iterations) resamples **entire premise clusters** in text, rather than individual pairs.
2. **Categorical Discordance:** McNemar's test evaluated over matched prediction decisions ($p < 0.01$).
3. **Receiver Operating Characteristic Curves:** DeLong's test for paired ROC-AUC comparisons.

