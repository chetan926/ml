# Dataset Accounting and Partition Integrity Audit (Version 3.5)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.4 (Final Independent Certification and Phase 3 Readiness Gate)  
**Document Version:** Version 3.5  
**Date:** November 10, 2026  
**Status:** INDEPENDENTLY AUDITED & CERTIFIED (Reproducible Artifacts Verified)

---

## 1. Executive Summary and Audit Scope

This document provides exhaustive, row-level empirical accounting of all benchmark datasets utilized in this research project:
1. **Academic Text Benchmark:** `datasets/processed/text_plagiarism_benchmark.csv` (19,666 raw rows).
2. **Academic Source Code Benchmark:** `datasets/processed/code_plagiarism_benchmark.csv` (733 candidate pairs) and underlying Java files in `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` (467 files).

All figures, counts, duplicate IDs, and overlap matrices were computed via the reproducible audit script [`run_phase_2_3_4_audit.py`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/run_phase_2_3_4_audit.py) and verified against machine-readable CSV/JSON ledgers in `research_upgrade/audits/phase_2_3_4/`.

---

## 2. Text Plagiarism Benchmark Empirical Accounting

### 2.1 File Characteristics & Data Integrity
- **Physical Disk Path:** `datasets/processed/text_plagiarism_benchmark.csv`
- **File Size:** 4,321,489 bytes (4.12 MB)
- **Total Raw Rows:** Exactly **19,666 rows**
- **Null / Empty Fields:** Exactly 0 null or whitespace-only entries across `reference_text` and `suspicious_text`.
- **Underlying Source Splits:** 
  - `snli_val.parquet`: 9,842 rows (labeled `split == 'val'`).
  - `snli_test.parquet`: 9,824 rows (labeled `split == 'test'`).
- **Class Balance:**
  - Plagiarized ($y=1$): 6,554 pairs (33.33%).
  - Non-plagiarized ($y=0$): 13,112 pairs (66.67%).
  - Natural 1 : 2 class ratio preserved.
- **Unique Texts:** 6,641 unique reference texts (premises); 19,112 unique suspicious texts (hypotheses).

---

### 2.2 Duplicate Pair Analysis
An exhaustive pairwise comparison across all 19,666 text pairs established:
1. **Exact Ordered Duplicates:** Exactly **4 duplicate rows (2 pair groups)**:
   - **Group 1 (`TEXT_DUPE_GRP_01`):** Row `TXT_002504` and `TXT_002505` (`"Two children play on a sidewalk."` vs `"Two children playing outside on a sidewalk."`, $y=1$, both in `val`).
   - **Group 2 (`TEXT_DUPE_GRP_02`):** Row `TXT_003940` and `TXT_003942` (`"A man in a black jacket and a black beanie is holding a skateboard."` vs `"A man holding a skateboard."`, $y=1$, both in `val`).
2. **Reversed Duplicate Pairs:** Exactly **0 reversed duplicate pairs** `(B, A)` vs `(A, B)`.
3. **Whitespace / Case Normalized Duplicates:** Exactly **4 rows** (identical to the exact ordered duplicates; no additional formatting duplicates).
4. **Ingestion Resolution:** The data loader retains the first instance (`TXT_002504` and `TXT_003940`) and purges the 2 redundant duplicate rows (`TXT_002505` and `TXT_003942`).

---

### 2.3 Cross-Split Contamination Audit (Premise Leakage)
A join across all unique premises in `snli_val` (3,319 unique) and `snli_test` (3,323 unique) identified **exactly one (1) shared premise sentence crossing the split boundary**:
- **Shared Premise String:** `"A dog catching a Frisbee."`
- **Affected Rows (6 total rows across 2 splits):**
  - **In `snli_val` (3 rows):**
    - `TXT_001181` ($y=1$, hypothesis: `"A dog is playing with a frisbee."`)
    - `TXT_001182` ($y=0$, hypothesis: `"A cat sits on a couch."`)
    - `TXT_001183` ($y=0$, hypothesis: `"A dog is sleeping."`)
  - **In `snli_test` (3 rows):**
    - `TXT_016302` ($y=1$, hypothesis: `"A dog jumps for a frisbee."`)
    - `TXT_016303` ($y=0$, hypothesis: `"A cat is climbing a tree."`)
    - `TXT_016304` ($y=0$, hypothesis: `"A dog is eating food."`)
- **Zero Collision Overlap:** The 6 contamination rows share zero overlap with the 4 duplicate rows.
- **Mandatory Leakage Resolution:** All 6 rows associated with this premise are explicitly purged by premise ID in the ingestion loader, guaranteeing strictly 0.0% cross-split leakage.

---

### 2.4 Final Text Accounting Reconciliation

The complete text dataset accounts for every single row with zero discrepancy:

$$\begin{aligned}
\text{Total Raw Rows} &= 19,666 \\
\text{Active Experimental Partitions} &= 10,501 \quad (D_{\text{train}}: 4000, D_{\text{val}}: 1500, D_{\text{cal-prob}}: 1000, D_{\text{cal-conf}}: 1000, D_{\text{test}}: 3001) \\
\text{Purged Cross-Split Contamination} &= 6 \quad (\text{Premise: "A dog catching a Frisbee."}) \\
\text{Purged Redundant Duplicate Rows} &= 2 \quad (\text{Rows: } \text{TXT\_002505}, \text{TXT\_003942}) \\
\text{Retained Development Reserve} &= 2,337 \quad (9842 - 7500 - 3 - 2) \\
\text{Retained Test Reserve} &= 6,820 \quad (9824 - 3001 - 3 - 0) \\
\text{Total Retained Reserves} &= 9,157 \quad (2,337 + 6,820)
\end{aligned}$$

**Arithmetic Sum Verification:**
$$10,501 + 6 + 2 + 2,337 + 6,820 = 19,666 \quad \implies \quad \mathbf{RECONCILIATION: PASS}$$

---

### 2.5 Five-Partition Premise Independence Overlap Matrix

The deterministic premise hash splitter maps unique premises across active partitions and reserves with **strictly zero cross-partition overlap**:

| Partition Name | Total Rows | Unique Premises | Train Overlap | Val Overlap | Cal-Prob Overlap | Cal-Conf Overlap | Test Overlap | Dev Reserve Overlap | Test Reserve Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train ($D_{\text{train}}$)** | 4,000 | 1,349 | **1,349** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Val ($D_{\text{val}}$)** | 1,500 | 508 | 0 | **508** | 0 | 0 | 0 | 0 | 0 |
| **Cal-Prob ($D_{\text{cal-prob}}$)** | 1,000 | 336 | 0 | 0 | **336** | 0 | 0 | 0 | 0 |
| **Cal-Conf ($D_{\text{cal-conf}}$)** | 1,000 | 334 | 0 | 0 | 0 | **334** | 0 | 0 | 0 |
| **Test ($D_{\text{test}}$)** | 3,001 | 1,015 | 0 | 0 | 0 | 0 | **1,015** | 0 | 0 |
| **Dev Reserve** | 2,337 | 791 | 0 | 0 | 0 | 0 | 0 | **791** | 0 |
| **Test Reserve** | 6,820 | 2,307 | 0 | 0 | 0 | 0 | 0 | 0 | **2,307** |

---

## 3. Academic Source Code Benchmark Accounting (IR-Plag)

### 3.1 Corpus Breakdown & Denominator Resolution Across Tasks

Audited directly against `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` and `datasets/processed/code_plagiarism_benchmark.csv`:

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
- **Raw Pairs ($N=105$ for `case-06`, $N=733$ total):** Historical baseline in `src/evaluation.py` evaluated on raw pairs without deduplication.
- **Policy A ($N=103$ for `case-06`, $N=720$ total):** Standard deduplication retaining the first instance of each duplicate group.
- **Policy B ($N=101$ for `case-06`, $N=709$ total):** Aggressive collision exclusion dropping all rows in colliding pairs.

---

### 3.2 Case-05 Group Structure & Cross-Fitting Degeneracy Audit

The group structure of `case-05` comprises **16 base program identities** (1 original reference implementation `original` and 15 independent non-plagiarized implementations `non-plag-01` to `non-plag-15`).

All 53 plagiarized pairs involve comparisons against `original/Main.java`. Simulating 5-fold cross-validation grouping by base program identities yields:

| Fold # | Assigned Held-Out Groups | Positives in Training ($N_{\text{pos}}^{\text{train}}$) | Negatives in Training ($N_{\text{neg}}^{\text{train}}$) | Positives in Held-Out ($N_{\text{pos}}^{\text{test}}$) | Negatives in Held-Out ($N_{\text{neg}}^{\text{test}}$) | Optimization Solver Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** | `original`, `non-plag-01`, `non-plag-02` | **0** | 39 | **53** | 15 | **DIVERGES TO $-\infty$ (Zero Positives in Training)** |
| **Fold 2** | `non-plag-03`, `non-plag-04`, `non-plag-05` | 53 | 40 | **0** | 14 | ROC-AUC / ECE Undefined (Zero Test Positives) |
| **Fold 3** | `non-plag-06`, `non-plag-07`, `non-plag-08` | 53 | 39 | **0** | 15 | ROC-AUC / ECE Undefined (Zero Test Positives) |
| **Fold 4** | `non-plag-09`, `non-plag-10`, `non-plag-11` | 53 | 39 | **0** | 15 | ROC-AUC / ECE Undefined (Zero Test Positives) |
| **Fold 5** | `non-plag-12`, `non-plag-13`, `non-plag-14`, `non-plag-15` | 53 | 42 | **0** | 12 | ROC-AUC / ECE Undefined (Zero Test Positives) |

**Empirical Conclusion:** 5-fold cross-fitting Platt scaling on `case-05` is mathematically degenerate. In Fold 1, the maximum likelihood loss is strictly decreasing as $B \to -\infty$, causing optimization divergence. In Folds 2–5, the test fold contains zero positive pairs, making precision, recall, and calibration error undefined. Code evaluation is therefore restricted to exploratory raw decision margin thresholding $z(\mathbf{x})$, with conformal prediction formally disclaimed.

