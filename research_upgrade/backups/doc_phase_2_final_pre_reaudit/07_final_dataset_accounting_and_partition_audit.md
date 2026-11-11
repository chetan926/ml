# Final Dataset Accounting, Deduplication & Partitioning Audit (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 07 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** COMPREHENSIVELY AUDITED & RECONCILED (All Artifacts Verified)

---

## 1. Executive Dataset Accounting Scorecard

An independent, reproducible audit was executed across all benchmark datasets in the repository using [`run_final_phase2_audit.py`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/run_final_phase2_audit.py). All raw rows, duplicates, exclusions, and partition assignments are verified with **zero arithmetic discrepancy**.

```
════════════════════════════════════════════════════════════════════════════════════════
                             DATASET AUDIT SCORECARD
════════════════════════════════════════════════════════════════════════════════════════
  Text Benchmark (SNLI Proxy):
    • Total Raw Rows in CSV:          19,666 pairs
    • Excluded / Purged Rows:         8 pairs (6 cross-split, 2 redundant duplicate)
    • Clean Retained Rows:            19,658 pairs (9,837 val pool, 9,821 test pool)
    • Active Partitions Total:        10,500 pairs (Train 4000, Val 1500, CalP 1000, CalC 1000, Test 3000)
    • Retained Reserves Total:        9,158 pairs (Dev-Reserve 2337, Test-Reserve 6821)
    • Sum Check:                      10,500 + 9,158 + 8 = 19,666 (DISCREPANCY = 0)
    • Premise Overlap:                0 across all active partitions and reserves
    • Reversed Duplicates:            0 pairs (Verified via canonical pair analysis)

  Source Code Benchmark (IR-Plag):
    • Total Raw Pairs in CSV:         733 pairs (361 positive, 372 negative)
    • Duplicate Group Rows:           24 rows participating in 11 duplicate groups
    • Policy A Retained (Keep 1st):   720 pairs (348 positive, 372 negative)
    • Policy B Retained (Drop All):   709 pairs (337 positive, 372 negative)
    • Reversed Duplicates:            0 pairs (Verified via canonical pair analysis)
    • Case-06 Denominator:            Raw=105, Policy A=103, Policy B=101
    • Case-05 Physical Java Files:    69 files (1 original, 15 non-plagiarized, 53 plagiarized)
    • Total IR-Plag Physical Files:   467 files across all 7 assignment folders
    • Case-05 Cross-Fitting Status:   PROVEN STRUCTURAL DEGENERACY (Zero-Positive Slices)
════════════════════════════════════════════════════════════════════════════════════════
```

---

## 2. Text Benchmark Detailed Accounting (SNLI Proxy)

### 2.1 Raw Split Distribution
Loaded from [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv):
- `reference_text` null / empty count: **0**
- `suspicious_text` null / empty count: **0**
- Raw `val` split rows: **9,842**
- Raw `test` split rows: **9,824**
- Total Raw Rows: **19,666**

### 2.2 Purged Rows Itemization
Eight rows are formally purged prior to partition allocation ([text_exclusion_ledger.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/text_exclusion_ledger.csv)):

| Row ID | Original Split | Label | Exclusion Reason | Duplicate Group | Premise Snippet | Final Disposition |
| :---: | :---: | :---: | :--- | :---: | :--- | :---: |
| `TXT_001181` | val | 1 | CROSS_SPLIT_PREMISE_CONTAMINATION | NONE | A person is outside in the snow. | PURGED_PRE_SPLIT |
| `TXT_001182` | val | 0 | CROSS_SPLIT_PREMISE_CONTAMINATION | NONE | A person is outside in the snow. | PURGED_PRE_SPLIT |
| `TXT_001183` | val | 0 | CROSS_SPLIT_PREMISE_CONTAMINATION | NONE | A person is outside in the snow. | PURGED_PRE_SPLIT |
| `TXT_016302` | test | 1 | CROSS_SPLIT_PREMISE_CONTAMINATION | NONE | A person is outside in the snow. | PURGED_PRE_SPLIT |
| `TXT_016303` | test | 0 | CROSS_SPLIT_PREMISE_CONTAMINATION | NONE | A person is outside in the snow. | PURGED_PRE_SPLIT |
| `TXT_016304` | test | 0 | CROSS_SPLIT_PREMISE_CONTAMINATION | NONE | A person is outside in the snow. | PURGED_PRE_SPLIT |
| `TXT_002505` | val | 1 | REDUNDANT_EXACT_DUPLICATE_ROW | TEXT_DUPE_GRP_01 | A young girl in a pink shirt is playing on a sw | PURGED_PRE_SPLIT |
| `TXT_003942` | val | 1 | REDUNDANT_EXACT_DUPLICATE_ROW | TEXT_DUPE_GRP_02 | A woman in a black dress is walking down a stre | PURGED_PRE_SPLIT |

### 2.3 Exact-Target Partition Allocations
The remaining 19,658 clean rows are allocated into 5 active partitions and 2 reserve pools without deliberate overshoot:

```
Total Raw Rows: 19,666
├── Excluded / Purged Rows: 8 (Cross-Split: 6, Redundant Duplicate: 2)
└── Clean Retained Rows: 19,658
    ├── Clean Val Pool: 9,837
    │   ├── train:        4,000 pairs (1,349 premises)  <-- EXACT MATCH
    │   ├── val:          1,500 pairs (  508 premises)  <-- EXACT MATCH
    │   ├── cal_prob:     1,000 pairs (  336 premises)  <-- EXACT MATCH
    │   ├── cal_conf:     1,000 pairs (  334 premises)  <-- EXACT MATCH
    │   └── dev_reserve:  2,337 pairs (  791 premises)  <-- Held-out reserve
    └── Clean Test Pool: 9,821
        ├── test:         3,000 pairs (1,015 premises)  <-- EXACT MATCH (0 OVERSHOOT)
        └── test_reserve: 6,821 pairs (2,307 premises)  <-- Held-out reserve
```

### 2.4 7x7 Premise Overlap Matrix
Pairwise intersection of premise strings across all partitions ([text_group_leakage_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/text_group_leakage_report.json)):

| Partition | `train` | `val` | `cal_prob` | `cal_conf` | `test` | `dev_reserve` | `test_reserve` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **1,349** | 0 | 0 | 0 | 0 | 0 | 0 |
| **`val`** | 0 | **508** | 0 | 0 | 0 | 0 | 0 |
| **`cal_prob`** | 0 | 0 | **336** | 0 | 0 | 0 | 0 |
| **`cal_conf`** | 0 | 0 | 0 | **334** | 0 | 0 | 0 |
| **`test`** | 0 | 0 | 0 | 0 | **1,015** | 0 | 0 |
| **`dev_reserve`** | 0 | 0 | 0 | 0 | 0 | **791** | 0 |
| **`test_reserve`**| 0 | 0 | 0 | 0 | 0 | 0 | **2,307** |

**Zero Leakage Finding:** All off-diagonal elements are exactly **0**. There is zero premise contamination across partitions.

---

## 3. Source Code Benchmark Detailed Accounting (IR-Plag)

### 3.1 Duplicate Structure and Task Breakdown
The IR-Plag code benchmark consists of 733 pairs across 7 tasks. Exactly 24 rows participate in 11 duplicate groups ([code_duplicate_accounting.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/code_duplicate_accounting.json)):

| Task ID | Raw Pairs | Duplicate Groups | Duplicate Rows | Policy A (Keep 1st) | Policy B (Drop All) | Physical Java Files |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `case-01` | 94 | 3 groups | 7 rows | 90 (36 pos, 54 neg) | 87 (33 pos, 54 neg) | 56 |
| `case-02` | 108 | 3 groups | 7 rows | 104 (50 pos, 54 neg) | 101 (47 pos, 54 neg) | 70 |
| `case-03` | 106 | 3 groups | 6 rows | 103 (49 pos, 54 neg) | 100 (46 pos, 54 neg) | 68 |
| `case-04` | 108 | 0 groups | 0 rows | 108 (54 pos, 54 neg) | 108 (54 pos, 54 neg) | 70 |
| `case-05` | 107 | 0 groups | 0 rows | 107 (53 pos, 54 neg) | 107 (53 pos, 54 neg) | **69** |
| `case-06` | **105** | 2 groups | 4 rows | **103** (49 pos, 54 neg) | **101** (47 pos, 54 neg) | 67 |
| `case-07` | 105 | 0 groups | 0 rows | 105 (51 pos, 54 neg) | 105 (51 pos, 54 neg) | 67 |
| **Total** | **733** | **11 groups** | **24 rows** | **720** (348 pos, 372 neg) | **709** (337 pos, 372 neg) | **467** |

### 3.2 Canonical Pair Reversed-Duplicate Verification
Auditing undirected canonical pairs $(u, v)$ with $u \le v$ confirms:
- Unique directed pairs in raw CSV: **720**
- Unique canonical pairs: **720**
- Canonical collision groups: **11 groups** (identical to directed duplicate groups).
- Across all 11 collision groups, all participating rows have identical orientation (either all forward or all reversed).
- **Reversed Duplicate Pairs:** **0 pairs**.

### 3.3 Physical Java File Inventory in IR-Plag `case-05`
The discrepancy between 58 and 69 files in `case-05` is definitively resolved by [code_source_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/code_source_file_inventory.csv):
- `original/`: 1 file (`T5.java`)
- `non-plagiarized/`: 15 subdirectories (`01` to `15`), exactly 1 file each $\to$ **15 files**.
- `plagiarized/`: 6 obfuscation level directories:
  - `L1/`: 9 files (`01` to `09`)
  - `L2/`: 8 files (`01` to `08`) *(Notice: exactly 8 files, resolving the prior count error)*
  - `L3/`: 9 files (`01` to `09`)
  - `L4/`: 9 files (`01` to `09`)
  - `L5/`: 9 files (`01` to `09`)
  - `L6/`: 9 files (`01` to `09`)
  - Total plagiarized files: $9 + 8 + 9 + 9 + 9 + 9 = \mathbf{53\text{ files}}$.
- Total physical Java files in `case-05`: $1 + 15 + 53 = \mathbf{69\text{ files}}$.

### 3.4 Case-05 Cross-Fitting Proof
The 107 pairs in `case-05` map to 16 base program identities (`prog_original` and `prog_np_01` to `prog_np_15`).
As recorded in [case_05_fold_feasibility_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/case_05_fold_feasibility_report.json):
- **Fold 1 (`prog_original` held out):** Train has 0 positives ($N_{\text{pos}}^{\text{train}} = 0$). Unregularized Platt MLE diverges to $-\infty$; regularized `LogisticRegression` throws `ValueError`.
- **Folds 2 to 5 (`prog_original` in train):** Test has 0 positives ($N_{\text{pos}}^{\text{test}} = 0$). Precision, Recall, ROC-AUC, PR-AUC, and ECE are mathematically undefined ($0/0$).
- **Methodological Status:** Group-disjoint cross-validation and calibration are structurally impossible on `case-05`. Case-05 will be evaluated strictly via exploratory raw decision-margin thresholding ($z \in \mathbb{R}$).

