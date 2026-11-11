# Phase 2.3.3 Comprehensive Correction and Verification Log

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.3 (Final Dataset, Routing, and Integrity Verification)  
**Document Version:** Version 3.4  
**Date:** March 2026  
**Status:** FULL AUDIT CERTIFIED (Empirical Ground-Truth Verified)

---

## 1. Executive Summary

This correction log documents the definitive resolution of all remaining dataset-accounting, routing, data-leakage, statistical-validity, and repository-integrity issues identified in the Phase 2.3.2 review. 

All corrections were validated by creating and executing the standalone audit script [`run_dataset_integrity_audit.py`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_3/run_dataset_integrity_audit.py) against the raw files on disk. All numerical claims in this document are directly backed by the machine-readable artifacts generated in [`research_upgrade/audits/phase_2_3_3/results/`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_3/results/).

---

## 2. Itemized Resolution Log

### Issue 1: Dataset Directory Path Discrepancy
- **Identified Problem:** Previous documentation cited paths starting with `data/` (e.g., `data/text_plagiarism_benchmark.csv` and `data/code_plagiarism_benchmark.csv`). Inspection of the filesystem revealed that no `data/` directory exists in the workspace.
- **Physical Disk Evidence:** 
  - `Test-Path "data"` returns `False`.
  - Processed CSV files are located in `datasets/processed/`.
  - Raw datasets are located in `datasets/raw/text_mit_plagiarism/` and `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/`.
- **Resolution:** All file paths across all Version 3.4 deliverables have been updated to reflect the true filesystem structure.

---

### Issue 2: Text Dataset Accounting & Arithmetic Reconciliation
- **Identified Problem:** In Version 3.3, the text dataset was described as having 19,666 total rows, with 10,500 active pairs and "9,166 reserve pairs" alongside "6 purged cross-split rows", which failed exact arithmetic reconciliation ($10,500 + 9,166 + 6 = 19,672 \neq 19,666$).
- **Empirical Ground-Truth Audit:**
  - Total raw rows in `datasets/processed/text_plagiarism_benchmark.csv`: exactly **19,666 rows** (`val`: 9,842; `test`: 9,824).
  - Purged cross-split contamination: exactly **6 rows** (premise `"A dog catching a Frisbee."`: 3 in `val`, 3 in `test`).
  - Purged duplicate rows: exactly **2 rows** (purged `TXT_002505` and `TXT_003942`, retaining first instances `TXT_002504` and `TXT_003940`).
  - Active experimental partitions: exactly **10,500 rows** ($D_{\text{train}}$: 4,000; $D_{\text{val}}$: 1,500; $D_{\text{cal-prob}}$: 1,000; $D_{\text{cal-conf}}$: 1,000; $D_{\text{test}}$: 3,000).
  - Retained Development Reserve: $9,842 - 7,500 - 3 - 2 =$ **2,337 rows**.
  - Retained Test Reserve: $9,824 - 3,000 - 3 - 0 =$ **6,821 rows**.
  - Total Retained Reserves: $2,337 + 6,821 =$ **9,158 rows**.
- **Reconciliation Arithmetic:**
  $$10,500 \text{ (active)} + 6 \text{ (purged crossing)} + 2 \text{ (purged dupes)} + 2,337 \text{ (dev)} + 6,821 \text{ (test)} = 19,666 \text{ rows}$$
- **Resolution:** Full arithmetic reconciliation verified and codified in `dataset_accounting_and_partition_audit_v3_4.md` and saved in `text_dataset_accounting_summary.csv`.

---

### Issue 3: Duplicate Pair & Directionality Audit
- **Identified Problem:** Previous audits noted the existence of duplicates but did not systematically test for reversed duplicate pairs `(B, A)` vs `(A, B)` or check whitespace/case normalization.
- **Empirical Ground-Truth Audit:**
  - **Text:** 
    - Exact ordered duplicate rows: **4 rows** (2 pair groups: `TXT_002504`/`05` and `TXT_003940`/`42`).
    - Reversed duplicate pairs: **0 pairs**.
    - Case/whitespace normalized duplicates: **4 rows** (identical to exact duplicates; no additional formatting duplicates).
  - **Code:** 
    - Exact duplicate rows: **24 rows** (12 pair groups: `case-01`: 7, `case-02`: 7, `case-03`: 6, `case-06`: 4).
    - Reversed duplicate pairs: **0 pairs**.
- **Resolution:** Complete duplicate ledgers generated and stored in `text_duplicate_audit.csv` and `code_source_file_overlap_report.json`. Data loaders explicitly purge redundant occurrences.

---

### Issue 4: Five-Partition Isolation & Full Overlap Matrix
- **Identified Problem:** Previous documentation verified validation vs test isolation but lacked a comprehensive $5 \times 5$ cross-partition overlap matrix proving mutual independence across all active subsets.
- **Empirical Ground-Truth Audit:**
  - Using deterministic group-premise SHA-256 hashing on clean data, all unique premises were partitioned into $D_{\text{train}}$ (1,349 premises), $D_{\text{val}}$ (508 premises), $D_{\text{cal-prob}}$ (336 premises), $D_{\text{cal-conf}}$ (334 premises), $D_{\text{test}}$ (1,015 premises), $D_{\text{dev-reserve}}$ (791 premises), and $D_{\text{test-reserve}}$ (2,307 premises).
  - Cross-partition premise overlap between **any pair of different partitions is strictly 0**.
- **Resolution:** The full $7 \times 7$ overlap matrix is published in `text_partition_overlap_matrix.csv`.

---

### Issue 5: Structural Impossibility of Platt Scaling on IR-Plag `case-05`
- **Identified Problem:** Previous documentation proposed evaluating `case-05` using 5-fold cross-fitting Platt scaling without auditing class balance across base program identities.
- **Empirical Ground-Truth Audit:**
  - `case-05` contains 107 total pairs across 16 program identities (1 original reference + 15 non-plagiarized).
  - All 53 positive pairs involve `original/Main.java`.
  - In a 5-fold group split:
    - **Fold 1** (holding out `original`): $N_{\text{pos}}^{\text{train}} = 0$, $N_{\text{neg}}^{\text{train}} = 39$, $N_{\text{pos}}^{\text{test}} = 53$, $N_{\text{neg}}^{\text{test}} = 15$. Logistic regression maximum likelihood diverges to $-\infty$.
    - **Folds 2–5** (holding out non-plag): $N_{\text{pos}}^{\text{train}} = 53$, $N_{\text{neg}}^{\text{train}} \approx 40$, $N_{\text{pos}}^{\text{test}} = 0$, $N_{\text{neg}}^{\text{test}} \approx 14$. Held-out test fold has zero positive events; ROC-AUC and calibration error are undefined.
- **Resolution:** Forced cross-fitting Platt scaling on `case-05` is formally rejected. Code is evaluated via exploratory raw margin thresholding $z(\mathbf{x})$, with conformal guarantees explicitly disclaimed.

---

### Issue 6: Conformal Escalation Condition Correction
- **Identified Problem:** Previous drafts defined escalation loosely as $s_{\text{cand}} \ge \theta_{\text{stage1}}$, failing to exclude direct high-risk bypass pairs from the conformal calibration set.
- **Mathematical Correction:**
  - Escalation slice: strictly $\theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}}$.
  - Direct high-risk bypass: $s_{\text{cand}}(x) \ge \theta_{\text{high}}$ bypasses Stage 2 and does NOT enter the conformal calibration set, ensuring the calibration population matches the deployed heavy-model execution slice.
  - Finite-sample attainability lower bound $n \ge \lceil(1-\epsilon)/\epsilon\rceil$ enforced; $\hat{q}_{1-\epsilon} = \infty$ if sample size is insufficient.
- **Resolution:** Formulations and pseudocode updated across `mathematical_formulations_v3_4.md` and `algorithm_pseudocode_v3_4.md`.

---

### Issue 7: Four-State Output Schema Harmonization
- **Identified Problem:** Discrepancies in output states and human adjudication roles across previous documents.
- **Methodological Correction:**
  - Standardized 4-state vocabulary: `LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, `ABSTAIN_UNRESOLVED`.
  - Text utilizes all 4 states; code utilizes 3 states (`ABSTAIN_UNRESOLVED` omitted).
  - Four-Level Adjudication Model enforced: machine learning outputs empirical statistical evidence; humans make institutional decisions.
- **Resolution:** Codified across all 11 Version 3.4 deliverables.

---

### Issue 8: Repository Integrity & Resolution of the 18 vs 28 File Discrepancy
- **Identified Problem:** One previous report mentioned 18 files in `src/`, while another checklist mentioned 28 files.
- **Empirical Ground-Truth Audit:**
  - Root of `src/` contains **17 Python files**.
  - Subpackage `src/code/` contains **6 Python files** (`esplag_detector.py`, `esplag_tokenizer.py`, `penalty_similarity.py`, `rkr_gst.py`, `token_indexer.py`, `__init__.py`).
  - Subpackage `src/text/` contains **5 Python files** (`hybrid_similarity.py`, `lexical_detector.py`, `semantic_detector.py`, `staged_hybrid_pipeline.py`, `__init__.py`).
  - Total Python files in `src/`: $17 + 6 + 5 =$ **28 files**.
  - The "18 files" figure was an artifact of an abstract template listing placeholder file names that never existed on disk.
  - Historical source code integrity is classified as `NOT_VERIFIABLE` due to the lack of pre-Phase-2 version control.
  - Current SHA-256 baseline manifest established for all 28 files and published in `repository_sha256_inventory.csv`.
- **Resolution:** Discrepancy fully resolved and documented.

