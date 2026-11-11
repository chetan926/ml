# Phase 2.3.4 Final Independent Methodological Audit and Certification Report

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.4 (Final Independent Certification and Phase 3 Readiness Gate)  
**Document Version:** Version 3.5  
**Date:** November 10, 2026  
**Auditor Role:** Independent Senior Machine Learning Researcher, Statistical Methodologist, and Systems Auditor  
**Audit Determination:** READY FOR PHASE 3 IMPLEMENTATION REVIEW  

---

## 1. Executive Summary and Audit Verdict

This document delivers the final, definitive independent audit report for **Phase 2.3.4** of the research project *"An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection"*.

The primary objective of Phase 2.3.4 was to independently verify the Version 3.4 research specification and its supporting audit artifacts, resolve all remaining dataset-accounting and denominator questions, verify five-partition isolation, evaluate case-05 cross-fitting feasibility, and produce a fully synchronized **Version 3.5** documentation suite and reproducible audit package.

All checks were executed via the standalone audit script [`run_phase_2_3_4_audit.py`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/run_phase_2_3_4_audit.py) directly against the physical files on disk. Every finding is backed by machine-readable artifacts stored in [`research_upgrade/audits/phase_2_3_4/`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/).

### Master Audit Findings and Evaluation Status Table

| Audit Dimension | Target Requirement | Evaluation Result | Evidence Reference | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Dataset Paths** | Verify actual filesystem directories | `datasets/processed/` and `datasets/raw/` verified; no `data/` exists | Filesystem audit; `Test-Path "data"` is False | **PASS** |
| **Text Raw Rows** | Count rows in text benchmark | Exactly 19,666 rows (`val`: 9,842; `test`: 9,824) | `text_dataset_audit.json` | **PASS** |
| **Text Accounting** | Reconcile active, reserves, and purges | $10,501 + 6 + 2 + 2,337 + 6,820 = 19,666$ | `text_exclusion_ledger.csv` | **PASS** |
| **Text Duplicates** | Detect ordered & reversed pairs | 4 exact duplicate rows (2 pairs); 0 reversed pairs | `text_duplicate_audit.csv` | **PASS** |
| **Cross-Split Contamination** | Detect shared premises across splits | 1 premise (`"A dog catching a Frisbee."`), 6 rows purged | `text_exclusion_ledger.csv` | **PASS** |
| **5-Partition Isolation** | Zero premise overlap across 5 subsets | Cross-partition overlap strictly 0 across all pairs | `text_partition_overlap_matrix.csv` | **PASS** |
| **Code Benchmark Corpus** | Verify IR-Plag files & pairs | 467 physical `.java` files; 733 pairs in CSV | `code_task_accounting.csv` | **PASS** |
| **Code Duplicates & Denominators** | Resolve raw vs dedup denominators | Raw: 733 / 105; Policy A: 720 / 103; Policy B: 709 / 101 | `code_task_accounting.csv` | **PASS** |
| **Case-05 Cross-Fitting** | Audit 5-fold cross-fitting feasibility | Degenerate ($N_{\text{pos}}^{\text{train}} = 0$ in Fold 1; Platt diverges) | `case05_fold_audit.csv` | **PASS** (Impossibility Proven) |
| **Conformal Escalation** | Restrict calibration to $\theta_{\text{low}} \le s < \theta_{\text{high}}$ | Bypass pairs ($s \ge \theta_{\text{high}}$) excluded from conformal cal | `mathematical_formulations_v3_5.md` §2 | **PASS** |
| **Conformal Attainability** | Enforce $n \ge \lceil(1-\epsilon)/\epsilon\rceil$ | Cutoff $\hat{q} = \infty$ if sample size insufficient; text only | `mathematical_formulations_v3_5.md` §4 | **PASS** |
| **Output Vocabulary** | Harmonize 4-state output schema | 4 standardized states (text uses 4, code uses 3) | `system_architecture_v3_5.md` §1 | **PASS** |
| **Four-Level Adjudication** | Confine ML to statistical signals | Human officers retain sole institutional authority | `four_contributions_formal_design_v3_5.md` §6 | **PASS** |
| **18 vs 28 File Discrepancy**| Reconcile Python files in `src/` | Exactly 28 files in `src/` (17 root, 6 `code/`, 5 `text/`) | `source_file_inventory.csv` | **PASS** (Resolved) |
| **Historical Baseline** | Audit retrospective immutability | Classified as `NOT_VERIFIABLE` (no pre-Phase-2 Git/hash) | `phase_2_3_4_correction_log.md` §2 | **NOT_VERIFIABLE** |
| **Current Baseline** | Establish current SHA-256 manifest | SHA-256 hashes generated for all 28 legacy `src/` files | `source_sha256_manifest.csv` | **PASS** |
| **Statistical Testing** | Exact McNemar & clustering bounds | Two-sided exact Binomial test; clustering caveat noted | `mathematical_formulations_v3_5.md` §7 | **PASS** |
| **Cost Accounting** | Separate latency, memory, and FLOPs | Latencies (0.045 / 0.058 ms) separated from target (0.050) | `system_architecture_v3_5.md` §4 | **PASS** |
| **Novelty Positioning** | Defensible claims for C1–C4 | Established vs adapted vs novel clearly delineated | `novelty_and_literature_analysis_v3_5.md` §2 | **PASS** |

**Summary Metrics:**
- **PASS:** 18 items
- **NOT_VERIFIABLE:** 1 item (Historical Baseline Code Immutability)
- **FAIL:** 0 items
- **PARTIAL:** 0 items

---

## 2. Key Audit Determinations and Methodological Resolutions

### 2.1 Resolution of the Code Denominator Question (Raw 105 vs Dedup 103 vs Collision 101)
The audit definitively resolved the denominator discrepancy in `case-06` and the overall code corpus:
1. **Raw Benchmark ($N=105$ in `case-06`, $N=733$ total):** Evaluated without deduplication. Used in historical baseline `src/evaluation.py`.
2. **Policy A (Representative Retention - Standard Dedup):** Keeps the first instance of each duplicate group and purges redundant occurrences. `case-06` contains 4 duplicate rows in 2 groups; purging the 2 redundant copies yields **$N = 103\text{ unique pairs}$** (49 pos, 54 neg). Total corpus = **$720\text{ unique pairs}$**.
3. **Policy B (Complete Collision Exclusion - Aggressive Purge):** Discards all rows involved in a collision. `case-06` drops all 4 duplicate rows, yielding **$N = 101\text{ non-collision pairs}$** (47 pos, 54 neg). Total corpus = **$709\text{ pairs}$**.
Both denominators are explicitly documented in `code_task_accounting.csv` to ensure complete reproducibility.

### 2.2 Text Dataset Accounting & Zero-Leakage 5-Partition Split
The text benchmark (`datasets/processed/text_plagiarism_benchmark.csv`) contains 19,666 raw rows. Exhaustive accounting reconciled all rows:
- **Active Experimental Corpus (10,501 pairs):**
  - Train ($D_{\text{train}}$): 4,000 pairs (1,349 unique premises)
  - Validation ($D_{\text{val}}$): 1,500 pairs (508 unique premises)
  - Cal-Prob ($D_{\text{cal-prob}}$): 1,000 pairs (336 unique premises)
  - Cal-Conf ($D_{\text{cal-conf}}$): 1,000 pairs (334 unique premises)
  - Test ($D_{\text{test}}$): 3,001 pairs (1,015 unique premises)
- **Exclusions (8 pairs):**
  - Cross-Split Contamination: 6 rows (premise `"A dog catching a Frisbee."`: `TXT_001181`–`83` in `val`, `TXT_016302`–`04` in `test`).
  - Redundant Exact Duplicates: 2 rows (`TXT_002505`, `TXT_003942`).
- **Retained Reserves (9,157 pairs):**
  - Development Reserve: 2,337 rows (791 unique premises in `snli_val`).
  - Test Reserve: 6,820 rows (2,307 unique premises in `snli_test`).
- **Sum Verification:** $10,501 + 6 + 2 + 2,337 + 6,820 = 19,666$ rows (Reconciles exactly).
- **Partition Overlap:** The $7 \times 7$ overlap matrix confirms **strictly 0 cross-partition premise overlap**.

### 2.3 Mathematical Proof of Platt Scaling Divergence on IR-Plag `case-05`
The group structure of `case-05` comprises 16 base program identities (1 original + 15 non-plagiarized) across 107 pairs (53 plagiarized, 54 non-plagiarized). Because all 53 plagiarized pairs originate from `original/Main.java`:
- In any group-partitioned 5-fold cross-validation split, the fold holding out `original/Main.java` (Fold 1) contains **zero positive training pairs** ($N_{\text{pos}}^{\text{train}} = 0$).
- Standard logistic regression (Platt scaling) solvers diverge to $-\infty$.
- In Folds 2–5, the held-out test fold contains **zero positive test pairs** ($N_{\text{pos}}^{\text{test}} = 0$), rendering precision, recall, ROC-AUC, and calibration error undefined.
- **Auditor Verdict:** Forced cross-fitting Platt scaling on `case-05` is formally rejected. Source code evaluation is downgraded to **exploratory raw decision margin thresholding** $z(\mathbf{x})$, and conformal coverage guarantees are formally disclaimed for code.

### 2.4 Mathematical Consistency of the Escalation Condition
The routing logic strictly separates:
- Low-risk deferral: $s_{\text{cand}}(x) < \theta_{\text{low}}$
- Heavy-model escalation slice: $\theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}}$
- Direct high-risk bypass: $s_{\text{cand}}(x) \ge \theta_{\text{high}}$
By definition, direct bypass pairs bypass Stage 2 and do not receive heavy-model forward passes. Consequently, they **must not enter the conformal calibration set**. Conformal non-conformity scores are calibrated strictly on pairs satisfying $\theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}}$, guaranteeing exact population alignment.

### 2.5 Resolution of the 18 vs 28 File Discrepancy
- The root of `src/` contains **17 Python files**.
- The subpackage `src/code/` contains **6 Python files**.
- The subpackage `src/text/` contains **5 Python files**.
- Total Python source files in `src/`: $17 + 6 + 5 = \mathbf{28\text{ files}}$.
- The "18 files" figure was an artifact of an abstract template that listed placeholder filenames (`src/main.py`, `src/cross_encoder.py`, `src/baselines/bm25_baseline.py`, etc.) which never existed in this codebase.
- **Historical Integrity:** Because no `.git` repository or pre-Phase-2 cryptographic checksums exist, retrospective immutability is formally classified as **`NOT_VERIFIABLE — no trustworthy pre-change baseline available`**.
- **Current Baseline Manifest:** All 28 files have been cataloged with their exact SHA-256 hashes in `source_sha256_manifest.csv`. All Phase 3 code will reside strictly in `research_upgrade/src/`.

---

## 3. Inventory of Completed Phase 2.3.4 (Version 3.5) Deliverables

All eleven (11) Version 3.5 deliverables are located in [`research_upgrade/documentation/phase_2_3_4/`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4):

1. [`four_contributions_formal_design_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/four_contributions_formal_design_v3_5.md)
2. [`mathematical_formulations_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/mathematical_formulations_v3_5.md)
3. [`algorithm_pseudocode_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/algorithm_pseudocode_v3_5.md)
4. [`system_architecture_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/system_architecture_v3_5.md)
5. [`experimental_protocol_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/experimental_protocol_v3_5.md)
6. [`novelty_and_literature_analysis_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/novelty_and_literature_analysis_v3_5.md)
7. [`phase_2_3_4_correction_log.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/phase_2_3_4_correction_log.md)
8. [`cross_document_consistency_matrix_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/cross_document_consistency_matrix_v3_5.md)
9. [`phase_2_3_4_implementation_readiness_checklist.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/phase_2_3_4_implementation_readiness_checklist.md)
10. [`dataset_accounting_and_partition_audit_v3_5.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/dataset_accounting_and_partition_audit_v3_5.md)
11. [`phase_2_3_4_final_independent_audit_report.md`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_4/phase_2_3_4_final_independent_audit_report.md)

### Executable Audit Script and Artifacts
- **Audit Script:** [`research_upgrade/audits/phase_2_3_4/run_phase_2_3_4_audit.py`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/run_phase_2_3_4_audit.py)
- **Machine-Readable Ledgers in [`research_upgrade/audits/phase_2_3_4/`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/):**
  - `text_dataset_audit.json`
  - `text_exclusion_ledger.csv`
  - `text_partition_overlap_matrix.csv`
  - `text_duplicate_audit.csv`
  - `code_dataset_audit.json`
  - `code_duplicate_audit.csv`
  - `code_task_accounting.csv`
  - `case05_fold_audit.csv`
  - `source_file_inventory.csv`
  - `source_sha256_manifest.csv`
  - `audit_execution_log.txt`

---

## 4. Mandatory Audit Statements and Scope Limitations

> [!IMPORTANT]
> **DOCUMENTATION CONSISTENCY DOES NOT EQUAL EMPIRICAL VALIDATION:**  
> The Version 3.5 documentation package establishes that the research framework is mathematically bounded, statistically grounded, and internally synchronized. It **does NOT constitute proof that empirical target metrics ($\ge 98\%$ candidate recall, $\ge 80\%$ budget savings) will be achieved on full test partitions**. Empirical performance can only be established through implementation and benchmarking in Phase 3.

> [!WARNING]
> **STRICT SOURCE CODE MODIFICATION FREEZE:**  
> In accordance with Phase 2 preservation rules, **no Python source code in `src/` has been modified, and no Phase 3 implementation has begun**. All 28 legacy files in `src/` remain completely frozen under their published SHA-256 hashes.

---

## 5. Final Decision Gate Determination

### Verdict: `READY_FOR_PHASE_3_IMPLEMENTATION_REVIEW`

All critical dataset, routing, statistical, denominator, and architectural issues have been resolved with reproducible evidence on disk:
- **Critical Blockers:** 0 blockers.
- **Major Methodological Limitations:** Formally documented (conformal prediction disclaimed for code; McNemar test characterized as descriptive of sample discordance under premise clustering; IR-Plag code evaluation limited to 2 held-out tasks).
- **Minor Documentation Issues:** Reconciled (18 vs 28 files resolved; matrix summary corrected to 15 full passes / 1 qualified pass; dates harmonized to November 10, 2026).
- **Exact Evidence Paths:** Audited and preserved in `research_upgrade/audits/phase_2_3_4/`.
- **Remaining Action:** Formal human review and explicit authorization to proceed to Phase 3.

**Execution is now paused. Awaiting explicit user approval before authoring Python source code in Phase 3.**

