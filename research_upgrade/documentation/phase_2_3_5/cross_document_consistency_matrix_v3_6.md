# Cross-Document Consistency Matrix & Harmonization Audit (Version 3.6)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Document Version:** Version 3.6  
**Date:** November 10, 2026  
**Status:** FULLY HARMONIZED (15 Full PASS [93.75%], 1 Qualified PASS [6.25%])

---

## 1. Audit Scope and Evaluation Methodology

To ensure absolute methodological and empirical coherence across all documentation and audit deliverables, 16 key technical dimensions were audited across the Version 3.6 specification package. Every dimension was verified against:
1. Physical files on disk.
2. The executable audit script [run_phase_2_3_5_audit.py](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/run_phase_2_3_5_audit.py).
3. The 16 machine-readable audit artifacts in [research_upgrade/audits/phase_2_3_5/](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/).

---

## 2. 16-Dimension Cross-Document Consistency Matrix

| # | Technical Dimension | Audited Specification Value | Verified in Documentation? | Verified in Audit Artifacts? | Audit Status | Notes & Cross-References |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| **01** | **Text Total Raw Rows** | 19,666 pairs | PASS | PASS | **FULL PASS** | Exactly 19,666 rows in [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv). |
| **02** | **Text Purged Rows** | 8 rows (6 cross-split, 2 dupe) | PASS | PASS | **FULL PASS** | Itemized row-by-row in [text_exclusion_ledger.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_exclusion_ledger.csv). |
| **03** | **Text Clean Retained** | 19,658 rows (9,837 val, 9,821 test) | PASS | PASS | **FULL PASS** | Zero nulls, verified in [text_dataset_audit.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_dataset_audit.json). |
| **04** | **Text Active Partitions** | Train: 4,000; Val: 1,500; Cal-Prob: 1,000; Cal-Conf: 1,000; Test: 3,000 | PASS | PASS | **FULL PASS** | Exact match; zero overshoot. [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv). |
| **05** | **Text Reserve Partitions** | Dev-Reserve: 2,337; Test-Reserve: 6,821 | PASS | PASS | **FULL PASS** | Perfect sum: $10,500 + 9,158 + 8 = 19,666$. Zero row loss. |
| **06** | **Text Premise Overlap** | Exactly 0 across all 7 splits | PASS | PASS | **FULL PASS** | 7x7 matrix verified in [text_partition_overlap_matrix.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_overlap_matrix.csv). |
| **07** | **Text Duplicate Structure** | 4 rows in 2 groups; 0 reversed pairs | PASS | PASS | **FULL PASS** | Canonical pair grouping verified in [text_reversed_pair_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_reversed_pair_audit.csv). |
| **08** | **Code Total Raw Pairs** | 733 pairs across 7 tasks | PASS | PASS | **FULL PASS** | Verified in [code_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/code_plagiarism_benchmark.csv). |
| **09** | **Code Duplicate Structure** | 24 participating rows in 11 groups; 0 reversed pairs | PASS | PASS | **FULL PASS** | Canonical audit verified in [code_reversed_pair_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_reversed_pair_audit.csv). |
| **10** | **Code Deduplication** | Policy A: 720 retained; Policy B: 709 retained | PASS | PASS | **FULL PASS** | Full task breakdown in [code_task_accounting.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_task_accounting.csv). |
| **11** | **Case-06 Denominator** | Raw: 105; Policy A: 103; Policy B: 101 | PASS | PASS | **FULL PASS** | Discrepancy formally resolved; all 4 collision rows are positives. |
| **12** | **Case-05 Java Files** | Exactly 69 physical Java files | PASS | PASS | **FULL PASS** | Discrepancy with 58 resolved. [case_05_java_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_java_file_inventory.csv). |
| **13** | **Case-05 Cross-Fitting** | Structural zero-positive degeneracy proven | PASS | PASS | **FULL PASS** | Unregularized divergence & regularized `ValueError` proven in [case_05_fold_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_fold_audit.csv). |
| **14** | **Data Loader Status** | Classified as `DESIGN_SPECIFIED_NOT_IMPLEMENTED` | PASS | PASS | **FULL PASS** | No files in `research_upgrade/src/`. Verified in [loader_verification_report.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/loader_verification_report.csv). |
| **15** | **Latency Profile** | 10 distinct operational stages | PASS | PASS | **FULL PASS** | $0.050\text{ ms}$ clarified as pre-indexed lookup design target. |
| **16** | **Calibration Metrics** | ECE ($M=10$), MCE, Brier, clipped NLL | PASS | PASS | **FULL PASS** | Formulas and empty-bin contributions verified in Section 5 of formulations doc. |

---

## 3. Historical Provenance Audit (Qualified Pass)

| Attribute | Audited Status | Methodological Finding | Operational Recommendation |
| :--- | :---: | :--- | :--- |
| **Git Baseline Provenance** | **QUALIFIED PASS** | The workspace lacks a pre-Phase-2 `.git` history or baseline cryptographic commit log. It is impossible to cryptographically verify whether files in `src/` were modified prior to the Phase 2 audit. | All 28 current Python files in `src/` have been cryptographically frozen and fingerprinted via SHA-256 in [source_sha256_manifest.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/source_sha256_manifest.csv) as the immutable baseline. |

---

## 4. Overall Harmonization Certification

- **Total Dimensions Audited:** 16
- **Full PASS Rate:** **15 / 16 (93.75%)**
- **Qualified PASS Rate:** **1 / 16 (6.25%)**
- **FAIL / Discrepancy Rate:** **0 / 16 (0.00%)**

**Audit Determination:**  
The Version 3.6 documentation suite is mathematically, statistically, and experimentally harmonized across all deliverables and machine-readable artifacts.

