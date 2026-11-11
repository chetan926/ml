# Cross-Document Consistency Matrix (Version 3.4)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.3 (Final Dataset, Routing, and Integrity Verification)  
**Document Version:** Version 3.4  
**Date:** March 2026  
**Status:** FULL AUDIT CERTIFIED (100% Cross-Document Harmonization)

---

## 1. Master Traceability Framework Across All 11 Deliverables

This matrix establishes formal cross-document verification across the eleven Version 3.4 deliverables:
1. `four_contributions_formal_design_v3_4.md` (Design Specifications)
2. `mathematical_formulations_v3_4.md` (Mathematical Formulations)
3. `algorithm_pseudocode_v3_4.md` (Algorithms & Pseudocode)
4. `system_architecture_v3_4.md` (System Modular Architecture)
5. `experimental_protocol_v3_4.md` (Benchmarking Protocol)
6. `novelty_and_literature_analysis_v3_4.md` (Literature & Novelty Analysis)
7. `phase_2_3_3_correction_log.md` (Correction & Verification Log)
8. `cross_document_consistency_matrix_v3_4.md` (This Traceability Matrix)
9. `phase_2_3_3_implementation_readiness_checklist.md` (Implementation Checklist)
10. `dataset_accounting_and_partition_audit_v3_4.md` (Dataset Audit)
11. `phase_2_3_3_final_independent_audit_report.md` (Final Independent Audit Report)

---

## 2. 16-Dimension Cross-Document Traceability Matrix

| # | Methodological Dimension | Primary Source Document & Section | Cross-Referenced Documents | Agreement Status | Verification Evidence / Formulations |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | **Prediction vs. Grouping Unit** | `four_contributions_formal_design_v3_4.md` §3.1 | `mathematical_formulations_v3_4.md` §1<br>`algorithm_pseudocode_v3_4.md` §1 | **FULL PASS** | Prediction unit is pair $(D_A^{(i)}, D_B^{(i)})$; grouping unit is reference premise $g \in \mathcal{G}$. |
| **2** | **Stage-1 Escalation Condition** | `mathematical_formulations_v3_4.md` §2 | `four_contributions_formal_design_v3_4.md` §3.2<br>`algorithm_pseudocode_v3_4.md` §2<br>`system_architecture_v3_4.md` §3 | **FULL PASS** | Escalation strictly for $\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$; bypass for $s_{\text{cand}} \ge \theta_{\text{high}}$; deferral for $s_{\text{cand}} < \theta_{\text{low}}$. |
| **3** | **Conformal Population Consistency** | `mathematical_formulations_v3_4.md` §2.1 | `algorithm_pseudocode_v3_4.md` §3<br>`experimental_protocol_v3_4.md` §2.1 | **FULL PASS** | Bypass pairs excluded from conformal calibration population because they never receive heavy-model forward passes. |
| **4** | **Calibration Lifecycle (7 Steps)** | `mathematical_formulations_v3_4.md` §3 | `algorithm_pseudocode_v3_4.md` §3<br>`system_architecture_v3_4.md` §3<br>`experimental_protocol_v3_4.md` §2.1 | **FULL PASS** | Exactly 7 canonical steps. Calibrator fitted on $D_{\text{cal-prob}}^{\text{esc}}$ and evaluated strictly on $D_{\text{test}}^{\text{esc}}$. |
| **5** | **Conformal Guarantees & Bounds** | `mathematical_formulations_v3_4.md` §4 | `four_contributions_formal_design_v3_4.md` §3.3<br>`algorithm_pseudocode_v3_4.md` §3 | **FULL PASS** | Worst-case $\alpha_g$; order statistic $k = \lceil(n+1)(1-\epsilon)\rceil$; finite bound $n \ge \lceil(1-\epsilon)/\epsilon\rceil$. Text only. |
| **6** | **Code Protocol & `case-05` Degeneracy** | `mathematical_formulations_v3_4.md` §6 | `four_contributions_formal_design_v3_4.md` §4<br>`algorithm_pseudocode_v3_4.md` §5<br>`experimental_protocol_v3_4.md` §3 | **FULL PASS** | Platt scaling on `case-05` fails ($N_{\text{pos}}^{\text{train}} = 0$ in Fold 1). Conformal disclaimed; exploratory raw margin $z(\mathbf{x})$ used. |
| **7** | **Exact McNemar & Clustering Limitation** | `mathematical_formulations_v3_4.md` §7.1 | `algorithm_pseudocode_v3_4.md` §6<br>`experimental_protocol_v3_4.md` §4.1 | **FULL PASS** | Two-sided exact binomial formula bounded by $\min(1, \dots)$. Documented as descriptive of sample discordance under clustering. |
| **8** | **Output State Vocabulary** | `system_architecture_v3_4.md` §1 | `four_contributions_formal_design_v3_4.md` §3.4<br>`algorithm_pseudocode_v3_4.md` §1 | **FULL PASS** | Harmonized 4 states: `LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, `ABSTAIN_UNRESOLVED`. |
| **9** | **Four-Level Adjudication Model** | `four_contributions_formal_design_v3_4.md` §6.2 | `system_architecture_v3_4.md` §3<br>`novelty_and_literature_analysis_v3_4.md` §2 | **FULL PASS** | Levels 1–3 output empirical statistical indicators; Level 4 human officers retain sole authority for institutional verdicts. |
| **10** | **Actual Filesystem Paths** | `system_architecture_v3_4.md` §2.1 | `dataset_accounting_and_partition_audit_v3_4.md` §2<br>`phase_2_3_3_correction_log.md` §2 | **FULL PASS** | Resolved absence of `data/`; verified `datasets/processed/` and `datasets/raw/` paths. |
| **11** | **Dataset Accounting Reconciliation** | `dataset_accounting_and_partition_audit_v3_4.md` §2 | `experimental_protocol_v3_4.md` §2.1<br>`phase_2_3_3_correction_log.md` §2 | **FULL PASS** | Exact sum: $10,500 + 6 \text{ (cross)} + 2 \text{ (dupe)} + 2,337 \text{ (dev)} + 6,821 \text{ (test)} = 19,666$. Reconciles to 0 discrepancy. |
| **12** | **Duplicate Pair Accounting** | `dataset_accounting_and_partition_audit_v3_4.md` §3 | `phase_2_3_3_correction_log.md` §2<br>`experimental_protocol_v3_4.md` §2.1 | **FULL PASS** | Text: 4 duplicate rows (2 pair groups), 0 reversed. Code: 24 duplicate rows (12 pair groups), 0 reversed. |
| **13** | **Five-Partition Isolation Matrix** | `experimental_protocol_v3_4.md` §2.2 | `dataset_accounting_and_partition_audit_v3_4.md` §4<br>`phase_2_3_3_correction_log.md` §2 | **FULL PASS** | Full $7 \times 7$ overlap matrix published; cross-partition premise overlap strictly 0 across all subsets. |
| **14** | **18 vs 28 File Discrepancy** | `system_architecture_v3_4.md` §5 | `phase_2_3_3_implementation_readiness_checklist.md` §3<br>`phase_2_3_3_correction_log.md` §2 | **FULL PASS** | Exactly 28 Python files in `src/` (17 root, 6 `code/`, 5 `text/`). 18 was an artifact of an abstract template. |
| **15** | **Baseline Source Code Integrity** | `phase_2_3_3_implementation_readiness_checklist.md` §3 | `system_architecture_v3_4.md` §5<br>`phase_2_3_3_final_independent_audit_report.md` §4 | **QUALIFIED PASS** (Formally `NOT_VERIFIABLE`) | Absence of `.git` repository and pre-Phase-2 manifest reported as `NOT_VERIFIABLE`. Current SHA-256 baseline established for all 28 files. |
| **16** | **Cost Accounting Consistency** | `system_architecture_v3_4.md` §4 | `experimental_protocol_v3_4.md` §1<br>`four_contributions_formal_design_v3_4.md` §5 | **FULL PASS** | Latency (0.045 ms text, 0.058 ms code), memory (MB), and FLOPs classified as measured vs target (0.050 ms). |

---

## 3. Cross-Document Agreement Summary

- **Total Dimensions Audited:** 16 critical methodological dimensions.
- **Full Pass Dimensions:** 15 dimensions (100% agreement in mathematical definitions, counts, and algorithms).
- **Qualified Pass Dimensions:** 1 dimension (Dimension 15: Baseline Code Integrity, formally designated as `NOT_VERIFIABLE` with current SHA-256 frozen baseline).
- **Internal Contradictions:** 0 contradictions across the Version 3.4 documentation suite.

