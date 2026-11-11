# Phase 2.3.5 Methodological Correction & Evidence Repair Log

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Date:** November 10, 2026  
**Auditor:** Senior Independent ML Researcher, Statistical Methodologist, & Reproducibility Specialist  
**Status:** ALL 8 ISSUES FORMALLY RESOLVED & AUDITED ON DISK

---

## 1. Executive Summary of Corrections

Phase 2.3.5 was initiated to resolve eight critical methodological, statistical, dataset, and implementation-readiness inconsistencies identified in the Version 3.5 specification. Rather than merely adjusting prose, every resolution is backed by deterministic Python assertions and machine-readable artifacts generated in [research_upgrade/audits/phase_2_3_5/](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/).

---

## 2. Detailed Issue-by-Issue Resolution Log

### Issue 1: Text Partition Target Deviation & Deliberate Overshoot
- **Problem in Version 3.5:** The deterministic allocator assigned entire premise clusters sequentially. At the test partition boundary, allocating a 3-pair cluster pushed the test partition to 3,001 rows (+1 overshoot beyond the target of 3,000), violating strict non-overshooting constraints.
- **Root Cause:** A greedy sequential allocator that does not look ahead to fill exact remaining integer capacities.
- **Resolution in Version 3.6:** Implemented an exact-target greedy allocator (`non_overshoot_allocator` in [run_phase_2_3_5_audit.py](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/run_phase_2_3_5_audit.py)). The algorithm scans the remaining premise pool to hit exact target integers without deliberate overshoot:
  - `train`: Exactly 4,000 rows (1,349 premises)
  - `val`: Exactly 1,500 rows (508 premises)
  - `cal_prob`: Exactly 1,000 rows (336 premises)
  - `cal_conf`: Exactly 1,000 rows (334 premises)
  - `test`: Exactly 3,000 rows (1,015 premises)
  - `dev_reserve`: 2,337 rows (791 premises)
  - `test_reserve`: 6,821 rows (2,307 premises)
  - Purged rows: 8 rows (3 premises)
  - **Reconciliation:** $4,000 + 1,500 + 1,000 + 1,000 + 3,000 + 2,337 + 6,821 + 8 = 19,666\text{ rows}$ (Difference = 0).
- **Supporting Artifacts:** [text_dataset_audit.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_dataset_audit.json), [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv).

---

### Issue 2: Physical Java File Inventory in IR-Plag `case-05` (69 vs 58)
- **Problem in Version 3.5:** Earlier narrative documents repeatedly cited "58 Java files in case-05", whereas physical inspection revealed 69 files.
- **Root Cause:** Confusion between the number of files and task folder counts in earlier drafts, or omitting some obfuscation subdirectories.
- **Resolution in Version 3.6:** Conducted an exhaustive physical filesystem audit of `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/case-05/`. Confirmed **exactly 69 physical `.java` files**:
  - `original`: 1 file (`original/T5.java`)
  - `non-plagiarized`: 15 files (`01` to `15`, 1 file each)
  - `plagiarized`: 53 files across 6 obfuscation levels (L1: 9, L2: 8, L3: 9, L4: 9, L5: 9, L6: 9 $\to$ exactly 53 files).
  - Sum: $1 + 15 + 53 = 69$ files.
- **Supporting Artifacts:** [case_05_java_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_java_file_inventory.csv), [case_05_pair_to_program_mapping.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_pair_to_program_mapping.csv).

---

### Issue 3: Platt Scaling Divergence (Unregularized vs Regularized Formulations)
- **Problem in Version 3.5:** Documentation claimed that Platt scaling "diverges" when positive instances are zero, without distinguishing between unregularized maximum likelihood and regularized library implementations.
- **Root Cause:** Incomplete mathematical characterization of logistic loss under single-class training folds.
- **Resolution in Version 3.6:** Derived both mathematical behaviors in Section 3 of [mathematical_formulations_v3_6.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_5/mathematical_formulations_v3_6.md):
  1. **Unregularized MLE:** $\frac{\partial \mathcal{L}}{\partial B} = \sum \sigma(A z_i + B) > 0$ strictly. The loss strictly decreases as $B \to -\infty$, causing numerical optimization to diverge.
  2. **Regularized Implementation:** In scikit-learn (`LogisticRegression`), single-class training triggers a fatal exception: `ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0`.
  3. Consequently, neither unregularized nor regularized Platt scaling can produce a valid binary calibrator on single-class data.
- **Supporting Artifacts:** [case_05_fold_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_fold_audit.csv).

---

### Issue 4: Derivation of Cross-Fitting Folds from Base Program Identities
- **Problem in Version 3.5:** The 5-fold cross-fitting array was hard-coded in scripts without programmatic grounding in base program identities.
- **Root Cause:** Ad-hoc scripting rather than identity-aware group derivation.
- **Resolution in Version 3.6:** Programmatically identified the 16 base program identities in `case-05` (`prog_original` and 15 non-plagiarized programs). Proved that because all 53 positive pairs involve `prog_original`, any group-disjoint split results in:
  - Fold 1 (holding out `prog_original`): Training positives $= 0$ (Platt scaling fails).
  - Folds 2–5 (`prog_original` in training): Test positives $= 0$ (Precision, Recall, ROC-AUC, PR-AUC, ECE are undefined).
  - Concluded that `case-05` must be evaluated using exploratory raw decision margins ($z \in \mathbb{R}$).
- **Supporting Artifacts:** [case_05_fold_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_fold_audit.csv).

---

### Issue 5: Reversed Duplicate Detection Methodology & Case-06 Denominator Clarification
- **Problem in Version 3.5:** Reversal checks were performed using indirect set subtraction; Case-06 pair counts were cited inconsistently as 105, 103, or 101.
- **Root Cause:** Ambiguity over whether duplicates were evaluated before or after deduplication policies.
- **Resolution in Version 3.6:** Implemented explicit canonical-key grouping with orientation tracking (`(u, v)` vs `(v, u)`). Confirmed **0 reversed duplicate pairs** in both text and code. Clarified duplicate terminology:
  - 24 participating rows in 11 duplicate groups.
  - Policy A (retain 1st) purges 13 redundant rows $\to$ 720 retained.
  - Policy B (drop all collisions) purges 24 collision rows $\to$ 709 retained.
  - Case-06: 105 raw pairs; 4 participating rows (2 groups); Policy A yields **103 pairs** (49 pos, 54 neg); Policy B yields **101 pairs** (47 pos, 54 neg).
- **Supporting Artifacts:** [text_reversed_pair_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_reversed_pair_audit.csv), [code_reversed_pair_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_reversed_pair_audit.csv), [code_duplicate_ledger.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_duplicate_ledger.csv), [code_task_accounting.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/code_task_accounting.csv).

---

### Issue 6: Formal Data Loader Status Verification
- **Problem in Version 3.5:** Earlier audit summaries mistakenly reported that data loaders were "implemented and verified", when in fact only the legacy baseline loader `src/data_loader.py` existed on disk.
- **Root Cause:** Conflating audit verification scripts with production loaders in `research_upgrade/src/`.
- **Resolution in Version 3.6:** Conducted an exhaustive inventory. Confirmed that `research_upgrade/src/` does not yet exist because Phase 3 implementation has not been authorized. Formally classified `TextBenchmarkDataLoader` and `CodeBenchmarkDataLoader` as **`DESIGN_SPECIFIED_NOT_IMPLEMENTED`**.
- **Supporting Artifacts:** [loader_verification_report.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/loader_verification_report.csv).

---

### Issue 7: Multi-Component Latency Breakdown
- **Problem in Version 3.5:** The latency budget was presented as a flat $0.050\text{ ms}$, creating the impression that entire candidate pairs could be tokenized, embedded, and classified in 50 microseconds.
- **Root Cause:** Conflating pre-indexed dot-product lookup with full unbatched end-to-end execution.
- **Resolution in Version 3.6:** Decomposed system execution into 10 distinct operational stages in Section 2 of [system_architecture_v3_6.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_5/system_architecture_v3_6.md). Clarified that $0.050\text{ ms}$ is a system design target for cached similarity lookups, whereas full unbatched embedding takes $1.85\text{ ms}$ and cross-encoder inference takes $4.046\text{ ms}$.

---

### Issue 8: Mathematical Precision in Calibration Metrics
- **Problem in Version 3.5:** Metrics were named without unambiguous mathematical definitions (e.g., bin count, handling of empty bins, and numerical clipping for NLL).
- **Root Cause:** Informal prose descriptions.
- **Resolution in Version 3.6:** Formally defined all four metrics in Section 5 of [mathematical_formulations_v3_6.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_3_5/mathematical_formulations_v3_6.md):
  - ECE with $M=10$ uniform bins and explicit 0-contribution for empty bins.
  - MCE over occupied bins.
  - Brier Score.
  - Clipped NLL with probability clipping to $[10^{-15}, 1 - 10^{-15}]$.
  - Evaluated strictly on the untouched test escalation population $D_{\text{test}}^{\text{esc}}$.

