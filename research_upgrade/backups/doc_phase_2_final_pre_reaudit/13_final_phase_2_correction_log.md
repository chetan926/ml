# Final Phase 2 Comprehensive Correction Log (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 13 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** ALL ISSUES EXHAUSTIVELY AUDITED & RESOLVED

---

## 1. Executive Summary of Corrections

This document records the definitive resolution of all mathematical, statistical, dataset, architectural, and documentation inconsistencies identified across the Phase 2 audit lifecycle. Every correction is backed by deterministic Python assertions and physical verification on disk.

```
════════════════════════════════════════════════════════════════════════════════════════
                        FINAL AUDIT METRIC TOTALS
════════════════════════════════════════════════════════════════════════════════════════
  Total Audited Requirements:         20
  • FULL PASS Count:                  19  (95.00%)
  • QUALIFIED PASS Count:              1  ( 5.00%)  [Pre-Phase-2 Git Commit Provenance]
  • FAIL Count:                        0  ( 0.00%)
  • NOT VERIFIABLE Count:              0  ( 0.00%)
  • NOT APPLICABLE Count:              0  ( 0.00%)
  Actual Full-Pass Percentage:        95.0%
════════════════════════════════════════════════════════════════════════════════════════
```

---

## 2. Itemized Correction Log Across Technical Domains

### 1. Consistency Matrix Status Harmonization
- **Prior Issue:** Earlier versions exhibited an accounting contradiction where the text claimed "15 Full Pass + 1 Qualified Pass" while the table listed all 16 items as Full Pass.
- **Definitive Fix:** Formulated a standardized 20-dimension matrix in [11_final_cross_document_consistency_matrix.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_final/11_final_cross_document_consistency_matrix.md) where every requirement carries exactly one status. Requirement 20 (Git Provenance) is explicitly marked **QUALIFIED PASS** in the table, yielding exactly **19 FULL PASS (95.0%) and 1 QUALIFIED PASS (5.0%)**.

### 2. Elimination of Deliberate Text Partition Overshoot
- **Prior Issue:** The sequential greedy allocator assigned a 3-pair premise cluster at the test boundary, causing the test split to reach 3,001 rows (+1 overshoot).
- **Definitive Fix:** Implemented `non_overshoot_allocator` in [`run_final_phase2_audit.py`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/run_final_phase2_audit.py), which searches ahead for exact cluster sizes to match target integers exactly:
  - `train`: 4,000 | `val`: 1,500 | `cal_prob`: 1,000 | `cal_conf`: 1,000 | `test`: 3,000
  - Reserves: `dev_reserve`: 2,337 | `test_reserve`: 6,821 | Purged: 8
  - Perfect sum: $10,500 + 9,158 + 8 = 19,666$ ($\Delta = 0$).

### 3. Case-05 Physical Java File Inventory Discrepancy (69 vs 58)
- **Prior Issue:** Earlier narrative drafts cited 58 Java files in `case-05`, conflicting with the filesystem.
- **Definitive Fix:** Filesystem audit of `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/case-05/` confirms **exactly 69 physical `.java` files**:
  - `original`: 1 file (`original/T5.java`)
  - `non-plagiarized`: 15 files (`01` to `15`, 1 file each)
  - `plagiarized`: 53 files across 6 levels (L1: 9, L2: 8, L3: 9, L4: 9, L5: 9, L6: 9 $\to$ 53 files).
  - Sum: $1 + 15 + 53 = 69$ files itemized in [code_source_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/code_source_file_inventory.csv).

### 4. Platt Scaling Failure Derivations on Single-Class Data
- **Prior Issue:** Previous texts asserted "divergence" without distinguishing unregularized from regularized implementations.
- **Definitive Fix:** Derived both behaviors in Section 3 of [03_final_mathematical_formulations.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_final/03_final_mathematical_formulations.md):
  1. *Unregularized MLE:* $\frac{\partial \mathcal{L}}{\partial B} = \sum \sigma(A z_i + B) > 0$ strictly; loss strictly decreases as $B \to -\infty$, causing optimization divergence.
  2. *Regularized Implementation:* In scikit-learn (`LogisticRegression`), single-class training triggers a fatal exception: `ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0`.

### 5. Programmatic Case-05 Cross-Fitting Proof
- **Prior Issue:** Cross-fitting folds were hard-coded without programmatic grounding in base program identities.
- **Definitive Fix:** Mapped all 107 pairs in `case-05` to 16 base program identities (`prog_original`, `prog_np_01` to `prog_np_15`). Proved that because all 53 positive pairs share `prog_original`:
  - Fold 1 (`prog_original` held out): Train positives $= 0$ (Platt scaling fails via divergence / `ValueError`).
  - Folds 2–5 (`prog_original` in train): Test positives $= 0$ (Precision, Recall, ROC-AUC, PR-AUC, ECE are undefined).
  - Enforced protocol: Case-05 evaluated strictly via exploratory raw decision margins ($z \in \mathbb{R}$).

### 6. Canonical Pair Reversed-Duplicate Verification & Case-06 Denominators
- **Prior Issue:** Reversal checks were performed using indirect set subtraction; Case-06 pair counts were cited ambiguously as 105, 103, or 101.
- **Definitive Fix:** Implemented canonical-key grouping with orientation tracking, proving **0 reversed pairs** in text and code. Clarified duplicate terminology (24 participating rows in 11 groups) and definitively resolved Case-06:
  - Raw: 105 pairs (51 pos, 54 neg)
  - Policy A (Keep 1st): 103 pairs (49 pos, 54 neg)
  - Policy B (Drop All): 101 pairs (47 pos, 54 neg).

### 7. Accurate Data Loader Implementation Classification
- **Prior Issue:** Earlier summaries conflated audit simulation scripts with production data loaders.
- **Definitive Fix:** Inspected `research_upgrade/src/`, confirmed it is empty (Phase 3 not authorized), and formally classified loaders as **`DESIGN_SPECIFIED`** in [component_implementation_status.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/component_implementation_status.json).

### 8. Latency Scope Decomposition (10 Operations)
- **Prior Issue:** A flat $0.050\text{ ms}$ latency was cited without explaining operational boundaries.
- **Definitive Fix:** Decomposed execution into 10 distinct operations in [08_final_cost_and_resource_accounting.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_final/08_final_cost_and_resource_accounting.md), clarifying that $0.050\text{ ms}$ is a design target for pre-indexed vector dot products, whereas unbatched embedding takes $1.850\text{ ms}$ and cross-encoder inference takes $4.046\text{ ms}$.

### 9. Multi-Modal Stage-2 Model Separation
- **Prior Issue:** A text cross-encoder (`ms-marco-MiniLM-L-6-v2`) was incorrectly referenced for both text and code.
- **Definitive Fix:** Architected independent Stage-2 components in [05_final_system_architecture.md](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/documentation/phase_2_final/05_final_system_architecture.md): Text Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) and Code Stage-2 Structural-Semantic Detector (hybrid AST subtree kernel + CodeBERT).

### 10. Exact Finite-Sample Group Conformal Order Statistic
- **Prior Issue:** Documentation referenced approximate percentiles or continuous quantile functions.
- **Definitive Fix:** Specified the exact finite-sample order statistic $k = \lceil (n+1)(1-\epsilon) \rceil$ with finite-sample attainability threshold $n \ge \lceil (1-\epsilon)/\epsilon \rceil$. Rejected interpolated quantiles.

### 11. Statistical Dependence Respect & Inference Protocol
- **Prior Issue:** Pairwise comparisons were treated as independent observations, inflating significance.
- **Definitive Fix:** Formulated group-aware bootstrap resampling over entire premise clusters ($B = 1,000$ iterations). Explicitly acknowledged that with only 2 independent test tasks in IR-Plag, code results are descriptive and generalization is exploratory.

