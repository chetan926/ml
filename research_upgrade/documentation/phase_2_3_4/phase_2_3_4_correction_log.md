# Phase 2.3.4 Comprehensive Correction and Verification Log

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.4 (Final Independent Certification and Phase 3 Readiness Gate)  
**Document Version:** Version 3.5  
**Date:** November 10, 2026  
**Status:** INDEPENDENTLY AUDITED & CERTIFIED (Reproducible Artifacts Verified)

---

## 1. Executive Summary

This correction log documents the definitive findings and methodological resolutions established during **Phase 2.3.4**. All claims are backed by physical inspection of repository files and execution of the standalone audit script [`run_phase_2_3_4_audit.py`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/run_phase_2_3_4_audit.py).

All machine-readable audit artifacts are preserved in [`research_upgrade/audits/phase_2_3_4/`](file:///d:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_4/).

---

## 2. Itemized Verification and Methodological Resolutions

### Issue 1: Text Dataset Accounting & Exclusion Ledger Verification
- **Empirical Audit Finding:** 
  - `datasets/processed/text_plagiarism_benchmark.csv` contains exactly **19,666 raw rows** (9,842 in `val`, 9,824 in `test`).
  - Exactly 6 rows cross splits on premise `"A dog catching a Frisbee."`: rows `TXT_001181`, `TXT_001182`, `TXT_001183` in `val`; rows `TXT_016302`, `TXT_016303`, `TXT_016304` in `test`.
  - Exactly 4 duplicate rows exist (2 pair groups): `TXT_002504`/`TXT_002505` and `TXT_003940`/`TXT_003942` (all in `val`).
  - **Zero Overlap:** The intersection between duplicate rows and contamination rows is strictly 0 rows (no double counting).
  - Deduplication keeps the first instance and purges 2 redundant duplicate rows (`TXT_002505`, `TXT_003942`).
  - Partitioning via deterministic premise hashing yields: Active Partitions = 10,501 pairs ($D_{\text{train}}$: 4,000; $D_{\text{val}}$: 1,500; $D_{\text{cal-prob}}$: 1,000; $D_{\text{cal-conf}}$: 1,000; $D_{\text{test}}$: 3,001); Development Reserve = 2,337 pairs; Test Reserve = 6,820 pairs.
  - **Arithmetic Sum Check:** $10,501 + 6 + 2 + 2,337 + 6,820 = 19,666$ rows (Discrepancy = 0).
- **Artifact:** Verified in `text_dataset_audit.json` and row-level ledger `text_exclusion_ledger.csv`.

---

### Issue 2: Code Dataset Deduplication & The Denominator Discrepancy
- **Empirical Audit Finding:**
  - `datasets/processed/code_plagiarism_benchmark.csv` contains exactly **733 raw pairs** across 467 physical `.java` files.
  - Exactly **24 duplicate rows** exist across 11 duplicate groups (3 groups in `case-01`, 3 in `case-02`, 3 in `case-03`, 2 in `case-06`).
  - **Resolution of the `case-06` Denominator (105 vs 101 vs 103):**
    - **Raw Benchmark (No Dedup):** 105 pairs in `case-06`, 733 pairs total across the corpus. This denominator was used in historical baseline `src/evaluation.py`.
    - **Policy A (Representative Retention - Standard Dedup):** Keeps the first instance of each duplicate group and purges the redundant occurrences. `case-06` has 4 duplicate rows in 2 groups, so $105 - 2 = \mathbf{103\text{ unique pairs}}$ (49 pos, 54 neg). Total corpus = $\mathbf{720\text{ unique pairs}}$.
    - **Policy B (Complete Collision Exclusion - Aggressive Purge):** Discards all rows involved in a collision. `case-06` drops all 4 duplicate rows, yielding $105 - 4 = \mathbf{101\text{ non-collision pairs}}$ (47 pos, 54 neg). Total corpus = $733 - 24 = \mathbf{709\text{ pairs}}$.
- **Artifact:** Reconciled and published in `code_task_accounting.csv` and `code_duplicate_audit.csv`.

---

### Issue 3: Five-Partition Premise Independence Overlap Matrix
- **Empirical Audit Finding:**
  - The $7 \times 7$ cross-partition overlap matrix was reconstructed using the actual grouping premises across $D_{\text{train}}$ (1,349 premises), $D_{\text{val}}$ (508 premises), $D_{\text{cal-prob}}$ (336 premises), $D_{\text{cal-conf}}$ (334 premises), $D_{\text{test}}$ (1,015 premises), $D_{\text{dev-reserve}}$ (791 premises), and $D_{\text{test-reserve}}$ (2,307 premises).
  - **Cross-partition overlap between any two distinct subsets is strictly 0.0%**.
- **Artifact:** Preserved in `text_partition_overlap_matrix.csv`.

---

### Issue 4: Mathematical Degeneracy of Platt Scaling on IR-Plag `case-05`
- **Empirical Audit Finding:**
  - `case-05` contains 107 pairs across 16 program identities (1 original reference + 15 non-plagiarized).
  - All 53 plagiarized pairs originate from `original/Main.java`.
  - In a 5-fold group split:
    - **Fold 1** (holding out `original`): Training set has **0 positive pairs** ($N_{\text{pos}}^{\text{train}} = 0$). Logistic loss is strictly decreasing as $B \to -\infty$; Platt scaling optimization **diverges to $-\infty$**.
    - **Folds 2–5** (holding out non-plag): Held-out test folds have **0 positive pairs** ($N_{\text{pos}}^{\text{test}} = 0$); ROC-AUC, Precision, Recall, and ECE are **undefined**.
  - **Resolution:** Forced cross-fitting Platt scaling is rejected. Code evaluation is downgraded to **exploratory raw decision margin thresholding** $z(\mathbf{x})$, with conformal guarantees formally disclaimed.
- **Artifact:** Documented in `case05_fold_audit.csv`.

---

### Issue 5: Conformal Population Alignment & Attainability Bounds
- **Mathematical Correction:**
  - The escalation condition is strictly $\theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}}$.
  - Direct high-risk bypass pairs ($s_{\text{cand}} \ge \theta_{\text{high}}$) bypass Stage 2 and **do not enter the conformal calibration set**.
  - Finite-sample order statistic $k = \lceil (n+1)(1-\epsilon) \rceil$ requires $n \ge \lceil (1-\epsilon)/\epsilon \rceil$. If $n < 19$ (for $\epsilon = 0.05$), $\hat{q}_{1-\epsilon} = \infty$, formally abstaining from finite coverage claims.

---

### Issue 6: Standardized 4-State Output Vocabulary & Adjudication Model
- **Vocabulary Harmonized:** `LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, `ABSTAIN_UNRESOLVED`.
- **Modality Mapping:** Text utilizes all 4 states; Code utilizes 3 states (`ABSTAIN_UNRESOLVED` omitted).
- **Four-Level Adjudication Architecture:** Levels 1–3 emit empirical statistical evidence; Level 4 human officers retain exclusive institutional authority for academic integrity findings.

---

### Issue 7: Cost Accounting Consistency
- **Latency Categorization:**
  - Text Stage-1 screening: **0.045 ms/pair** (historical measured, warm-cache vector dot product).
  - Code Stage-1 screening: **0.058 ms/pair** (historical measured, partial token regex).
  - System Stage-1 target: **0.050 ms/pair** (design constraint, not yet achieved).
  - Heavy transformer inference: **4.046 ms/pair** (MiniLM forward pass on CPU).
  - Code AST parsing: **2.050 ms/pair**.
  - Evidence diff generation: **1.850 ms/pair**.
- **Memory & FLOPs:** Reported separately (MB / FLOPs); never added to latency milliseconds.

---

### Issue 8: Statistical Validity & Inference Bounds
- **Exact McNemar Formula:** Bounded two-sided exact Binomial test $p = \min(1, 2\sum_{k=0}^{\min(b,c)} \binom{b+c}{k} 0.5^{b+c})$.
- **Clustering Caveat:** McNemar p-values on text are descriptive of paired discordance under premise clustering, not cluster-robust hypothesis testing.
- **Independent Task Reporting:** `case-06` and `case-07` reported separately; pooled statistics descriptive only.

---

### Issue 9: Correction of Matrix Summary Consistency Claim
- **Correction:** The cross-document consistency matrix audits **16 dimensions**:
  - **Full Pass:** 15 dimensions (93.75%).
  - **Qualified Pass:** 1 dimension (6.25% — Baseline Code Integrity, formally `NOT_VERIFIABLE`).
  - Prior claims stating "all 16 are full passes" or "15 dimensions represent 100% agreement" have been corrected.

---

### Issue 10: Date Harmonization & Source Code Inventory
- **Execution Date:** Harmonized to **November 10, 2026** across all Version 3.5 deliverables.
- **Repository Source Inventory:** Exactly **28 Python source files** in `src/` (17 in root, 6 in `src/code/`, 5 in `src/text/`).
- **Historical Integrity:** Classified as `NOT_VERIFIABLE` due to the lack of pre-Phase-2 version control.
- **Current Baseline Manifest:** All 28 files cataloged with current SHA-256 hashes in `source_sha256_manifest.csv` and frozen.

