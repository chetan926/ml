# Phase 2.3 Methodological Audit Correction Log
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2.3 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Completed Audit & Resolution Ledger  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Executive Summary

This document provides a comprehensive, forensic audit ledger detailing the resolution of all seven blocking methodological inconsistencies identified in the Phase 2.2 review. All corrections have been systematically incorporated into the Version 3.1 specifications in:  
`d:\ml rearcch project ad\research_upgrade\documentation\phase_2_3/`

---

## 2. Point-by-Point Correction Ledger for the Seven Blocking Issues

### Issue 1: Separate Threshold Selection and Calibration Evaluation
- **Original Inconsistency in Version 3.0:**
  Threshold optimization, probability calibration, and calibration diagnostics (ECE, Brier score) were insufficiently separated, risking data reuse where the calibrator was evaluated on the same conditional slice used for fitting or threshold search.
- **Methodological Correction in Version 3.1:**
  1. Strictly defined 5 group-disjoint partitions:
     - $D_{\text{train}}$: Base candidate generator and heavy model fitting.
     - $D_{\text{val}}$: Routing threshold optimization $(\theta_{\text{low}}, \theta_{\text{high}})$.
     - $D_{\text{cal-prob}}$: Probability calibrator fitting on conditional escalated slice $D_{\text{cal-prob}}^{\text{esc}}$.
     - $D_{\text{cal-conf}}$: Conformal quantile estimation on conditional escalated slice $D_{\text{cal-conf}}^{\text{esc}}$ (Text).
     - $D_{\text{test}}$: Final, untouched evaluation partition (evaluated once).
  2. Calibration diagnostics (ECE, Brier score) are strictly evaluated on independent evaluation data, never on the fitting split.
  3. Thresholds are frozen on $D_{\text{val}}$ before any calibrator is fitted.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 1.2, 3.2)
  - `mathematical_formulations_v3_1.md` (Section 3.1)
  - `algorithm_pseudocode_v3_1.md` (Algorithm 2, Algorithm 7)
  - `experimental_protocol_v3_1.md` (Section 1.1, 2.0)
- **Resolution Status:** **Fully Resolved for Text; Data-Constrained for Code.** (Text has sufficient premises for 5 splits; code uses `case-05` for validation and exploratory calibration).

---

### Issue 2: Correct Threshold Optimization and Calibration Sequence
- **Original Inconsistency in Version 3.0:**
  Threshold optimization was described as searching across thousands of candidate cutoffs while calibration and conformal procedures assumed a fixed escalation policy, creating a circular dependency where changing thresholds alters the calibration population.
- **Methodological Correction in Version 3.1:**
  1. Formalized the exact nine-step sequential execution lifecycle:
     - Step 1: Train candidate generator and heavy models on $D_{\text{train}}$.
     - Step 2: Select routing thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ on $D_{\text{val}}$ via discretized Pareto surface search.
     - Step 3: **FREEZE** routing thresholds.
     - Step 4: Apply frozen policy to $D_{\text{cal-prob}}$, extracting escalated slice $D_{\text{cal-prob}}^{\text{esc}}$.
     - Step 5: Fit probability calibrator $g_{\text{cal}}$ on $D_{\text{cal-prob}}^{\text{esc}}$.
     - Step 6: Apply frozen policy to $D_{\text{cal-conf}}$, extracting escalated slice $D_{\text{cal-conf}}^{\text{esc}}$ (Text).
     - Step 7: Estimate empirical conformal quantile $\hat{q}_{1-\epsilon}$ on $D_{\text{cal-conf}}^{\text{esc}}$ (Text).
     - Step 8: **FREEZE** the complete pipeline.
     - Step 9: Evaluate once on untouched held-out test split $D_{\text{test}}$.
  2. Established the inviolable invariant: Any change to routing thresholds invalidates conditional slices and requires complete re-calibration before test evaluation.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 1.2, 4.3)
  - `algorithm_pseudocode_v3_1.md` (Algorithm 7, Algorithm 4)
  - `system_architecture_v3_1.md` (Section 2, Component layout)
  - `experimental_protocol_v3_1.md` (Section 2.0)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 3: Correct Code-Modality Conformal Prediction Claims
- **Original Inconsistency in Version 3.0:**
  Conformal coverage guarantees were asserted for source code despite `case-05` containing only 15 reference Java programs, which mathematically cannot satisfy finite-sample sample size bounds ($N \ge \lceil (1-\epsilon)/\epsilon \rceil$).
- **Methodological Correction in Version 3.1:**
  1. Strictly separated modality claims:
     - **Text Modality:** Retains formal split inductive conformal prediction with verified marginal coverage $P(Y \in \mathcal{C}(X) \mid \text{Escalated}) \ge 1 - \epsilon$.
     - **Code Modality:** Formally **disclaimed mathematical conformal coverage guarantees**. Code uncertainty is designated as **exploratory calibrated uncertainty routing** using calibrated posterior probability $\hat{p}$, predictive entropy $H(\hat{p})$, and decision bands $[\tau_{\text{non-plag}}, \tau_{\text{plag}}]$.
  2. For source code records, `conformal_prediction_set` is set to `None (Disclaimed for Code)`.
  3. The 15-group constraint is formally documented as a primary experimental limitation.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 3.3)
  - `mathematical_formulations_v3_1.md` (Section 3.3)
  - `algorithm_pseudocode_v3_1.md` (Algorithm 3, Algorithm 6)
  - `system_architecture_v3_1.md` (Section 3, 5)
  - `novelty_and_literature_analysis_v3_1.md` (Section 3.2, 4.0)
- **Resolution Status:** **Fully Resolved (Methodologically Corrected & Disclaimed).**

---

### Issue 4: Resolve Group Dependency and Text Split Leakage
- **Original Inconsistency in Version 3.0:**
  Version 3.0 claimed "100% zero premise overlap" for text while prior audit logs noted one shared premise between `snli_val.parquet` and `snli_test.parquet`. Furthermore, code pairs were at risk of correlation across partitions.
- **Methodological Correction in Version 3.1:**
  1. **Text Modality:**
     - Identified grouping unit as `reference_text` (Premise ID).
     - Explicitly disclosed the forensic reality: `snli_val.parquet` (3,304 premises) and `snli_test.parquet` (3,261 premises) share exactly **one premise sentence** ($99.98\%$ premise isolation).
     - Mandated that the data loader explicitly isolates and excludes this single shared premise sentence during partitioning, ensuring complete group isolation across all 5 splits.
     - Enforced that zero-overlap positive statistics are re-computed dynamically on the new test split rather than assuming legacy figures (119 pairs / 10.57%).
  2. **Code Modality:**
     - Identified grouping unit as programming assignment task / reference program.
     - Enforced strict task boundaries: `case-01` to `case-04` ($N=416$) train, `case-05` ($N=107$) validation/exploratory calibration, `case-06` ($N=105$) and `case-07` ($N=105$) test.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 2.2, 2.4)
  - `experimental_protocol_v3_1.md` (Section 1.1, 1.2)
  - `system_architecture_v3_1.md` (Section 3, Data contracts)
  - `novelty_and_literature_analysis_v3_1.md` (Section 4.0)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 5: Correct Tier-1 High-Risk Flag Semantics
- **Original Inconsistency in Version 3.0:**
  The state `HIGH_RISK_FLAGGED` was described as detecting near-verbatim overlap, but was triggered by a fused logistic candidate score ($s_{\text{cand}} \ge \theta_{\text{high}}$). A candidate score alone does not prove verbatim overlap or misconduct.
- **Methodological Correction in Version 3.1:**
  1. Systematically renamed the operational state:
     `HIGH_RISK_FLAGGED` $\to$ `HIGH_RISK_CANDIDATE_FLAGGED`
  2. Clarified operational semantics:
     - It is a fast **triage signal** indicating extreme similarity across multiple lightweight lexical/token views.
     - It is **NOT proof of academic misconduct or verbatim copying**.
     - Human committee review is strictly required (`human_review_required = True`).
     - Heavy inference is bypassed to conserve computational budget; an `EvidenceDossier` is generated to assist human reviewers.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 1.1, 1.3, 6.0)
  - `mathematical_formulations_v3_1.md` (Section 1.1, 4.1)
  - `algorithm_pseudocode_v3_1.md` (Algorithm 1, 4, 5, 6)
  - `system_architecture_v3_1.md` (Section 3, Dataclasses)
  - `experimental_protocol_v3_1.md` (Section 3.1)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 6: Make Cost Accounting Complete and Reproducible
- **Original Inconsistency in Version 3.0:**
  Cost accounting omitted explicit accounting for AST parsing in evidence generation, lacked clear distinction between cold-start and warm-cache profiling, and conflated sublinear term weighting with sublinear algorithmic complexity.
- **Methodological Correction in Version 3.1:**
  1. Defined a comprehensive **ten-component cost accounting model**:
     $C_{\text{prep}}, C_{\text{vec-text}}, C_{\text{tok-code}}, C_{\text{ast}}, C_{\text{cache}}, C_{\text{Tier1}}, C_{\text{infer-heavy}}, C_{\text{explain}}, M_{\text{RSS}}, C_{\text{e2e}}$.
  2. Explicitly separated cold-start latency from warm-cache latency profiling.
  3. Highlighted cohort scaling: $O(M)$ document representation caching vs. $O(M^2)$ pairwise comparisons.
  4. Clarified that sublinear TF-IDF weighting $\left(1+\log(\text{tf})\right)$ does not imply sublinear time complexity; vectorization remains $O(|D|)$ linear in document length.
  5. Enforced that evidence extraction ($C_{\text{explain}}$) is charged **strictly when human review is required**.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 4.1, 5.2)
  - `mathematical_formulations_v3_1.md` (Section 2.1, 4.1)
  - `algorithm_pseudocode_v3_1.md` (Algorithm 4, Algorithm 6)
  - `system_architecture_v3_1.md` (Section 4.0)
  - `experimental_protocol_v3_1.md` (Section 3.4)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 7: Correct Statistical Testing and Generalization Claims
- **Original Inconsistency in Version 3.0:**
  Code test performance was reported as a pooled metric over 210 pairs, using pair-level McNemar tests to claim statistical significance and implying generalization to unseen programming tasks despite having only two held-out tasks (`case-06` and `case-07`).
- **Methodological Correction in Version 3.1:**
  1. **Mandatory Disaggregated Reporting:** `case-06` (Palindrome, $N=105$) and `case-07` (Sorting, $N=105$) must be reported **separately**. Pooled metrics are strictly descriptive.
  2. **Generalization Disclaimer:** Prohibited claims of cross-task generalization based on a two-task test set.
  3. **Exact Binomial McNemar Test:** Mandated the exact binomial test whenever discordant cell counts $b + c < 25$, replacing the invalid asymptotic $\chi^2$ approximation.
  4. **Multiple Testing Correction:** Mandated the Benjamini-Hochberg False Discovery Rate (FDR) procedure across all comparisons at $q = 0.05$.
  5. **Group-Aware Bootstrap:** Enforced 1,000-resample cluster bootstrap on unique groups for all confidence intervals.
- **Files Updated:**
  - `four_contributions_formal_design_v3_1.md` (Section 6.0)
  - `mathematical_formulations_v3_1.md` (Section 6.0)
  - `system_architecture_v3_1.md` (Section 2, Component layout)
  - `experimental_protocol_v3_1.md` (Section 1.2, 4.0)
  - `novelty_and_literature_analysis_v3_1.md` (Section 4.0)
- **Resolution Status:** **Fully Resolved (Protocol & Statistical Rules Established).**

---
*End of Document: phase_2_3_correction_log.md*

