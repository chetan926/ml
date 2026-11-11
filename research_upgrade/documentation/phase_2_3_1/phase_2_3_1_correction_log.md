# Phase 2.3.1 Independent Audit Correction Log
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2.3.1 Deliverable)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Completed Audit & Resolution Ledger  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Executive Summary

This document provides a forensic record of all mathematical, algorithmic, architectural, and experimental corrections implemented in the Version 3.2 deliverables under:  
`d:\ml rearcch project ad\research_upgrade\documentation\phase_2_3_1/`

All eight issues raised in the Phase 2.3.1 audit have been systematically resolved.

---

## 2. Point-by-Point Correction Ledger for the Eight Issues

### Issue 1: Independent Calibration Evaluation
- **Original Inconsistency in Version 3.1:**
  Version 3.1 stated that calibration diagnostics would be evaluated on independent validation data, but did not formally enforce that final calibration metrics (ECE, Brier score, NLL) must be calculated on the untouched held-out test set ($D_{\text{test}}^{\text{esc}}$) strictly where probabilities are actually produced, without influencing threshold or model selection.
- **Methodological Correction in Version 3.2:**
  1. Established the 7-step lifecycle: Model training ($D_{\text{train}}$) $\to$ Threshold optimization ($D_{\text{val}}$) $\to$ Threshold freezing $\to$ Probability calibration ($D_{\text{cal-prob}}^{\text{esc}}$) $\to$ Text conformal quantile estimation ($D_{\text{cal-conf}}^{\text{esc}}$) $\to$ Pipeline freezing $\to$ Final evaluation on untouched test data ($D_{\text{test}}$).
  2. Enforced that final calibration metrics (ECE, Brier score, NLL) are evaluated strictly on the untouched test escalation slice $D_{\text{test}}^{\text{esc}}$.
  3. Explicitly distinguished in-sample diagnostics, calibration-fitting performance, independent calibration evaluation, and final test performance.
  4. Enforced that bypassed pairs carry `None` for heavy scores and calibrated probabilities.
- **Files Updated:**
  - `four_contributions_formal_design_v3_2.md` (Section 1.4, 3.2)
  - `mathematical_formulations_v3_2.md` (Section 3.1)
  - `algorithm_pseudocode_v3_2.md` (Algorithm 2, 7)
  - `experimental_protocol_v3_2.md` (Section 2, 3.3)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 2: Text Conformal Prediction and Group Dependence
- **Original Inconsistency in Version 3.1:**
  Version 3.1 described empirical percentile calculation using a generic percentile formula rather than the exact finite-sample order statistic $k = \lceil (n+1)(1-\epsilon) \rceil$, and did not mathematically define what happens when $k > n$ or formalize group-level conformal scores.
- **Methodological Correction in Version 3.2:**
  1. Defined exact finite-sample order statistic: $k = \lceil (n + 1)(1 - \epsilon) \rceil$ for sample size $n = |D_{\text{cal-conf}}^{\text{esc}}|$.
  2. Formulated the finite-sample attainability condition: If $n < \lceil (1-\epsilon)/\epsilon \rceil$, then $k > n$, meaning the quantile cannot be attained without extrapolation; the system must abstain from claiming conformal guarantees and fall back to exploratory uncertainty routing.
  3. Formulated premise-level group conformal prediction: $\alpha_g = \max_{i \in g} \alpha_i$ where $g$ is the reference premise cluster.
  4. Explicitly stated the scope of the guarantee: Marginal conditional coverage $P(Y \in \mathcal{C}(X) \mid \text{Escalated}) \ge 1 - \epsilon$. Disclaimed individual-instance coverage, coverage under distribution shift, and recall guarantees.
- **Files Updated:**
  - `four_contributions_formal_design_v3_2.md` (Section 3.3)
  - `mathematical_formulations_v3_2.md` (Section 3.2)
  - `algorithm_pseudocode_v3_2.md` (Algorithm 3)
  - `experimental_protocol_v3_2.md` (Section 1.1, 3.3)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 3: Code Probability Calibration and 15-Group Limitation
- **Original Inconsistency in Version 3.1:**
  Version 3.1 disclaimed code conformal prediction, but left the exploratory calibration procedure on `case-05` underspecified, risking that cross-fitted results might be misinterpreted as an independent external test.
- **Methodological Correction in Version 3.2:**
  1. Specified a group-aware 5-fold cross-fitting protocol on `case-05` (3 reference programs per fold) for exploratory calibration and threshold validation.
  2. Strictly separated exploratory calibration fitting from test evaluation on `case-06` and `case-07`.
  3. Reported the exact distribution in `case-05`: 1 original reference program, 15 non-plagiarized files, 53 plagiarized variants.
  4. Prohibited describing raw SVM margins or Random Forest tree vote fractions as calibrated probabilities.
  5. Enforced predictive Shannon entropy $H(\hat{p})$ and calibrated decision bands with mandatory human review.
- **Files Updated:**
  - `four_contributions_formal_design_v3_2.md` (Section 3.3)
  - `mathematical_formulations_v3_2.md` (Section 3.3)
  - `algorithm_pseudocode_v3_2.md` (Algorithm 3)
  - `experimental_protocol_v3_2.md` (Section 1.2)
  - `novelty_and_literature_analysis_v3_2.md` (Section 4.0)
- **Resolution Status:** **Fully Resolved (Protocol & Disclaimers Established).**

---

### Issue 4: Rename Model Decision Outputs & Adjudication Disambiguation
- **Original Inconsistency in Version 3.1:**
  Version 3.1 used `LOW_RISK_PRUNED` and `CONFIRMED_PLAGIARIZED`, which could be misconstrued as the model definitively proving plagiarism or establishing innocence.
- **Methodological Correction in Version 3.2:**
  1. Renamed all operational states across the entire project to the mandated standardized nomenclature:
     - `LOW_RISK_REVIEW_DEFERRED` (replaces `LOW_RISK_PRUNED` / `CLEARED_NON_PLAGIARIZED`)
     - `HIGH_RISK_CANDIDATE_FLAGGED` (replaces `HIGH_RISK_FLAGGED` / `CONFIRMED_PLAGIARIZED`)
     - `AMBIGUOUS_HUMAN_REVIEW`
     - `ABSTAIN_UNRESOLVED`
  2. Established the Four-Level Adjudication Model:
     Level 1 (Model Output) $\to$ Level 2 (Evidence Generated) $\to$ Level 3 (Human Adjudication) $\to$ Level 4 (Institutional Verdict).
  3. Explicitly stated that singleton conformal sets $\{0\}$ or $\{1\}$ are model outputs, NOT proof of innocence or guilt.
- **Files Updated:**
  - `four_contributions_formal_design_v3_2.md` (Section 1.2, 1.3)
  - `mathematical_formulations_v3_2.md` (Section 1.1, 3.2)
  - `algorithm_pseudocode_v3_2.md` (Algorithms 1, 3, 4, 5, 6)
  - `system_architecture_v3_2.md` (Section 3, 5)
  - `experimental_protocol_v3_2.md` (Section 3.1)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 5: Correct Cost and Resource Model (Time vs. Resources)
- **Original Inconsistency in Version 3.1:**
  Version 3.1 included resident memory ($M_{\text{RSS}}$) in the total cost model formula alongside latency terms, and had minor discrepancies between 0.045 ms (text benchmark) and 0.050 ms (engineering target).
- **Methodological Correction in Version 3.2:**
  1. Strictly separated Time measurements ($T$ in milliseconds) and Resource measurements ($M$ in megabytes). Memory bytes are **never** added to latency milliseconds.
  2. Formulated expected per-pair wall-clock latency: $\bar{T}_{\text{total}}(\boldsymbol{\theta}) = T_{\text{prep}} + T_{\text{Tier1}} + P(\text{Esc}) \cdot [T_{\text{feat}} + T_{\text{infer}} + T_{\text{router}}] + P(\text{Rev}) \cdot T_{\text{explain}}$.
  3. Formulated memory vector: $\mathbf{M} = [M_{\text{RSS}}, M_{\text{cache}}, M_{\text{model}}]$.
  4. Reconciled and classified all latency values:
     - 0.045 ms: Measured benchmark for warm-cache text candidate scoring on CPU.
     - 0.058 ms: Historical result from Report 7 for partial code feature routing.
     - 0.050 ms: Proposed engineering target budget for Tier-1 screening.
     - 2.050 ms: Measured benchmark for Java AST parsing on CPU.
     - 4.046 ms: Measured benchmark for Sentence-BERT embedding on CPU.
  5. Specified reproducible benchmarking protocol (hardware, software, warmup, repetitions, median/IQR).
  6. Re-clarified that sublinear TF weighting is mathematical log scaling, not sublinear time complexity.
- **Files Updated:**
  - `four_contributions_formal_design_v3_2.md` (Section 4.1, 4.2)
  - `mathematical_formulations_v3_2.md` (Section 2.1A, 4.1, 4.2)
  - `algorithm_pseudocode_v3_2.md` (Algorithm 4)
  - `system_architecture_v3_2.md` (Section 4.0, 4.1, 4.2)
  - `experimental_protocol_v3_2.md` (Section 3.4)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 6: Correct Statistical Inference and Disaggregated Tasks
- **Original Inconsistency in Version 3.1:**
  Version 3.1 required reporting `case-06` and `case-07` separately, but did not fully elaborate on the limitation of having only two independent test clusters when calculating bootstrap confidence intervals.
- **Methodological Correction in Version 3.2:**
  1. Mandated that `case-06` (Palindrome, $N=105$) and `case-07` (Sorting, $N=105$) are reported **strictly as separate task case studies**. Pooled metrics are descriptive summary statistics only.
  2. Explicitly disclaimed broad task generalization; stated that 1,000 bootstrap resamples on 2 clusters cannot establish unseen-task generalization.
  3. Enforced the Exact Binomial McNemar test whenever discordant cell count $b + c < 25$.
  4. Specified cluster bootstrap resampling unique premises for text.
  5. Mandated Benjamini-Hochberg FDR correction across comparisons at $q = 0.05$.
- **Files Updated:**
  - `four_contributions_formal_design_v3_2.md` (Section 6.0)
  - `mathematical_formulations_v3_2.md` (Section 6.1, 6.2, 6.3)
  - `experimental_protocol_v3_2.md` (Section 1.2, 4.0)
  - `novelty_and_literature_analysis_v3_2.md` (Section 4.0)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 7: Reconcile Dataset Accounting (The 9,166 Text Reserve Pairs)
- **Original Inconsistency in Version 3.1:**
  Version 3.1 specified 10,500 active text pairs across 5 partitions, but did not explain the status of the remaining 9,166 pairs from the 19,666 pairs available in `text_plagiarism_benchmark.csv`.
- **Methodological Correction in Version 3.2:**
  1. Authored dedicated forensic document: `dataset_accounting_and_partition_audit.md`.
  2. Reconciled text pairs:
     - Total available in raw files: 19,666 pairs (9,842 in `snli_val`, 9,824 in `snli_test`).
     - Active experimental partitions: 10,500 pairs (4,000 train, 1,500 val, 1,000 cal-prob, 1,000 cal-conf, 3,000 test).
     - Reserve development pairs: $9,842 - 7,500 = 2,342$ unselected development pairs.
     - Reserve test pairs: $9,824 - 3,000 = 6,824$ unselected test pairs.
     - Total reserve pairs: $2,342 + 6,824 = 9,166$ pairs held in reserve for sample size sensitivity analysis.
  3. Reconciled code corpus: 467 Java files across 7 tasks yielding 733 pairs (378 non-plagiarized, 355 plagiarized). Detailed breakdown for each case.
- **Files Updated:**
  - `experimental_protocol_v3_2.md` (Section 1.1, 1.2)
  - `dataset_accounting_and_partition_audit.md` (Entire document)
- **Resolution Status:** **Fully Resolved.**

---

### Issue 8: Correct Implementation-Readiness Claims (5-Tier Taxonomy)
- **Original Inconsistency in Version 3.1:**
  Checklist marked requirements as PASS based on architectural design alone, conflating specification readiness with runtime implementation and empirical verification.
- **Methodological Correction in Version 3.2:**
  1. Replaced generic PASS with the 5-tier evidence taxonomy:
     - `DESIGN_SPECIFIED`: Architecture described and mathematically formulated.
     - `SOURCE_CODE_IMPLEMENTED`: Legacy module inspected and verified in `src/`.
     - `UNIT_TESTED`: Unit tests executed and verified.
     - `INTEGRATION_TESTED`: Integrated pipeline verified.
     - `EMPIRICALLY_VERIFIED`: Quantitative results reproduced and measured.
  2. Applied `NOT VERIFIABLE` to pre-execution runtime targets (e.g. latency $\le 0.050\text{ ms}$, recall $\ge 0.980$).
  3. Verified legacy codebase preservation by comparing actual files in `src/`, `experiments/`, and `reports/`.
- **Files Updated:**
  - `phase_2_3_1_implementation_readiness_checklist.md` (Entire document)
  - `phase_2_3_1_final_audit_report.md` (Section 2)
- **Resolution Status:** **Fully Resolved.**

---
*End of Document: phase_2_3_1_correction_log.md*

