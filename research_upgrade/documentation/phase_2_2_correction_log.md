# Phase 2.2 Forensic Audit Correction Log & Consistency Matrix
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2.2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Completed Audit & Consistency Ledger  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Executive Summary

This document provides a forensic record of all mathematical, algorithmic, architectural, and experimental corrections implemented in the Version 3.0 deliverables under:  
`d:\ml rearcch project ad\research_upgrade\documentation/`

All 11 critique areas raised in the Phase 2.2 audit have been systematically resolved.

---

## 2. Point-by-Point Correction Ledger

### Correction 1: Selection-Aware Calibration and Conformal Prediction
- **Problem Identified:** Tier-1 screening applies to all pairs, but the heavy classifier and conformal router apply only to escalated pairs ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$). Fitting calibrators or quantiles on unselected marginal data introduces selection bias / distribution shift.
- **Correction Made:**
  1. The Tier-1 model $g_{\text{cand}}$ is trained on $D_{\text{train}}$.
  2. Routing thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ are tuned on $D_{\text{val}}$ and **frozen**.
  3. Probability calibration is fitted on the **conditional escalation split**:
     $$D_{\text{cal-prob}}^{\text{esc}} = \{(\mathbf{x}_i, y_i) \in D_{\text{cal-prob}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}}\}$$
  4. Conformal quantile estimation is performed on an independent conditional escalation split:
     $$D_{\text{cal-conf}}^{\text{esc}} = \{(\mathbf{x}_i, y_i) \in D_{\text{cal-conf}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}}\}$$
  5. Conformal validity holds for the conditional escalation sub-population:
     $$P(Y \in \mathcal{C}(X) \mid \text{Escalated}) \ge 1 - \epsilon$$

---

### Correction 2: Resolution of the IR-Plag 15-Group Problem
- **Problem Identified:** `case-05` contains only 15 unique reference Java programs. 5-fold cross-conformal prediction yields only $\approx 3$ reference programs per fold, violating sample size requirements for reliable quantile estimation ($N \ge \lceil (1 - \epsilon)/\epsilon \rceil$).
- **Correction Made:** We explicitly disclaim mathematical conformal guarantees for the source code modality. We state that the code uncertainty mechanism is **exploratory calibrated routing**, and formally document that finite-sample conformal guarantees for code require expanding the corpus to 50+ independent assignment tasks (e.g. SOAP benchmark).

---

### Correction 3: Complete Four-Set Conformal Operational Semantics
- **Problem Identified:** Conformal sets include $\{0\}$, $\{1\}$, $\{0, 1\}$, and the empty set $\emptyset$. The empty set is not, by itself, proof of out-of-distribution (OOD) data.
- **Correction Made:**
  - $\{0\}$: Model-supported negative prediction $\implies$ Cleared Non-Plagiarized (`human_review_required = False`).
  - $\{1\}$: Model-supported positive prediction $\implies$ Flagged Plagiarized (`human_review_required = True`).
  - $\{0, 1\}$: Ambiguous prediction $\implies$ Escalated to Human Review (`human_review_required = True`).
  - $\emptyset$: Abstention / Unresolved prediction $\implies$ Abstain, Mandatory Human Review (`human_review_required = True`).
  - Removed unsubstantiated OOD claims; OOD claims require an independent generative/density OOD detector.

---

### Correction 4: Master Pipeline Data Contracts
- **Problem Identified:** Algorithm 6 previously assigned conformal sets $\{0\}$ and $\{1\}$ to pairs that bypassed the heavy detector.
- **Correction Made:** Strictly separated fields in dataclasses:
  - For pairs that bypass the heavy model (`LOW_RISK_PRUNED` and `HIGH_RISK_FLAGGED`):
    `heavy_model_raw_score = None`, `calibrated_probability = None`, `conformal_prediction_set = None`, `predictive_entropy = None`.
  - A direct Tier-1 flag carries `tier1_routing_state = 'HIGH_RISK_FLAGGED'`, `human_review_required = True`, `conformal_prediction_set = None`.

---

### Correction 5: Calibration Evaluation Protocol
- **Problem Identified:** Algorithm 2 previously evaluated ECE and Brier score on the training data used to fit the calibrator.
- **Correction Made:** Separated fitting diagnostics from independent evaluation. Calibrator is fitted on $D_{\text{cal-prob}}^{\text{esc}}$, and independent diagnostic ECE/Brier scores are evaluated on separate validation escalation data $D_{\text{val}}^{\text{esc}}$. Final test data is never exposed.

---

### Correction 6: Group-Aware Dataset Audit & Statistical Reality
- **Problem Identified:** Lack of verified group counts and claiming zero leakage without an executed audit.
- **Correction Made:**
  - Text: Verified that `snli_val` ($N=9,842$, $3,304$ unique premises) and `snli_test` ($N=9,824$, $3,261$ unique premises) share only 1 reference sentence ($99.98\%$ isolation). Partitioning enforces `GroupShuffleSplit` on `reference_text`.
  - Code: Audited all 7 tasks; documented that `case-05` has 15 reference programs. Acknowledged the severe sample size limitation of having only 2 test tasks (`case-06` and `case-07`). Task-specific results will be reported separately for each.

---

### Correction 7: Computational Cost Model Harmonization
- **Problem Identified:** Evidence dossiers were previously generated for all escalated pairs in pseudocode, while the cost equation charged evidence only for review cases.
- **Correction Made:** Harmonized both: evidence extraction is executed **strictly when `human_review_required == True`**.
  $$\bar{C}_{\text{total}}(\boldsymbol{\theta}) = \bar{C}_{\text{prep}} + \bar{C}_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ \bar{C}_{\text{feat-heavy}} + \bar{C}_{\text{infer-heavy}} + \bar{C}_{\text{router}} \right] + P(\text{Review} \mid \boldsymbol{\theta}) \cdot \bar{C}_{\text{explain}}$$
  Where $P(\text{Review} \mid \boldsymbol{\theta}) = P(\text{Flagged}_{\text{Tier1}}) + P(\text{Escalated}) \cdot P(\mathcal{C} \in \{\{1\}, \{0, 1\}, \emptyset\} \mid \text{Escalated})$.
  Distinguished cached vs. uncached, per-submission vs. per-pair, and process RSS via `psutil`.

---

### Correction 8: Candidate Generation Definitions
- **Problem Identified:** Ambiguities between set Jaccard and multiset Jaccard; handling of empty inputs and zero denominators.
- **Correction Made:** Explicitly defined set Jaccard over unique character tri-grams and unique tokens. Added explicit zero-division guards ($0.0$ returned if denominator is zero). Defined regex token normalization preserving Java keywords, operators, and atomic string literals (`STRING_LITERAL`).

---

### Correction 9: Budget Optimizer Disambiguation & Infeasibility
- **Problem Identified:** Mislabeled as a "novel Lagrangian optimizer" and used hardcoded fallback thresholds.
- **Correction Made:** Accurately renamed: **"Recall-Constrained Routing-Policy Optimization via Discretized Pareto Surface Search"**. Removed hardcoded fallback; if constraints cannot be met on $D_{\text{val}}$, outputs **`STATUS: INFEASIBLE_BUDGET`** and reports constraint violations. Dominated grid points are filtered out.

---

### Correction 10: Statistical Evaluation Limitations
- **Problem Identified:** Two-cluster bootstrap on code cannot represent broad task-level variance.
- **Correction Made:** Disclosed that code bootstrap resampling is constrained by having only 2 test tasks (`case-06` and `case-07`). Specified task-specific reporting for each task, Exact McNemar test for small discordant cells ($b + c < 25$), and Benjamini-Hochberg FDR correction.

---

### Correction 11: Novelty Claims Reassessment
- **Problem Identified:** Risk of overclaiming integration of established techniques as novel algorithms.
- **Correction Made:** Reclassified every component into a transparent 5-tier taxonomy (Established, Adaptations, Proposed Integrations, Proposed Mechanisms, Empirical Contributions). Clearly stated what is and is not novel. Added a 21-entry verifiable bibliography with DOIs.

---

## 3. Comprehensive Consistency Matrix

```
┌────────────────────────────────┬──────────────────────────┬────────────────────────────────────────────────────────┬──────────────┬──────────────┬──────────────┬────────────────────────────────────────────────┐
│ Issue                          │ Previous v2.0 Location   │ Correction Made in Version 3.0                         │ Math Consist.│ Algo Consist.│ Exp Feasib.  │ Remaining Limitation                           │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 1. Selection-Aware Calibration │ four_contributions §3.2  │ Calibrator & conformal quantile fitted on D_cal_esc    │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Conditional coverage holds on escalation slice;│
│    and Conformal Shift         │ mathematical_form §3.1   │ (conditioned on theta_low <= s_cand < theta_high)      │              │              │              │ does not provide marginal coverage over pruned.│
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 2. IR-Plag 15-Group Problem    │ experimental_protocol §1 │ Disclaimed mathematical conformal guarantee on code;   │ VERIFIED     │ VERIFIED     │ VERIFIED     │ IR-Plag has only 15 reference programs in c-05;│
│    (Small Sample Starvation)   │ mathematical_form §3.3   │ designated code uncertainty as exploratory routing.    │              │              │              │ requires 50+ tasks for formal finite guarantee.│
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 3. Conformal Set Semantics     │ algorithm_pseudocode §3  │ Defined all 4 sets: {0}, {1}, {0,1}, and Ø (Abstain).  │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Empty set is model abstention, not proof of    │
│    and Empty Set Ø Policy      │ four_contributions §3.3  │ Removed unsubstantiated OOD claims.                    │              │              │              │ out-of-distribution without an OOD detector.   │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 4. Master Pipeline Data        │ algorithm_pseudocode §6  │ Bypassed pairs carry heavy_raw = None, prob = None,    │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Direct Tier-1 flags rely on near-verbatim      │
│    Contracts & None Values     │ system_architecture §3   │ conf_set = None. Flagged pairs carry review_req = True │              │              │              │ lexical overlap, not heavy model scores.       │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 5. Calibration Evaluation      │ algorithm_pseudocode §2  │ Independent ECE and Brier evaluated on D_val_esc;      │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Sample size in small escalation slices limits  │
│    Independence Protocol       │ mathematical_form §3.2   │ calibration fitting performed strictly on D_cal_esc.   │              │              │              │ ECE bin precision when escalation rate is low. │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 6. Group-Aware Data Leakage    │ experimental_protocol §1 │ Text grouped by reference premise; IR-Plag grouped by  │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Evaluated test performance is restricted to    │
│    and Split Isolation         │ novelty_analysis §3      │ source program; test evaluated strictly out-of-sample. │              │              │              │ two Java assignment tasks (case-06 & case-07). │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 7. Cost Model Execution        │ mathematical_form §4.1   │ Harmonized cost equation: evidence extraction charged  │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Sub-0.05 ms Tier-1 latency is a benchmark      │
│    Branch Alignment            │ algorithm_pseudocode §4  │ strictly when human_review_required == True.           │              │              │              │ target pending Phase 3 timed profiling.        │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 8. Candidate Feature Zero      │ mathematical_form §2.1   │ Formulated explicit set Jaccard, regex token norms,    │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Precomputed inverted index must be maintained  │
│    Handling & Pure Tier-1      │ four_contributions §2.2  │ operator vectors, and zero-division fallback guards.   │              │              │              │ in RAM during high-throughput evaluation.      │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 9. Budget Optimizer Labeling   │ four_contributions §4.1  │ Accurately labeled as Discretized Pareto Surface       │ VERIFIED     │ VERIFIED     │ VERIFIED     │ 2D grid search is optimal for 2 parameters;    │
│    and Infeasibility Reporting │ algorithm_pseudocode §4  │ Search. Removed fallback; reports INFEASIBLE_BUDGET.   │              │              │              │ higher dimensions would require MOEA / BO.     │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 10. Statistical Evaluation     │ experimental_protocol §3 │ Added task-specific reporting for case-06/07, Exact    │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Code test set has only 2 clusters; cannot      │
│     and Code Sample Bounds     │ novelty_analysis §4      │ McNemar testing (b+c < 25), and Benjamini-Hochberg FDR.│              │              │              │ claim universal programming generalization.    │
├────────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────────────────────────────┤
│ 11. Honest Novelty Demarcation │ novelty_analysis §2      │ Established 5-tier taxonomy (Established, Adaptations, │ VERIFIED     │ VERIFIED     │ VERIFIED     │ Integrated framework novelty resides in        │
│     and Claim Boundaries       │ four_contributions §6    │ Integrations, Proposed Mechanisms, Empirical Contrib). │              │              │              │ selection-aware cascading and Pareto search.   │
└────────────────────────────────┴──────────────────────────┴────────────────────────────────────────────────────────┴──────────────┴──────────────┴──────────────┴────────────────────────────────────────────────┘
```

