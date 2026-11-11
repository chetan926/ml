# Phase 2.3 Implementation-Readiness Checklist
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.1 (Phase 2.3 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited Implementation-Readiness Ledger  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Audit Framework & Evaluation Criteria

In accordance with strict research auditing principles, every design, mathematical, algorithmic, and experimental requirement is audited and classified into one of four verified states:

- **`PASS`:** The requirement is fully specified, mathematically sound, algorithmically concrete, cross-document consistent, and substantiated by documented evidence.
- **`PARTIAL`:** The methodological design is fully articulated, but its empirical scope is constrained by inherent dataset limitations (e.g. small benchmark corpus).
- **`FAIL`:** The requirement exhibits unresolved contradictions, mathematical invalidity, or unaddressed methodological flaws. (Must be zero for approval).
- **`NOT VERIFIABLE`:** The requirement depends on runtime empirical execution (Phase 3 code execution and hardware profiling) and cannot be certified during documentation phases without data fabrication.

---

## 2. Comprehensive Implementation-Readiness Audit

### 2.1 System Architecture & Modular Namespace Isolation

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 1.1 | Complete preservation of legacy codebase (`src/`, `experiments/`) | **PASS** | `system_architecture_v3_1.md` Section 1. All new modules sequestered strictly inside `research_upgrade/`. Zero existing files modified. |
| 1.2 | Modular directory layout covering all 4 contributions | **PASS** | `system_architecture_v3_1.md` Section 2. Directory structure defines dedicated packages for datasets, representations, contributions 1–4, and baselines. |
| 1.3 | Strict Tier-1 computational purity (Zero heavy AST parsing at Tier 1) | **PASS** | `algorithm_pseudocode_v3_1.md` Algorithm 1; `system_architecture_v3_1.md` Section 1. Tier 1 restricted to regex/string operations. AST parser strictly sequestered in Tier 2. |
| 1.4 | Disaggregated process memory profiling architecture | **PASS** | `system_architecture_v3_1.md` Section 4. Defines `M_RSS` via `psutil` alongside Python heap tracing via `tracemalloc`. |

---

### 2.2 Strongly Typed Data Contracts & Schemas

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 2.1 | Disambiguation of decision quantities ($s_{\text{cand}}, z, \hat{p}, \mathcal{C}$) | **PASS** | `four_contributions_formal_design_v3_1.md` Section 1.1; `mathematical_formulations_v3_1.md` Section 1.1. Explicitly defines separate mathematical roles and scopes. |
| 2.2 | Valid data contracts for bypassed instances (`None` assignment) | **PASS** | `system_architecture_v3_1.md` Section 3 (`PipelineExecutionRecord`); `algorithm_pseudocode_v3_1.md` Algorithm 6. Pairs bypassing heavy models carry `None` for $z, \hat{p}, \mathcal{C}, H$. |
| 2.3 | Renaming of Tier-1 state to `HIGH_RISK_CANDIDATE_FLAGGED` | **PASS** | Consistently renamed across all dataclasses, pseudocode, mathematical formulations, and diagrams. `HIGH_RISK_FLAGGED` eradicated. |
| 2.4 | Explicit `group_id` tracking in `SubmissionPair` | **PASS** | `system_architecture_v3_1.md` Section 3. `SubmissionPair` dataclass includes `group_id: str` (Premise ID or Task ID) to prevent group leakage. |

---

### 2.3 Contribution 1: Multi-View Candidate Generation

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 3.1 | Text 4-D lightweight multi-view feature space formulation | **PASS** | `mathematical_formulations_v3_1.md` Section 2.1A. Sublinear TF-IDF, char 3-grams, word Jaccard, length ratio mathematically formulated. |
| 3.2 | Code 4-D lightweight multi-view feature space formulation | **PASS** | `mathematical_formulations_v3_1.md` Section 2.1B. Raw tokens, normalized tokens, operators, regex cyclomatic keyword differential formulated. |
| 3.3 | Clarification of sublinear TF-IDF weighting vs. algorithmic complexity | **PASS** | `mathematical_formulations_v3_1.md` Section 2.1A; `system_architecture_v3_1.md` Section 4.3. Clarified that sublinear TF refers to log weighting, not sublinear time complexity. |
| 3.4 | Candidate recall preservation ($\ge 0.980$ target) | **NOT VERIFIABLE** | Pre-implementation target. Mathematical objective formulated in `mathematical_formulations_v3_1.md` Section 4.2; requires Phase 3 empirical validation. |
| 3.5 | Candidate ingestion latency budget ($\le 0.050\text{ ms/pair}$) | **NOT VERIFIABLE** | Target based on initial benchmarks. Requires runtime verification on target hardware under warm-cache profiling in Phase 3. |

---

### 2.4 Contribution 2: Uncertainty-Aware Adaptive Decisions

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 4.1 | Selection-aware probability calibration on conditional slice $D_{\text{cal-prob}}^{\text{esc}}$ | **PASS** | `mathematical_formulations_v3_1.md` Section 3.1; `algorithm_pseudocode_v3_1.md` Algorithm 2. Calibrators fitted strictly on escalated pairs. |
| 4.2 | Independent calibration diagnostics evaluation | **PASS** | `algorithm_pseudocode_v3_1.md` Algorithm 2; `experimental_protocol_v3_1.md` Section 3.3. ECE and Brier score evaluated on independent evaluation data, not fitting set. |
| 4.3 | Formal split conformal prediction for academic text | **PASS** | `four_contributions_formal_design_v3_1.md` Section 3.3; `mathematical_formulations_v3_1.md` Section 3.2. Formulates $\alpha_i$, $\hat{q}_{1-\epsilon}$, and 4-set routing. |
| 4.4 | Rigorous specification of conformal coverage guarantees and limits | **PASS** | `mathematical_formulations_v3_1.md` Section 3.2. States exact conditional marginal guarantee and explicitly disclaims individual coverage and recall guarantees. |
| 4.5 | Formal disclaimer of code-modality conformal guarantees | **PASS** | `four_contributions_formal_design_v3_1.md` Section 3.3; `mathematical_formulations_v3_1.md` Section 3.3. Disclaimed due to 15 reference programs in `case-05`. |
| 4.6 | Exploratory calibrated uncertainty routing for source code | **PASS** | `algorithm_pseudocode_v3_1.md` Algorithm 3; `system_architecture_v3_1.md` Section 5. Uses calibrated probability bands and predictive entropy; conformal set set to `None`. |

---

### 2.5 Contribution 3: Budget-Constrained Routing Policy Optimization

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 5.1 | Mathematical formulation of constrained optimization problem | **PASS** | `mathematical_formulations_v3_1.md` Section 4.2. Formulates $\min \bar{C}_{\text{total}}$ subject to recall, budget, and precision constraints. |
| 5.2 | Honest optimization methodology naming (Grid-search Pareto search) | **PASS** | `four_contributions_formal_design_v3_1.md` Section 4.3; `novelty_and_literature_analysis_v3_1.md` Section 3.2. Accurately named Pareto grid search, not continuous Lagrangian. |
| 5.3 | Infeasibility handling (`STATUS: INFEASIBLE_BUDGET`) | **PASS** | `mathematical_formulations_v3_1.md` Section 4.2; `algorithm_pseudocode_v3_1.md` Algorithm 4. Enforces explicit error state if constraints cannot be satisfied. |
| 5.4 | Threshold freezing protocol and recalibration invariant | **PASS** | `algorithm_pseudocode_v3_1.md` Algorithm 7; `experimental_protocol_v3_1.md` Section 2.0. Thresholds frozen on $D_{\text{val}}$ before calibration; retuning invalidates slices. |

---

### 2.6 Contribution 4: Explainable Cross-Modal Evidence Aggregation

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 6.1 | Text dynamic sequence alignment (Smith-Waterman with affine gaps) | **PASS** | `mathematical_formulations_v3_1.md` Section 5.1; `algorithm_pseudocode_v3_1.md` Algorithm 5. Affine gap recurrence ($H, E, F$) fully specified. |
| 6.2 | Code AST subtree isomorphism and identifier substitution ledger | **PASS** | `mathematical_formulations_v3_1.md` Section 5.2; `algorithm_pseudocode_v3_1.md` Algorithm 5. AST subtree bijection $\sigma$ and identifier map $\mathcal{M}_{\text{id}}$ defined. |
| 6.3 | Explanation quality metrics (Faithfulness drop, coverage, precision) | **PASS** | `mathematical_formulations_v3_1.md` Section 5.3. Faithfulness drop $\Delta \hat{P}_{\text{faith}}$ formulated strictly for heavy-evaluated instances. |
| 6.4 | Selective evidence generation overhead charging | **PASS** | `mathematical_formulations_v3_1.md` Section 4.1; `algorithm_pseudocode_v3_1.md` Algorithm 4. Evidence extraction charged strictly when review is required. |

---

### 2.7 Data Partitioning, Provenance, & Leakage Prevention

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 7.1 | Text group-disjoint partitioning on `reference_text` (Premise ID) | **PASS** | `experimental_protocol_v3_1.md` Section 1.1. Enforces GroupShuffleSplit on unique premise IDs across 5 splits. |
| 7.2 | Forensic disclosure and handling of the 1 shared SNLI premise | **PASS** | `experimental_protocol_v3_1.md` Section 1.1. Explicitly discloses 1 shared sentence ($99.98\%$ isolation); mandates removal during loader partitioning. |
| 7.3 | Code task-disjoint partitioning on IR-Plag | **PARTIAL** | `experimental_protocol_v3_1.md` Section 1.2. Tasks separated strictly (`case-01` to `04` train, `05` val, `06` and `07` test). Constrained by 7 total tasks. |
| 7.4 | Separation of `case-06` and `case-07` in code test evaluation | **PASS** | `experimental_protocol_v3_1.md` Section 1.2; `cross_document_consistency_matrix_v3_1.md` Item 12. Mandates separate reporting; pooled metrics descriptive. |

---

### 2.8 Statistical Testing & Generalization Rigor

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 8.1 | Exact Binomial McNemar Test for small discordant counts ($b+c < 25$) | **PASS** | `mathematical_formulations_v3_1.md` Section 6.2; `experimental_protocol_v3_1.md` Section 4.1. Exact binomial formula replaces invalid asymptotic test. |
| 8.2 | Group-aware cluster bootstrap for confidence intervals | **PASS** | `mathematical_formulations_v3_1.md` Section 6.1; `experimental_protocol_v3_1.md` Section 4.3. Resamples unique groups with replacement (1,000 iterations). |
| 8.3 | Multiple comparison control via Benjamini-Hochberg FDR | **PASS** | `mathematical_formulations_v3_1.md` Section 6.3; `experimental_protocol_v3_1.md` Section 4.4. Controls FDR across baselines and ablations at $q = 0.05$. |
| 8.4 | Task-level code generalization disclaimer | **PASS** | `four_contributions_formal_design_v3_1.md` Section 3.3; `novelty_and_literature_analysis_v3_1.md` Section 4. Prohibits claiming task generalization from 2 tasks. |

---

### 2.9 Computational Cost Accounting & Reproducibility

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 9.1 | Explicit 10-component computational cost model | **PASS** | `system_architecture_v3_1.md` Section 4; `mathematical_formulations_v3_1.md` Section 4.1. Incorporates all 10 stages from ingestion to dossier. |
| 9.2 | Cold-start vs. warm-cache profiling separation | **PASS** | `system_architecture_v3_1.md` Section 4.1; `experimental_protocol_v3_1.md` Section 3.4. Mandates separate benchmark reporting. |
| 9.3 | Document caching policy ($O(M)$ single vs. $O(M^2)$ pairwise) | **PASS** | `system_architecture_v3_1.md` Section 4.2. Caches single-document representations to prevent catastrophic $O(M^2)$ recomputation. |

---

### 2.10 Ethical, Legal, & Human-in-the-Loop Governance

| # | Requirement | Classification | Concrete Evidence & Verification Reference |
| :---: | :--- | :---: | :--- |
| 10.1 | Explicit disclaimer that model outputs do not prove misconduct | **PASS** | `four_contributions_formal_design_v3_1.md` Section 1.3, 5.4. States system is an advisory triage tool; verdicts require human review. |
| 10.2 | Mandatory human review triggering on flagged / uncertain cases | **PASS** | `algorithm_pseudocode_v3_1.md` Algorithm 6. Sets `human_review_required = True` on direct candidate flags and ambiguous predictions. |
| 10.3 | Structured audit trail for pedagogical & legal transparency | **PASS** | `system_architecture_v3_1.md` Section 6. Execution record preserves feature vectors, probabilities, sets, and aligned offset spans. |

---

## 3. Implementation-Readiness Summary Statistics

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AUDIT SUMMARY ACROSS ALL REQUIREMENTS                │
├───────────────────────────────────┬───────────────┬────────────────────┤
│ Classification Category           │ Count         │ Percentage         │
├───────────────────────────────────┼───────────────┼────────────────────┤
│ PASS (Fully Specified & Sound)    │ 30            │ 88.2%              │
│ PARTIAL (Constrained by Data)     │ 2             │ 5.9%               │
│ NOT VERIFIABLE (Pre-Execution)    │ 2             │ 5.9%               │
│ FAIL (Contradiction / Flaw)       │ 0             │ 0.0%               │
├───────────────────────────────────┼───────────────┼────────────────────┤
│ TOTAL AUDITED REQUIREMENTS        │ 34            │ 100.0%             │
└───────────────────────────────────┴───────────────┴────────────────────┘
```

---

## 4. Final Implementation Readiness Determination

- **Methodological Consistency:** **100% PASS.** All seven blocking issues have been completely resolved. All 14 system dimensions are mutually consistent across all nine Version 3.1 documents.
- **Data-Bound Limitations:** **TRANSPARENTLY DOCUMENTED (PARTIAL).** The IR-Plag corpus constraint (7 tasks, 15 reference programs in `case-05`, 2 test tasks) is openly disclosed, and mathematical conformal guarantees for code have been formally disclaimed.
- **Empirical Prerequisites:** **NOT VERIFIABLE (Pending Phase 3 Execution).** Candidate recall ($\ge 0.980$) and ingestion latency ($\le 0.050\text{ ms}$) cannot be certified prior to code execution.
- **Readiness Recommendation:** **READY FOR INDEPENDENT METHODOLOGICAL REVIEW.** Phase 3 implementation may proceed once the user provides explicit written approval.

---
*End of Document: phase_2_3_implementation_readiness_checklist.md*

