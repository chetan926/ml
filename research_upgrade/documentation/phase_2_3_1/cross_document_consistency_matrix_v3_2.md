# Cross-Document Consistency Matrix (Version 3.2)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.2 (Phase 2.3.1 Deliverable)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited Cross-Document Verification Matrix  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Executive Summary

This matrix audits the semantic, mathematical, algorithmic, and experimental consistency across all Version 3.2 documents in `research_upgrade/documentation/phase_2_3_1/`:
1. `four_contributions_formal_design_v3_2.md` (Design)
2. `mathematical_formulations_v3_2.md` (Math)
3. `algorithm_pseudocode_v3_2.md` (Algorithm)
4. `system_architecture_v3_2.md` (Architecture)
5. `experimental_protocol_v3_2.md` (Protocol)
6. `novelty_and_literature_analysis_v3_2.md` (Novelty)
7. `dataset_accounting_and_partition_audit.md` (Dataset Audit)

Every core dimension has been reconciled to ensure 100% mutual consistency, eliminating all circular dependencies, terminological discrepancies, and unbacked claims.

---

## 2. Comprehensive Cross-Document Consistency Matrix

| # | Topic / Dimension | Formal Design Specification (`four_contributions_formal_design_v3_2.md`) | Mathematical Formulation (`mathematical_formulations_v3_2.md`) | Algorithmic Specification (`algorithm_pseudocode_v3_2.md`) | System Architecture & Contracts (`system_architecture_v3_2.md`) | Experimental Protocol (`experimental_protocol_v3_2.md`) | Consistency Status | Remaining Limitations |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | **Decision Quantities Disambiguation** | Disambiguates $s_{\text{cand}}, z, \hat{p}, \mathcal{C}_{1-\epsilon}$. Bypassed pairs carry `None`. | Formally defines $s_{\text{cand}} \in [0, 1]$, $z \in \mathbb{R}$, $\hat{p} \in [0, 1]$, $\mathcal{C}_{1-\epsilon} \subseteq \{0, 1\}$. | Master Pipeline returns `None` for bypassed quantities. | Dataclasses enforce `Optional[float]` and `Optional[Set[int]]`. | Tracks and evaluates each quantity on its designated population. | **CONSISTENT** | Enforced across all schemas. |
| **2** | **Conditional Escalation Population** | Defines $\mathcal{D}_{\text{esc}}(\boldsymbol{\theta}) = \{\mathbf{x} \mid \theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}\}$. | Formulates calibration and quantile estimation strictly on $D^{\text{esc}}$. | Filters $D_{\text{cal}}$ into $D^{\text{esc}}$ using frozen thresholds. | Architecture routes escalated instances to Tier-2 heavy modules. | Mandates conditional slice filtering before fitting calibrators. | **CONSISTENT** | Changing thresholds invalidates conditional slices. |
| **3** | **Standardized Output States (Issue 4)** | Renamed: `LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, `ABSTAIN_UNRESOLVED`. | Evaluated via candidate cutoffs and conformal sets mapping to the 4 states. | All algorithms output strictly the 4 standardized states. | Dataclass `final_adjudication_state` enforces the 4 states. | Evaluates metrics across the 4 operational categories. | **CONSISTENT** | Requires human committee for disciplinary verdict. |
| **4** | **Four-Level Adjudication Model** | Defines Level 1 (Model), Level 2 (Evidence), Level 3 (Human), Level 4 (Institution). | Mathematical scores represent Level 1 algorithmic outputs. | Level 2 invoked strictly when Level 1 sets review flag. | Architecture diagram explicitly models the 4 levels. | Protocol specifies advisory triage evaluation. | **CONSISTENT** | Model does not automate institutional verdicts. |
| **5** | **Data Partitions Separation** | 5 group-disjoint splits: $D_{\text{train}}, D_{\text{val}}, D_{\text{cal-prob}}, D_{\text{cal-conf}}, D_{\text{test}}$. | Formulates objective loss on train, constraints on val, ECE on test. | Algorithms take explicit, distinct partition arguments. | Dedicated group loaders enforce split isolation. | Defines active splits ($10.5\text{k}$) and reserve ($9.2\text{k}$). | **CONSISTENT** | Code modality constrained by IR-Plag 7 tasks. |
| **6** | **Sequential Optimization Lifecycle** | 7-step execution lifecycle. Thresholds frozen before calibration. | Optimization on $D_{\text{val}}$ precedes calibration on $D_{\text{cal}}^{\text{esc}}$. | Algorithm 7 formally implements the complete sequence. | Component layout follows sequential execution pipeline. | Protocol enforces irreversible 7-step lifecycle. | **CONSISTENT** | Re-optimizing thresholds requires recalibration. |
| **7** | **Text Conformal Calibration (Issue 2)** | Split conformal on $D_{\text{cal-conf}}^{\text{esc}}$. Exact order statistic $k = \lceil (n+1)(1-\epsilon) \rceil$. | Formulates non-conformity $\alpha_i$, order statistic $k$, and $n \ge \lceil (1-\epsilon)/\epsilon \rceil$. | Implements exact order statistic indexing `sorted_scores[k-1]`. | `text_conformal_engine.py` implements 4-set conformal router. | Evaluates empirical coverage, set size, efficiency. | **CONSISTENT** | Marginal coverage; does not guarantee individual coverage. |
| **8** | **Code Uncertainty Routing (Issue 3)** | Formally disclaims conformal guarantees. Exploratory calibrated routing. | Formulates predictive entropy $H(\hat{p})$ and calibrated decision bands. | Implements `ROUTE_CODE_EXPLORATORY_UNCERTAINTY`; returns `None` for set. | `code_uncertainty_router.py` implements band routing; sets set to `None`. | Evaluates selective accuracy and ECE on code tasks. | **CONSISTENT** | Conformal guarantees disclaimed due to sample size. |
| **9** | **IR-Plag Group Structure (Issue 3)** | Identifies 15 reference programs in `case-05` (53 plag pairs share 1 file). | Demonstrates $n = 15 < 19$ violates finite-sample bound. | Evaluates `case-05` via group-aware 5-fold cross-fitting. | Task loader structures code by task and reference ID. | Documents limitation; uses `case-05` for exploratory tuning. | **CONSISTENT** | Requires expanding to 50+ tasks for formal code conformal. |
| **10** | **Text Dataset Reconciliation (Issue 7)** | Reconciles 19,666 pairs (10,500 active, 9,166 reserve). Discloses 1 shared premise. | Formulates cluster bootstrap grouping on Premise IDs. | Group loader isolates premises; removes shared sentence. | `text_group_loader.py` enforces GroupShuffleSplit. | Full reconciliation table matching disk files. | **CONSISTENT** | Dynamic test stats re-calculated on new split. |
| **11** | **Separation of Time & Resources (Issue 5)**| Strictly separates Time $T$ (ms) and Resources $M$ (MB). No adding bytes to ms. | Models $\bar{T}_{\text{total}}(\boldsymbol{\theta})$ in ms and $\mathbf{M}$ in MB independently. | Algorithm 4 optimizes latency model in milliseconds. | Dataclass records `latency_breakdown_ms` and `memory_breakdown_mb`. | Benchmarks time and memory on separate axes. | **CONSISTENT** | Hardware-specific timings must be measured in Phase 3. |
| **12** | **Document Caching Policy** | Enforces $O(M)$ submission caching vs. $O(M^2)$ pairwise comparisons. | Explains sublinear TF weighting is mathematical, not time complexity. | Get-or-compute pattern amortizes representation extraction. | `DocumentCache` caches preprocessed representations. | Profiles cold-start vs. warm-cache performance. | **CONSISTENT** | Cache memory scales linearly $O(M)$ with cohort size. |
| **13** | **Disaggregated Code Test (Issue 6)** | Reports `case-06` and `case-07` separately. Pooled metrics descriptive. | Separate metrics vector $\mathbf{m}_6, \mathbf{m}_7$; pooled descriptive. | Evaluation harness runs independent loops for `case-06` and `case-07`. | Batch runner supports task-disaggregated reporting. | Protocol mandates separate tables for `case-06` and `case-07`. | **CONSISTENT** | 2 tasks cannot support cross-task generalization claims. |
| **14** | **Statistical Significance (Issue 6)** | Replaces asymptotic test with Exact Binomial McNemar when $b+c < 25$. | Formulates Exact Binomial $p$-value and BH-FDR procedure. | Statistical module implements exact test and FDR correction. | `significance_tests.py` implements exact binomial test. | Protocol mandates exact test for small discordant counts. | **CONSISTENT** | Limited statistical power on very small samples ($N=105$). |
| **15** | **Evidence Dossier Synthesis** | Structured dossier for human review; clear non-misconduct disclaimer. | Formulates faithfulness drop $\Delta \hat{P}$, coverage, alignment precision. | Algorithm 5 invoked strictly when `human_review_required == True`. | `EvidenceDossier` dataclass enforces strict field typing. | Evaluates faithfulness drop strictly on escalated positive pairs. | **CONSISTENT** | Faithfulness drop not applicable to Tier-1 direct flags. |

---
*End of Document: cross_document_consistency_matrix_v3_2.md*

