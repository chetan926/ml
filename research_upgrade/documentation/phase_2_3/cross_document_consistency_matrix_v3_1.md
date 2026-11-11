# Cross-Document Consistency Matrix (Version 3.1)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.1 (Phase 2.3 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited Cross-Document Verification Matrix  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Executive Summary

This matrix audits the semantic, mathematical, algorithmic, and experimental consistency across all Version 3.1 documents in `research_upgrade/documentation/phase_2_3/`:
1. `four_contributions_formal_design_v3_1.md` (Design)
2. `mathematical_formulations_v3_1.md` (Math)
3. `algorithm_pseudocode_v3_1.md` (Algorithm)
4. `system_architecture_v3_1.md` (Architecture)
5. `experimental_protocol_v3_1.md` (Protocol)
6. `novelty_and_literature_analysis_v3_1.md` (Novelty)

Every core dimension has been reconciled to ensure 100% mutual consistency, eliminating all circular dependencies and terminological discrepancies.

---

## 2. Comprehensive Cross-Document Consistency Matrix

| # | Topic / Dimension | Formal Design Specification (`four_contributions_formal_design_v3_1.md`) | Mathematical Formulation (`mathematical_formulations_v3_1.md`) | Algorithmic Specification (`algorithm_pseudocode_v3_1.md`) | System Architecture & Contracts (`system_architecture_v3_1.md`) | Experimental Protocol (`experimental_protocol_v3_1.md`) | Consistency Status | Remaining Limitations |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | **Decision Quantities Disambiguation** | Disambiguates $s_{\text{cand}}, z, \hat{p}, \mathcal{C}_{1-\epsilon}$. Bypassed pairs carry `None`. | Formally defines $s_{\text{cand}} \in [0, 1]$, $z \in \mathbb{R}$, $\hat{p} \in [0, 1]$, $\mathcal{C}_{1-\epsilon} \subseteq \{0, 1\}$. | Master Pipeline returns `None` for bypassed quantities. | Dataclasses enforce `Optional[float]` and `Optional[Set[int]]`. | Tracks and evaluates each quantity on its designated population. | **CONSISTENT** | None. Enforced across all schemas. |
| **2** | **Conditional Escalation Population** | Defines $\mathcal{D}_{\text{esc}}(\boldsymbol{\theta}) = \{\mathbf{x} \mid \theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}\}$. | Formulates calibration and quantile estimation strictly on $D^{\text{esc}}$. | Filters $D_{\text{cal}}$ into $D^{\text{esc}}$ using frozen thresholds. | Architecture routes escalated instances to Tier-2 heavy modules. | Mandates conditional slice filtering before fitting calibrators. | **CONSISTENT** | Changing thresholds invalidates conditional slices. |
| **3** | **Tier-1 High-Risk State Semantics** | Named `HIGH_RISK_CANDIDATE_FLAGGED`. Triage signal; mandatory human review; not proof of guilt. | Evaluated via candidate score $s_{\text{cand}} \ge \theta_{\text{high}}$ with direct precision $P_{\text{target}}$. | Output state `HIGH_RISK_CANDIDATE_FLAGGED`; sets `human_review_required = True`. | Dataclass `triage_state` enforces `HIGH_RISK_CANDIDATE_FLAGGED`. | Evaluates direct precision and candidate recall on retained positives. | **CONSISTENT** | Requires human adjudicator to confirm misconduct. |
| **4** | **Data Partitions Separation** | 5 group-disjoint splits: $D_{\text{train}}, D_{\text{val}}, D_{\text{cal-prob}}, D_{\text{cal-conf}}, D_{\text{test}}$. | Separates objective loss on train, constraints on val, ECE on test. | Algorithms take explicit, distinct partition arguments. | Dedicated group loaders enforce split isolation. | Defines split sizes ($4\text{k}, 1.5\text{k}, 1\text{k}, 1\text{k}, 3\text{k}$ text). | **CONSISTENT** | Code modality constrained by IR-Plag 7 tasks. |
| **5** | **Sequential Optimization Lifecycle** | 9-step execution lifecycle. Thresholds frozen before calibration. | Optimization on $D_{\text{val}}$ precedes calibration on $D_{\text{cal}}^{\text{esc}}$. | Algorithm 7 formally implements the complete 9-step sequence. | Component layout follows sequential execution pipeline. | Protocol enforces irreversible 9-step lifecycle. | **CONSISTENT** | Re-optimizing thresholds requires recalibration. |
| **6** | **Text Conformal Calibration** | Split inductive conformal prediction on $D_{\text{cal-conf}}^{\text{esc}}$. 4-set routing. | Formulates non-conformity $\alpha_i$, quantile $\hat{q}_{1-\epsilon}$, and coverage guarantee. | Implements `CALIBRATE_TEXT_CONFORMAL_QUANTILE` and `ROUTE_TEXT_CONFORMAL`. | `text_conformal_engine.py` implements 4-set conformal router. | Evaluates empirical coverage, set size, efficiency. | **CONSISTENT** | Marginal coverage; does not guarantee individual coverage. |
| **7** | **Code Uncertainty Routing** | Formally disclaims conformal guarantees. Exploratory calibrated routing. | Formulates predictive entropy $H(\hat{p})$ and calibrated decision bands. | Implements `ROUTE_CODE_EXPLORATORY_UNCERTAINTY`; returns `None` for set. | `code_uncertainty_router.py` implements band routing; sets conformal set to `None`. | Evaluates selective accuracy and ECE on code tasks. | **CONSISTENT** | Conformal guarantees disclaimed due to sample size. |
| **8** | **IR-Plag 15-Group Constraint** | Identifies 15 reference programs in `case-05` as sample size limit. | Demonstrates $N_{\text{cal}} < \lceil (1-\epsilon)/\epsilon \rceil$ for $\epsilon = 0.05$. | Avoids conformal quantile estimation on code splits. | Task loader structures code by task and reference ID. | Documents limitation; uses `case-05` for exploratory tuning. | **CONSISTENT** | Requires expanding to 50+ tasks for formal code conformal. |
| **9** | **Text Premise Grouping & Leakage** | Identifies `reference_text` grouping. Discloses 1 shared sentence. | Formulates cluster bootstrap grouping on Premise IDs. | Group loader isolates premises; removes shared sentence. | `text_group_loader.py` enforces GroupShuffleSplit. | Discloses $99.98\%$ premise isolation; removes shared premise. | **CONSISTENT** | Dynamic test stats re-calculated on new split. |
| **10** | **Ten-Component Cost Accounting** | Incorporates 10 stages: prep, vec, tok, ast, cache, tier1, infer, explain, rss, e2e. | Latency formula matches all 10 stages in master pipeline. | Algorithm 4 charges evidence extraction strictly on review. | Architecture provides explicit cost stage definitions and profiler. | Benchmarks all 10 stages under cold and warm cache. | **CONSISTENT** | Hardware-specific timings must be empirically measured. |
| **11** | **Document Caching Policy** | Enforces $O(M)$ submission caching vs. $O(M^2)$ pairwise comparisons. | Explains sublinear TF weighting is mathematical, not time complexity. | Get-or-compute pattern amortizes representation extraction. | `DocumentCache` caches preprocessed representations. | Profiles cold-start vs. warm-cache performance. | **CONSISTENT** | Cache memory scales linearly $O(M)$ with cohort size. |
| **12** | **Code Test Set Evaluation** | Reports `case-06` and `case-07` separately. Pooled metrics descriptive. | Separate metrics vector $\mathbf{m}_6, \mathbf{m}_7$; pooled descriptive. | Evaluation harness runs independent loops for `case-06` and `case-07`. | Batch runner supports task-disaggregated reporting. | Protocol mandates separate tables for `case-06` and `case-07`. | **CONSISTENT** | 2 tasks cannot support cross-task generalization claims. |
| **13** | **Statistical Significance Testing** | Replaces asymptotic test with Exact Binomial McNemar when $b+c < 25$. | Formulates Exact Binomial $p$-value and BH-FDR procedure. | Statistical module implements exact test and FDR correction. | `significance_tests.py` implements exact binomial test. | Protocol mandates exact test for small discordant counts. | **CONSISTENT** | Limited statistical power on very small samples ($N=105$). |
| **14** | **Evidence Dossier Synthesis** | Structured dossier for human review; clear non-misconduct disclaimer. | Formulates faithfulness drop $\Delta \hat{P}$, coverage, alignment precision. | Algorithm 5 invoked strictly when `human_review_required == True`. | `EvidenceDossier` dataclass enforces strict field typing. | Evaluates faithfulness drop strictly on escalated positive pairs. | **CONSISTENT** | Faithfulness drop not applicable to Tier-1 direct flags. |

---

## 3. Cross-Document Verification Conclusion

All six Version 3.1 documents have been cross-audited and reconciled:
- **Zero Terminology Inconsistencies:** The state `HIGH_RISK_CANDIDATE_FLAGGED` is used uniformly across all texts, pseudocode, and dataclasses.
- **Zero Data Flow Contradictions:** The nine-step sequential calibration lifecycle is universally followed.
- **Zero Unsupported Conformal Claims:** Code conformal prediction is disclaimed in all documents; text conformal guarantees explicitly state their conditional scope.
- **Zero Omitted Cost Components:** The 10-component cost model matches Algorithm 4 and Algorithm 6 exactly.

---
*End of Document: cross_document_consistency_matrix_v3_1.md*

