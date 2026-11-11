# System Architecture and Modular Design (Version 3.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.0 (Phase 2.2 Methodological Audit Revision)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Architectural Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Architectural Principles & Isolation Safeguards

To comply with the mandatory preservation rule, the upgraded system is organized inside an isolated top-level namespace: `research_upgrade/`.

### Key Architectural Safeguards:
1. **Zero Contamination:** No existing file in `src/` or `experiments/` is modified.
2. **Strict Tier-1 Ingestion Purity:** Tier-1 feature extractors are strictly restricted to $O(N)$ string and token operations ($\le 0.045\text{ ms/pair}$). Heavy AST parsing ($2.050\text{ ms/pair}$) and Sentence-BERT forward passes ($4.000\text{ ms/pair}$) are sequestered in Tier 2.
3. **Disaggregated Profiling:** System profiling measures process-level Resident Set Size (RSS via `psutil`) alongside Python heap tracing (`tracemalloc`), and reports cached vs. uncached latencies separately.
4. **Valid Data Contracts:** Pairs bypassing the heavy model carry `None` for heavy scores and conformal sets. Bypassed direct flags are never falsely labeled as conformal predictions.

---

## 2. Directory Structure and Component Layout

```
research_upgrade/
│
├── config/
│   ├── text_config.yaml              # Sentence-BERT, TF-IDF, char n-gram parameters
│   └── code_config.yaml              # SVM, Random Forest, AST, token parameters
│
├── datasets/
│   ├── text_group_loader.py          # Group-aware SNLI premise partitioner
│   ├── code_task_loader.py           # Task-disjoint IR-Plag Java loader
│   └── split_verifier.py             # Automated split contamination & leakage auditor
│
├── representations/
│   ├── text/
│   │   ├── lightweight_views.py      # TF-IDF, Word Jaccard, Char 3-Gram, Length ratio
│   │   └── sbert_encoder.py          # Offline Sentence-BERT miniLM wrapper
│   └── code/
│       ├── lightweight_views.py      # Regex raw tokens, Normalized tokens, Operators
│       └── heavy_ast_views.py        # Java AST node bigrams, McCabe cyclomatic parser
│
├── contribution_1_multiview/
│   ├── text_candidate_generator.py   # Tier-1 text candidate screening (< 0.045 ms)
│   ├── code_candidate_generator.py   # Tier-1 code candidate screening (< 0.045 ms)
│   └── redundancy_auditor.py         # Variance Inflation Factor & correlation tests
│
├── contribution_2_uncertainty/
│   ├── probability_calibrator.py     # Platt scaling & Isotonic regression on D_cal_prob_esc
│   ├── calibration_metrics.py        # ECE (10-bin), MCE, Brier Score, Predictive Entropy
│   └── conformal_router.py           # Split conformal quantile & complete 4-set triage
│
├── contribution_3_budget/
│   ├── cost_model.py                 # Multi-stage latency & peak RSS profiler
│   ├── pareto_surface_search.py      # Discretized 2D grid search on D_val
│   └── budget_policy_executor.py     # Runtime routing policy & infeasibility handler
│
├── contribution_4_explainability/
│   ├── text_smith_waterman.py        # Affine gap local sequence alignment
│   ├── code_ast_isomorphism.py       # AST subtree matcher & identifier dictionary
│   ├── dossier_schema.py             # Strongly typed EvidenceDossier dataclass
│   └── faithfulness_evaluator.py     # Decision probability ablation drop engine
│
├── baselines/
│   ├── setu_adapter.py               # Setu et al. (2025) baseline interface
│   ├── gandhi_adapter.py             # Gandhi et al. (2024) SVM interface
│   ├── mehsen_adapter.py             # Mehsen & Joshi (2024) Random Forest interface
│   └── esplag_adapter.py             # Sulistiani & Karnalim (2019) RKRGST interface
│
├── integration/
│   ├── master_pipeline.py            # Unified end-to-end framework
│   └── batch_runner.py               # Cohort-level batch evaluation harness
│
├── evaluation/
│   ├── classification_metrics.py     # Accuracy, Precision, Recall, Specificity, F1
│   ├── candidate_metrics.py          # Candidate Recall, Pruning Ratio, Call Rate
│   └── explainability_metrics.py     # Faithfulness Drop, Coverage, Alignment Precision
│
├── statistical_analysis/
│   ├── group_bootstrap_ci.py         # Group-aware 1,000-resample bootstrap 95% CIs
│   └── significance_tests.py         # Paired Wilcoxon & Exact McNemar test
│
├── visualizations/
│   ├── pareto_plots.py               # Vector graphics of Pareto frontier curves
│   ├── calibration_curves.py         # Reliability diagrams and ECE bin distributions
│   └── evidence_diff_renderers.py    # Side-by-side colorized visual evidence HTML
│
└── documentation/
    ├── four_contributions_formal_design_v3.md
    ├── mathematical_formulations_v3.md
    ├── algorithm_pseudocode_v3.md
    ├── system_architecture_v3.md
    ├── experimental_protocol_v3.md
    ├── novelty_and_literature_analysis_v3.md
    └── phase_2_2_correction_log.md
```

---

## 3. Strongly Typed Data Contracts

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any, Union

@dataclass(frozen=True)
class SubmissionPair:
    pair_id: str
    modality: str  # 'text' | 'code'
    reference_content: str
    suspicious_content: str
    ground_truth: int  # 0 or 1
    group_id: str  # Premise ID (text) or Program/Task ID (code)
    obfuscation_level: Optional[str] = None

@dataclass(frozen=True)
class Tier1CandidateResult:
    pair_id: str
    modality: str
    feature_vector: Tuple[float, ...]
    candidate_score: float
    triage_state: str  # 'LOW_RISK_PRUNED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_FLAGGED'
    is_retained: bool

@dataclass(frozen=True)
class Tier2HeavyResult:
    pair_id: str
    raw_decision_margin: float
    calibrated_probability: float
    predictive_entropy: float
    conformal_set: Set[int]  # {0}, {1}, {0, 1}, or empty set set()
    conformal_action: str  # 'CLEARED_NON_PLAGIARIZED' | 'CONFIRMED_PLAGIARIZED' | 'AMBIGUOUS_HUMAN_REVIEW' | 'ABSTAIN_UNRESOLVED'
    human_review_required: bool

@dataclass(frozen=True)
class BudgetPolicy:
    target_candidate_recall: float
    max_latency_budget_ms: float
    target_direct_precision: float
    theta_low: float
    theta_high: float
    expected_latency_ms: float
    status: str  # 'FEASIBLE' | 'STATUS: INFEASIBLE_BUDGET'

@dataclass
class EvidenceDossier:
    pair_id: str
    modality: str
    operational_state: str
    conformal_action: str
    calibrated_probability: Optional[float]  # None for Tier-1 direct flags
    conformal_set: Optional[List[int]]       # None for Tier-1 direct flags
    predictive_entropy: Optional[float]      # None for Tier-1 direct flags
    matching_spans: List[Dict[str, Any]]
    transformation_ledger: Dict[str, Any]
    faithfulness_score: Optional[float]      # None for Tier-1 direct flags
    coverage_score: float
    narrative_summary: str
    provenance_metadata: Dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class PipelineExecutionRecord:
    pair_id: str
    modality: str
    tier1_candidate_score: float
    tier1_routing_state: str                 # 'LOW_RISK_PRUNED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_FLAGGED'
    heavy_model_raw_score: Optional[float]   # None if bypassed
    calibrated_probability: Optional[float]  # None if bypassed
    conformal_prediction_set: Optional[Set[int]] # None if bypassed
    predictive_entropy: Optional[float]      # None if bypassed
    human_review_required: bool
    evidence_dossier: Optional[EvidenceDossier]
```

---

## 4. Class Design and Component Interfaces

### 4.1 Interface Contract: Contribution 1 $\to$ Master Pipeline
- **Class:** `LightweightTextCandidateGenerator` and `LightweightCodeCandidateGenerator`.
- **API:** `generate_candidate(pair: SubmissionPair, theta_low: float, theta_high: float) -> Tier1CandidateResult`.
- **Behavioral Guarantee:**
  - Guaranteed execution latency $\le 0.045\text{ ms/pair}$.
  - Returns `LOW_RISK_PRUNED` if $s_{\text{cand}} < \theta_{\text{low}}$ (immediate termination, 0 heavy compute).
  - Returns `HIGH_RISK_FLAGGED` if $s_{\text{cand}} \ge \theta_{\text{high}}$ (direct flag for human review, 0 heavy compute).
  - Returns `UNCERTAIN_ESCALATED` if $\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$ (escalates to Tier 2).

### 4.2 Interface Contract: Contribution 2 $\to$ Master Pipeline
- **Class:** `ProbabilityCalibrator` and `ConformalRouter`.
- **Training API:**
  - `fit_calibrator(D_cal_prob_esc) -> CalibratorModel`.
  - `calibrate_conformal_quantile(D_cal_conf_esc, calibrator, epsilon=0.05) -> float`.
- **Inference API:** `route(raw_margin: float) -> Tier2HeavyResult`.
- **Four-Set Output Guarantee:**
  - $\{0\} \implies \text{human\_review\_required = False}$
  - $\{1\} \implies \text{human\_review\_required = True}$
  - $\{0, 1\} \implies \text{human\_review\_required = True}$
  - $\emptyset \implies \text{human\_review\_required = True}$ (`ABSTAIN_UNRESOLVED`)

### 4.3 Interface Contract: Contribution 3 $\to$ Master Pipeline
- **Class:** `ParetoSurfaceOptimizer`.
- **API:** `optimize(D_val, target_recall=0.98, budget_ms=1.50, min_precision=0.90) -> BudgetPolicy`.
- **Safety Guarantee:** If no policy satisfies the constraints on $D_{\text{val}}$, returns `status = 'STATUS: INFEASIBLE_BUDGET'`. Never substitutes an arbitrary hardcoded fallback.

### 4.4 Interface Contract: Contribution 4 $\to$ Master Pipeline
- **Class:** `TextEvidenceExtractor` (Smith-Waterman) and `CodeEvidenceExtractor` (AST subtree isomorphism & identifier mapping).
- **API:** `extract_dossier(pair: SubmissionPair, tier1_res, tier2_res) -> EvidenceDossier`.
- **Invocation Invariant:** Invoked **strictly when `human_review_required == True`**.

---

## 5. Latency Profiling and Memory Measurement Specifications

To ensure scientific measurement rigor:
1. **10-Item Disaggregated Latency Reporting:**
   Every experimental run logs:
   - $t_{\text{prep}}$ (per-submission parsing)
   - $t_{\text{Tier1}}$ (screening)
   - $t_{\text{feat-heavy}}$ (Tier-2 feature extraction)
   - $t_{\text{infer-heavy}}$ (heavy model forward pass)
   - $t_{\text{router}}$ (calibration & quantile lookup)
   - $t_{\text{explain}}$ (evidence extraction)
2. **Warm-Up Protocol:**
   All timed benchmarks execute 5 initial warm-up passes (discarded) followed by 50 timed iterations. We report **Median, Mean, and 95th-Percentile Latency**.
3. **Memory Tracking:**
   Process-level peak Resident Set Size (RSS) is measured via `psutil.Process().memory_info().rss` to capture full OS allocation, supplemented by `tracemalloc` for fine-grained Python heap inspection. Cached vs. uncached measurements are reported in separate columns.

