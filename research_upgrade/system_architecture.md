# System Architecture and Modular Design
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Complete Architectural Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Architectural Philosophy & Isolation Principles

To satisfy the mandatory requirement that **all existing code, datasets, and baseline experimental results remain strictly untouched**, the upgraded system is organized inside an isolated top-level namespace: `research_upgrade/`.

Key principles:
1. **Zero Contamination:** No existing file in `src/` or `experiments/` is modified.
2. **Strict Component Isolation:** Each of the four contributions lives in its own dedicated package with an independent test suite.
3. **Strongly Typed Data Contracts:** All inter-module communication is governed by immutable `dataclasses` with explicit types and validation invariants.
4. **Offline Determinism:** External model weights (`sentence-transformers/all-MiniLM-L6-v2`) and datasets execute in strict offline mode (`HF_HUB_OFFLINE=1`).

---

## 2. Directory Structure & Module Layout

```
research_upgrade/
│
├── config/
│   ├── default_config.yaml           # Global experimental configuration
│   ├── text_config.yaml              # Text hyperparameters (TF-IDF, SBERT, thresholds)
│   └── code_config.yaml              # Code hyperparameters (AST, SVM, RF, thresholds)
│
├── datasets/
│   ├── text_loader.py                # Provenance-verified SNLI proxy loader
│   ├── code_loader.py                # Task-disjoint IR-Plag Java loader
│   └── split_manager.py              # Strict 4-way split manager (train/val/cal/test)
│
├── representations/
│   ├── text/
│   │   ├── lexical_views.py          # Word TF-IDF, Word Jaccard, Char 3-Gram
│   │   └── embedding_views.py        # Fast token embedding proxy, SBERT adapter
│   └── code/
│       ├── token_views.py            # Raw token Jaccard, Identifier-normalized Jaccard
│       ├── ast_views.py              # AST node bigrams, statement sequence patterns
│       └── complexity_views.py       # Operator distribution, cyclomatic complexity
│
├── contribution_1_multiview/
│   ├── text_candidate_generator.py   # Multi-view text candidate screening engine
│   ├── code_candidate_generator.py   # Multi-view code candidate screening engine
│   └── redundancy_analyzer.py        # VIF & feature cross-correlation audit
│
├── contribution_2_uncertainty/
│   ├── calibrator.py                 # Platt scaling & Isotonic regression trainers
│   ├── uncertainty_metrics.py        # ECE, MCE, Brier score, predictive entropy
│   └── conformal_router.py           # Inductive split conformal prediction engine
│
├── contribution_3_budget/
│   ├── cost_model.py                 # Comprehensive per-pair latency & memory accounting
│   ├── pareto_optimizer.py           # 2D grid search & constrained Lagrangian solver
│   └── budget_policy.py              # Runtime routing policy executor
│
├── contribution_4_explainability/
│   ├── text_aligner.py               # Smith-Waterman local alignment & paraphrase finder
│   ├── code_aligner.py               # AST subtree isomorphism & identifier mapping
│   ├── dossier_schema.py             # Strongly typed EvidenceDossier dataclass
│   └── faithfulness_evaluator.py     # Feature ablation drop metric engine
│
├── baselines/
│   ├── setu_adapter.py               # Setu et al. (2025) baseline interface
│   ├── gandhi_adapter.py             # Gandhi et al. (2024) SVM interface
│   ├── mehsen_adapter.py             # Mehsen & Joshi (2024) RF interface
│   └── esplag_adapter.py             # Sulistiani & Karnalim (2019) RKRGST interface
│
├── integration/
│   ├── master_pipeline.py            # Unified end-to-end framework
│   └── batch_evaluator.py            # Cohort-level evaluation harness
│
├── evaluation/
│   ├── classification_metrics.py     # Acc, Prec, Rec, F1, Specificity, ROC-AUC
│   ├── candidate_metrics.py          # Candidate Recall, Pruning Ratio, Call Rate
│   └── explainability_metrics.py     # Faithfulness drop, Coverage, Align precision
│
├── statistical_analysis/
│   ├── bootstrap_ci.py               # 1,000-iteration bootstrap confidence intervals
│   └── significance_tests.py         # Paired Wilcoxon signed-rank & McNemar tests
│
├── visualizations/
│   ├── pareto_plots.py               # Publication-quality Pareto frontier figures
│   ├── calibration_curves.py         # Reliability diagrams and ECE bin plots
│   └── evidence_renderers.py         # HTML/Markdown visual diff & evidence heatmaps
│
├── experiments/
│   ├── run_all_upgraded.py           # Master 10-phase reproduction runner
│   ├── exp1_multiview_sweep.py       # Contribution 1 experiments
│   ├── exp2_calibration_sweep.py     # Contribution 2 experiments
│   ├── exp3_budget_sweep.py          # Contribution 3 experiments
│   ├── exp4_explainability_sweep.py  # Contribution 4 experiments
│   └── exp5_ablation_matrix.py       # Full system ablation suite
│
├── tests/
│   ├── test_multiview.py             # Unit tests for multi-view features
│   ├── test_calibration.py           # Unit tests for ECE & conformal coverage
│   ├── test_budget_optimizer.py      # Unit tests for budget feasibility & costs
│   ├── test_explainability.py        # Unit tests for alignment & faithfulness
│   └── test_leakage.py               # Automated split-isolation verification
│
└── documentation/
    ├── four_contributions_formal_design.md
    ├── mathematical_formulations.md
    ├── algorithm_pseudocode.md
    ├── system_architecture.md
    ├── experimental_protocol.md
    └── novelty_and_literature_analysis.md
```

---

## 3. Strongly Typed Data Contracts

All data passing between components is modeled via typed Python dataclasses:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any

@dataclass(frozen=True)
class SubmissionPair:
    pair_id: str
    modality: str  # 'text' | 'code'
    reference_content: str
    suspicious_content: str
    ground_truth: int  # 0 or 1
    obfuscation_level: Optional[str] = None
    task_id: Optional[str] = None

@dataclass(frozen=True)
class MultiViewFeatures:
    pair_id: str
    modality: str
    feature_vector: Tuple[float, ...]
    feature_names: Tuple[str, ...]
    raw_candidate_score: float
    retention_decision: str  # 'PRUNED' | 'RETAINED'

@dataclass(frozen=True)
class CalibratedPrediction:
    pair_id: str
    raw_margin: float
    calibrated_probability: float
    predictive_entropy: float
    conformal_set: Set[int]
    operational_state: str  # 'LOW_RISK_PRUNED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_FLAGGED'

@dataclass(frozen=True)
class BudgetPolicy:
    target_candidate_recall: float
    max_latency_budget_ms: float
    theta_low: float
    theta_high: float
    expected_latency_ms: float
    is_feasible: bool

@dataclass
class EvidenceDossier:
    pair_id: str
    modality: str
    calibrated_probability: float
    operational_state: str
    predictive_entropy: float
    conformal_set: Set[int]
    matching_spans: List[Dict[str, Any]]
    transformation_ledger: Dict[str, Any]
    faithfulness_score: float
    coverage_score: float
    narrative_summary: str
    provenance_metadata: Dict[str, str] = field(default_factory=dict)
```

---

## 4. Class Design & Component Interfaces

### 4.1 Interface Between Contribution 1 and Contribution 2
- **Interface Contract:** `MultiViewCandidateGenerator` $\to$ `SubmissionPair` or `MultiViewFeatures`.
- **Output Condition:**
  - If `retention_decision == 'PRUNED'`, the candidate pair immediately terminates with `OperationalState.LOW_RISK_PRUNED`.
  - If `retention_decision == 'RETAINED'`, the pair passes to `CalibratedUncertaintyRouter`.

### 4.2 Interface Between Contribution 2 and Contribution 3
- **Interface Contract:** `BudgetConstrainedOptimizer` determines `theta_low` and `theta_high` based on validation cost curves and conformal quantiles $\hat{q}_{1-\epsilon}$.
- **Routing Invariant:**
  - `p_calibrated < theta_low` $\implies$ `LOW_RISK_PRUNED` (Zero expensive calls).
  - `p_calibrated >= theta_high` $\implies$ `HIGH_RISK_FLAGGED` (Direct Tier-1 flag; zero heavy calls).
  - `theta_low <= p_calibrated < theta_high` $\implies$ `UNCERTAIN_ESCALATED` (Triggers heavy classifier: SBERT / SVM / RF).

### 4.3 Interface Between Contribution 3 and Contribution 4
- **Interface Contract:** Any pair not pruned (i.e., `FLAGGED` or `ESCALATED`) is passed to `EvidenceAggregator`.
- **Evidence Extraction Invariant:**
  - Text pairs invoke `TextEvidenceExtractor` (Smith-Waterman token alignment + paraphrase span attribution).
  - Code pairs invoke `CodeEvidenceExtractor` (AST subtree isomorphism + identifier mapping dictionary).
  - Both return an instance of `EvidenceDossier`.

---

## 5. Architectural Reliability, Caching, and Resource Management

1. **Sublinear Inverted-Index Caching:**
   - Vocabulary dictionaries and document frequencies are pre-computed during dataset initialization.
   - Per-pair lookup time for TF-IDF is restricted to $< 0.020\text{ ms}$.
2. **Deterministic Seed Locks:**
   - All probabilistic operations, shuffle actions, and bootstrap samples are locked to `seed = 42`.
3. **Graceful Degradation & Safe Fallback:**
   - If an AST parse fails due to an unexpected Java syntax error, the code pre-filter automatically falls back to raw token Jaccard and anonymized keyword histograms, flagging the parse failure in the `EvidenceDossier` provenance metadata.

