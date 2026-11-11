# System Architecture and Modular Design (Version 3.3)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.3 (Phase 2.3.2 Methodological Audit & Certification)  
**Author:** Senior Machine Learning Researcher, Statistical Methodologist & Independent Research Auditor  
**Date:** November 2026  
**Status:** Certified Final Architectural Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals  
**Source Baseline:** Audited from Version 3.2 (`research_upgrade/documentation/phase_2_3_1/`)

---

## 1. Architectural Principles & Isolation Safeguards

To maintain strict scientific and experimental integrity, the upgraded system is organized inside an isolated top-level namespace: `research_upgrade/`.

### Core Architectural Safeguards:
1. **Zero Contamination:** No existing file in `src/`, `experiments/`, or `datasets/` is modified.
2. **Tier-1 Ingestion Purity:** Tier-1 feature extractors are strictly restricted to $O(|D|)$ string and regex operations ($\le 0.050\text{ ms/pair}$). Heavy AST parsing ($2.050\text{ ms/pair}$) and Sentence-BERT forward passes ($4.050\text{ ms/pair}$) are sequestered exclusively in Tier 2.
3. **Disaggregated Profiling:** Strict dimensional separation of time metrics (milliseconds) and resource metrics (megabytes). Adding memory bytes to latency milliseconds is mathematically prohibited.
4. **Valid Data Contracts:** Pairs bypassing the heavy model carry `None` for heavy scores and conformal sets. Bypassed direct flags are never falsely labeled as conformal predictions.
5. **Four-Level Adjudication Model:** Algorithmic outputs and generated evidence serve strictly as advisory triage tools; legal and disciplinary authority resides solely with human academic committees.

---

## 2. Directory Structure and Component Layout

```
research_upgrade/
│
├── config/
│   ├── text_config.yaml              # Sentence-BERT, TF-IDF, char n-gram parameters
│   └── code_config.yaml              # SVM, Random Forest, AST, regex token parameters
│
├── datasets/
│   ├── text_group_loader.py          # Group-aware SNLI premise partitioner
│   ├── code_task_loader.py           # Task-disjoint IR-Plag Java loader
│   └── split_verifier.py             # Automated split contamination & leakage auditor
│
├── caching/
│   ├── document_cache.py             # O(M) submission-level representation cache
│   └── cache_profiler.py             # Cold-start vs. warm-cache latency auditor
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
│   ├── text_candidate_generator.py   # Tier-1 text candidate screening (< 0.050 ms)
│   ├── code_candidate_generator.py   # Tier-1 code candidate screening (< 0.050 ms)
│   └── redundancy_auditor.py         # Variance Inflation Factor & correlation tests
│
├── contribution_2_uncertainty/
│   ├── probability_calibrator.py     # Platt scaling & Isotonic regression on D_cal_prob_esc (Text)
│   ├── text_group_conformal.py       # Group-level inductive split conformal engine (Text)
│   └── code_raw_router.py            # Exploratory raw score thresholding engine (Code)
│
├── contribution_3_optimization/
│   ├── cost_model.py                 # Time and resource profiling models
│   ├── pareto_surface_search.py      # Discretized 2D grid Pareto search on D_val
│   └── policy_enforcer.py            # Frozen threshold deployment & validation
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
│   ├── master_pipeline.py            # Unified end-to-end framework (Algorithm 6)
│   └── sequential_calibrator.py      # Sequential optimization protocol (Algorithm 7)
│
├── evaluation/
│   ├── classification_metrics.py     # Accuracy, Precision, Recall, Specificity, F1
│   ├── candidate_metrics.py          # Candidate Recall, Pruning Ratio, Escalation Rate
│   └── explainability_metrics.py     # Faithfulness Drop, Coverage, Alignment Precision
│
├── statistical_analysis/
│   ├── group_bootstrap_ci.py         # Group-aware 1,000-resample bootstrap 95% CIs
│   └── significance_tests.py         # Paired Wilcoxon & Exact Binomial McNemar test
│
└── documentation/
    └── phase_2_3_2/                  # Version 3.3 finalized research design
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
    triage_state: str  # 'LOW_RISK_REVIEW_DEFERRED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_CANDIDATE_FLAGGED'
    is_retained: bool

@dataclass(frozen=True)
class Tier2HeavyResult:
    pair_id: str
    raw_decision_margin: float
    calibrated_probability: Optional[float]  # None for Code
    conformal_set: Optional[Set[int]]       # None for Code; {0}, {1}, {0, 1}, set() for Text
    routing_action: str                     # Standardized 4 states
    human_review_required: bool

@dataclass(frozen=True)
class BudgetPolicy:
    target_candidate_recall: float
    max_latency_budget_ms: float
    target_direct_precision: float
    theta_low: float
    theta_high: float
    is_feasible: bool

@dataclass(frozen=True)
class EvidenceDossier:
    pair_id: str
    modality: str
    operational_state: str
    routing_action: str
    calibrated_probability: Optional[float]
    conformal_set: Optional[List[int]]
    raw_decision_margin: Optional[float]
    matching_spans: List[Dict[str, Any]]
    transformations: Dict[str, Any]
    faithfulness_score: Optional[float]
    coverage_score: float
    human_review_required: bool
    narrative_summary: str

@dataclass(frozen=True)
class PipelineExecutionRecord:
    pair_id: str
    modality: str
    group_id: str
    ground_truth: int
    tier1_candidate_score: float
    tier1_routing_state: str                    # 'LOW_RISK_REVIEW_DEFERRED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_CANDIDATE_FLAGGED'
    heavy_model_raw_score: Optional[float]       # None if bypassed
    calibrated_probability: Optional[float]      # None if bypassed or code
    conformal_prediction_set: Optional[Set[int]] # None if bypassed or code
    human_review_required: bool
    final_adjudication_state: str                # Standardized 4 states (Issue 5)
    evidence_dossier: Optional[EvidenceDossier]
    latency_breakdown_ms: Dict[str, float]       # Explicitly time units (ms)
    memory_breakdown_mb: Dict[str, float]        # Explicitly resource units (MB)
```

---

## 4. Cost and Resource Model Architecture (Issue 5)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DISAGGREGATED TIME AND RESOURCE PROFILING                                 │
├────┬───────────────────────┬──────────────┬────────────────────────────────────────────────────────────┤
│ #  │ Dimension Name        │ Unit         │ Profiling Scope & Measurement Tool                         │
├────┼───────────────────────┼──────────────┼────────────────────────────────────────────────────────────┤
│ 1  │ T_prep                │ Milliseconds │ Ingestion, lowercasing, regex stripping (perf_counter_ns)  │
│ 2  │ T_vec-text            │ Milliseconds │ Sublinear TF-IDF vectorization (Amortized O(M) via cache)  │
│ 3  │ T_tok-code            │ Milliseconds │ Code regex tokenization & operator counting (Amortized)   │
│ 4  │ T_ast                 │ Milliseconds │ Java AST parsing (Strictly Tier 2; never at Tier 1)        │
│ 5  │ T_cache               │ Milliseconds │ In-memory Python hash lookup overhead                      │
│ 6  │ T_Tier1               │ Milliseconds │ 4-D dot product + logistic evaluation (< 0.050 ms/pair)   │
│ 7  │ T_infer-heavy         │ Milliseconds │ Sentence-BERT / AST Random Forest forward inference        │
│ 8  │ T_explain             │ Milliseconds │ Smith-Waterman or AST isomorphism (Charged on review only) │
├────┼───────────────────────┼──────────────┼────────────────────────────────────────────────────────────┤
│ 9  │ M_RSS                 │ Megabytes    │ Peak process Resident Set Size (psutil.Process.memory_info)│
│ 10 │ M_cache               │ Megabytes    │ In-memory document cache footprint (sys.getsizeof/tracemall│
│ 11 │ M_model               │ Megabytes    │ Weights resident in memory (PyTorch/joblib serialized size)│
└────┴───────────────────────┴──────────────┴────────────────────────────────────────────────────────────┘
```

### 4.1 Reconciliation of Claimed Latency Values

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               RECONCILIATION OF SYSTEM LATENCY METRICS                                 │
├──────────────┬───────────────┬──────────────────────┬──────────────────────────────────────────────────┤
│ Operation    │ Reported Value│ Status Classification│ Forensic Provenance & Context                    │
├──────────────┼───────────────┼──────────────────────┼──────────────────────────────────────────────────┤
│ Tier-1 Text  │ 0.045 ms/pair │ Measured Benchmark   │ Warm-cache 4-D dot product & logistic on CPU     │
│ Tier-1 Code  │ 0.058 ms/pair │ Historical Result    │ Partial regex feature routing reported in Rep 7  │
│ Tier-1 Target│ 0.050 ms/pair │ Proposed Target      │ Design constraint for budget optimization        │
│ AST Parsing  │ 2.050 ms/pair │ Measured Benchmark   │ Java AST parsing via javalang on single thread   │
│ SBERT Text   │ 4.046 ms/pair │ Measured Benchmark   │ miniLM-L6-v2 embedding forward pass on CPU       │
└──────────────┴───────────────┴──────────────────────┴──────────────────────────────────────────────────┘
```

### 4.2 Standardized Benchmarking Protocol Specification
- **Hardware Environment:** Dedicated x86-64 CPU (pinned core, Intel Core i7 / AMD Ryzen), 32 GB RAM, no GPU acceleration enabled during CPU latency profiling.
- **Software Environment:** Python 3.10+, PyTorch 2.0+, scikit-learn 1.3+, OS Windows / Linux.
- **Warm-Up Protocol:** 100 burn-in iterations prior to measurement collection to prime CPU instruction caches.
- **Repetitions & Aggregation:** 1,000 independent evaluation trials. Timings measured using `time.perf_counter_ns()`. Aggregation reported as **Median and Interquartile Range (IQR)** to eliminate OS scheduler outliers.

### 4.3 Cohort Scaling: $O(M)$ Document Caching vs. $O(M^2)$ Pairwise Comparisons
- In an academic cohort of $M$ submissions, single-document representations (TF-IDF vectors, regex token sets, ASTs) are computed **once** and stored in `DocumentCache` ($O(M)$ operations).
- Pairwise operations (cosine, Jaccard, candidate scoring) scale as $\binom{M}{2} \approx \frac{1}{2} M^2$.
- Without caching, recomputing document representations for every pair scales catastrophically as $O(M^2)$. Caching is strictly enforced.

---

## 5. Four-Level Adjudication Model Architecture (Issue 4)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUR-LEVEL ADJUDICATION FLOW ARCHITECTURE                                │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ LEVEL 1: Algorithmic Output                             │
                       │ • Raw Candidate Score s_cand in [0, 1]                  │
                       │ • Calibrated Posterior Probability p_hat (Text only)    │
                       │ • Conformal Set C in {{0}, {1}, {0,1}, Ø} (Text)        │
                       │ • Raw Decision Margin z in R (Code)                     │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ LEVEL 2: Evidence Dossier Synthesis                     │
                       │ • Invoked strictly when review is required              │
                       │ • Extracts character offsets & AST isomorphic subtrees  │
                       │ • Builds identifier alpha-conversion mapping ledger     │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ LEVEL 3: Human Expert Adjudication                      │
                       │ • Course instructor / Academic integrity committee      │
                       │ • Inspects synthesized EvidenceDossier side-by-side     │
                       │ • Verifies contextual legitimacy (permitted templates)  │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ LEVEL 4: Final Institutional Verdict                    │
                       │ • Disciplinary or pedagogical finding                   │
                       │ • Sole authority of educational institution             │
                       └─────────────────────────────────────────────────────────┘
```

---
*End of Document: system_architecture_v3_3.md*

