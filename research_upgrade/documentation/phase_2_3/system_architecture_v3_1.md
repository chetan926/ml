# System Architecture and Modular Design (Version 3.1)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.1 (Phase 2.3 Methodological Consistency & Implementation Readiness)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Finalized Architectural Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Architectural Principles & Isolation Safeguards

To maintain strict scientific and experimental integrity, the upgraded system is organized inside an isolated top-level namespace: `research_upgrade/`.

### Core Architectural Safeguards:
1. **Zero Contamination:** No existing file in `src/`, `experiments/`, or `datasets/` is modified.
2. **Tier-1 Ingestion Purity:** Tier-1 feature extractors are strictly restricted to $O(|D|)$ string and regex operations ($\le 0.045\text{ ms/pair}$). Heavy AST parsing ($2.050\text{ ms/pair}$) and Sentence-BERT forward passes ($4.050\text{ ms/pair}$) are sequestered exclusively in Tier 2.
3. **Disaggregated Profiling:** System profiling measures process-level Resident Set Size (RSS via `psutil`) alongside Python heap tracing (`tracemalloc`), and reports cached vs. uncached latencies separately.
4. **Valid Data Contracts:** Pairs bypassing the heavy model carry `None` for heavy scores and conformal sets. Bypassed direct flags are never falsely labeled as conformal predictions.
5. **Human Review Mandate:** High-risk candidate flags and conformal review actions require human adjudication; the system is an intelligent triage assistant, not an automated disciplinary engine.

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
│   ├── text_candidate_generator.py   # Tier-1 text candidate screening (< 0.045 ms)
│   ├── code_candidate_generator.py   # Tier-1 code candidate screening (< 0.045 ms)
│   └── redundancy_auditor.py         # Variance Inflation Factor & correlation tests
│
├── contribution_2_uncertainty/
│   ├── probability_calibrator.py     # Platt scaling & Isotonic regression on D_cal_prob_esc
│   ├── text_conformal_engine.py      # Inductive split conformal 4-set engine (Text)
│   └── code_uncertainty_router.py    # Exploratory calibrated band router (Code)
│
├── contribution_3_optimization/
│   ├── cost_model.py                 # 10-component computational cost model
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
    └── phase_2_3/                    # Version 3.1 finalized research design
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
    triage_state: str  # 'LOW_RISK_PRUNED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_CANDIDATE_FLAGGED'
    is_retained: bool

@dataclass(frozen=True)
class Tier2HeavyResult:
    pair_id: str
    raw_decision_margin: float
    calibrated_probability: float
    predictive_entropy: float
    conformal_set: Optional[Set[int]]  # {0}, {1}, {0, 1}, set(), or None (for Code)
    conformal_action: str  # 'CLEARED_NON_PLAGIARIZED' | 'CONFIRMED_PLAGIARIZED' | 'AMBIGUOUS_HUMAN_REVIEW' | 'ABSTAIN_UNRESOLVED'
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
    predictive_entropy: Optional[float]
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
    tier1_routing_state: str  # 'LOW_RISK_PRUNED' | 'UNCERTAIN_ESCALATED' | 'HIGH_RISK_CANDIDATE_FLAGGED'
    heavy_model_raw_score: Optional[float]       # None if bypassed
    calibrated_probability: Optional[float]      # None if bypassed
    conformal_prediction_set: Optional[Set[int]] # None if bypassed or code
    predictive_entropy: Optional[float]          # None if bypassed
    human_review_required: bool
    final_adjudication_state: str
    evidence_dossier: Optional[EvidenceDossier]
    latency_breakdown_ms: Dict[str, float]
    memory_rss_bytes: int
```

---

## 4. Ten-Component Computational Cost Accounting Architecture

The system implements a rigorous, disaggregated cost model to track computational overhead:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               TEN EXPLICIT COMPUTATIONAL COST STAGES                                   │
├────┬───────────────────────┬───────────────────────────┬───────────────────────────────────────────────┤
│ #  │ Stage Identifier      │ Scope                     │ Accounting Semantics & Measurement Method     │
├────┼───────────────────────┼───────────────────────────┼───────────────────────────────────────────────┤
│ 1  │ C_prep                │ Ingestion & Normalization │ Regex stripping, lowercasing, comment removal │
│ 2  │ C_vec-text            │ Text Vectorization        │ Sublinear TF-IDF (Amortized O(M) via cache)   │
│ 3  │ C_tok-code            │ Code Tokenization         │ Regex lexer token counts & operator histogram │
│ 4  │ C_ast                 │ Java AST Parsing          │ Strictly Tier 2! Never evaluated at Tier 1.   │
│ 5  │ C_cache               │ Cache Lookup              │ In-memory Python dict hash retrieval (O(1))   │
│ 6  │ C_Tier1               │ Candidate Scoring         │ 4-D dot product + logistic evaluation (<0.045)│
│ 7  │ C_infer-heavy         │ Heavy Classifier Forward  │ Sentence-BERT / Structural Random Forest      │
│ 8  │ C_explain             │ Evidence Synthesis        │ Charged strictly when human review required   │
│ 9  │ M_RSS                 │ Process Memory Footprint  │ Resident Set Size (psutil) + heap tracemalloc │
│ 10 │ C_e2e                 │ End-to-End Latency        │ Full wall-clock duration from ingest to dossier│
└────┴───────────────────────┴───────────────────────────┴───────────────────────────────────────────────┘
```

### 4.1 Cold-Start vs. Warm-Cache Latency Profiling
- **Cold-Start Latency:** Evaluates processing when submission representations must be parsed and indexed from raw disk files. Includes file I/O, regex tokenization, and initial caching.
- **Warm-Cache Latency:** Evaluates pipeline throughput when submission representations are resident in the in-memory `DocumentCache`. Reflects operational deployment in high-volume submission servers.

### 4.2 Cohort Scaling: $O(M)$ Submission Caching vs. $O(M^2)$ Pairwise Comparisons
In an academic class cohort of $M$ students, there are $\binom{M}{2} = \frac{M(M-1)}{2} \approx \frac{1}{2} M^2$ pairwise comparisons.
- **Single-Document Operations ($O(M)$):** Preprocessing, sublinear TF-IDF vectorization, regex token extraction, and AST parsing are single-document operations. By caching representations in `DocumentCache`, each document is processed **exactly once** ($M$ operations total).
- **Pairwise Operations ($O(M^2)$):** Cosine similarities, Jaccard overlaps, candidate scoring, and heavy comparisons occur across all $\approx \frac{1}{2} M^2$ pairs.
- **Catastrophic Failure of Naive Architectures:** Without document caching, recomputing document representations for every pair scales as $O(M^2)$, inflating computation by orders of magnitude. The system architecture strictly enforces $O(M)$ document caching.

### 4.3 Sublinear TF-IDF Clarification
Sublinear term-frequency weighting refers to the formula:
$$\text{tf-sublinear}(t, d) = 1 + \log(\text{tf}(t, d)) \quad \text{for } \text{tf}(t, d) > 0$$
This is a mathematical transformation to prevent repeated words from dominating cosine distance. It has **no relationship to sublinear algorithmic time complexity**; computing term frequencies across a document requires reading every token, which is $O(|D|)$ linear in text length.

---

## 5. Modality-Specific Execution Engine Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODALITY-SPECIFIC EXECUTION ENGINES                                       │
├──────────────────────────────────┬─────────────────────────────────────────────────────────────────────┤
│ Architectural Dimension          │ Academic Text Pipeline              │ Source Code Pipeline          │
├──────────────────────────────────┼─────────────────────────────────────┼───────────────────────────────┤
│ Tier-1 Representation Space      │ TF-IDF, Char 3-Gram, Word Jaccard   │ Regex Tokens, Operators, Cyc  │
│ Tier-1 Computational Overhead    │ < 0.045 ms / pair                   │ < 0.045 ms / pair             │
│ Tier-2 Heavy Representation      │ Sentence-BERT contextual embeddings │ 7-D AST & McCabe structural   │
│ Tier-2 Classifier                │ Bi-Encoder Cosine Margin Classifier │ Random Forest Ensemble (100t) │
│ Uncertainty Calibration          │ Isotonic / Platt on D_cal_prob_esc  │ Platt Scaling on D_cal_prob_esc│
│ Uncertainty Routing Mechanism    │ Formal Split Conformal 4-Set Router │ Exploratory Calibrated Band   │
│ Conformal Coverage Guarantee     │ P(Y in C | Esc) >= 1 - epsilon      │ Formally Disclaimed (15 grp)  │
│ Evidence Extraction Technique    │ Smith-Waterman Affine Gap Alignment │ AST Subtree Isomorphism + Id  │
└──────────────────────────────────┴─────────────────────────────────────┴───────────────────────────────┘
```

---

## 6. Evidence Synthesis & Human-in-the-Loop Triage Architecture

The architecture enforces a strict human-in-the-loop governance model:
1. **Automated Triage, Human Verdict:** The system flags suspicious patterns and prepares structured evidence dossiers; human committees retain exclusive authority to determine whether misconduct occurred.
2. **Selective Evidence Generation:** Evidence extraction is charged computationally **only when a case is flagged for human review** (`s_cand >= theta_high` or `conformal_action in {'CONFIRMED_PLAGIARIZED', 'AMBIGUOUS_HUMAN_REVIEW', 'ABSTAIN_UNRESOLVED'}`). Cleared pairs bypass evidence generation entirely.
3. **Structured Audit Trail:** Every execution record contains full feature vectors, operational states, timestamps, memory metrics, and latency breakdowns to guarantee reproducibility and legal auditability.

---
*End of Document: system_architecture_v3_1.md*

