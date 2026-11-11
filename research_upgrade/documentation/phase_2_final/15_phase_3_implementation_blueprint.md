# Phase 3 Implementation Blueprint: Modular Engineering Specifications (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 15 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** ACTIONABLE & IMPLEMENTATION-READY (Blocked Pending Approval)

---

## 1. Blueprint Overview & Engineering Principles

This blueprint provides the definitive, modular implementation specification for Phase 3. Implementation is structured into eight sequential, test-driven phases (Phase 3.1 to Phase 3.8).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3 IMPLEMENTATION ROADMAP                              │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
    ┌───────────────────┬────────────────┼───────────────────┬─────────────────┐
    ▼                   ▼                ▼                   ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Phase 3.1:    │ │ Phase 3.2:    │ │ Phase 3.3:    │ │ Phase 3.4:    │ │ Phase 3.5:    │
│ Governance &  │ │ Data Loaders  │ │ Multi-View    │ │ Uncertainty & │ │ Pareto Budget │
│ Baseline      │ │ & Feature     │ │ Screening     │ │ Conformal     │ │ Optimization  │
│ Freezing      │ │ Extractors    │ │ Fusion        │ │ Routing       │ │ Engine        │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │                 │                 │
        └─────────────────┼─────────────────┼─────────────────┴─────────────────┘
                          │                 │
                          ▼                 ▼
                  ┌───────────────┐ ┌───────────────┐
                  │ Phase 3.6:    │ │ Phase 3.7 &   │
                  │ Multi-Modal   │ │ Phase 3.8:    │
                  │ Attribution   │ │ Evaluation &  │
                  │ Dossiers      │ │ Publication   │
                  └───────────────┘ └───────────────┘
```

---

## 2. Phase 3.1 — Repository Preparation, Baseline Freezing & Governance

- **Objective:** Freeze legacy baseline code, verify the Python runtime, and initialize experiment tracking.
- **Modules to Create:**
  - `research_upgrade/src/__init__.py`
  - `research_upgrade/src/config.py`: Global paths, deterministic random seeds (42), and device configuration.
  - `research_upgrade/src/governance.py`: Runtime verification of baseline hashes against [repository_sha256_manifest.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/repository_sha256_manifest.csv).
- **Interfaces:**
  - Input: Workspace filesystem paths.
  - Output: Boolean `governance_verified` flag, initialized configuration dict.
- **Acceptance Criteria:**
  - All 28 files in `src/` match their registered SHA-256 hashes exactly.
  - PyTorch 2.5 and Hugging Face transformers initialize without warning.
- **Unit & Integration Tests:**
  - `tests/test_governance.py`: Asserts hash matching; verifies non-modification of `src/`.
- **Execution Command:** `.venv\Scripts\pytest tests/test_governance.py`
- **Failure Conditions:** Any checksum mismatch in `src/` aborts execution immediately.

---

## 3. Phase 3.2 — Dataset Loaders & Feature Extraction Pipelines

- **Objective:** Implement group-aware dataset loaders enforcing exact non-overshooting partition allocations and multi-view feature extraction.
- **Modules to Create:**
  - `research_upgrade/src/text_data_loader.py`: Implements `TextBenchmarkDataLoader` using [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/text_exclusion_ledger.csv).
  - `research_upgrade/src/code_data_loader.py`: Implements `CodeBenchmarkDataLoader` supporting Policy A (720 pairs) and Policy B (709 pairs).
  - `research_upgrade/src/text_multiview.py`: Implements `TextMultiViewExtractor` (Char 3-5 gram, Word 1-2 gram, `all-MiniLM-L6-v2`, surface delta).
  - `research_upgrade/src/code_multiview.py`: Implements `CodeMultiViewExtractor` (Normalized token edit distance, AST subtree hashes, 8D software metrics).
  - `research_upgrade/src/code_stage2_detector.py`: Implements hybrid structural-semantic code detector combining AST isomorphism kernels and token embeddings.
- **Interfaces:**
  - `TextBenchmarkDataLoader.get_partition(name: str) -> pd.DataFrame`
  - `TextMultiViewExtractor.extract_features(doc_a: str, doc_b: str) -> np.ndarray` (Shape: `[4]`, range: `[0, 1]`)
  - `CodeMultiViewExtractor.extract_features(code_a: str, code_b: str) -> np.ndarray` (Shape: `[3]`, range: `[0, 1]`)
- **Acceptance Criteria:**
  - Text partition row counts hit exact integers: `train` (4,000), `val` (1,500), `cal_prob` (1,000), `cal_conf` (1,000), `test` (3,000).
  - Pairwise premise overlap across all partitions is exactly 0.
  - Feature values strictly bounded in $[0, 1]$; zero-denominator inputs return $0.0$.
- **Unit & Integration Tests:**
  - `tests/test_text_loader.py`: Verifies row counts, hash membership, zero leakage.
  - `tests/test_code_loader.py`: Verifies Case-06 denominators (103 for Policy A, 101 for Policy B).
  - `tests/test_feature_extractors.py`: Verifies feature vector bounds and empty-string handling.
- **Execution Command:** `.venv\Scripts\pytest tests/test_text_loader.py tests/test_code_loader.py`
- **Failure Conditions:** Any partition count mismatch or cross-split premise leak fails the test suite.

---

## 4. Phase 3.3 — Stage-1 Screening & Linear Fusion Engine

- **Objective:** Fit validation-optimized linear fusion weights $\mathbf{w}^*$ on $D_{\text{train}}$ and select candidate thresholds on $D_{\text{val}}$.
- **Modules to Create:**
  - `research_upgrade/src/fusion_model.py`: Logistic fusion model fitting and inference.
  - `research_upgrade/src/screening_evaluator.py`: Computes $\text{Recall}@K$, pruning ratio $\eta$, and recall on the 119 zero-overlap text pairs.
- **Interfaces:**
  - `FusionModel.fit(X_train: np.ndarray, y_train: np.ndarray)`
  - `FusionModel.predict_score(X: np.ndarray) -> np.ndarray`
- **Acceptance Criteria:**
  - Candidate recall on $D_{\text{val}}^{\text{pos}}$ satisfies $\ge 98.0\%$.
  - Multi-view candidate pool achieves high empirical candidate recall on the known 119 zero-overlap positive text pairs (targeting $\ge 95.0\%$, overcoming lexical screening failure).
  - Benign pair pruning ratio $\ge 70.0\%$.
- **Unit & Integration Tests:**
  - `tests/test_screening_fusion.py`: Evaluates recall constraints and ablation configurations.
- **Execution Command:** `.venv\Scripts\pytest tests/test_screening_fusion.py`

---

## 5. Phase 3.4 — Uncertainty Calibration & Conformal Routing Engine

- **Objective:** Implement Platt probability calibration, exact group split conformal prediction, and 4-state operational routing.
- **Modules to Create:**
  - `research_upgrade/src/calibration.py`: `PlattCalibrator` with single-class protection.
  - `research_upgrade/src/conformal.py`: `GroupConformalPredictor` computing exact finite-sample order statistic $k = \lceil (n+1)(1-\epsilon) \rceil$.
  - `research_upgrade/src/routing_engine.py`: Maps scores to `LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, and `ABSTAIN_UNRESOLVED`.
- **Interfaces:**
  - `PlattCalibrator.fit(logits: np.ndarray, labels: np.ndarray)`
  - `GroupConformalPredictor.calibrate(group_scores: list[float], epsilon: float) -> float`
  - `RoutingEngine.route_pair(...) -> PipelineOutputRecord`
- **Acceptance Criteria:**
  - Platt calibrator fitted strictly on $D_{\text{cal-prob}}^{\text{esc}}$.
  - Conformal quantile $\hat{q}_{1-\epsilon}$ computed strictly over escalated premise groups in $D_{\text{cal-conf}}^{\text{esc}}$ using exact order statistic.
  - Empty conformal sets map strictly to `ABSTAIN_UNRESOLVED`.
  - Code modality uses exploratory raw decision-margin thresholding.
- **Unit & Integration Tests:**
  - `tests/test_conformal.py`: Asserts finite-sample order statistic calculation, coverage, and abstention routing.
  - `tests/test_calibration.py`: Asserts single-class exception handling.
- **Execution Command:** `.venv\Scripts\pytest tests/test_conformal.py tests/test_calibration.py`

---

## 6. Phase 3.5 — Pareto Budget-Constrained Optimization Engine

- **Objective:** Characterize the empirical Pareto frontier across recall, latency, and human review workload over $D_{\text{val}}$.
- **Modules to Create:**
  - `research_upgrade/src/cost_profiler.py`: Measures isolated component wall-clock times, FLOP counts, and RAM footprint.
  - `research_upgrade/src/pareto_optimizer.py`: Dense grid search over $(\theta_{\text{low}}, \theta_{\text{high}}, \gamma_{\text{lower}}, \gamma_{\text{upper}})$.
- **Interfaces:**
  - `ParetoOptimizer.search_frontier(val_data: pd.DataFrame) -> pd.DataFrame`
- **Acceptance Criteria:**
  - Evaluates recall constraints ($\ge 0.95$) and review workload constraints ($\le 0.15$).
  - Outputs full Pareto-optimal parameter set.
- **Unit & Integration Tests:**
  - `tests/test_pareto_optimizer.py`: Validates non-dominated sorting and constraint filtering.
- **Execution Command:** `.venv\Scripts\pytest tests/test_pareto_optimizer.py`

---

## 7. Phase 3.6 — Multi-Modal Explainable Evidence Synthesis

- **Objective:** Implement token-level Myers diff alignment and AST subtree isomorphism extraction.
- **Modules to Create:**
  - `research_upgrade/src/attribution_text.py`: Myers $\mathcal{O}(ND)$ token diff generator and cross-attention extractor.
  - `research_upgrade/src/attribution_code.py`: Canonical AST subtree comparison and identifier renaming mapper.
  - `research_upgrade/src/dossier_generator.py`: Generates standardized HTML/JSON statistical evidence dossiers.
- **Interfaces:**
  - `DossierGenerator.synthesize(record: PipelineOutputRecord) -> Path`
- **Acceptance Criteria:**
  - Evidence dossiers generated for all flagged pairs.
  - Legal disclaimer separating statistical evidence from judicial accusations strictly embedded in all outputs.
- **Unit & Integration Tests:**
  - `tests/test_attribution.py`: Asserts diff generation correctness and JSON schema validity.
- **Execution Command:** `.venv\Scripts\pytest tests/test_attribution.py`

---

## 8. Phase 3.7 & 3.8 — Comprehensive Experimental Evaluation & Reporting

- **Objective:** Execute full evaluation over untouched test sets ($D_{\text{test}}^{\text{esc}}$ for text; held-out tasks for code), run statistical inference, and generate publication-ready tables.
- **Modules to Create:**
  - `research_upgrade/src/run_experiments.py`: Automated master runner executing baselines and proposed framework across all splits.
  - `research_upgrade/src/statistical_tests.py`: Group-aware bootstrap ($B = 1,000$), McNemar tests, DeLong ROC tests.
  - `research_upgrade/src/report_generator.py`: Outputs LaTeX tables, confusion matrices, reliability diagrams, and cost curves.
- **Expected Artifacts:**
  - `research_upgrade/results/tables/candidate_screening_summary.tex`
  - `research_upgrade/results/tables/calibration_metrics_summary.tex` (ECE, MCE, Brier, clipped NLL)
  - `research_upgrade/results/figures/reliability_diagrams.pdf`
  - `research_upgrade/results/figures/pareto_frontier_curves.pdf`
- **Execution Command:** `.venv\Scripts\python research_upgrade/src/run_experiments.py`

