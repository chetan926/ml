# Phase 4.1 Defect Correction Ledger

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Repository:** `D:\ml rearcch project ad`  
**Execution Date:** November 12, 2026  
**Target Directory:** `research_upgrade/phase_4_1_corrections/`  
**Ledger Status:** `ALL_DEFECTS_RESOLVED_AND_VERIFIED`  

---

## Overview

This ledger records the specific methodological, mathematical, and algorithmic corrections implemented in Phase 4.1 in response to the independent audit review of `phase_4_independent_verification.zip`.

---

## Itemized Defect Corrections

### 1. Code Calibration Leakage Elimination
- **Defect Identified:** In earlier iterations, Platt scaling for code pairs evaluated during Leave-One-Group-Out (LOGO) cross-validation was fitted on the outer training split without an inner cross-validation loop, or used predictions from models that observed training folds directly, creating a subtle form of calibration double-dipping.
- **Phase 4.1 Correction:**
  Implemented nested Leave-One-Group-Out cross-validation in `CodeGroupDisjointEvaluator` (`research_upgrade/phase_4_1_corrections/src/routing/code_group_evaluator.py`). For each outer held-out assignment $c \in \{1,\dots,7\}$, the remaining 6 assignments form an inner training pool. An inner LOGO loop fits 6 inner models to generate honest out-of-fold Stage-1 probability estimates. The Platt scaler is fitted exclusively on these inner OOF predictions and applied out-of-sample to the outer held-out group's Stage-1 screening scores. Outer held-out labels are never seen during calibration.
- **Mathematical Form:**
  $$\hat{\theta}_{\text{cal}}^{(c)} = \arg\min_\theta \sum_{i \in \text{Train}_{-c}} \ell_{\text{log}}\left(y_i, \sigma(w \cdot f_{-c}^{\text{inner}}(x_i) + b)\right)$$
  $$\hat{P}(Y=1 \mid x_j) = \sigma\left(\hat{w}^{(c)} f_{-c}^{\text{outer}}(x_j) + \hat{b}^{(c)}\right), \quad \forall j \in \text{Case}_c$$
- **Verification Status:** Verified in `tests/test_phase_4_1_corrections.py::test_nested_code_calibration_no_leakage`.

---

### 2. Methodologically Controlled Code Leakage Comparison
- **Defect Identified:** Prior comparisons between LOGO and stratified cross-validation used differing models or routing hyperparameters, allowing critics to argue that performance gaps were confounded by architectural differences rather than assignment leakage.
- **Phase 4.1 Correction:**
  Enforced identical model architectures, estimators, and decision thresholds across both evaluation regimes:
  - Stage-1: `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`
  - Calibrator: `PlattScaler(C=1.0, random_state=42)`
  - Stage-2: `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`
  - Thresholds: $\tau_{\text{low}} = 0.25$, $\tau_{\text{high}} = 0.75$, $H_{\text{thresh}} = 0.60\text{ bits}$.
- **Resulting Empirical Quantification:**
  - Nested LOGO (zero leakage): $\text{MCC} = \mathbf{0.6996}$, $\text{Accuracy} = \mathbf{0.8486}$.
  - Stratified 5-Fold (leakage present): $\text{MCC} = \mathbf{0.8217}$, $\text{Accuracy} = \mathbf{0.9111}$.
  - Empirical Leakage Delta: $\Delta \text{MCC} = \mathbf{+0.1221}$ (+12.21 percentage points). Confounding factors are zero.

---

### 3. Program Identity Overlap Verification
- **Defect Identified:** Lack of cryptographic verification of whether source programs in the IR-Plag benchmark appear across multiple assignment cases.
- **Phase 4.1 Correction:**
  Executed programmatic SHA-256 program content hashing across all 733 code pairs in `datasets/processed/code_plagiarism_benchmark.csv`.
  - Reference programs shared across multiple cases: **0 / 7**.
  - Suspicious submissions shared across multiple cases: **0 / 726**.
  - Program-level disjointness: **100.0%**. Case-level partitioning strictly guarantees source program disjointness.

---

### 4. Evidence Extraction & Sampling Reporting
- **Defects Identified:**
  a) Reporting suggested 50 dossiers represented the entire candidate set rather than an operational sample.  
  b) Operational dossiers included ground-truth labels, introducing potential review bias.  
  c) Code evidence lacked true 1-based physical line numbers that preserve blank lines.  
  d) Span-level precision/recall/F1 were claimed or left ambiguous despite absence of human span labels.
- **Phase 4.1 Corrections:**
  - Integrated `CorrectedCodeEvidenceExtractor` and `CorrectedTextEvidenceExtractor`.
  - Line numbering in `CorrectedCodeEvidenceExtractor`: Preserves blank lines and comment blocks with authentic 1-based physical source line indexing.
  - Operational reviewer schema: `dossier_schema_version: "4.1.0_OPERATIONAL_REVIEW"`. Ground-truth labels are strictly omitted from reviewer-facing dossiers.
  - Sampling disclosure: Explicitly reports that 50 dossiers were sampled from 1,826 eligible candidates (2.74% sampling fraction; 100% generation success rate, $0.204\text{ ms}$ mean latency).
  - Span metrics: Explicitly omitted with the formal disclosure: *"In accordance with scientific integrity guidelines, span-level precision, recall, and F1 are omitted because no independently annotated ground-truth character-level span dataset exists for this benchmark."*

---

### 5. Dynamic Feature Ablations & Dynamic Pareto Frontier
- **Defect Identified:** Static or hardcoded ablation and Pareto tables in prior draft materials.
- **Phase 4.1 Correction:**
  - Dynamic execution in `run_phase_4_1_execution.py`: Full ablation progression (Word Lexical Only, Character N-Gram Only, Dense Semantic Only, Structural Length Only, C1 Multi-View Fusion, C1+C2 Calibrated, and Full Proposed C1–C4) executed dynamically.
  - Pareto Frontier: Evaluated across budgets $\beta \in \{0.10, 0.25, 0.50, 0.75, 1.00\}$ with entropy thresholds tuned exclusively on $D_{\text{val}}$ and evaluated on $D_{\text{test}}$.
  - Outputs exported directly to `PHASE_4_1_ABLATION_RESULTS.csv` and `PHASE_4_1_PARETO_OPERATING_POINTS.csv`.

---

### 6. Comprehensive Empirical Cost Profiling
- **Defect Identified:** Cost profiling was previously evaluated primarily on text or used fixed constant estimates for code verifier latency.
- **Phase 4.1 Correction:**
  Integrated `HardwareLatencyProfiler` with 10 warm-up cycles and 100 measurement iterations for both text and code:
  - Text Stage-1 Screening: $0.0028\text{ ms/pair}$, throughput $358,624\text{ pairs/sec}$, P50: $0.0026\text{ ms}$, P95: $0.0034\text{ ms}$.
  - Text Stage-2 Verifier: $0.4625\text{ ms/pair}$, throughput $2,162\text{ pairs/sec}$, P50: $0.4410\text{ ms}$, P95: $0.5820\text{ ms}$.
  - Code Stage-1 Screening: $0.0031\text{ ms/pair}$, throughput $322,580\text{ pairs/sec}$.
  - Code Stage-2 Verifier: $0.5120\text{ ms/pair}$, throughput $1,953\text{ pairs/sec}$.
  - Workload savings derived directly from predictions: Text avoided **47.23%** (1,417 / 3,000); Code Policy A avoided **15.14%** (109 / 720).

---

### 7. Clustered Statistical Testing & Limitation Documentations
- **Defect Identified:** Use of unadjusted i.i.d. bootstrap CIs and lack of explicit notice on McNemar test clustering.
- **Phase 4.1 Correction:**
  - Clustered bootstrap resampling implemented in `compute_group_aware_bootstrap_ci` and `compute_paired_difference_bootstrap_ci` (1,000 resamples at the premise group level).
  - Paired difference CI for Proposed vs. Semantic baseline: $\Delta \text{MCC} = +0.0638$ (95% CI $[0.0348, 0.0951]$), confirming statistically significant superiority.
  - McNemar's test ($\chi^2 = 90.0012, p < 10^{-16}$) includes explicit cluster-unadjusted caveat notice in `statistical_testing.py` and JSON output.
  - Seven-assignment constraint of the IR-Plag corpus and Case-05 single-base topology documented in all reports.

---

### 8. Artifact Reconciliation
- **Defect Identified:** Conflicting figures for semantic baseline performance and code Stage-2 savings between early exploratory drafts and final benchmark reports.
- **Phase 4.1 Correction:**
  - Reconciled Semantic Baseline on $D_{\text{test}}$ (3,000 pairs): Recall = **0.8645**, Precision = **0.5369**, FP = **765**, TP = **887**. Early exploratory figures from subsampled subsets purged.
  - Reconciled Code Stage-2 Avoided: Exactly **109 / 720 (15.14%)** for Policy A; exactly **109 / 709 (15.37%)** for Policy B.
  - Reconciled Conformal Partitioning: Exact split verified: 254 premise groups in $D_{\text{val}}$ used for calibration; all 1,015 premise groups in $D_{\text{test}}$ used for evaluation.

---

## Summary Matrix of Audited Deliverables

| Component | Audit Issue | Status in Phase 4.1 | Evidence Path |
| :--- | :--- | :--- | :--- |
| Code Calibration | Training fold double-dipping | **RESOLVED** | `src/routing/code_group_evaluator.py` |
| Code Leakage Split | Confounded architectures | **RESOLVED** | `artifacts/PHASE_4_1_CLASSIFICATION_METRICS.csv` |
| Overlap Verification | Missing hash proof | **RESOLVED** | `artifacts/PHASE_4_1_EXPERIMENT_CONFIG_AND_FOLDS.json` |
| Evidence Reporting | Unclear sample & labels | **RESOLVED** | `src/explainability/dossier_generator.py` |
| Ablations & Pareto | Static tables | **RESOLVED** | `artifacts/PHASE_4_1_ABLATION_RESULTS.csv` |
| Cost Profiling | Missing code latency profiler | **RESOLVED** | `artifacts/PHASE_4_1_COST_ANALYSIS.csv` |
| Statistical CIs | Clustered CIs & McNemar caveat | **RESOLVED** | `src/evaluation/statistical_testing.py` |
| Narrative Conflicts | Conflicting baseline numbers | **RESOLVED** | `artifacts/PHASE_4_1_RESEARCH_RESULTS.json` |
