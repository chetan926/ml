# Phase 1: Complete Project Audit & Research Gap Analysis
## Q1 Journal Upgrade: "An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection"

**Document Type:** Formal Scientific Audit & Architectural Blueprint (Phase 1)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Completed & Pending Committee Approval for Phase 2  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals (e.g., *IEEE Transactions on Knowledge and Data Engineering*, *ACM Transactions on Computing Education*, *Computers & Education*, *Information Systems*)

---

## 1. Executive Summary & Objective

The objective of this research initiative is to elevate the existing college-level academic integrity detection prototype into a **rigorous, reproducible, Q1-publication-oriented research framework**. 

The upgraded framework, titled:
> *"An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection"*

addresses the critical tradeoff in automated plagiarism screening: **minimizing expensive inference costs (deep transformers, graph algorithms, combinatorial string tiling) while guaranteeing high candidate recall, well-calibrated decision uncertainty, and interpretable evidence for human adjudicators**.

This Phase 1 document delivers:
1. A forensic audit of the existing codebase, experimental artifacts, and baseline metrics.
2. An unsparing examination of dataset provenance, label validity, and structural limitations.
3. An audit of reference and recent literature implementations.
4. A formal scientific gap analysis detailing why the current prototype cannot withstand Q1 peer review.
5. A comprehensive architectural and mathematical blueprint for the four proposed technical contributions, implemented in an isolated `research_upgrade/` workspace.

---

## 2. Comprehensive Inventory of Existing Project Assets

### 2.1 Codebase Architecture (`src/`)
The existing project contains 18 Python source files in `src/` and submodules `src/code/` and `src/text/`:

| Module | LOC | Primary Functionality | Implementation Maturity |
| :--- | :---: | :--- | :--- |
| [`src/text/lexical_detector.py`](file:///d:/ml%20rearcch%20project%20ad/src/text/lexical_detector.py) | 50 | Sublinear TF-IDF + Cosine distance | Baseline (Stable) |
| [`src/text/semantic_detector.py`](file:///d:/ml%20rearcch%20project%20ad/src/text/semantic_detector.py) | 65 | Sentence-BERT (`all-MiniLM-L6-v2`, 384-D) | Baseline (Stable, CPU/GPU) |
| [`src/text/hybrid_similarity.py`](file:///d:/ml%20rearcch%20project%20ad/src/text/hybrid_similarity.py) | 40 | $\max(S_{\text{TF-IDF}}, S_{\text{SBERT}})$ | Setu et al. (2025) E1 Baseline |
| [`src/text/staged_hybrid_pipeline.py`](file:///d:/ml%20rearcch%20project%20ad/src/text/staged_hybrid_pipeline.py) | 260 | Dual-threshold lexical routing $[\theta_{low}, \theta_{high}]$ | Setu E2 Staged Extension |
| [`src/code_plagiarism.py`](file:///d:/ml%20rearcch%20project%20ad/src/code_plagiarism.py) | 190 | 7-D AST & syntactic feature extractor + RBF SVM | Gandhi et al. (2024) Baseline |
| [`src/code_random_forest.py`](file:///d:/ml%20rearcch%20project%20ad/src/code_random_forest.py) | 142 | 100-tree Random Forest ($\text{depth}=6$) | Mehsen & Joshi (2024) Baseline |
| [`src/code/rkr_gst.py`](file:///d:/ml%20rearcch%20project%20ad/src/code/rkr_gst.py) | 115 | Running-Karp-Rabin Greedy String Tiling ($m=2$) | Sulistiani & Karnalim (2019) Core |
| [`src/code/esplag_detector.py`](file:///d:/ml%20rearcch%20project%20ad/src/code/esplag_detector.py) | 217 | Dynamic cosine pre-filter + RKRGST + Penalty | ES-Plag Complete Pipeline |
| [`src/typed_decision_pipeline.py`](file:///d:/ml%20rearcch%20project%20ad/src/typed_decision_pipeline.py) | 310 | Modality detection + Tri-band routing + Triage | Original Prototype Architecture |
| [`src/threshold_optimizer.py`](file:///d:/ml%20rearcch%20project%20ad/src/threshold_optimizer.py) | 163 | Grid sweep for $\min(\text{Cost})$ s.t. $\text{Recall} \ge \tau$ | Basic Recall-Constrained Sweep |
| [`src/cost_profiler.py`](file:///d:/ml%20rearcch%20project%20ad/src/cost_profiler.py) | 88 | `tracemalloc` + high-res wall clock timing | Basic Cost Accounting |
| [`src/evaluation.py`](file:///d:/ml%20rearcch%20project%20ad/src/evaluation.py) | 480 | Test runner for text and code partitions | Metric Calculation |
| [`src/experiment_evaluator.py`](file:///d:/ml%20rearcch%20project%20ad/src/experiment_evaluator.py) | 710 | Multi-condition benchmarker and logger | Benchmark Pipeline |

### 2.2 Verified Empirical Baseline Metrics (Run 7 Truth)

All baseline numbers are strictly preserved and documented in `experiments/results/final_comparison_table.csv` and `experiments/results/recall_constrained_multi_method_comparison.csv`:

```
Existing Baseline Performance Matrix:
┌───────────────────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────────┬──────────────┬──────────────┐
│ Model / Pipeline Condition            │ Accuracy │ Precision│ Recall   │ F1-Score │ Cand. Recall │ Workload Red.│ Latency/Pair │
├───────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┼──────────────┤
│ Text: TF-IDF Alone (Salton/Sajid)     │ 0.5723   │ 0.4417   │ 0.7846   │ 0.5652   │ N/A          │ 0.00%        │ 0.020 ms     │
│ Text: SBERT Contextual (Setu 2025)    │ 0.7200   │ 0.5697   │ 0.8570   │ 0.6844   │ 1.0000       │ 0.00%        │ 4.046-4.505ms│
│ Text: Setu Full Hybrid E1 (Calib)     │ 0.7373   │ 0.5950   │ 0.8100   │ 0.6861   │ 1.0000       │ 0.00%        │ 4.505 ms     │
│ Text: Setu Staged Hybrid E2 (Ext.)    │ 0.7303   │ 0.5934   │ 0.7592   │ 0.6661   │ 0.8881       │ 25.70%       │ 3.484 ms     │
│ Text: Proposed Typed Pipeline (Cond C)│ 0.7180   │ 0.5751   │ 0.7817   │ 0.6627   │ 0.8598       │ 29.97%       │ 3.138-3.167ms│
├───────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┼──────────────┤
│ Code: Token Jaccard Alone             │ 0.5286   │ 0.5083   │ 0.9020   │ 0.6502   │ N/A          │ N/A          │ 0.050 ms     │
│ Code: Standalone Gandhi SVM (2024)    │ 0.7000   │ 0.6258   │ 0.9510   │ 0.7549   │ 1.0000       │ 0.00%        │ 0.013 ms*    │
│ Code: Standalone Mehsen RF (2024)     │ 0.7333   │ 0.6643   │ 0.9118   │ 0.7686   │ 1.0000       │ 0.00%        │ 0.025 ms*    │
│ Code: Full RKRGST (Sulistiani 2019)   │ 0.4905   │ 0.4878   │ 0.9804   │ 0.6515   │ 1.0000       │ 0.00%        │ 11.236-13.8ms│
│ Code: ES-Plag Dynamic Filter (0.50)   │ 0.4524   │ 0.4656   │ 0.8627   │ 0.6048   │ 0.8627       │ 10.00%       │ 9.914-16.97ms│
│ Code: Proposed Typed Pipeline (Cond C)│ 0.6762   │ 0.6049   │ 0.9608   │ 0.7424   │ 1.0000       │ 36.67%       │ 0.058 ms*    │
└───────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────────┴──────────────┴──────────────┘
*Note: Code inference latencies are evaluated over pre-extracted vectors. Offline 7-D AST feature extraction requires 2.050 ms/pair.
```

---

## 3. Dataset Audit & Methodological Integrity

### 3.1 Natural Language Text Dataset (SNLI / Pereira Proxy)
- **Primary Source:** Stanford Natural Language Inference (SNLI) corpus (Bowman et al., 2015, EMNLP), derived from Flickr30k image captions.
- **Secondary Hosting:** Kaggle `ruvelpereira/mit-plagairism-detection-dataset`.
- **Files:** `datasets/raw/text_mit_plagiarism/snli_val.parquet` (413 KB), `snli_test.parquet` (411 KB).
- **Extracted Benchmark:** `datasets/processed/text_plagiarism_benchmark.csv` (19,666 pairs).
- **Split Construction:** Deterministic sampling (`random_state=42`): Validation sample $N = 3,000$ (1,031 Pos, 1,969 Neg); Test sample $N = 3,000$ (1,063 Pos, 1,937 Neg).
- **Data Integrity Audit:**
  - Zero duplicate rows across the corpus.
  - Zero cross-split contamination (only 1 premise sentence shared, $99.98\%$ isolation).
- **Critical Vulnerability / Scientific Reality:**
  - *The Zero-Overlap Phenomenon:* Exactly **10.57% of positive pairs (119/1,063 test positives) share zero common vocabulary ($\text{TF-IDF} = 0.0000$)**.
  - Any single-view lexical pre-filter with $\theta_{low} > 0$ automatically drops these pairs, creating an empirical ceiling of **88.81% candidate recall**.

### 3.2 Source Code Dataset (IR-Plag Java Corpus)
- **Primary Source:** Oscar Karnalim (2019, *Informatics in Education*, DOI: 10.15388/infedu.2019.15).
- **Corpus:** 467 authentic Java source files across 7 introductory assignments (`case-01` to `case-07`).
- **Benchmark Manifest:** `datasets/processed/code_plagiarism_benchmark.csv` (733 total pairs: 355 Pos, 378 Neg).
- **Partitioning Protocol (Strictly Task-Disjoint):**
  - **Train:** `case-01` to `case-04` ($N = 416$ pairs: 200 Pos, 216 Neg).
  - **Validation:** `case-05` ($N = 107$ pairs: 53 Pos, 54 Neg).
  - **Test:** `case-06` and `case-07` ($N = 210$ pairs: 102 Pos, 108 Neg).
- **Obfuscation Stratification:** Stratified across 6 Faidhi-Robinson levels (L1: Comments, L2: Identifiers, L3: Statement reordering, L4: Control flow, L5: Method abstraction, L6: Compound attack).
- **Data Leakage Verification:** Zero leakage between training and testing. Test tasks (`case-06` Palindrome, `case-07` Bubble Sort) were completely unseen during training and threshold calibration.

---

## 4. Deep Research Gap Analysis (Why Current Implementation Falls Short of Q1)

Although the existing implementation achieved working status, an audit reveals four fundamental methodological shortcomings that would prevent acceptance in a top-tier Q1 journal:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CRITICAL RESEARCH GAPS IN CURRENT PIPELINE                       │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ Existing Prototype Limitation │ Q1 Manuscript Deficiency                               │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Gap 1: Single-View Candidate  │ Relying on a single scalar (TF-IDF or Token Jaccard)   │
│ Generation is Brittle         │ imposes an artificial ceiling (88.81% text recall;     │
│                               │ 0.48% code pruning) due to extreme vocabulary shifts.  │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Gap 2: Ad-Hoc Decision        │ Scalar thresholds ignore prediction uncertainty.       │
│ Boundaries Lack Calibration   │ SVM/RF probability outputs are uncalibrated (high ECE),│
│                               │ offering no statistical confidence guarantees.         │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Gap 3: Cost Accounting Lacks  │ Threshold tuning uses unconstrained heuristics rather  │
│ Formal Optimization           │ than a mathematically principled constrained           │
│                               │ optimization framework (budget-constrained Lagrangian).│
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ Gap 4: Opacity of Scalar      │ The pipeline outputs only a similarity score or label. │
│ Plagiarism Scores             │ Academic integrity review requires granular, faithful  │
│                               │ cross-modal evidence (AST diffs, token maps, tokens).  │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

### Detailed Breakdown of the Four Gaps:

#### Gap 1: Brittle Single-View Candidate Generation
- **In Text:** The existing pipeline filters pairs using TF-IDF cosine similarity alone. Because 10.57% of positive pairs share zero content words, TF-IDF cannot differentiate between independent documents and conceptual paraphrases.
- **In Code:** The existing pipeline filters pairs using basic token multiset Jaccard overlap. Because all students in a class use identical Java boilerplate (`public class`, `main`, `Scanner`), token Jaccard clusters near 0.40–0.70, yielding almost zero pruning capacity (0.48% pruned).
- **Required Innovation:** Multi-view candidate generation combining complementary orthogonal representations (lexical n-grams, character-level n-grams, fast syntactic finger-printing, AST token density) into a unified, validation-learned candidate generator that guarantees $\ge 98\%$ candidate recall with meaningful pruning.

#### Gap 2: Uncalibrated Ad-Hoc Decision Boundaries
- The existing pipeline partitions pairs into three bands using fixed scalar cutoffs $[\theta_{low}, \theta_{high}]$ chosen by simple grid search.
- Neither the SVM decision values nor the Random Forest vote proportions are calibrated probabilities.
- Standard classification metrics (F1, Accuracy) conceal dangerous miscalibration: a detector might predict $\hat{y} = 1$ with "confidence" $0.90$ while being wrong $40\%$ of the time.
- **Required Innovation:** An adaptive uncertainty-aware decision mechanism incorporating **Isotonic / Platt probability calibration**, **Expected Calibration Error (ECE)** minimization, **Brier score** optimization, and **Conformal Prediction set guarantees** for selective classification.

#### Gap 3: Absence of a Formal Budget-Constrained Optimization Framework
- The current `src/threshold_optimizer.py` evaluates a basic grid search over candidate recall without an explicit mathematical objective function.
- In production, institutions face hard operational constraints: e.g., "Maximum 5.0 seconds per 1,000 submissions" or "Maximum $B$ FLOPs/student cohort".
- **Required Innovation:** A formal Lagrangian optimization framework:
  $$\min_{\boldsymbol{\theta}} \mathcal{L}_{\text{total}}(\boldsymbol{\theta}) \quad \text{subject to} \quad \text{CandidateRecall}(\boldsymbol{\theta}) \ge 1 - \alpha \quad \text{and} \quad \text{Latency}(\boldsymbol{\theta}) \le B$$
  Deriving the empirical Pareto frontier across compute budgets and candidate recall constraints.

#### Gap 4: Complete Absence of Explainable Evidence Traces
- A single score (e.g., $s = 0.75$) provides no actionable proof for an academic integrity review board.
- When an instructor accuses a student, they must inspect exact verbatim fragments, substituted synonyms, rearranged code statements, and AST subtree transformations.
- **Required Innovation:** An explainable cross-modal evidence extraction engine that generates:
  1. Phrase-level lexical and semantic alignment heatmaps for text.
  2. Concrete identifier-mapping dictionaries (`i` $\to$ `counter`) and AST node permutation mappings for code.
  3. Automated human-readable audit dossiers with uncertainty margins and evidence fidelity scores.

---

## 5. Architectural Blueprint for `research_upgrade/`

To guarantee **100% preservation of all existing files and experimental results**, the upgraded research framework will be developed in an independent, fully isolated top-level directory:

```
d:\ml rearcch project ad\
├── src/                                  <-- EXISTING: Strictly Preserved
├── experiments/results/                  <-- EXISTING: Strictly Preserved
├── reports/                              <-- EXISTING: Strictly Preserved
├── research_upgrade/                     <-- NEW: Isolated Q1 Implementation
│   ├── config/                           <-- Experiment configurations (YAML/JSON)
│   ├── datasets/                         <-- Dataset loaders & provenance verifiers
│   ├── representations/                  <-- Multi-view feature representations
│   │   ├── text/                         <-- Lexical, Char-ngram, Fast-embed, SBERT
│   │   └── code/                         <-- Token, Identifier-norm, AST-graph, Operators
│   ├── contribution_1_multiview/         <-- Robust Multi-View Candidate Generator
│   ├── contribution_2_uncertainty/       <-- Calibrated Uncertainty & Conformal Router
│   ├── contribution_3_budget/            <-- Budget-Constrained Optimization Engine
│   ├── contribution_4_explainability/    <-- Cross-Modal Evidence Aggregator
│   ├── baselines/                        <-- Standardized baseline adaptors
│   ├── integration/                      <-- Complete unified master framework
│   ├── evaluation/                       <-- Calibration, selective risk, Pareto metrics
│   ├── statistical_analysis/             <-- Bootstrap CI, Wilcoxon, McNemar tests
│   ├── visualizations/                   <-- Publication-quality vector plots (300+ DPI)
│   ├── experiments/                      <-- Execution runners and manifests
│   ├── results/                          <-- Upgraded CSV/JSON experiment ledgers
│   ├── reports/                          <-- Publication-ready manuscript artifacts
│   ├── tests/                            <-- Pytest unit & integration test suite
│   └── documentation/                    <-- Mathematical formulations & proofs
```

---

## 6. Mathematical Formulation Roadmap for the 4 Contributions

### Contribution 1: Robust Multi-View Candidate Generation
- **Text Multi-View Tuple:**
  $$\mathbf{v}_{\text{text}}(D_A, D_B) = \left[ S_{\text{word-tfidf}}, \, S_{\text{char-ngram}}, \, S_{\text{jaccard}}, \, S_{\text{sbert}}, \, \Delta_{\text{len}} \right]$$
- **Code Multi-View Tuple:**
  $$\mathbf{v}_{\text{code}}(C_A, C_B) = \left[ S_{\text{raw-token}}, \, S_{\text{norm-token}}, \, S_{\text{ast-ngram}}, \, S_{\text{operator-hist}}, \, \Delta_{\text{cyclomatic}} \right]$$
- **Filtering Formulation:** Validation-trained projection $g(\mathbf{v}) \in [0, 1]$ maximizing candidate recall at strict target pruning ratios.

### Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism
- **Calibration Engine:** Fit Isotonic Regression and Platt scaling to mapping logits $z$ to calibrated probabilities $P(Y=1 \mid \mathbf{x})$.
- **Uncertainty Quantification:** Measure predictive entropy $\mathcal{H}(p)$ and Expected Calibration Error (ECE):
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Conformal Prediction Band:** Construct non-conformity scores $\alpha_i = 1 - \hat{P}(Y_i \mid \mathbf{x}_i)$ yielding distribution-free coverage guarantee:
  $$P\left(Y_{N+1} \in \mathcal{C}(\mathbf{x}_{N+1})\right) \ge 1 - \epsilon$$

### Contribution 3: Budget-Constrained Routing Optimization
- **Constrained Lagrangian Objective:**
  $$\min_{\boldsymbol{\theta}} \left[ \beta \cdot \text{Cost}(\boldsymbol{\theta}) + (1 - \beta) \cdot \text{Risk}(\boldsymbol{\theta}) \right] \quad \text{s.t.} \quad \text{CandidateRecall}(\boldsymbol{\theta}) \ge R_{\text{target}}$$
- **Empirical Pareto Frontier:** Map candidate recall vs. per-pair latency (ms) across discrete budget levels ($B \in [0.1\text{ ms}, 5.0\text{ ms}]$).

### Contribution 4: Explainable Cross-Modal Evidence Aggregation
- **Text Evidence Engine:** Token sequence alignment via Smith-Waterman local alignment and semantic paraphrase span extraction.
- **Code Evidence Engine:** Normalized AST subtree isomorphism, identifier substitution tables ($\{ \text{old\_id} \to \text{new\_id} \}$), and control-flow statement permutation traces.
- **Evidence Dossier:** Structured JSON/Markdown summary with evidence faithfulness scores and verifiable provenance links.

---

## 7. Recommended Next Steps & Request for Approval

Phase 1 audit and gap analysis is complete. The existing codebase is fully documented and safeguarded.

We are ready to proceed to:
- **PHASE 2: Mathematical Formulation and Detailed Architectural Design**
  1. Formalize the mathematical definitions for all four contributions.
  2. Define the exact interfaces, classes, and data contracts.
  3. Create the `research_upgrade/` directory tree and base configuration files.

*Action Required:* Please review this Phase 1 audit and indicate approval to commence Phase 2.

