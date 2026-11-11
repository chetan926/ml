# Experimental Protocol and Evaluation Framework
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Complete Experimental Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Strict 4-Way Data Partitioning Protocol

To eliminate data leakage, threshold overfitting, and optimistic calibration bias, we mandate a strict **4-way data partition**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FULL BENCHMARK DATASET                                            │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
         ┌───────────────────┬──────────────────────┴───────────────┬───────────────────┐
         ▼                   ▼                                      ▼                   ▼
┌─────────────────┐ ┌─────────────────┐                    ┌─────────────────┐ ┌─────────────────┐
│  TRAIN SPLIT    │ │ VALIDATION      │                    │ CALIBRATION     │ │ TEST SPLIT      │
│  (D_train)      │ │ SPLIT (D_val)   │                    │ SPLIT (D_cal)   │ │ (D_test)        │
├─────────────────┤ ├─────────────────┤                    ├─────────────────┤ ├─────────────────┤
│ • Heavy Model   │ │ • Multi-View    │                    │ • Platt Scaling │ │ • Final Unseen  │
│   Fitting       │   Weights         │                    │ • Isotonic Reg  │   Out-of-Sample │
│ • TF-IDF Vocab  │ │ • Budget Pareto │                    │ • Conformal     │   Evaluation    │
│ • Feature Scaler│   Optimization    │                    │   Quantile q_hat│ • Reported Once │
└─────────────────┘ └─────────────────┘                    └─────────────────┘ └─────────────────┘
```

### 1.1 Natural Language Text Partitioning (SNLI / Pereira Benchmark)
- **Total Corpus Size:** $N = 6,000$ pairs sampled deterministically (`random_state=42`) from the processed benchmark.
- **Partition Allocation:**
  1. **Train Split ($D_{\text{train}}^{\text{text}}$):** $N = 2,000$ pairs ($34.5\%$ positive, $65.5\%$ negative). Used for TF-IDF vocabulary extraction, word/character n-gram indexing, and multi-view logistic weights.
  2. **Validation Split ($D_{\text{val}}^{\text{text}}$):** $N = 1,500$ pairs ($35.0\%$ positive, $65.0\%$ negative). Used exclusively for multi-view feature selection and budget-constrained threshold grid search.
  3. **Calibration Split ($D_{\text{cal}}^{\text{text}}$):** $N = 1,000$ pairs ($34.8\%$ positive, $65.2\%$ negative). Used exclusively for Platt scaling / Isotonic regression fitting and conformal non-conformity quantile estimation.
  4. **Test Split ($D_{\text{test}}^{\text{text}}$):** $N = 1,500$ pairs ($35.2\%$ positive, $64.8\%$ negative). Completely frozen and evaluated once at the conclusion of experiments.

### 1.2 Source Code Partitioning (IR-Plag Java Benchmark)
The 7 programming assignment tasks are strictly segregated to enforce **task-disjoint isolation** (0.00% data leakage):
1. **Train Split ($D_{\text{train}}^{\text{code}}$):** `case-01` to `case-04` ($N = 416$ pairs: 200 Pos, 216 Neg). Used for training Gandhi SVM, Mehsen Random Forest, and extracting AST grammar frequencies.
2. **Validation Split ($D_{\text{val}}^{\text{code}}$):** First half of `case-05` ($N = 54$ pairs: 27 Pos, 27 Neg). Used for budget-constrained routing parameter sweeps.
3. **Calibration Split ($D_{\text{cal}}^{\text{code}}$):** Second half of `case-05` ($N = 53$ pairs: 26 Pos, 27 Neg). Used for probability calibrator fitting and conformal quantile estimation.
4. **Test Split ($D_{\text{test}}^{\text{code}}$):** `case-06` (String Palindrome) and `case-07` (Array Search & Bubble Sort) ($N = 210$ pairs: 102 Pos, 108 Neg). Evaluated strictly out-of-sample across all 6 Faidhi-Robinson obfuscation levels.

---

## 2. Comprehensive Metric Evaluation Suite

### 2.1 Classification Performance Metrics
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- **Specificity:** $\frac{TN}{TN + FP}$
- **F1-Score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC & PR-AUC:** Area under receiver operating characteristic and precision-recall curves.

### 2.2 Candidate Generation & Cost Metrics
- **Candidate Recall:** $\frac{\text{Retained True Positives}}{\text{Total Ground-Truth Positives}}$
- **Pruning Ratio (%):** $\frac{\text{Pruned Candidate Pairs}}{\text{Total Candidate Pairs}} \times 100$
- **Expensive Detector Call Rate (%):** $\frac{\text{Pairs Evaluated by SBERT / SVM / RF}}{\text{Total Candidate Pairs}} \times 100$
- **Total End-to-End Latency:** Mean execution time per pair (ms), disaggregated into preprocessing, Tier-1 screening, heavy classifier, and router lookup.
- **Peak RAM (MB):** Measured via `tracemalloc`.

### 2.3 Uncertainty Quantification Metrics
- **Expected Calibration Error (ECE):** With $M = 10$ bins.
- **Brier Score (BS):** Mean squared error of probabilistic outputs.
- **Empirical Conformal Coverage:** Proportion of test pairs where true label is in prediction set:
  $$\text{Coverage}_{\text{emp}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}(y_i \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_i))$$
- **Selective Classification Risk & Inefficiency:** Mean size of prediction sets $|\mathcal{C}|$.

### 2.4 Code-Specific Obfuscation Resilience Metrics
- Breakdown of detection recall across:
  - Level 1 (Comments & Whitespace)
  - Level 2 (Identifier Renaming)
  - Level 3 (Statement Reordering)
  - Level 4 (Control Flow Replacement)
  - Level 5 (Method Abstraction)
  - Level 6 (Compound Multi-Transform Attack)

### 2.5 Explainability & Interpretability Metrics
- **Evidence Faithfulness ($\Delta \hat{P}$):** Average probability drop when evidence tokens/nodes are ablated.
- **Evidence Coverage (%):** Proportion of candidate tokens covered by identified evidence.
- **Transformation Alignment Precision (%):** On known obfuscations, the accuracy of extracted identifier mappings and statement permutations.

---

## 3. Statistical Significance & Validation Protocol

To guarantee that reported improvements are statistically robust:
1. **Bootstrap Confidence Intervals:**
   - 1,000 bootstrap resamples on the test split.
   - Report 95% confidence intervals $[Q_{0.025}, Q_{0.975}]$ for F1-score, Candidate Recall, Latency, and ECE.
2. **Paired Statistical Significance Tests:**
   - **McNemar's Test:** For discrete binary classifications comparing the Proposed Framework against Standalone Baselines and Fixed-Threshold Cascades:
     $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^1$$
     Where $b$ denotes cases where Proposed succeeds and Baseline fails, and $c$ denotes vice-versa.
   - **Wilcoxon Signed-Rank Test:** For paired latency and continuous score distributions (significance threshold $\alpha = 0.01$).

---

## 4. Full Ablation Study Matrix

To isolate the individual contribution of each component, we will evaluate 5 distinct system configurations:

| Configuration ID | Multi-View Generation (C1) | Uncertainty Calibration (C2) | Budget Optimization (C3) | Explainable Evidence (C4) | Primary Investigation Objective |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Framework** | Yes (5-view tuple) | Yes (Isotonic + Conformal) | Yes (Pareto Lagrangian) | Yes (Dual engine) | Full Proposed Q1 System |
| **Ablation 1** | **No** (Single TF-IDF/Jaccard) | Yes | Yes | Yes | Measures candidate recall drop under zero-overlap / obfuscation |
| **Ablation 2** | Yes | **No** (Raw uncalibrated score) | Yes | Yes | Measures increase in ECE and loss of conformal coverage |
| **Ablation 3** | Yes | Yes | **No** (Heuristic grid sweep) | Yes | Measures latency penalty and violation of compute budgets |
| **Ablation 4** | Yes | Yes | Yes | **No** (Scalar score only) | Measures opacity and lack of actionable evidence for reviewers |

