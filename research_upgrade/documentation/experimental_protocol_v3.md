# Experimental Protocol and Evaluation Framework (Version 3.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.0 (Phase 2.2 Methodological Audit Revision)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Experimental Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Group-Aware Data Partitioning Protocols & Leakage Audit

To guarantee 0.00% data leakage, all data splits enforce strict **group-level boundaries**.

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
│ • Model Fitting │ │ • Feature       │                    │ • Calibrator    │ │ • Held-out test │
│ • Vocab & Index │   Selection       │                    │   Fitting       │   tasks / groups  │
│ • Group Isolated│ │ • Pareto Policy │                    │ • Conformal     │ • Evaluated ONCE  │
│                   │   Optimization  │                    │   Quantile      │   at the end      │
└─────────────────┘ └─────────────────┘                    └─────────────────┘ └─────────────────┘
```

### 1.1 Text Modality Group-Aware Partitioning (SNLI / Pereira Benchmark)

#### Forensic Provenance & Grouping Audit:
In the SNLI text corpus, multiple hypothesis sentences are generated from the exact same reference premise sentence (an average of 3 hypotheses per reference text). A naive random split scatters pairs sharing the identical premise across train, validation, and test partitions, creating **premise-level contamination**.

#### Partition Protocol:
To eliminate this leakage, we partition by **unique `reference_text` (Premise ID)** using `GroupShuffleSplit`:
- **Pool 1: Development Pool (derived from `snli_val.parquet`, $N = 9,842$ pairs across 3,304 unique premises):**
  - **Train Split ($D_{\text{train}}^{\text{text}}$):** $N = 4,000$ pairs ($1,340$ unique premises). Used for vocabulary extraction and candidate scorer training.
  - **Validation Split ($D_{\text{val}}^{\text{text}}$):** $N = 1,500$ pairs ($505$ unique premises, 100% disjoint from train). Used for Pareto budget threshold optimization.
  - **Probability Calibration Split ($D_{\text{cal-prob}}^{\text{text}}$):** $N = 1,000$ pairs ($335$ unique premises, 100% disjoint). Used for fitting Platt scaling / Isotonic regression on conditional escalated pairs.
  - **Conformal Calibration Split ($D_{\text{cal-conf}}^{\text{text}}$):** $N = 1,000$ pairs ($335$ unique premises, 100% disjoint). Used for calculating empirical conformal quantile $\hat{q}_{1-\epsilon}$ on conditional escalated pairs.
- **Pool 2: Held-Out Test Split ($D_{\text{test}}^{\text{text}}$, derived from `snli_test.parquet`, $N = 3,000$ pairs across 1,010 unique premises):**
  - Completely isolated from Pool 1 (zero premise overlap with train, validation, or calibration).
  - Evaluated exactly once for final reporting.

*Re-verification of Empirical Statistics:*
All class proportions, duplicate statistics, and zero-overlap statistics must be re-computed dynamically on the newly partitioned test split. The historical $119$-pair ($10.57\%$) zero-overlap figure from the legacy random split is **not** assumed; it will be formally recalculated and reported on $D_{\text{test}}^{\text{text}}$.

---

### 1.2 Source Code Modality: Task-Disjoint Partitioning & Statistical Limitations

#### Forensic Task & File Structure Audit:
The authentic IR-Plag Java corpus contains 7 distinct programming assignment tasks.
Inspection reveals:
- `case-01` to `case-04` ($N = 416$ pairs): Training tasks.
- `case-05` ($N = 107$ pairs): Exactly 15 unique reference programs and 68 suspicious submissions.
- `case-06` (Palindrome, $N = 105$ pairs: 51 Pos, 54 Neg): Test task.
- `case-07` (Sorting, $N = 105$ pairs: 51 Pos, 54 Neg): Test task.

#### Critical Scientific Limitation Acknowledged:
The Phase 2.1 audit reveals that splitting `case-05` naively into two halves causes the **same 15 reference Java programs to appear in both splits**, violating file-level independence. Furthermore, allocating an entire 8th task for conformal calibration is impossible because IR-Plag contains only 7 tasks.

#### Methodological Solution:
1. **Task-Disjoint Primary Allocation:**
   - **Training Set ($D_{\text{train}}^{\text{code}}$):** `case-01`, `case-02`, `case-03`, `case-04` ($N = 416$ pairs: 200 Pos, 216 Neg).
   - **Tuning & Calibration Set:** `case-05` ($N = 107$ pairs: 53 Pos, 54 Neg).
   - **Held-Out Test Set ($D_{\text{test}}^{\text{code}}$):** `case-06` and `case-07` ($N = 210$ pairs: 102 Pos, 108 Neg).
2. **K-Fold Cross-Conformal Prediction on `case-05`:**
   To prevent calibration leakage while avoiding sample starvation ($N \approx 50$), we employ **5-fold Cross-Conformal Prediction (CV-Conformal)** grouped by source program identity on `case-05`:
   - In each fold, 12 reference programs train the calibrator, and 3 held-out reference programs estimate $\hat{q}$.
   - Aggregated quantiles establish the decision boundary, preserving program-level group isolation.
3. **Explicit Publication Limitation:**
   We explicitly document in the manuscript that evaluating on only two independent code test tasks (`case-06` and `case-07`) limits statistical power. A 2-cluster bootstrap cannot estimate broad task-level variance. Future work must validate the pipeline on expanded multi-task corpora (e.g., the 50-task SOAP benchmark). Task-specific metrics will be reported separately for `case-06` and `case-07`.

---

## 2. Comprehensive Evaluation Metrics

### 2.1 Classification Metrics
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, and PR-AUC.

### 2.2 Candidate Generation & Cost Metrics
- **Candidate Recall:** Proportion of true plagiarized submissions retained by Tier 1.
- **Pruning Ratio (%):** Proportion of candidate pairs pruned at Tier 1 without heavy evaluation.
- **Expensive Detector Call Rate (%):** Proportion of pairs escalated to heavy models.
- **Disaggregated Latency:** Reported as Median, Mean, and 95th Percentile over 50 warmed-up iterations.
- **Peak RAM:** Process-level Resident Set Size (RSS via `psutil`) and Python heap (`tracemalloc`).

### 2.3 Uncertainty Metrics
- **Expected Calibration Error (ECE):** Evaluated over 10 equal-width bins on independent validation data.
- **Brier Score (BS):** Mean squared error of calibrated probabilities.
- **Conformal Marginal Coverage (%):** Proportion of test instances where $y_i \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_i) \mid \text{Escalated}$.
- **Conformal Set Size Distribution:** Breakdown of $|\mathcal{C}| \in \{0, 1, 2\}$ and frequency of the empty set $\emptyset$.

### 2.4 Code Obfuscation Breakdown
- Stratified detection recall across Faidhi-Robinson levels: L1 (Comments), L2 (Renaming), L3 (Reordering), L4 (Control Flow), L5 (Abstraction), and L6 (Compound Disguise).

### 2.5 Explainability & Interpretability Metrics
- **Evidence Faithfulness ($\Delta \hat{P}_{\text{faith}}$):** Average probability drop when identified evidence tokens/nodes are masked.
- **Evidence Coverage (%):** Percentage of suspicious document tokens covered by identified evidence.
- **Transformation Alignment Precision (%):** On known obfuscations, precision of extracted identifier mappings ($\mathcal{M}_{\text{id}}$) and statement permutation traces.
- **Reviewer Usefulness Protocol:** Qualitative review by human educators scoring clarity and evidentiary sufficiency on a 5-point Likert scale.

---

## 3. Statistical Testing and Significance Protocol

1. **Group-Aware Bootstrap Resampling:**
   - Rather than naive i.i.d. pair resampling, we resample **clusters/groups** (resampling premises for text; resampling task clusters for code) over $B = 1,000$ bootstrap iterations.
   - Report 95% confidence intervals $[Q_{0.025}, Q_{0.975}]$ for all primary metrics and metric differences ($\Delta \text{F1}$, $\Delta \text{Latency}$).
2. **Paired Statistical Significance Tests:**
   - **Exact McNemar's Test:** Computed using the exact binomial distribution:
     $$p = 2 \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} 0.5^{b + c}$$
     when discordant cell counts $b + c < 25$. Standard $\chi^2$ approximation is used only when $b + c \ge 25$.
   - **Wilcoxon Signed-Rank Test:** For paired continuous latency distributions ($\alpha = 0.01$).
3. **Multiple-Comparison Correction:**
   - When evaluating multi-method comparisons across 5 baseline models, apply the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure to control false positive discoveries.

---

## 4. Full Ablation Study Matrix

To isolate the individual contribution of each component, we evaluate 5 distinct configurations under identical data partitions:

| Condition ID | Multi-View Screening (C1) | Probability Calibration (C2) | Conformal Four-Set Triage (C2) | Pareto Budget Optimizer (C3) | Explainable Evidence (C4) | Primary Investigation Objective |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Framework** | Yes (Lightweight 4-view) | Yes (Platt / Isotonic) | Yes (4-set policy) | Yes (Pareto surface search) | Yes (Dual engine) | Full Proposed Q1 Framework |
| **Ablation 1** | **No** (Single TF-IDF / Jaccard) | Yes | Yes | Yes | Yes | Isolates candidate recall loss on zero-overlap text and L2 renaming |
| **Ablation 2** | Yes | **No** (Raw uncalibrated score) | **No** (Fixed scalar cutoff) | Yes | Yes | Isolates calibration error (ECE surge) and loss of statistical coverage |
| **Ablation 3** | Yes | Yes | Yes | **No** (Heuristic grid sweep) | Yes | Isolates latency inefficiency and constraint violations under fixed budgets |
| **Ablation 4** | Yes | Yes | Yes | Yes | **No** (Scalar score only) | Isolates opacity and lack of actionable evidence for academic review boards |

