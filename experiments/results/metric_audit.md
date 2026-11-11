# Comprehensive Metric Calculation & Mathematical Formulation Audit

**Project Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Auditor Role:** Senior Machine Learning Research Engineer & Independent Research Evaluator  
**Audit Scope:** Mathematical Definitions, Code Implementation, Denominator Consistency, and Tri-Band Accounting  
**Audit Date:** November 2026  

---

## 1. Executive Metric Audit & Resolution of the Denominator Question

### The Critical Issue Investigated:
In the initial source-code evaluation, the pipeline reported:
- **Pruning Ratio = 0.5% (0.48%)**
- **Expensive Detector Call Rate = 63.3% (63.33%)**

The evaluation committee raised the question:
> *"Determine whether these metrics use different denominators or contain an implementation bug."*

### Mathematical Finding & Resolution:
**There is NO bug in the code, and both metrics use the EXACT SAME denominator ($N_{\text{total}} = 210$ total eligible pairs in the test set).**

The apparent numerical discrepancy ($0.48\% + 63.33\% = 63.81\% \ne 100\%$) arises from a fundamental architectural difference between a traditional **binary pre-filter** and the **tri-band typed-decision pipeline**:

1. **In a Traditional Binary Filter (Condition B):**
   Every submission pair has exactly two mutually exclusive outcomes:
   - Outcome 1: Pruned ($s < \theta$)
   - Outcome 2: Escalated to Expensive Detector ($s \ge \theta$)
   $$\text{Pruned Pairs} + \text{Escalated Pairs} = \text{Total Pairs}$$
   $$\text{Pruning Ratio} + \text{Expensive Call Rate} = 100.00\%$$
   *(Example from Text Condition B: $42.87\% \text{ pruned} + 57.13\% \text{ called} = 100.00\%$).*

2. **In the Proposed Tri-Band Typed Pipeline (Condition C):**
   Every submission pair has **three** mutually exclusive outcomes:
   - Band 1 (`LOW_RISK_PRUNED`): $s < \theta_{low}$ $\rightarrow$ Filtered out at zero heavy cost ($N_{\text{pruned}} = 1$ pair).
   - Band 2 (`UNCERTAIN_ESCALATED`): $\theta_{low} \le s < \theta_{high}$ $\rightarrow$ Escalated to heavy detector ($N_{\text{call}} = 133$ pairs).
   - Band 3 (`HIGH_RISK_FLAGGED`): $s \ge \theta_{high}$ $\rightarrow$ Directly flagged for instructor review, **bypassing heavy detector** ($N_{\text{flag}} = 76$ pairs).

   Summing all three partitions over the common denominator $N_{\text{total}} = 210$:
   $$\frac{N_{\text{pruned}}}{N_{\text{total}}} + \frac{N_{\text{call}}}{N_{\text{total}}} + \frac{N_{\text{flag}}}{N_{\text{total}}} = \frac{1 + 133 + 76}{210} = \frac{210}{210} = 100.00\%$$

   $$\mathbf{0.48\% \text{ (Pruned)}} + \mathbf{63.33\% \text{ (Heavy SVM Calls)}} + \mathbf{36.19\% \text{ (Directly Flagged)}} = \mathbf{100.00\%}$$

### Workload Reduction Accounting
The total computational workload reduction achieved by the pipeline is:
$$\text{Workload Reduction Factor} = 1.0 - \text{Expensive Call Rate} = \frac{N_{\text{pruned}} + N_{\text{flag}}}{N_{\text{total}}} = 0.48\% + 36.19\% = \mathbf{36.67\%}$$

The expensive SVM detector was completely avoided for **77 out of 210 pairs (36.67%)**, while retaining **100.00% Candidate Recall** (all 102 true positives preserved).

---

## 2. Line-by-Line Code Audit of All Evaluation Metrics

All metric calculations in [`src/typed_decision_pipeline.py`](file:///d:/ml%20rearcch%20project%20ad/src/typed_decision_pipeline.py) and [`src/evaluation.py`](file:///d:/ml%20rearcch%20project%20ad/src/evaluation.py) were audited against canonical statistical definitions:

### 2.1 Standard Classification Metrics

| Metric Name | Mathematical Definition | Code Implementation (`typed_decision_pipeline.py`) | Audit Status |
| :--- | :---: | :--- | :---: |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | `accuracy_score(y_test, preds)` | **CORRECT** |
| **Precision** | $\frac{TP}{TP + FP}$ | `precision_score(y_test, preds, zero_division=0)` | **CORRECT** |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | `recall_score(y_test, preds, zero_division=0)` | **CORRECT** |
| **F1-Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | `f1_score(y_test, preds, zero_division=0)` | **CORRECT** |
| **Specificity** | $\frac{TN}{TN + FP}$ | `float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0` | **CORRECT** |
| **False Positive Rate (FPR)** | $\frac{FP}{TN + FP} = 1 - \text{Specificity}$ | `float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0` | **CORRECT** |

---

### 2.2 Pipeline-Specific Operational Metrics

| Metric Name | Mathematical Definition | Code Implementation | Population & Denominator | Audit Status |
| :--- | :---: | :--- | :---: | :---: |
| **Candidate Recall** | $\frac{\sum_{i=1}^N \mathbb{I}(s_i \ge \theta_{low} \land y_i = 1)}{\sum_{i=1}^N \mathbb{I}(y_i = 1)}$ | `int(((cheap_scores >= theta_low) & (y_test == 1)).sum()) / total_positives` | Ground-truth positive pairs ($Y=1$) | **AUDITED & FIXED** |
| **Candidate Precision** | $\frac{\sum_{i=1}^N \mathbb{I}(s_i \ge \theta_{low} \land y_i = 1)}{\sum_{i=1}^N \mathbb{I}(s_i \ge \theta_{low})}$ | `retained_positives / retained_candidates` | Retained pairs ($s \ge \theta_{low}$) | **CORRECT** |
| **Pruning Ratio** | $1 - \frac{N_{\text{retained}}}{N_{\text{total}}} = \frac{N_{\text{pruned}}}{N_{\text{total}}}$ | `float(filtered_count / n_pairs)` | Total test pairs ($N_{\text{total}}$) | **CORRECT** |
| **Expensive Call Rate** | $\frac{N_{\text{escalated}}}{N_{\text{total}}}$ | `float(len(escalate_indices) / n_pairs)` | Total test pairs ($N_{\text{total}}$) | **CORRECT** |
| **Direct Flag Rate** | $\frac{N_{\text{flagged}}}{N_{\text{total}}}$ | `float(direct_flag_count / n_pairs)` | Total test pairs ($N_{\text{total}}$) | **CORRECT** |

---

### 2.3 Computational Latency Metrics

To ensure latency measurements are scientifically meaningful, the execution time was instrumented at both the per-stage and end-to-end levels:

| Metric Name | Formula / Measurement | Code Implementation | Audit Status |
| :--- | :---: | :--- | :---: |
| **Total Runtime (s)** | $\Delta t_{\text{wall}} = t_{\text{end}} - t_{\text{start}}$ | `time.perf_counter() - start_total` | **CORRECT** |
| **Average Latency (ms)** | $\frac{1}{N} \sum_{i=1}^N \text{latency}_i \times 1000$ | `float(np.mean(latencies) * 1000.0)` | **CORRECT** |
| **Median Latency (ms)** | $\text{Median}(\{\text{latency}_i\}_{i=1}^N) \times 1000$ | `float(np.median(latencies) * 1000.0)` | **CORRECT** |
| **P95 Latency (ms)** | $\text{95th percentile}(\{\text{latency}_i\}_{i=1}^N) \times 1000$ | `float(np.percentile(latencies, 95) * 1000.0)` | **CORRECT** |

---

## 3. Verified Audit Table Across All Experimental Populations

| Experimental Population | Condition | Total Pairs ($N$) | Pruned Pairs ($N_{prune}$) | Pruning Ratio (%) | Heavy Calls ($N_{call}$) | Call Rate (%) | Flagged Pairs ($N_{flag}$) | Flag Rate (%) | Candidate Recall | F1-Score | Mean Latency | P95 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text Test Partition** | Condition A (Full) | 3,000 | 0 | 0.00% | 3,000 | 100.00% | 0 | 0.00% | 1.0000 | 0.6844 | 4.143 ms | 4.250 ms |
| **Text Test Partition** | Condition B (Fixed 50%) | 3,000 | 1,286 | 42.87% | 1,714 | 57.13% | 0 | 0.00% | 0.7366 | 0.6252 | 2.904 ms | 4.163 ms |
| **Text Test Partition** | **Condition C (Proposed Tri-Band)** | **3,000** | **796** | **26.53%** | **2,101** | **70.03%** | **103** | **3.43%** | **0.8598** | **0.6627** | **3.167 ms** | **4.163 ms** |
| **Source Code Test Partition** | Condition A (Full) | 210 | 0 | 0.00% | 210 | 100.00% | 0 | 0.00% | 1.0000 | 0.7549 | 0.013 ms | 0.020 ms |
| **Source Code Test Partition** | Condition B (Fixed 50%) | 210 | 52 | 24.76% | 158 | 75.24% | 0 | 0.00% | 0.7745 | 0.6724 | 0.060 ms | 0.063 ms |
| **Source Code Test Partition** | **Condition C (Proposed Tri-Band)** | **210** | **1** | **0.48%** | **133** | **63.33%** | **76** | **36.19%** | **1.0000** | **0.7424** | **0.058 ms** | **0.059 ms** |

### Key Analytical Takeaways:
1. **Total Population Conservation:** For both modalities under Condition C:
   $$\text{Pruning Ratio} + \text{Call Rate} + \text{Flag Rate} = 100.00\%$$
   There is zero population leakage or uncounted pairs.
2. **Recall Collapse in Binary Filtering:** Condition B incurs severe candidate recall loss (dropping to 73.66% on text and 77.45% on code). Condition C eliminates candidate recall loss on code (100.00%) and preserves 85.98% candidate recall on text.

