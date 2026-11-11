# Final Statistical Validity & Inference Protocol Report (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 10 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** STATISTICALLY AUDITED & METHODOLOGICALLY RIGOROUS

---

## 1. Accounting for Statistical Dependence in Academic Cohorts

A critical flaw in standard machine learning evaluations is treating all pairwise comparisons as independent and identically distributed (i.i.d.). In academic integrity detection, pairwise comparisons exhibit strong structural dependencies:
- **Text Premise Clustering:** In [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv), multiple suspicious hypotheses are compared against the same reference premise sentence. Errors on hypotheses sharing a premise are correlated.
- **Source Code Base-Program Topology:** In IR-Plag, multiple student submissions are compared against the same assignment specification or starter code. In `case-05`, all 53 plagiarized variants share the same underlying program (`prog_original`).
- **Assignment-Level Topic Correlation:** Submissions within an assignment share vocabulary, boilerplate code, libraries, and function signatures.

Treating correlated pairs as independent leads to artificially deflated standard errors, inflated $t$-statistics, and spurious claims of statistical significance.

---

## 2. Resampling Units & Group-Aware Bootstrap

To account for intra-cluster correlation, all statistical confidence intervals in this research are computed using **group-aware block bootstrap resampling**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GROUP-AWARE BOOTSTRAP PROTOCOL                              │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
    ┌────────────────────────────────────┴────────────────────────────────────┐
    ▼                                                                         ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ Text Modality Resampling Unit         │   │ Code Modality Resampling Unit         │
│ • Unit: Independent Premise Cluster   │   │ • Unit: Programming Assignment Task   │
│ • Sample with replacement over G      │   │ • Due to small task count (N=7),      │
│ • Preserves all pairs within cluster  │   │   pooled results remain DESCRIPTIVE   │
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Empirical Distribution & 95% Confidence Intervals                               │
│ • B = 1,000 bootstrap iterations over clusters (NOT independent pairs)          │
│ • Percentile method: [q_0.025, q_0.975]                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

> [!WARNING]
> **Code Generalization Limitation:**  
> In the IR-Plag corpus, only **two independent tasks** are available for held-out testing. Therefore, we explicitly state that **pooled code results are descriptive**, and **task-level generalization claims are exploratory**. We do not claim population-level generalization from two tasks.

---

## 3. Calibration Lifecycle & Population Separation

To ensure absolute separation of model selection, probability calibration, and final evaluation, five distinct populations are strictly enforced:

```
Total Clean Retained Rows: 19,658
├── Clean Development Pool (from val split): 9,837
│   ├── D_train:      4,000 pairs (1,349 premises) -> Train linear fusion weights w*
│   ├── D_val:        1,500 pairs (  508 premises) -> Tune routing thresholds (theta_low, theta_high)
│   ├── D_cal-prob:   1,000 pairs (  336 premises) -> Fit Platt calibration params (A*, B*)
│   ├── D_cal-conf:   1,000 pairs (  334 premises) -> Compute exact conformal quantile q_hat
│   └── D_dev-res:    2,337 pairs (  791 premises) -> Held-out development reserve
└── Clean Test Pool (from test split): 9,821
    ├── D_test:       3,000 pairs (1,015 premises) -> UNTOUCHED TEST BENCHMARK
    └── D_test-res:   6,821 pairs (2,307 premises) -> Held-out test reserve
```

### Strict Calibration Rules:
1. **Frozen Thresholds:** The Stage-1 screening thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ are tuned on $D_{\text{val}}$ and permanently frozen prior to calibration.
2. **Conformal Population Consistency:** Only candidate pairs that are **actually escalated by Stage 1** ($\theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}}$) enter the conformal calibration set $D_{\text{cal-conf}}^{\text{esc}}$. Pairs routed to `HIGH_RISK_DIRECT_BYPASS` never receive heavy-model forward passes and are excluded from the conformal population.
3. **Untouched Test Escalation Diagnostic:** All final calibration metrics (ECE, MCE, Brier, NLL) are evaluated strictly on the untouched test escalation population $D_{\text{test}}^{\text{esc}}$. Test set outcomes are never used to adjust thresholds, fusion weights, or calibrator parameters.

---

## 4. Exact Finite-Sample Group Conformal Prediction

For escalated text pairs in premise group $g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}$, the group non-conformity score is:
$$\alpha_g = \max_{i \in \mathcal{P}_g^{\text{esc}}} \left( 1 - \hat{P}_{\text{cal}}(Y = y_i \mid z_i) \right)$$

Let $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ be the number of calibration premise groups. Sorted scores are $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$.  
The exact finite-sample quantile index is:
$$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$

```python
# Exact finite-sample implementation
scores = np.sort(np.asarray(group_non_conformities))
n_groups = len(scores)
k = int(np.ceil((n_groups + 1) * (1.0 - epsilon)))

if k > n_groups:
    q_hat = float("inf")  # Finite-sample guarantee mathematically unattainable
else:
    q_hat = float(scores[k - 1])  # Exact order statistic
```

- **Attainability Check:** $n \ge \lceil (1 - \epsilon)/\epsilon \rceil$. At $\epsilon = 0.05$, $n \ge 19$. In our text benchmark, $D_{\text{cal-conf}}$ contains **334 independent premise groups**, easily satisfying $n \gg 19$.
- **Empty Prediction Sets ($\mathcal{C}(x) = \emptyset$):** Mapped to the operational state **`ABSTAIN_UNRESOLVED`**. Indicates predictive ambiguity/model inconsistency, not out-of-distribution proof.

---

## 5. Calibration Evaluation Metrics Formulations

All calibration diagnostics are evaluated strictly on $D_{\text{test}}^{\text{esc}}$:
1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N_{\text{test}}^{\text{esc}}} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|$$
   Uniform binning $M = 10$. Empty bins ($|B_m| = 0$) contribute exactly $0$.
2. **Maximum Calibration Error (MCE):**
   $$\text{MCE} = \max_{m \in \{1, \dots, 10\} \,:\, |B_m| > 0} |\text{acc}(B_m) - \text{conf}(B_m)|$$
3. **Brier Score:**
   $$\text{BS} = \frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} (\hat{p}_i - y_i)^2 \in [0, 1]$$
4. **Clipped Negative Log-Likelihood (NLL):**
   $$\text{NLL} = -\frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} [y_i \log(\tilde{p}_i) + (1 - y_i) \log(1 - \tilde{p}_i)]$$
   where $\tilde{p}_i = \text{clip}(\hat{p}_i, 10^{-15}, 1 - 10^{-15})$.

