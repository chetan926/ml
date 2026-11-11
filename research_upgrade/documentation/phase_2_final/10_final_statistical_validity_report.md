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
> **Code Generalization & Case-05 Three-Dimensional Analysis:**  
> In the IR-Plag corpus, only **two independent tasks** are available for held-out testing, bounding task-level generalization claims to exploratory status. Furthermore, in `case-05` (all 53 positives share `prog_original`), three distinct methodological dimensions must be distinguished:
> 1. **Can Platt calibration be fitted on training folds?**  
>    In Fold 1 (`prog_original` held out): **NO** ($N_{\text{pos}}^{\text{train}} = 0$; unregularized MLE diverges to $-\infty$, regularized LogisticRegression raises `ValueError`).  
>    In Folds 2–5 (`prog_original` in training): **YES** (both classes present: $+53$, $-36\text{ to }-38$).
> 2. **Are test metrics mathematically defined?**  
>    In Fold 1: **YES** (test fold contains both classes: $+53$, $-21$).  
>    In Folds 2–5: **PARTIALLY DEGENERATE** ($N_{\text{pos}}^{\text{test}} = 0$). Recall and ROC-AUC are undefined ($0/0$); Precision is $0.0$ if $\text{FP} > 0$ but **undefined** when there are no predicted positives ($\text{TP}+\text{FP}=0 \implies 0/0$); Brier Score is mathematically defined ($\frac{1}{N}\sum \hat{p}_i^2$); and ECE is **mathematically computable** across bins, collapsing into the mean predicted positive probability $\frac{1}{N}\sum \hat{p}_i$ because positive empirical frequency $\bar{y}_m = 0.0$ in all bins. (Note: if top-label classification accuracy were used, true negatives with $\hat{p} < 0.5$ would have classification accuracy $1.0$, not $0.0$).
> 3. **Can those metrics meaningfully evaluate the intended calibrated model?**  
>    **NO across all folds.** In Fold 1, test metrics cannot evaluate a calibrated model because training failed. In Folds 2–5, an all-negative test fold cannot evaluate detection sensitivity or discrimination: a trivial constant predictor $\hat{p} \equiv 0$ achieves a "perfect" Brier Score and ECE of $0.0$ despite being entirely useless for detecting plagiarism. Thus, cross-fitting cannot validate calibration on `case-05`, justifying exploratory raw decision margins.

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

### 4.1 Selected Premise-Group Sampling Unit & Stage-1 Filtering
In academic text matching, multiple student sentences are compared against the same reference premise sentence $g \in \mathcal{G}$. Intra-cluster error correlation invalidates standard pair-level i.i.d. assumptions.
1. **Sampling Unit:** The primary sampling unit is the **premise group cluster** $g = (D_{\text{ref}}^{(g)}, \mathcal{P}_g)$, where $\mathcal{P}_g = \{ (x_{g, j}, y_{g, j}) \}_{j=1}^{K_g}$ is the set of candidate pairs sharing reference premise $g$.
2. **Stage-1 Filtering:** Each pair within group $g$ is evaluated by the frozen Stage-1 screener $s_{\text{cand}}(x_{g, j})$. The escalated candidate pairs form the escalated sub-group:
   $$\mathcal{P}_g^{\text{esc}} = \left\{ x_{g, j} \in \mathcal{P}_g \;\middle|\; \theta_{\text{low}} \le s_{\text{cand}}(x_{g, j}) < \theta_{\text{high}} \right\}, \quad m_g = |\mathcal{P}_g^{\text{esc}}|$$

### 4.2 Handling Groups with Zero Escalated Pairs ($m_g = 0$) & Identical Filter
- **Zero-Escalation Groups ($m_g = 0$):** Groups where all pairs are pruned ($s_{\text{cand}} < \theta_{\text{low}}$) or directly bypassed ($s_{\text{cand}} \ge \theta_{\text{high}}$) require zero Stage-2 passes and are **excluded from the calibration set $\mathcal{G}_{\text{cal-conf}}^{\text{esc}}$ and test set $\mathcal{G}_{\text{test}}^{\text{esc}}$**.
- **Active Population:** The conformal population is strictly defined over the **active selected premise groups**:
  $$\mathcal{G}^{\text{esc}} = \left\{ g \in \mathcal{G} \;\middle|\; m_g = |\mathcal{P}_g^{\text{esc}}| \ge 1 \right\}$$
  The calibration size $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ counts only active groups with $m_g \ge 1$.
- **Identical Selection Mechanism:** The Stage-1 screening model $\mathbf{w}^*$ and thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ are tuned on $D_{\text{val}}$ and **permanently frozen**. Calibration and test premise groups undergo the exact same deterministic screening filter, guaranteeing that calibration and test active groups share the identical selection mechanism.

### 4.3 Group Non-Conformity Score & Exchangeability Assumptions
1. **Individual Pair Non-Conformity Score:** For each escalated pair $(x_{g, j}, y_{g, j})$:
   $$s(x_{g, j}, y_{g, j}) = 1 - \hat{P}_{\text{cal}}(Y = y_{g, j} \mid z(x_{g, j})) = \begin{cases} 1 - \hat{p}(x_{g, j}) & \text{if } y_{g, j} = 1 \\ \hat{p}(x_{g, j}) & \text{if } y_{g, j} = 0 \end{cases}$$
2. **Group-Level Calibration Score:** Aggregates worst-case non-conformity across all $m_g$ escalated pairs sharing premise $g$:
   $$\alpha_g = \max_{j \in \{1, \dots, m_g\}} s(x_{g, j}, y_{g, j}) \in [0, 1]$$
3. **Required Exchangeability Assumptions:** Conformal validity requires that the sequence of active selected group scores $(\alpha_{g_1}, \dots, \alpha_{g_n}, \alpha_{g^*})$ is **exchangeable** conditional on $g \in \mathcal{G}^{\text{esc}}$. This is supported by:
   - Premise-level disjoint partitioning with 0 cross-split overlap.
   - Premise groups sampled as exchangeable cluster draws from $\mathcal{D}_{\mathcal{G}}$.
   - Symmetric, identical Stage-1 selection filter applied across splits.
   - Permutation-invariant group max-aggregation.

### 4.4 Exact Quantile Index & Finite-Sample Attainability
On $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ sorted group scores $\alpha_{(1)} \le \dots \le \alpha_{(n)}$:
$$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil, \quad \hat{q}_{1-\epsilon} = \alpha_{(k)} \text{ (if } k \le n \text{)}$$
Finite-sample attainability requires $n \ge \lceil (1-\epsilon)/\epsilon \rceil$ ($n \ge 19$ for $\epsilon = 0.05$). Text calibration split contains $n = 334 \gg 19$.

### 4.5 Individual Pair-Level Prediction Sets for Both Classes
For any test pair $x$ escalated to Stage 2 with calibrated probability $\hat{p}(x)$, the conformal prediction set $\mathcal{C}(x) \subseteq \{0, 1\}$ evaluates conformity for both binary classes:
$$\mathcal{C}(x) = \left\{ y \in \{0, 1\} \;\middle|\; 1 - \hat{P}_{\text{cal}}(Y = y \mid z(x)) \le \hat{q}_{1-\epsilon} \right\}$$
- **Class $y = 1$ (Plagiarism):** Included iff $1 - \hat{p}(x) \le \hat{q}_{1-\epsilon} \iff \hat{p}(x) \ge 1 - \hat{q}_{1-\epsilon}$.
- **Class $y = 0$ (Non-Plagiarism):** Included iff $\hat{p}(x) \le \hat{q}_{1-\epsilon}$.

This produces four explicit operational outcomes:
$$\mathcal{C}(x) = \begin{cases}
\{0, 1\} & \text{if } 1 - \hat{q}_{1-\epsilon} \le \hat{p}(x) \le \hat{q}_{1-\epsilon} \quad (\text{Ambiguous: both classes conforming}) \\
\{1\} & \text{if } \hat{p}(x) \ge 1 - \hat{q}_{1-\epsilon} \text{ and } \hat{p}(x) > \hat{q}_{1-\epsilon} \quad (\text{Confident Plagiarism}) \\
\{0\} & \text{if } \hat{p}(x) \le \hat{q}_{1-\epsilon} \text{ and } \hat{p}(x) < 1 - \hat{q}_{1-\epsilon} \quad (\text{Confident Non-Plagiarism}) \\
\emptyset & \text{if } \hat{q}_{1-\epsilon} < \hat{p}(x) < 1 - \hat{q}_{1-\epsilon} \quad (\text{Model Inconsistency / Ambiguity} \implies \text{ABSTAIN\_UNRESOLVED})
\end{cases}$$

### 4.6 Simultaneous Group Coverage Guarantee & Stage-1 Bypassed Exclusion
1. **Simultaneous Group Guarantee Under Exchangeability:** Because calibration scores are aggregated via maximum non-conformity across pairs, $\alpha_{g^*} \le \hat{q}_{1-\epsilon} \iff \forall j \in \{1, \dots, m_{g^*}\}, y_{g^*, j} \in \mathcal{C}(x_{g^*, j})$. Thus, under active group exchangeability:
   $$\mathbb{P}\left( \forall j \in \{1, \dots, m_{g^*}\}, \, y_{g^*, j} \in \mathcal{C}(x_{g^*, j}) \;\middle|\; g^* \in \mathcal{G}_{\text{test}}^{\text{esc}} \right) \ge 1 - \epsilon$$
2. **Exclusion of Stage-1 Bypassed Pairs:** Pairs pruned as low risk ($s_{\text{cand}} < \theta_{\text{low}}$) or bypassed as high risk ($s_{\text{cand}} \ge \theta_{\text{high}}$) bypass Stage 2 completely and **receive no conformal prediction set**. The guarantee holds **strictly conditional on Stage-1 escalation** ($g^* \in \mathcal{G}_{\text{test}}^{\text{esc}}$). Unconditional marginal coverage over the entire raw input cohort is explicitly neither claimed nor mathematically valid.

---

## 5. Calibration Evaluation Metrics Formulations

All calibration diagnostics are evaluated strictly on $D_{\text{test}}^{\text{esc}}$:
1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N_{\text{test}}^{\text{esc}}} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|$$
   Uniform binning $M = 10$. $\text{acc}(B_m) = \bar{y}_m = \frac{1}{|B_m|}\sum_{i \in B_m} y_i$ denotes positive-class empirical prevalence (fraction of positive labels in bin $m$, distinct from top-label classification accuracy). $\text{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} \hat{p}_i$. Empty bins ($|B_m| = 0$) contribute exactly $0$.
2. **Maximum Calibration Error (MCE):**
   $$\text{MCE} = \max_{m \in \{1, \dots, 10\} \,:\, |B_m| > 0} |\text{acc}(B_m) - \text{conf}(B_m)|$$
3. **Brier Score:**
   $$\text{BS} = \frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} (\hat{p}_i - y_i)^2 \in [0, 1]$$
4. **Clipped Negative Log-Likelihood (NLL):**
   $$\text{NLL} = -\frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} [y_i \log(\tilde{p}_i) + (1 - y_i) \log(1 - \tilde{p}_i)]$$
   where $\tilde{p}_i = \text{clip}(\hat{p}_i, 10^{-15}, 1 - 10^{-15})$.

