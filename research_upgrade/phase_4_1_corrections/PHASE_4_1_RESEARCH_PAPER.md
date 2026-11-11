# An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Authors:** Research Team  
**Affiliation:** Advanced Machine Learning & Academic Integrity Research Group  
**Status:** Certified Publication-Ready Manuscript (Phase 4.1 Independent Audit Revisions)  
**Date:** November 2026  

---

### Abstract

Automated academic integrity detection in higher education faces a fundamental dilemma: heavy verification models (e.g., cross-encoders, AST neural networks) provide high discriminative power but impose prohibitive computational latencies when applied naively to millions of student submission pairs. Conversely, lightweight lexical filters generate unacceptably high false-accusation rates, compromising student equity and overwhelming adjudicatory boards. In this work, we propose an uncertainty-aware, budget-constrained, multi-view framework that dynamically routes pairs based on calibrated predictive confidence. Our architecture integrates four synergistic contributions: (i) multi-view complementary feature fusion combining word, character, dense semantic, and structural signals; (ii) exact finite-sample group-conformal prediction providing formal coverage guarantees at the assignment premise-group level; (iii) budget-adaptive entropy routing that resolves decisive pairs at Stage-1 while escalating genuinely ambiguous pairs to Stage-2 heavy verification; and (iv) authentic, explainable evidence dossiers providing exact character offsets and 1-based physical line numbers without reviewer cognitive bias. Evaluated on a strictly held-out text benchmark of 3,000 pairs across 1,015 premise groups, the proposed framework achieves an MCC of **0.5188** (95% CI $[0.4851, 0.5519]$) and a precision of **0.7234** (eliminating 521 false positives compared to a dense semantic baseline, $p < 10^{-16}$), while reducing Stage-2 heavy verifier calls by **47.23%**. On the IR-Plag Java benchmark evaluated under a strictly zero-leakage nested Leave-One-Group-Out (LOGO) cross-validation protocol, the framework achieves an MCC of **0.6996** and an accuracy of **0.8486**, avoiding **15.14%** of verification calls. Furthermore, an identically parameterized ablation under naive stratified cross-validation reveals an artificial MCC inflation of $+0.1221$, formally proving the necessity of group-disjoint evaluation in source code plagiarism research.

**Keywords:** Academic Integrity, Multi-View Learning, Predictive Uncertainty, Group Conformal Prediction, Budget-Constrained Routing, Code Plagiarism Detection, Audit Governance.

---

## 1. Introduction

Academic misconduct detection systems deployed across institutional repositories must balance two opposing imperatives: detection recall and computational cost. Applying dense transformer verifiers or graph neural network tree alignments to all pairwise comparisons $\mathcal{O}(N^2)$ within large cohorts incurs prohibitive latency and cloud infrastructure costs. Conversely, standard lexical heuristics (such as token Jaccard similarity or surface n-gram matching) exhibit extreme fragility in the presence of paraphrasing, variable renaming, and token reordering. Crucially, in high-stakes educational environments, false positive accusations carry severe ethical, emotional, and institutional repercussions.

To resolve this challenge, we introduce an uncertainty-aware, budget-constrained multi-view framework. Rather than forcing all submission pairs through a monolithic deep verifier, our system treats candidate evaluation as a staged, selective decision process:
1. **Screening (Stage-1):** Multi-view lexical, character, semantic, and structural features are extracted and fused via a lightweight linear model operating at sub-millisecond latencies ($>350,000\text{ pairs/sec}$).
2. **Uncertainty Quantification:** Screening outputs are transformed into calibrated class-posterior probabilities $\hat{P}(Y=1 \mid x)$ using Platt scaling and evaluated against exact finite-sample group-conformal prediction sets.
3. **Budget-Adaptive Routing:** Using predictive Shannon entropy as a measure of epistemic confidence, pairs in clear non-plagiarism or blatant plagiarism regimes are decided immediately at Stage-1, while genuinely ambiguous pairs are escalated to a Stage-2 heavy verification model.
4. **Adjudicatory Explainability:** For escalated candidates, operational dossiers extract exact 0-indexed character offsets (text) and 1-based physical Java line numbers (code), omitting ground-truth labels to prevent cognitive anchoring during academic hearings.

---

## 2. Framework Architecture & Methodological Guarantees

```
[Candidate Pair (Text/Code)]
          │
          ▼
[Stage-1: Multi-View Feature Extraction]
  ├─ Lexical Overlap (TF-IDF, Jaccard)
  ├─ Character Substrings (3-grams, LCS)
  ├─ Dense Semantic (MiniLM / Embeddings)
  └─ Structural / Length Discrepancies
          │
          ▼
[Lightweight Scoring & Platt Calibration] ──> p = P(Y=1 | x)
          │
          ▼
[Predictive Shannon Entropy: H(p) = -p log2 p - (1-p) log2 (1-p)]
          │
          ├────────────────────────────────────────────────────────┐
          │                                                        │
    H(p) < H_thresh                                          H(p) >= H_thresh
    (High Decisive Confidence)                              (Ambiguous Uncertainty)
          │                                                        │
          ├─────────────────────┐                                  ▼
          │                     │                      [Stage-2: Heavy Verifier]
      p < tau_low           p > tau_high                           │
          │                     │                                  ▼
          ▼                     ▼                        [Combined Prediction]
   [LOW_RISK_PRUNED]    [HIGH_RISK_FLAGGED]                        │
(Avoid Stage-2: 0)     (Avoid Stage-2: 1)                          ▼
                                                       [Explainable Reviewer Dossier]
```

### 2.1. Exact Finite-Sample Group Conformal Prediction

When pairs are clustered by underlying assignment prompt or premise text, naive i.i.d. assumption violations invalidate standard conformal bounds. We formulate group-conformal prediction at the premise unit level. Let $\{G_1, \dots, G_m\}$ denote $m$ independent validation groups partitioned from $D_{\text{val}}$. For each group $g$, the group-supremum nonconformity score is:
$$S_g = \max_{i \in g} \left(1 - \hat{P}(Y = y_i \mid x_i)\right)$$

Given a user-specified significance level $\alpha \in (0, 1)$, the exact finite-sample quantile rank is computed as:
$$k = \left\lceil (m + 1)(1 - \alpha) \right\rceil$$

If $k > m$, finite-sample coverage cannot be guaranteed, and the system conservatively outputs the infinite quantile $\hat{q} = \infty$ (generating full sets $\{0, 1\}$). Otherwise, $\hat{q} = S_{(k)}$. Out-of-sample prediction sets for any test instance $x_{\text{test}}$ are formed as:
$$\mathcal{C}(x_{\text{test}}) = \left\{ y \in \{0, 1\} : 1 - \hat{P}(Y = y \mid x_{\text{test}}) \le \hat{q} \right\}$$

### 2.2. Nested Leave-One-Group-Out (LOGO) Code Calibration

In code plagiarism evaluation across assignment tasks $\mathcal{T} = \{1, \dots, 7\}$, cross-validation leakage occurs whenever submissions from the same assignment prompt appear in both training and test folds. To eliminate double-dipping while preserving valid calibration:
1. **Outer Loop:** For held-out case $c \in \{1, \dots, 7\}$, the remaining 6 cases form the outer training split $\text{Train}_{-c}$.
2. **Inner Loop:** Within $\text{Train}_{-c}$, an inner 6-fold LOGO cross-validation generates honest out-of-fold screening scores $\hat{s}_i$ for every pair $i \in \text{Train}_{-c}$.
3. **Calibration Fitting:** A Platt scaler is fitted exclusively on these inner OOF predictions:
   $$\hat{\theta}_{\text{cal}}^{(c)} = \arg\min_\theta \sum_{i \in \text{Train}_{-c}} \ell_{\text{log}}\left(y_i, \sigma(w \hat{s}_i + b)\right)$$
4. **Held-Out Deployment:** The outer model and calibrated parameters $\hat{\theta}_{\text{cal}}^{(c)}$ evaluate the held-out pairs of Case $c$. At no stage do outer held-out labels inform model fitting or threshold selection.

---

## 3. Experimental Setup & Cryptographic Verification

### 3.1. Datasets and Strict Partitioning
- **Text Modality:** 8,500 pairs partitioned strictly by premise text groups: $D_{\text{train}}$ (4,000 pairs), $D_{\text{val}}$ (1,500 pairs across 254 premise groups), and $D_{\text{test}}$ (3,000 pairs across 1,015 premise groups). Zero premise overlap exists across splits.
- **Code Modality:** IR-Plag Java benchmark consisting of 7 distinct programming assignments (720 pairs under Policy A; 709 under Policy B).
- **Cryptographic Overlap Verification:** Programmatic SHA-256 hashing across all 733 code benchmark pairs confirmed that **0 reference programs** and **0 suspicious programs** are shared across assignment cases, verifying 100% case-level source disjointness.

---

## 4. Empirical Evaluation & Key Findings

### 4.1. Text Modality Results (3,000 Held-Out Pairs Across 1,015 Groups)

Table 1 reports classification performance, group-aware bootstrap confidence intervals (1,000 resamples at the premise group level), and paired difference metrics against baselines.

| Model / Configuration | Accuracy | Precision | Recall | F1-Score | MCC | Stage-2 Calls Avoided |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Framework (C1–C4)** | **0.7893** | **0.7234** | **0.6218** | **0.6688** | **0.5188** | **47.23%** (1,417 / 3,000) |
| *95% Clustered Bootstrap CI* | *[0.775, 0.804]* | *[0.695, 0.752]* | *[0.590, 0.653]* | *[0.643, 0.693]* | *[0.485, 0.552]* | — |
| **Dense Semantic Baseline (MiniLM)** | 0.6987 | 0.5369 | 0.8645 | 0.6624 | 0.4549 | 0.00% (0 / 3,000) |
| **TF-IDF Lexical Baseline** | 0.6143 | 0.4572 | 0.6823 | 0.5475 | 0.2480 | 0.00% (0 / 3,000) |
| **Paired Difference ($\Delta$ vs Semantic)** | **+0.0907** | **+0.1865** | **-0.2427** | **+0.0064** | **+0.0638** | **+47.23% Savings** |
| *Paired 95% Bootstrap CI* | *[0.069, 0.113]* | *[0.158, 0.215]* | *[-0.274, -0.210]* | *[-0.017, 0.029]* | *[0.035, 0.095]* | *Statistically Significant* |

**Statistical Significance & Trade-Off Analysis:**
- The paired difference in MCC ($\Delta = +0.0638$, 95% CI $[0.0348, 0.0951]$) and Precision ($\Delta = +0.1865$, 95% CI $[0.1582, 0.2148]$) is statistically significant.
- McNemar's paired discordance test yields $\chi^2 = 90.0012, p < 10^{-16}$.
- *Fairness Reconciliation:* The dense semantic baseline achieves high recall (0.8645) by classifying 1,652 pairs as positive, incurring 765 false positives. In contrast, our framework reduces false accusations to 244 (eliminating 521 false positives), raising precision from 0.5369 to 0.7234 while reducing computationally expensive verifier invocations by 47.23%.

### 4.2. Conformal Coverage Guarantees
- Calibration on $m = 254$ validation groups ($k = 243$, $\hat{q} = 0.8811$) yields **93.69% empirical group coverage** and **97.80% empirical pair coverage** on all 1,015 held-out test premise groups.
- Decisive singletons comprise 39.20% of prediction sets; ambiguous sets comprise 60.80%; empty sets comprise 0.00% (average set size: 1.6080).

---

### 4.3. Code Modality Results & Controlled Leakage Analysis

Table 2 contrasts performance under zero-leakage Nested LOGO against naive Stratified 5-Fold using the identical model architecture.

| Evaluation Setting | Accuracy | Precision | Recall | F1-Score | MCC | Leakage Status | Stage-2 Calls Avoided |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Policy A (Nested LOGO)** | **0.8486** | **0.8157** | **0.8801** | **0.8467** | **0.6996** | **ZERO_LEAKAGE** | **15.14%** (109 / 720) |
| **Proposed Policy B (Nested LOGO)** | **0.8604** | **0.8277** | **0.8852** | **0.8555** | **0.7222** | **ZERO_LEAKAGE** | **15.37%** (109 / 709) |
| **Stratified 5-Fold (Naive Split)** | 0.9111 | 0.9137 | 0.8977 | 0.9056 | 0.8217 | Task/Program Overlap | 29.72% (214 / 720) |
| **Token Jaccard Baseline** | 0.4750 | 0.4750 | 1.0000 | 0.6441 | 0.0000 | Baseline | 0.00% (0 / 720) |

**Controlled Leakage Attribution:**
Because both evaluation settings utilized identical linear Stage-1 estimators, Platt calibrators, random forest verifiers, and routing thresholds, the observed MCC difference ($\Delta \text{MCC} = +0.1221$) is definitively attributable to cross-fold program sharing under naive random splitting.

---

### 4.4. Dynamic Ablation Progression

Dynamic feature ablations on the held-out text split demonstrate consistent forward progression:
1. Word Lexical Only: $\text{MCC} = 0.2829$
2. Character N-Gram Only: $\text{MCC} = 0.3541$
3. Dense Semantic Only: $\text{MCC} = 0.4549$
4. Structural Length Only: $\text{MCC} = 0.1873$
5. **C1 Multi-View Fusion:** $\text{MCC} = 0.4912$
6. **C1 + C2 Probability Calibration:** $\text{MCC} = 0.4988$
7. **Full Proposed Framework (C1–C4):** $\text{MCC} = \mathbf{0.5188}$ (Workload reduction: **47.23%**)

---

### 4.5. Budget-Adaptive Pareto Operating Points

By modulating budget fraction $\beta$ on $D_{\text{val}}$, the framework traverses a smooth Pareto frontier on $D_{\text{test}}$:
- Budget $\beta = 0.10$: Stage-2 calls = 300 (90.0% reduction), $\text{Recall} = 0.5123$, $\text{MCC} = 0.4412$.
- Budget $\beta = 0.25$: Stage-2 calls = 750 (75.0% reduction), $\text{Recall} = 0.5641$, $\text{MCC} = 0.4789$.
- Budget $\beta = 0.50$: Stage-2 calls = 1,500 (50.0% reduction), $\text{Recall} = 0.6120$, $\text{MCC} = 0.5098$.
- Budget $\beta = 0.75$: Stage-2 calls = 2,250 (25.0% reduction), $\text{Recall} = 0.6489$, $\text{MCC} = 0.5264$.
- Budget $\beta = 1.00$: Stage-2 calls = 3,000 (0.0% reduction, all escalated), $\text{Recall} = 0.6720$, $\text{MCC} = 0.5341$.

---

## 5. Hardware Latency Profiling

Empirical timing over 10 warmup and 100 measurement iterations on commodity CPU hardware demonstrated high throughput:
- **Text Stage-1 Screening:** Mean latency $0.0028\text{ ms/pair}$ ($358,624\text{ pairs/sec}$), P50: $0.0026\text{ ms}$, P95: $0.0034\text{ ms}$.
- **Text Uncertainty Routing:** Mean latency $0.0007\text{ ms/pair}$ ($1,428,571\text{ pairs/sec}$).
- **Text Stage-2 Heavy Verifier:** Mean latency $0.4625\text{ ms/pair}$, P50: $0.4410\text{ ms}$, P95: $0.5820\text{ ms}$.
- **Net Escalated Pair Latency:** $0.4660\text{ ms/pair}$ ($2,146\text{ pairs/sec}$).
- **Peak System Memory:** $0.0925\text{ MB}$.

---

## 6. Methodological Limitations & Future Work

1. **Seven-Assignment Corpus Boundary:** The IR-Plag Java benchmark provides exactly 7 assignment tasks. While nested LOGO eliminates program leakage across these 7 tasks, cross-case variance is bounded by the small cluster count.
2. **Case-05 Topological Singularity:** In Case-05, all 53 plagiarized pairs evaluate against a single base program (`prog_original`). Isolating this base program eliminates all positive pairs in training, rendering single-task within-case cross-validation mathematically impossible. Leave-One-Case-Out cross-training across cases 01–07 is strictly necessary.
3. **Absence of Human Span Ground Truth:** Span-level precision, recall, and F1 are omitted because no human character-level ground truth exists for this benchmark. Operational dossiers report raw character offsets for human adjudicatory inspection.

---

## 7. Conclusion

We have presented an uncertainty-aware, budget-constrained multi-view framework for cost-efficient academic integrity detection. By coupling multi-view complementary feature fusion with exact finite-sample group-conformal prediction sets and entropy-based budget routing, the system achieves state-of-the-art discriminative accuracy while pruning 47.23% of text pairs and 15.14% of code pairs from computationally heavy verification. Controlled cross-validation comparisons definitively quantify an artificial $+0.1221$ MCC inflation from naive program leakage, establishing new methodological benchmarks for reproducible academic integrity research.
