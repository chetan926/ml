# Mathematical Formulations and Theoretical Foundations (Version 3.3)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.3 (Phase 2.3.2 Methodological Audit & Certification)  
**Author:** Senior Machine Learning Researcher, Statistical Methodologist & Independent Research Auditor  
**Date:** November 2026  
**Status:** Certified Final Mathematical Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals  
**Source Baseline:** Audited from Version 3.2 (`research_upgrade/documentation/phase_2_3_1/`)

---

## 1. Formal Mathematical Variables & Problem Setting

Let $\mathcal{D} = \{(D_A^{(i)}, D_B^{(i)}, y_i, g_i)\}_{i=1}^N$ denote a benchmark dataset of submission pairs, where:
- $D_A^{(i)}$ represents a reference document (source text or base program).
- $D_B^{(i)}$ represents a suspicious candidate submission.
- $y_i \in \{0, 1\}$ represents ground-truth academic integrity ($y_i = 1$ denotes plagiarism / illicit reuse; $y_i = 0$ denotes independent work).
- $g_i \in \mathcal{G}$ represents the grouping cluster identifier (unique source premise ID for text; unique reference programming task / program ID for source code).
- $\mathcal{M} \in \{\text{text}, \text{code}\}$ denotes the operational modality.

### 1.1 Disambiguated Decision Quantities

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Tier-1 Candidate Score    s_cand(x) in [0, 1]  --> Fast candidate score on 100% of candidate pairs │
│ 2. Heavy Raw Margin          z(x) in R            --> Margin from Heavy Model on ESCALATED pairs only │
│ 3. Calibrated Probability    p_hat(x) in [0, 1]   --> Calibrated posterior on ESCALATED pairs only    │
│ 4. Conformal Prediction Set  C(x) in {{0},{1},{0,1},Ø} -> Set output on ESCALATED TEXT pairs only     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Candidate Score ($s_{\text{cand}}(\mathbf{x}) \in [0, 1]$):** Computed by the Tier-1 fast generator on all $100\%$ candidate pairs. Governed by routing cutoffs $(\theta_{\text{low}}, \theta_{\text{high}})$. It is an uncalibrated ranking score, not a posterior probability.
2. **Heavy Raw Margin ($z(\mathbf{x}) = f_{\text{heavy}}(\mathbf{x}) \in \mathbb{R}$):** Computed by the heavy classifier (Sentence-BERT for text; 7-D structural classifier for code) **strictly on escalated pairs**. For non-escalated pairs ($s_{\text{cand}} < \theta_{\text{low}}$ or $s_{\text{cand}} \ge \theta_{\text{high}}$), $z(\mathbf{x}) = \text{None}$.
3. **Calibrated Posterior Probability ($\hat{p}(\mathbf{x}) \in [0, 1]$):** Estimated probability $P(Y = 1 \mid z(\mathbf{x}))$ produced by a post-hoc calibrator fitted on escalated text data. For non-escalated pairs, $\hat{p}(\mathbf{x}) = \text{None}$. For code, calibrated probabilities are formally disclaimed.
4. **Conformal Prediction Set ($\mathcal{C}_{1-\epsilon}(\mathbf{x}) \subseteq \{0, 1\}$):** Set-valued output derived from $\hat{p}(\mathbf{x})$ and non-conformity threshold $\hat{q}_{1-\epsilon}$ on escalated pairs. Validated **strictly for the text modality**. For source code, conformal prediction is explicitly disclaimed, and $\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \text{None}$.

---

## 2. Contribution 1: Multi-View Candidate Generation

### 2.1 Multi-View Representation Spaces (Purely Lightweight Tier 1)

Tier-1 feature extractors are strictly restricted to $O(|D|)$ string and regex operations ($\le 0.050\text{ ms/pair}$). **Heavy AST parsing is strictly prohibited at Tier 1.**

#### A. Natural Language Text Multi-View Vector $\mathbf{v}_{\text{text}} \in [0, 1]^4$:

1. **Sublinear TF-IDF Cosine Similarity ($v_1$):**
   $$\text{tf-idf}(t, d) = \left( 1 + \log(\text{tf}(t, d)) \right) \cdot \log\left( \frac{1 + |D_{\text{train}}|}{1 + \text{df}(t)} \right) + 1 \quad \text{for } \text{tf}(t, d) > 0$$
   Where vocabulary $\mathcal{V}$ and document frequencies $\text{df}(t)$ are fitted strictly on $D_{\text{train}}$.
   $$v_1 = S_{\text{word-tfidf}}(D_A, D_B) = \begin{cases} \frac{\langle \mathbf{t}_A, \mathbf{t}_B \rangle}{\|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2}, & \text{if } \|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2 > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
   *Computational Complexity Clarification:* Sublinear term-frequency weighting refers to the logarithmic scaling of term counts $\left(1 + \log(\text{tf})\right)$ to prevent high-frequency words from dominating document vectors. It does **not** imply sublinear algorithmic time complexity. Vectorizing a document $D$ requires scanning all tokens, which remains $O(|D|)$ linear in document length.

2. **Character 3-Gram Set Jaccard Overlap ($v_2$):**
   Let $\mathcal{G}_3(D)$ denote the set of unique character tri-grams in lowercased string $D$:
   $$v_2 = S_{\text{char-3gram}}(D_A, D_B) = \begin{cases} \frac{|\mathcal{G}_3(D_A) \cap \mathcal{G}_3(D_B)|}{|\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)|}, & \text{if } |\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)| > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

3. **Word Token Set Jaccard Overlap ($v_3$):**
   Let $\mathcal{W}(D)$ denote the set of whitespace- and punctuation-delimited word tokens:
   $$v_3 = S_{\text{word-jaccard}}(D_A, D_B) = \begin{cases} \frac{|\mathcal{W}(D_A) \cap \mathcal{W}(D_B)|}{|\mathcal{W}(D_A) \cup \mathcal{W}(D_B)|}, & \text{if } |\mathcal{W}(D_A) \cup \mathcal{W}(D_B)| > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

4. **Length-Ratio Symmetry Metric ($v_4$):**
   $$v_4 = \Delta_{\text{len}}(D_A, D_B) = \frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}}, 1)}$$

#### B. Source Code Multi-View Vector $\mathbf{u}_{\text{code}} \in [0, 1]^4$ (Pure Regex/Lexer Tier 1):

1. **Raw Token Set Jaccard Overlap ($u_1$):** Lexical token overlap after stripping Java comments and whitespace.
2. **Normalized Identifier Token Set Jaccard ($u_2$):** Lexical token overlap where identifiers are replaced with canonical placeholder `VAR_TOKEN` via regex.
3. **Operator Frequency Profile Cosine ($u_3$):** Cosine similarity between 13-dimensional operator count vectors.
4. **Cyclomatic Keyword Profile Differential ($u_4$):** Normalized difference in decision keyword counts (`if`, `else`, `while`, `for`, `case`, `catch`, `?`).

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decisions

### 3.1 Text Modality: Complete Group-Level Conformal Formulation (Issue 1)

#### A. Prediction Unit and Grouping Unit Definitions:
- **Prediction Unit:** An individual submission pair $(D_A^{(i)}, D_B^{(i)})$ with ground-truth binary label $y_i \in \{0, 1\}$.
- **Grouping Unit:** The reference premise / source document identity $g \in \mathcal{G}$. Each group $g$ contains $m_g \ge 1$ pairs sharing the same reference document.

#### B. Selection-Aware Group Calibration Population:
Calibration operates strictly on the **conditional escalation slice**:
$$\mathcal{G}_{\text{cal-conf}}^{\text{esc}} = \left\{ g \in \mathcal{G}_{\text{cal-conf}} \mid \exists i \in g, \, \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$
Let $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ denote the number of independent escalated premise groups in calibration.

#### C. Group-Level Non-Conformity Score:
For an escalated pair $i$ with calibrated probability $\hat{p}_i = \hat{P}(Y=1 \mid \mathbf{x}_i)$, the pair-level non-conformity is $\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$.
The group-level non-conformity score is defined as the worst-case non-conformity among all escalated pairs in group $g$:
$$\alpha_g = \max_{i \in g, \, \text{escalated}} \alpha_i = \max_{i \in g, \, \text{escalated}} \left( 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i) \right)$$

#### D. Exact Finite-Sample Order Statistic Quantile:
For user-chosen significance level $\epsilon \in (0, 1)$ (e.g. $\epsilon = 0.05$ for $95\%$ nominal coverage):
$$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$

- **Finite-Sample Attainability Condition:**
  If $k > n$, which occurs when:
  $$n < \left\lceil \frac{1 - \epsilon}{\epsilon} \right\rceil$$
  (e.g., $n < 19$ for $\epsilon = 0.05$), the empirical quantile cannot be attained without extrapolation. In this case, we set:
  $$\hat{q}_{1-\epsilon} = \infty$$
  and the system explicitly **abstains** from claiming a finite-sample conformal guarantee.
- **Quantile Definition (when $k \le n$):**
  $$\hat{q}_{1-\epsilon} = \alpha_{(k)} \quad (\text{the } k\text{-th smallest value in } \{\alpha_g\}_{g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}})$$

#### E. Pair-Level Prediction Set Construction:
For an escalated query pair $\mathbf{x}_{\text{query}}$ belonging to a new test group $g_{\text{test}}$:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}_{\text{query}}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}_{\text{query}}) \le \hat{q}_{1-\epsilon} \right\}$$

#### F. Theoretical Coverage Guarantee and Statistical Properties:
1. **Simultaneous Group Coverage Theorem:** Assuming premise groups are exchangeable:
   $$\mathbb{P}\left( \forall i \in g_{\text{test}}, \, y_i \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_i) \mid \text{Escalated} \right) = \mathbb{P}\left( \max_{i \in g_{\text{test}}} \alpha_i \le \hat{q}_{1-\epsilon} \mid \text{Escalated} \right) \ge 1 - \epsilon$$
2. **Conservative Pair-Level Coverage Implication:** Because $\alpha_i \le \max_{j \in g} \alpha_j$, it follows directly that:
   $$\mathbb{P}\left( y_i \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_i) \mid \text{Escalated} \right) \ge \mathbb{P}\left( \forall j \in g_{\text{test}}, \, y_j \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_j) \mid \text{Escalated} \right) \ge 1 - \epsilon$$
   Individual pair coverage is at least $1-\epsilon$, but is **conservative** (over-coverage for individual pairs).
3. **Efficiency Trade-Off with Unequal Group Sizes:** Groups with more hypothesis pairs contribute higher maximum non-conformity scores, inflating $\hat{q}_{1-\epsilon}$ and producing more ambiguous prediction sets ($\{0, 1\}$).

---

### 3.2 Code Modality: Repair of Cross-Fitting and Class-Group Imbalance (Issue 3)

#### A. Proof of Structural Failure for 5-Fold Platt Scaling on `case-05`:
In the authentic IR-Plag corpus, `case-05` contains $N = 107$ pairs derived from 16 Java programs:
- 1 positive-bearing program (`original/Main.java`), which accounts for all 53 plagiarized pairs.
- 15 negative-bearing programs (`non-plag-01` to `non-plag-15`), which account for all 54 non-plagiarized pairs.

Let $\mathcal{F}_1, \dots, \mathcal{F}_5$ be a 5-fold group-aware split of the 16 programs. The positive-bearing group `original` can belong to at most one fold (say $\mathcal{F}_1$).
- In Fold 1 evaluation, the training set $\bigcup_{j=2}^5 \mathcal{F}_j$ contains **0 positive pairs** ($N_{\text{pos}}^{\text{train}} = 0$).
- Binary logistic regression minimizes cross-entropy:
  $$\mathcal{L}(A, B) = -\sum_{i=1}^{N_{\text{train}}} \left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log(1 - \sigma(A z_i + B)) \right]$$
  When $\forall i, y_i = 0$, the loss simplifies to $-\sum_i \log(1 - \sigma(A z_i + B))$, which is minimized by driving $B \to -\infty$ (or $A z_i + B \to -\infty$). The optimization diverges, and Platt scaling **cannot be fitted**.
- Conversely, in Folds 2 through 5, the held-out test fold contains **0 positive pairs**, rendering out-of-fold calibration diagnostics completely uninformative for the positive class.

#### B. Methodological Correction: Exploratory Raw Score Routing:
Forced 5-fold Platt scaling is **strictly rejected**. Code probability calibration is downgraded to **exploratory score thresholding** on uncalibrated decision margins $z \in \mathbb{R}$:
$$\text{Action}_{\text{code}}(z) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED}, & \text{if } z < \tau_{\text{low}}^{\text{raw}} \implies \text{Review: False} \\
\text{HIGH\_RISK\_CANDIDATE\_FLAGGED}, & \text{if } z \ge \tau_{\text{high}}^{\text{raw}} \implies \text{Review: True} \\
\text{AMBIGUOUS\_HUMAN\_REVIEW}, & \text{if } \tau_{\text{low}}^{\text{raw}} \le z < \tau_{\text{high}}^{\text{raw}} \implies \text{Review: True}
\end{cases}$$
Raw scores must **never** be described as calibrated probabilities. Conformal sets are assigned `None (Disclaimed for Code)`.

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Strict Dimensional Separation of Time and Resources (Issue 5)

Latency (time) and memory (resources) are kept in strictly separate mathematical models:
- **Latency Vector (ms):** $\mathbf{T} = [T_{\text{prep}}, T_{\text{vec}}, T_{\text{tok}}, T_{\text{ast}}, T_{\text{cache}}, T_{\text{Tier1}}, T_{\text{infer}}, T_{\text{explain}}]$.
- **Resource Vector (MB):** $\mathbf{M} = [M_{\text{RSS}}, M_{\text{cache}}, M_{\text{model}}]$.

**Never add memory bytes to latency milliseconds!**

#### Expected Per-Pair Wall-Clock Latency:
$$\bar{T}_{\text{total}}(\boldsymbol{\theta}) = T_{\text{prep}} + T_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ T_{\text{feat-heavy}} + T_{\text{infer-heavy}} + T_{\text{router}} \right] + P(\text{Review} \mid \boldsymbol{\theta}) \cdot T_{\text{explain}}$$

Where:
- $P(\text{Escalate} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}})$.
- $P(\text{Review} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \left[ \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \wedge \text{ReviewRequired}_i) \right]$.

### 4.2 Constrained Optimization Objective on $D_{\text{val}}$:
$$\min_{\boldsymbol{\theta}} \quad \bar{T}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \bar{T}_{\text{total}}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Text Evidence: Smith-Waterman Affine Gap Recurrence
$$H_{i, j} = \max \left(0, \, H_{i-1, j-1} + s(a_i, b_j), \, E_{i, j}, \, F_{i, j}\right)$$
Where:
$$E_{i, j} = \max\left(H_{i-1, j} - u - v, \, E_{i-1, j} - v\right)$$
$$F_{i, j} = \max\left(H_{i, j-1} - u - v, \, F_{i, j-1} - v\right)$$
Parameters: match reward $s(a, b) = +2.0$, mismatch penalty $s(a, b) = -1.0$, gap open $u = 2.0$, gap extend $v = 0.5$.

### 5.2 Code Evidence: AST Subtree Isomorphism & Identifier Ledger
Bijection $\sigma: \text{Vars}(t_A) \to \text{Vars}(t_B)$ such that $\text{GrammarType}(t_A) \equiv \text{GrammarType}(t_B)$ and $\sigma(\text{LeafTokens}(t_A)) = \text{LeafTokens}(t_B)$.
Mapping ledger $\mathcal{M}_{\text{id}} = \{ (v_A, v_B) \mid \text{Count}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \}$.

### 5.3 Explanation Quality Metrics:
1. **Evidence Faithfulness (Feature Ablation Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   Evaluated strictly on true positive pairs escalated to Tier 2. (Assigned `None` for Tier-1 direct flags).
2. **Evidence Coverage:** $\text{Coverage}(D_A, D_B) = \frac{|\mathcal{E}_{\text{tokens}}(D_B)|}{|D_B|_{\text{tokens}}}$.
3. **Transformation Alignment Precision:** $\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$.

---

## 6. Statistical Inference Formulations (Issue 4)

### 6.1 Two-Sided Exact Binomial McNemar Test
Let $b$ denote pairs where Proposed Model is correct and Baseline is wrong; let $c$ denote pairs where Proposed Model is wrong and Baseline is correct.
The two-sided exact binomial $p$-value is:
$$p = \min\left(1.0, \, 2 \cdot \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} \left(\frac{1}{2}\right)^{b + c}\right)$$

#### Assumptions and Clustering Limitations:
- **Assumption 1 (Paired Observations):** Both classifiers evaluate the exact same instances.
- **Assumption 2 (Conditional Independence):** Under the null hypothesis $H_0: p_b = p_c = 0.5$, discordant outcomes are conditionally independent given $n_{\text{disc}} = b + c$.
- **Clustering Limitation:** When pairs share underlying reference premises, discordant pairs may be clustered. The exact binomial formula corrects for finite-sample discreteness, but **does not correct for cluster correlation**. Cluster bootstrap is required to assess sampling variance.

### 6.2 Group-Aware Cluster Bootstrap Confidence Intervals
Resampling unit is the unique cluster $g \in \mathcal{G}$ (`reference_text` premise for text; assignment task for code).
For $b = 1, \dots, B$ ($B = 1,000$):
- Sample $K$ group identifiers $\mathcal{G}^{*(b)}$ with replacement.
- Assemble resampled dataset $D^{*(b)} = \bigcup_{g \in \mathcal{G}^{*(b)}} \{(\mathbf{x}_i, y_i) \mid g_i = g\}$.
- Compute metric $\theta^{*(b)} = M(D^{*(b)})$.
- Compute $95\%$ percentile interval: $\text{CI}_{95\%} = [\theta^{*}_{(\lfloor 0.025 \cdot B \rfloor)}, \, \theta^{*}_{(\lceil 0.975 \cdot B \rceil)}]$.

### 6.3 Benjamini-Hochberg False Discovery Rate (FDR) Procedure
Sort raw $p$-values: $p_{(1)} \le \dots \le p_{(K)}$. Find largest $k^*$ such that:
$$k^* = \max \left\{ k \in \{1, \dots, K\} \mid p_{(k)} \le \frac{k}{K} \cdot q^* \right\} \quad (q^* = 0.05)$$
Reject null hypotheses $H_{(1)}, \dots, H_{(k^*)}$.

---
*End of Document: mathematical_formulations_v3_3.md*

