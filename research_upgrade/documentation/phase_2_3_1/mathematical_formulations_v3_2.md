# Mathematical Formulations and Theoretical Foundations (Version 3.2)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.2 (Phase 2.3.1 Independent Audit Revision)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Finalized Mathematical Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

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
3. **Calibrated Posterior Probability ($\hat{p}(\mathbf{x}) \in [0, 1]$):** Estimated probability $P(Y = 1 \mid z(\mathbf{x}))$ produced by a post-hoc calibrator fitted on escalated data. For non-escalated pairs, $\hat{p}(\mathbf{x}) = \text{None}$.
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

1. **Raw Token Set Jaccard Overlap ($u_1$):**
   Let $\mathcal{T}_{\text{raw}}(C)$ denote unique lexical tokens after stripping Java comments and whitespace:
   $$u_1 = S_{\text{raw-token}}(C_A, C_B) = \frac{|\mathcal{T}_{\text{raw}}(C_A) \cap \mathcal{T}_{\text{raw}}(C_B)|}{|\mathcal{T}_{\text{raw}}(C_A) \cup \mathcal{T}_{\text{raw}}(C_B)|}$$

2. **Normalized Identifier Token Set Jaccard ($u_2$):**
   Let $\mathcal{T}_{\text{norm}}(C)$ denote tokens where user-defined identifiers are replaced with canonical placeholder `VAR_TOKEN` via regex:
   $$u_2 = S_{\text{norm-token}}(C_A, C_B) = \frac{|\mathcal{T}_{\text{norm}}(C_A) \cap \mathcal{T}_{\text{norm}}(C_B)|}{|\mathcal{T}_{\text{norm}}(C_A) \cup \mathcal{T}_{\text{norm}}(C_B)|}$$

3. **Operator Frequency Profile Cosine ($u_3$):**
   Let $\mathbf{o}(C) \in \mathbb{N}^{13}$ be the count histogram of 13 Java operators (`+`, `-`, `*`, `/`, `%`, `&&`, `||`, `==`, `!=`, `<`, `>`, `++`, `--`):
   $$u_3 = S_{\text{operator}}(C_A, C_B) = \begin{cases} \frac{\langle \mathbf{o}(C_A), \mathbf{o}(C_B) \rangle}{\|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2}, & \text{if } \|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2 > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

4. **Cyclomatic Keyword Profile Differential ($u_4$):**
   Let $k(C)$ denote the count of decision keywords (`if`, `else`, `while`, `for`, `case`, `catch`, `?`) extracted via fast regex:
   $$u_4 = \Delta_{\text{cyc-proxy}}(C_A, C_B) = 1.0 - \frac{|k(C_A) - k(C_B)|}{\max(k(C_A), k(C_B), 1)}$$

### 2.2 Fused Candidate Scoring Function

The multi-view feature vector $\mathbf{x} \in [0, 1]^4$ is mapped to candidate score $s_{\text{cand}}(\mathbf{x})$:
$$s_{\text{cand}}(\mathbf{x}) = \sigma\left(\mathbf{w}^T \mathbf{x} + b\right) = \frac{1}{1 + \exp\left(-\left(\sum_{j=1}^4 w_j x_j + b\right)\right)}$$
Parameters $\mathbf{w} \in \mathbb{R}^4, b \in \mathbb{R}$ are estimated on $D_{\text{train}}$ by minimizing cross-entropy with $L_2$ regularization:
$$\min_{\mathbf{w}, b} \left[ -\frac{1}{|D_{\text{train}}|} \sum_{i \in D_{\text{train}}} \left( y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i + b) + (1 - y_i) \log(1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 \right]$$

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decisions

### 3.1 Probability Calibration on Conditional Escalation Slice

Because the heavy classifier operates only on escalated pairs, probability calibrators fitted on unselected marginal data suffer from distribution shift.
Let the conditional escalation split be:
$$D_{\text{cal-prob}}^{\text{esc}}(\boldsymbol{\theta}) = \left\{ (\mathbf{x}_i, y_i) \in D_{\text{cal-prob}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$

#### Calibration Formulations:
1. **Platt Scaling (Parametric Sigmoid):**
   $$\hat{P}(Y = 1 \mid z) = \sigma(A \cdot z + B) = \frac{1}{1 + \exp(-(A \cdot z + B))}$$
   Parameters $A, B \in \mathbb{R}$ are fitted on $D_{\text{cal-prob}}^{\text{esc}}$ by minimizing binary cross-entropy.
2. **Isotonic Regression (Non-Parametric Monotonic Step):**
   $$\min_{\mathbf{m}} \sum_{i=1}^{N_{\text{cal-prob}}^{\text{esc}}} \left( y_i - m_i \right)^2 \quad \text{subject to } m_1 \le m_2 \le \dots \le m_{N_{\text{cal-prob}}^{\text{esc}}}$$
   Solved in $O(N)$ via the Pair-Adjacent Violators (PAV) algorithm.

#### Independent Calibration Diagnostics (Issue 1):
Calibration diagnostics are evaluated strictly on independent evaluation data ($D_{\text{test}}^{\text{esc}}$):
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Maximum Calibration Error (MCE):** $\text{MCE} = \max_{m} |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Brier Score:** $\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$.
- **Negative Log-Likelihood (NLL):** $\text{NLL} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$.

---

### 3.2 Modality A: Academic Text Conformal Prediction (Issue 2)

For academic text, split conformal prediction is applied to the independent conditional escalation split:
$$D_{\text{cal-conf}}^{\text{esc}}(\boldsymbol{\theta}) = \left\{ (\mathbf{x}_i, y_i) \in D_{\text{cal-conf}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$
Let $n = |D_{\text{cal-conf}}^{\text{esc}}|$ denote the calibration sample size.

#### Non-Conformity Score:
For an instance with true label $y_i \in \{0, 1\}$ and calibrated probability $\hat{p}_i$:
$$\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i) = \begin{cases} 1 - \hat{p}_i, & \text{if } y_i = 1 \\ \hat{p}_i, & \text{if } y_i = 0 \end{cases}$$

#### Exact Finite-Sample Conformal Order Statistic (Issue 2):
For user significance level $\epsilon \in (0, 1)$ (e.g. $\epsilon = 0.05$ for $95\%$ nominal coverage):
$$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$
The empirical non-conformity threshold $\hat{q}_{1-\epsilon}$ is the $k$-th smallest value in $\{\alpha_1, \dots, \alpha_n\}$:
$$\hat{q}_{1-\epsilon} = \alpha_{(k)}$$

#### Mathematical Condition for Coverage Attainability:
If $k > n$, which occurs when:
$$n < \left\lceil \frac{1 - \epsilon}{\epsilon} \right\rceil$$
(e.g., $n < \lceil 0.95/0.05 \rceil = 19$ for $\epsilon = 0.05$), the quantile cannot be attained without extrapolation. In this case, nominal coverage $1-\epsilon$ cannot be mathematically guaranteed. The system must abstain from claiming a conformal guarantee and fall back to exploratory uncertainty routing.

#### Premise-Level Group Conformal Formulation:
When respecting premise-level clusters $\mathcal{G} = \{g_1, \dots, g_K\}$, a group non-conformity score is defined:
$$\alpha_g = \max_{i \in g} \alpha_i$$
Where $K_n = |\mathcal{G}_{\text{cal}}|$ is the number of unique reference premises. The order statistic is $k_g = \lceil (K_n + 1)(1 - \epsilon) \rceil$.

#### Conformal Prediction Set:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}_{\text{query}}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}_{\text{query}}) \le \hat{q}_{1-\epsilon} \right\}$$

#### Operational Routing Actions (Issue 4):
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \begin{cases}
\{0\}, & \text{Negative Prediction} \implies \text{Action: LOW\_RISK\_REVIEW\_DEFERRED, Review: False} \\
\{1\}, & \text{Positive Flag} \implies \text{Action: HIGH\_RISK\_CANDIDATE\_FLAGGED, Review: True} \\
\{0, 1\}, & \text{Conflict / Uncertainty} \implies \text{Action: AMBIGUOUS\_HUMAN\_REVIEW, Review: True} \\
\emptyset, & \text{Empirical Abstention} \implies \text{Action: ABSTAIN\_UNRESOLVED, Review: True}
\end{cases}$$

---

### 3.3 Modality B: Source Code Exploratory Calibrated Routing (Issue 3)

#### Mathematical Justification for Disclaiming Code Conformal Guarantees:
IR-Plag contains only 15 unique reference Java programs in `case-05` ($N_{\text{ref}} = 15$). Because:
$$N_{\text{ref}} = 15 < \left\lceil \frac{1 - \epsilon}{\epsilon} \right\rceil = 19 \quad (\text{for } \epsilon = 0.05)$$
the order statistic index $k = \lceil 16 \times 0.95 \rceil = 16 > 15$. Hence, finite-sample conformal guarantees for source code cannot be mathematically attained and are **strictly disclaimed**.

#### Group-Aware Exploratory Cross-Fitting Formulation:
`case-05` is partitioned into 5 folds $\mathcal{F}_1, \dots, \mathcal{F}_5$, grouped by reference program identity (3 reference programs per fold). For fold $m$:
- Train Platt calibrator $g_{\text{cal}}^{(-m)}$ on $\bigcup_{j \neq m} \mathcal{F}_j$.
- Predict out-of-fold calibrated probabilities $\hat{p}_i = g_{\text{cal}}^{(-m)}(z_i)$ for $i \in \mathcal{F}_m$.
- Evaluate predictive Shannon entropy:
  $$H(\hat{p}) = -\hat{p} \log_2(\hat{p}) - (1 - \hat{p}) \log_2(1 - \hat{p})$$
- Exploratory decision band routing:
  $$\text{Action}_{\text{code}}(\hat{p}, H) = \begin{cases}
  \text{LOW\_RISK\_REVIEW\_DEFERRED}, & \text{if } \hat{p} < \tau_{\text{non-plag}} \text{ and } H(\hat{p}) \le \tau_{\text{entropy}} \implies \text{Review: False} \\
  \text{HIGH\_RISK\_CANDIDATE\_FLAGGED}, & \text{if } \hat{p} \ge \tau_{\text{plag}} \text{ and } H(\hat{p}) \le \tau_{\text{entropy}} \implies \text{Review: True} \\
  \text{AMBIGUOUS\_HUMAN\_REVIEW}, & \text{otherwise} \implies \text{Review: True}
  \end{cases}$$
  For code records, `conformal_prediction_set = None (Disclaimed for Code)`.

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Strict Separation of Time and Resource Metrics (Issue 5)

To prevent dimensional conflation, latency (time) and memory (resources) are kept in strictly separate mathematical models:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DISAGGREGATED COST AND RESOURCE VECTORS                                   │
├───────────────────────────────────┬────────────────────────────────────────────────────────────────────┤
│ Vector Category                   │ Formal Mathematical Definition                                     │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ Latency Vector T (ms)             │ T = [T_prep, T_vec, T_tok, T_ast, T_cache, T_Tier1, T_infer, T_exp]│
│ Resource Vector M (MB)            │ M = [M_RSS, M_cache, M_model]                                      │
└───────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

**Never add memory bytes to latency milliseconds!**

#### Expected Per-Pair Wall-Clock Latency:
$$\bar{T}_{\text{total}}(\boldsymbol{\theta}) = T_{\text{prep}} + T_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ T_{\text{feat-heavy}} + T_{\text{infer-heavy}} + T_{\text{router}} \right] + P(\text{Review} \mid \boldsymbol{\theta}) \cdot T_{\text{explain}}$$

Where:
- $P(\text{Escalate} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}})$.
- $P(\text{Review} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \left[ \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \wedge \text{ReviewRequired}_i) \right]$.

### 4.2 Constrained Mathematical Optimization Objective

Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with $\theta_{\text{low}} < \theta_{\text{high}}$.
$$\min_{\boldsymbol{\theta}} \quad \bar{T}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \bar{T}_{\text{total}}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

#### Infeasibility Condition:
If the feasible set $\Omega = \{ \boldsymbol{\theta} \mid g_1(\boldsymbol{\theta}) \le 0 \wedge g_2(\boldsymbol{\theta}) \le 0 \wedge g_3(\boldsymbol{\theta}) \le 0 \} = \emptyset$, the optimizer returns:
$$\boldsymbol{\theta}^* = \text{None}, \quad \text{Status: INFEASIBLE\_BUDGET}$$

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Text Evidence: Smith-Waterman Dynamic Alignment (Affine Gap)
$$H_{i, j} = \max \left(0, \, H_{i-1, j-1} + s(a_i, b_j), \, E_{i, j}, \, F_{i, j}\right)$$
Where:
$$E_{i, j} = \max\left(H_{i-1, j} - u - v, \, E_{i-1, j} - v\right)$$
$$F_{i, j} = \max\left(H_{i, j-1} - u - v, \, F_{i, j-1} - v\right)$$
Parameters: match $s(a, b) = +2.0$, mismatch $s(a, b) = -1.0$, gap open $u = 2.0$, gap extend $v = 0.5$.

### 5.2 Code Evidence: AST Subtree Isomorphism & Identifier Ledger
Bijection $\sigma: \text{Vars}(t_A) \to \text{Vars}(t_B)$ such that:
$$\text{GrammarType}(t_A) \equiv \text{GrammarType}(t_B) \quad \text{and} \quad \sigma(\text{LeafTokens}(t_A)) = \text{LeafTokens}(t_B)$$
Identifier substitution ledger:
$$\mathcal{M}_{\text{id}} = \left\{ (v_A, v_B) \in \text{Vars}(C_A) \times \text{Vars}(C_B) \mid \text{Count}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \right\}$$

### 5.3 Explanation Quality Metrics:
1. **Evidence Faithfulness (Decision Probability Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   Evaluated strictly on true positive pairs escalated to Tier 2. (Assigned `None` for Tier-1 direct flags).
2. **Evidence Coverage:** $\text{Coverage}(D_A, D_B) = \frac{|\mathcal{E}_{\text{tokens}}(D_B)|}{|D_B|_{\text{tokens}}}$.
3. **Transformation Alignment Precision:** $\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$.

---

## 6. Statistical Analysis Formulations (Issue 6)

### 6.1 Group-Aware Cluster Bootstrap Confidence Intervals
Resampling unit is the unique cluster $g \in \mathcal{G}$ (`reference_text` premise for text; assignment task for code).
For $b = 1, \dots, B$ ($B = 1,000$):
- Sample $K$ group identifiers $\mathcal{G}^{*(b)}$ with replacement.
- Assemble resampled dataset $D^{*(b)} = \bigcup_{g \in \mathcal{G}^{*(b)}} \{(\mathbf{x}_i, y_i) \mid g_i = g\}$.
- Compute metric $\theta^{*(b)} = M(D^{*(b)})$.
- Compute $95\%$ percentile interval: $\text{CI}_{95\%} = [\theta^{*}_{(\lfloor 0.025 \cdot B \rfloor)}, \, \theta^{*}_{(\lceil 0.975 \cdot B \rceil)}]$.

### 6.2 Exact Binomial McNemar Test for Paired Classifiers
Let $b$ be Proposed correct & Baseline wrong; let $c$ be Proposed wrong & Baseline correct.
- If $b + c \ge 25$: $\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1$.
- If $b + c < 25$ (mandatory for code tasks `case-06` and `case-07`):
  $$p = 2 \cdot \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} \left(\frac{1}{2}\right)^{b + c}$$

### 6.3 Benjamini-Hochberg False Discovery Rate (FDR) Procedure
Sort raw $p$-values: $p_{(1)} \le \dots \le p_{(K)}$. Find largest $k^*$ such that:
$$k^* = \max \left\{ k \in \{1, \dots, K\} \mid p_{(k)} \le \frac{k}{K} \cdot q^* \right\} \quad (q^* = 0.05)$$
Reject null hypotheses $H_{(1)}, \dots, H_{(k^*)}$.

---
*End of Document: mathematical_formulations_v3_2.md*

