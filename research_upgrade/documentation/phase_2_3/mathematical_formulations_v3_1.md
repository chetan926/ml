# Mathematical Formulations and Theoretical Foundations (Version 3.1)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.1 (Phase 2.3 Methodological Consistency & Implementation Readiness)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
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

To prevent methodological conflation across the triage cascade, four decision quantities are formally distinguished:

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

Tier-1 feature extractors are strictly restricted to $O(|D|)$ string and regex operations ($\le 0.045\text{ ms/pair}$). **Heavy AST parsing is strictly prohibited at Tier 1.**

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

Because the heavy classifier operates only on escalated pairs, probability calibrators fitted on unselected marginal data suffer from severe distribution shift.
Let the conditional escalation split be:
$$D_{\text{cal-prob}}^{\text{esc}}(\boldsymbol{\theta}) = \left\{ (\mathbf{x}_i, y_i) \in D_{\text{cal-prob}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$

#### Calibration Formulations:
1. **Platt Scaling (Parametric Sigmoid):**
   $$\hat{P}(Y = 1 \mid z) = \sigma(A \cdot z + B) = \frac{1}{1 + \exp(-(A \cdot z + B))}$$
   Parameters $A, B \in \mathbb{R}$ are fitted on $D_{\text{cal-prob}}^{\text{esc}}$ by minimizing binary cross-entropy.
2. **Isotonic Regression (Non-Parametric Monotonic Step):**
   $$\min_{\mathbf{m}} \sum_{i=1}^{N_{\text{cal-prob}}^{\text{esc}}} \left( y_i - m_i \right)^2 \quad \text{subject to } m_1 \le m_2 \le \dots \le m_{N_{\text{cal-prob}}^{\text{esc}}}$$
   Solved in $O(N)$ via the Pair-Adjacent Violators (PAV) algorithm.

#### Independent Calibration Diagnostics:
To evaluate calibration quality without training contamination, metrics are evaluated on an independent evaluation partition (e.g. held-out validation slice or $D_{\text{test}}^{\text{esc}}$):
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  Where $B_m$ represents probability bins of equal width $\frac{1}{M}$, $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$, and $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$.
- **Maximum Calibration Error (MCE):**
  $$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score:**
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \left( \hat{p}_i - y_i \right)^2$$
- **Negative Log-Likelihood (NLL):**
  $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

### 3.2 Modality A: Academic Text Conformal Prediction

For academic text, split conformal prediction is applied to the independent conditional escalation split:
$$D_{\text{cal-conf}}^{\text{esc}}(\boldsymbol{\theta}) = \left\{ (\mathbf{x}_i, y_i) \in D_{\text{cal-conf}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$

#### Non-Conformity Score:
For an instance with true label $y_i \in \{0, 1\}$ and calibrated probability $\hat{P}(Y = 1 \mid \mathbf{x}_i) = \hat{p}_i$:
$$\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i) = \begin{cases} 1 - \hat{p}_i, & \text{if } y_i = 1 \\ \hat{p}_i, & \text{if } y_i = 0 \end{cases}$$

#### Empirical Conformal Quantile:
For user-defined error significance level $\epsilon \in (0, 1)$ (e.g. $\epsilon = 0.05$ for $95\%$ nominal coverage):
$$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (N_{\text{cal-conf}}^{\text{esc}} + 1)(1 - \epsilon) \rceil}{N_{\text{cal-conf}}^{\text{esc}}}, \, \{\alpha_i\}_{i=1}^{N_{\text{cal-conf}}^{\text{esc}}} \right)$$

#### Conformal Prediction Set:
For an escalated query pair $\mathbf{x}_{\text{query}}$:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}_{\text{query}}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}_{\text{query}}) \le \hat{q}_{1-\epsilon} \right\}$$
Concretely:
$$\begin{aligned}
0 \in \mathcal{C}_{1-\epsilon} &\iff \hat{p}_{\text{query}} \le \hat{q}_{1-\epsilon} \\
1 \in \mathcal{C}_{1-\epsilon} &\iff (1 - \hat{p}_{\text{query}}) \le \hat{q}_{1-\epsilon}
\end{aligned}$$

#### Theoretical Guarantee & Precise Scope:
- **Theorem (Conditional Marginal Coverage):** If the calibration pairs $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N_{\text{cal-conf}}^{\text{esc}}}$ and escalated test pair $(\mathbf{x}_{N+1}, y_{N+1})$ are conditionally exchangeable given $\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}) < \theta_{\text{high}}$, then:
  $$P\left( y_{N+1} \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_{N+1}) \mid \text{Escalated} \right) \ge 1 - \epsilon$$
- **What is NOT guaranteed:**
  1. *Not individual-conditional coverage:* $P(Y \in \mathcal{C} \mid X = x)$ may deviate from $1-\epsilon$ for specific sub-populations.
  2. *Not guaranteed under distribution shift:* If student writing style shifts significantly from calibration data, exchangeability is violated.
  3. *Not a recall guarantee:* $P(1 \in \mathcal{C} \mid Y = 1)$ is not guaranteed to equal $1-\epsilon$; a non-plagiarized true label $Y=0$ covered by $\{0\}$ satisfies coverage while excluding $1$.

#### Complete Four-Set Operational Routing:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \begin{cases}
\{0\}, & \text{Model-supported negative} \implies \text{Action: CLEARED\_NON\_PLAGIARIZED, Human Review: False} \\
\{1\}, & \text{Model-supported positive} \implies \text{Action: CONFIRMED\_PLAGIARIZED, Human Review: True} \\
\{0, 1\}, & \text{High predictive conflict} \implies \text{Action: AMBIGUOUS\_HUMAN\_REVIEW, Human Review: True} \\
\emptyset, & \text{Empirical abstention} \implies \text{Action: ABSTAIN\_UNRESOLVED, Human Review: True}
\end{cases}$$

---

### 3.3 Modality B: Source Code Exploratory Calibrated Uncertainty Routing

#### Justification for Disclaiming Conformal Guarantee on Code:
For split conformal prediction at error rate $\epsilon = 0.05$, the required calibration sample size is:
$$N_{\text{cal}} \ge \left\lceil \frac{1 - \epsilon}{\epsilon} \right\rceil = \left\lceil \frac{0.95}{0.05} \right\rceil = 19 \text{ independent groups}$$
In the IR-Plag corpus, `case-05` contains only **15 unique reference Java programs**. Because submissions derived from the same reference program are correlated, treating pairs as independent violates exchangeability. Therefore, **finite-sample conformal guarantees for code cannot be justified and are formally disclaimed**.

#### Exploratory Calibrated Routing Formulation:
The heavy code model (Random Forest / SVM over 7 AST features) outputs calibrated probability $\hat{p} = g_{\text{cal}}(z)$.
Uncertainty is quantified via predictive Shannon entropy:
$$H(\hat{p}) = -\hat{p} \log_2(\hat{p}) - (1 - \hat{p}) \log_2(1 - \hat{p}) \quad \text{for } \hat{p} \in (0, 1)$$
Decision bands $[\tau_{\text{non-plag}}, \tau_{\text{plag}}]$ (e.g. $[0.20, 0.80]$) and entropy cutoff $\tau_{\text{entropy}}$ (e.g. $0.72$ bits, corresponding to $\hat{p} \in [0.20, 0.80]$) govern exploratory routing:
$$\text{Action}_{\text{code}}(\hat{p}, H) = \begin{cases}
\text{CLEARED\_NON\_PLAGIARIZED}, & \text{if } \hat{p} < \tau_{\text{non-plag}} \text{ and } H(\hat{p}) \le \tau_{\text{entropy}} \implies \text{Review: False} \\
\text{CONFIRMED\_PLAGIARIZED}, & \text{if } \hat{p} \ge \tau_{\text{plag}} \text{ and } H(\hat{p}) \le \tau_{\text{entropy}} \implies \text{Review: True} \\
\text{AMBIGUOUS\_HUMAN\_REVIEW}, & \text{if } \tau_{\text{non-plag}} \le \hat{p} < \tau_{\text{plag}} \text{ or } H(\hat{p}) > \tau_{\text{entropy}} \implies \text{Review: True}
\end{cases}$$
For code records, `conformal_prediction_set = None (Disclaimed for Code)`.

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Ten-Component Computational Cost Accounting Model

The expected per-pair computational cost $\bar{C}_{\text{total}}(\boldsymbol{\theta})$ accounts for all 10 stages of the end-to-end pipeline:
$$\bar{C}_{\text{total}}(\boldsymbol{\theta}) = C_{\text{prep}} + C_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ C_{\text{feat-heavy}} + C_{\text{infer-heavy}} + C_{\text{router}} \right] + P(\text{Review} \mid \boldsymbol{\theta}) \cdot C_{\text{explain}}$$

Where the 10 explicit cost components are:
1. $C_{\text{prep}}$: Submission loading, regex normalization, and comment stripping.
2. $C_{\text{vec-text}}$: Sublinear TF-IDF vectorization (amortized via document cache).
3. $C_{\text{tok-code}}$: Code regex tokenization and operator counting (amortized via cache).
4. $C_{\text{ast}}$: Heavy Java AST parsing (strictly Tier 2; included in $C_{\text{feat-heavy}}$ for code).
5. $C_{\text{cache}}$: In-memory hash lookup for precomputed representations.
6. $C_{\text{Tier1}}$: Tier-1 4-D dot product and logistic evaluation ($< 0.045\text{ ms}$).
7. $C_{\text{infer-heavy}}$: Forward inference of Sentence-BERT or 7-D structural classifier.
8. $C_{\text{explain}}$: Smith-Waterman alignment or AST isomorphism extraction.
9. $M_{\text{RSS}}$: Process resident set size memory overhead (tracked via `psutil`).
10. $C_{\text{e2e}}$: Total end-to-end wall-clock latency per pair.

#### Operational Probabilities:
- **Escalation Probability:**
  $$P(\text{Escalate} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}})$$
- **Evidence Review Probability (Charged strictly when review is required):**
  $$P(\text{Review} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \left[ \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \wedge \text{ReviewRequired}_i) \right]$$

### 4.2 Constrained Mathematical Optimization Problem

Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with $\theta_{\text{low}} < \theta_{\text{high}}$.
$$\min_{\boldsymbol{\theta}} \quad \bar{C}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \bar{C}_{\text{total}}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

Where:
$$\text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{low}})}{\sum_{i \in D_{\text{val}}} y_i}$$
$$\text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}})}{\sum_{i \in D_{\text{val}}} \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + 10^{-7}}$$

#### Infeasibility Condition:
If the feasible region $\Omega = \left\{ \boldsymbol{\theta} \mid g_1(\boldsymbol{\theta}) \le 0 \wedge g_2(\boldsymbol{\theta}) \le 0 \wedge g_3(\boldsymbol{\theta}) \le 0 \right\}$ is empty ($\Omega = \emptyset$), the optimizer returns:
$$\boldsymbol{\theta}^* = \text{None}, \quad \text{Status: INFEASIBLE\_BUDGET}$$

#### Pareto Dominance:
A policy $\boldsymbol{\theta}_A$ Pareto-dominates $\boldsymbol{\theta}_B$ if:
$$\text{CandidateRecall}(\boldsymbol{\theta}_A) \ge \text{CandidateRecall}(\boldsymbol{\theta}_B) \quad \text{and} \quad \bar{C}_{\text{total}}(\boldsymbol{\theta}_A) \le \bar{C}_{\text{total}}(\boldsymbol{\theta}_B)$$
with at least one strict inequality.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Text Evidence: Smith-Waterman Dynamic Alignment (Affine Gap)

For token sequences $\mathbf{a} = (a_1, \dots, a_m)$ and $\mathbf{b} = (b_1, \dots, b_n)$:
$$H_{i, j} = \max \begin{cases}
0 \\
H_{i-1, j-1} + s(a_i, b_j) & (\text{Diagonal Match / Mismatch}) \\
E_{i, j} & (\text{Gap in sequence } \mathbf{a}) \\
F_{i, j} & (\text{Gap in sequence } \mathbf{b})
\end{cases}$$
Where:
$$E_{i, j} = \max\left(H_{i-1, j} - u - v, \, E_{i-1, j} - v\right)$$
$$F_{i, j} = \max\left(H_{i, j-1} - u - v, \, F_{i, j-1} - v\right)$$
- Match reward $s(a, b) = +2.0$ if $a = b$; mismatch penalty $s(a, b) = -1.0$.
- Gap opening penalty $u = 2.0$; gap extension penalty $v = 0.5$.

### 5.2 Code Evidence: AST Subtree Isomorphism & Identifier Ledger

Let $T_A, T_B$ be Abstract Syntax Trees parsed at Tier 2. Two subtrees $t_A \subset T_A, t_B \subset T_B$ are isomorphic under identifier renaming if there exists a bijective variable mapping $\sigma: \text{Vars}(t_A) \to \text{Vars}(t_B)$ such that:
$$\text{GrammarType}(t_A) \equiv \text{GrammarType}(t_B) \quad \text{and} \quad \sigma(\text{LeafTokens}(t_A)) = \text{LeafTokens}(t_B)$$
The identifier substitution ledger $\mathcal{M}_{\text{id}}$ is:
$$\mathcal{M}_{\text{id}} = \left\{ (v_A, v_B) \in \text{Vars}(C_A) \times \text{Vars}(C_B) \mid \text{Count}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \right\}$$

### 5.3 Explanation Quality Metrics

1. **Evidence Faithfulness (Decision Probability Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   Evaluated strictly on true positive pairs escalated to Tier 2. (Not applicable to Tier-1 direct flags where heavy probability is bypassed).
2. **Evidence Coverage:**
   $$\text{Coverage}(D_A, D_B) = \frac{|\mathcal{E}_{\text{tokens}}(D_B)|}{|D_B|_{\text{tokens}}}$$
3. **Transformation Alignment Precision (Code):**
   $$\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$$

---

## 6. Statistical Analysis Formulations

### 6.1 Group-Aware Cluster Bootstrap Confidence Intervals

Because submission pairs sharing a reference source (premise in text; assignment task in code) are correlated, standard pair-level bootstrap produces anti-conservative (artificially narrow) confidence intervals.
Let $\mathcal{G} = \{g_1, \dots, g_K\}$ denote unique grouping identifiers in the evaluation set.
1. For bootstrap iteration $b = 1, \dots, B$ (with $B = 1,000$):
   - Sample $K$ group identifiers $\mathcal{G}^{*(b)}$ from $\mathcal{G}$ with replacement.
   - Assemble resampled dataset $D^{*(b)} = \bigcup_{g \in \mathcal{G}^{*(b)}} \{(\mathbf{x}_i, y_i) \mid g_i = g\}$.
   - Compute metric of interest $\theta^{*(b)} = M(D^{*(b)})$.
2. Calculate empirical $95\%$ percentile confidence interval:
   $$\text{CI}_{95\%} = \left[ \theta^{*}_{(\lfloor 0.025 \cdot B \rfloor)}, \, \theta^{*}_{(\lceil 0.975 \cdot B \rceil)} \right]$$

### 6.2 Exact Binomial McNemar Test for Paired Classifiers

Let $b$ denote pairs where Proposed Model is correct and Baseline is incorrect; let $c$ denote pairs where Proposed Model is incorrect and Baseline is correct.
- If discordant cell count $b + c \ge 25$, use continuity-corrected $\chi^2$ statistic:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1$$
- If discordant cell count $b + c < 25$ (common in small code subsets `case-06` and `case-07`), the asymptotic approximation is invalid. We compute the **Exact Binomial Two-Sided $p$-value**:
  $$p = 2 \cdot \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} \left(\frac{1}{2}\right)^{b + c}$$

### 6.3 Benjamini-Hochberg False Discovery Rate (FDR) Procedure

When testing hypotheses across $K$ baseline comparisons and ablation configurations:
1. Sort raw $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(K)}$.
2. Find largest index $k^*$ such that:
   $$k^* = \max \left\{ k \in \{1, \dots, K\} \mid p_{(k)} \le \frac{k}{K} \cdot q^* \right\}$$
   Where $q^* = 0.05$ is the target FDR bound.
3. Reject null hypotheses $H_{(1)}, \dots, H_{(k^*)}$; retain $H_{(k^*+1)}, \dots, H_{(K)}$.

---
*End of Document: mathematical_formulations_v3_1.md*

