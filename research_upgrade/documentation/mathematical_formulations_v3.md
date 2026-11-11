# Mathematical Formulations and Theoretical Foundations (Version 3.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.0 (Phase 2.2 Methodological Audit Revision)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Mathematical Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Formal Mathematical Variables & Problem Setting

Let $\mathcal{D} = \{(D_A^{(i)}, D_B^{(i)}, y_i)\}_{i=1}^N$ denote a dataset of submission pairs, where $D_A^{(i)}$ represents a reference submission, $D_B^{(i)}$ represents a suspicious candidate submission, and $y_i \in \{0, 1\}$ represents ground-truth academic integrity ($y_i = 1$ denotes plagiarism/text reuse; $y_i = 0$ denotes independent work).

The system operates across two modalities: $\mathcal{M} \in \{\text{text}, \text{code}\}$.

### Disambiguated Decision Quantities:
1. **$s_{\text{cand}}(\mathbf{x}) \in [0, 1]$:** Multi-view candidate score produced by the Tier-1 fast generator on all $100\%$ candidate pairs. Governed by routing thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$.
2. **$z(\mathbf{x}) = f_{\text{heavy}}(\mathbf{x}) \in \mathbb{R}$:** Raw decision function margin from the heavy classifier on **escalated pairs only**. For non-escalated pairs, $z(\mathbf{x}) = \text{None}$.
3. **$\hat{p}(\mathbf{x}) \in [0, 1]$:** Calibrated posterior probability $P(Y = 1 \mid z)$ produced by the heavy detector + calibrator on **escalated pairs only**. For non-escalated pairs, $\hat{p}(\mathbf{x}) = \text{None}$.
4. **$\mathcal{C}_{1-\epsilon}(\mathbf{x}) \subseteq \{0, 1\}$:** Conformal prediction set constructed from $\hat{p}(\mathbf{x})$ and quantile $\hat{q}_{1-\epsilon}$ on **escalated pairs only**. For non-escalated pairs, $\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \text{None}$.

---

## 2. Contribution 1: Multi-View Candidate Generation

### 2.1 Multi-View Representation Spaces (Purely Lightweight Tier 1)

Tier-1 features contain **zero heavy AST parsing operations** to ensure sub-millisecond execution.

#### A. Natural Language Text Multi-View Vector $\mathbf{v}_{\text{text}} \in [0, 1]^4$:
1. **Word-Level Sublinear TF-IDF Cosine Similarity ($v_1$):**
   $$\text{tf-idf}(t, d) = \left( 1 + \log(\text{tf}(t, d)) \right) \cdot \log\left( \frac{1 + |D_{\text{train}}|}{1 + \text{df}(t)} \right) + 1 \quad \text{for } \text{tf}(t, d) > 0$$
   Vocabulary and $\text{df}(t)$ are fitted strictly on $D_{\text{train}}$.
   $$v_1 = S_{\text{word-tfidf}}(D_A, D_B) = \begin{cases} \frac{\langle \mathbf{t}_A, \mathbf{t}_B \rangle}{\|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2}, & \text{if } \|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2 > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
2. **Character 3-Gram Set Jaccard Overlap ($v_2$):**
   Let $\mathcal{G}_3(D)$ denote the set of unique character tri-grams in lowercased text $D$:
   $$v_2 = S_{\text{char-3gram}}(D_A, D_B) = \begin{cases} \frac{|\mathcal{G}_3(D_A) \cap \mathcal{G}_3(D_B)|}{|\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)|}, & \text{if } |\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)| > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
3. **Word Token Set Jaccard Overlap ($v_3$):**
   Let $\mathcal{W}(D)$ denote the set of unique whitespace- and punctuation-delimited word tokens:
   $$v_3 = S_{\text{word-jaccard}}(D_A, D_B) = \begin{cases} \frac{|\mathcal{W}(D_A) \cap \mathcal{W}(D_B)|}{|\mathcal{W}(D_A) \cup \mathcal{W}(D_B)|}, & \text{if } |\mathcal{W}(D_A) \cup \mathcal{W}(D_B)| > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
4. **Length-Ratio Symmetry Metric ($v_4$):**
   $$v_4 = \Delta_{\text{len}}(D_A, D_B) = \frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}}, 1)}$$

#### B. Source Code Multi-View Vector $\mathbf{u}_{\text{code}} \in [0, 1]^4$ (Purely Regex/Lexer Tier 1):
1. **Raw Token Set Jaccard ($u_1$):**
   Let $\mathcal{T}_{\text{raw}}(C)$ denote unique lexical tokens after stripping Java comments:
   $$u_1 = S_{\text{raw-token}}(C_A, C_B) = \frac{|\mathcal{T}_{\text{raw}}(C_A) \cap \mathcal{T}_{\text{raw}}(C_B)|}{|\mathcal{T}_{\text{raw}}(C_A) \cup \mathcal{T}_{\text{raw}}(C_B)|}$$
2. **Identifier-Normalized Token Set Jaccard ($u_2$):**
   Let $\phi_{\text{regex}}$ replace all user-defined identifiers with canonical symbol `ID` while preserving language keywords (`class`, `for`, `while`, `int`), operators, and string literals:
   $$u_2 = S_{\text{norm-token}}(C_A, C_B) = \frac{|\phi_{\text{regex}}(\mathcal{T}(C_A)) \cap \phi_{\text{regex}}(\mathcal{T}(C_B))|}{|\phi_{\text{regex}}(\mathcal{T}(C_A)) \cup \phi_{\text{regex}}(\mathcal{T}(C_B))|}$$
3. **Operator Frequency Profile Cosine ($u_3$):**
   Normalized frequency vector $\mathbf{o}(C) \in \mathbb{R}^{14}$ of Java operators (`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`):
   $$u_3 = S_{\text{operator}}(C_A, C_B) = \begin{cases} \frac{\langle \mathbf{o}(C_A), \mathbf{o}(C_B) \rangle}{\|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2}, & \text{if } \|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2 > 0 \\ 0.0, & \text{otherwise} \end{cases}$$
4. **Token Length Ratio Asymmetry ($u_4$):**
   $$u_4 = \Delta_{\text{tok-len}}(C_A, C_B) = \frac{\min(|\mathcal{T}(C_A)|, |\mathcal{T}(C_B)|)}{\max(|\mathcal{T}(C_A)|, |\mathcal{T}(C_B)|, 1)}$$

### 2.2 Fused Candidate Scoring Rule
Fused candidate score $s_{\text{cand}} \in [0, 1]$ via a constrained logistic model fitted on $D_{\text{train}}$:
$$s_{\text{cand}}(\mathbf{x}) = \sigma\left( \mathbf{w}^T \mathbf{x} + b \right) = \frac{1}{1 + \exp\left( -(\mathbf{w}^T \mathbf{x} + b) \right)}, \quad \mathbf{w} \ge 0$$

#### Tier-1 Triage Decision Rule:
$$\text{Triage}_{\text{Tier1}}(\mathbf{x}) = \begin{cases}
\text{STATE 1: LOW\_RISK\_PRUNED}, & \text{if } s_{\text{cand}}(\mathbf{x}) < \theta_{\text{low}} \\
\text{STATE 2: UNCERTAIN\_ESCALATED}, & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}) < \theta_{\text{high}} \\
\text{STATE 3: HIGH\_RISK\_FLAGGED}, & \text{if } s_{\text{cand}}(\mathbf{x}) \ge \theta_{\text{high}}
\end{cases}$$

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Selection-Aware Calibration

Because the heavy detector executes strictly on pairs passing Tier 1, the probability calibrator $g_{\text{cal}}$ is fitted on the **conditional escalation split**:
$$D_{\text{cal-prob}}^{\text{esc}} = \left\{ (z_i, y_i) \mid \mathbf{x}_i \in D_{\text{cal-prob}} \wedge \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$

#### A. Parametric Calibration: Platt Scaling
$$P_{\text{Platt}}(Y = 1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
$$\min_{A, B} -\sum_{i \in D_{\text{cal-prob}}^{\text{esc}}} \left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log (1 - \sigma(A z_i + B)) \right]$$

#### B. Non-Parametric Calibration: Isotonic Regression
Fitted via the Pool Adjacent Violators (PAV) algorithm:
$$\min_{m} \sum_{i \in D_{\text{cal-prob}}^{\text{esc}}} (y_i - m(z_i))^2 \quad \text{s.t.} \quad m(z_i) \le m(z_j) \text{ whenever } z_i \le z_j$$

### 3.2 Quantitative Calibration & Uncertainty Metrics

1. **Expected Calibration Error (ECE):**
   Evaluated over $M = 10$ equally spaced bins $B_m = \left( \frac{m-1}{M}, \frac{m}{M} \right]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   Where $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$ and $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$.
2. **Maximum Calibration Error (MCE):**
   $$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
3. **Brier Score (Strictly Proper Scoring Rule):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
4. **Predictive Entropy ($\mathcal{H}$):**
   $$\mathcal{H}(\hat{p}) = -\hat{p} \log_2(\hat{p}) - (1 - \hat{p}) \log_2(1 - \hat{p})$$

### 3.3 Inductive Conformal Prediction & Four-Set Semantics

On the frozen conditional escalation split $D_{\text{cal-conf}}^{\text{esc}} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N_{\text{cal-conf}}^{\text{esc}}}$, compute non-conformity scores:
$$\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$$
For error significance $\epsilon \in (0, 1)$, compute empirical quantile:
$$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (N_{\text{cal-conf}}^{\text{esc}} + 1)(1 - \epsilon) \rceil}{N_{\text{cal-conf}}^{\text{esc}}}, \, \{\alpha_1, \dots, \alpha_{N_{\text{cal-conf}}^{\text{esc}}}\} \right)$$
For escalated query instance $\mathbf{x}$, the prediction set $\mathcal{C}_{1-\epsilon}(\mathbf{x})$ is:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}) \le \hat{q}_{1-\epsilon} \right\}$$

#### Theoretical Assumptions and Validity Guarantee:
- **Exchangeability Assumption:** Assumes calibration pairs $(X_i, Y_i)_{i=1}^{N_{\text{cal-conf}}^{\text{esc}}}$ and escalated test pairs $(X_{N+1}, Y_{N+1})$ are exchangeable conditioned on escalation ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$).
- **Conditional Validity Theorem:** Under conditional exchangeability:
  $$P\left( Y_{N+1} \in \mathcal{C}_{1-\epsilon}(X_{N+1}) \mid \text{Escalated} \right) \ge 1 - \epsilon$$
- **Operational Meaning of the Four Conformal Sets:**
  $$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \begin{cases}
  \{0\}, & \text{Model-supported negative prediction} \implies \text{Cleared Non-Plagiarized} \\
  \{1\}, & \text{Model-supported positive prediction} \implies \text{Flagged for Human Review} \\
  \{0, 1\}, & \text{Ambiguous prediction} \implies \text{Escalated to Human Review} \\
  \emptyset, & \text{Abstention / Unresolved prediction} \implies \text{Abstain, Mandatory Human Review}
  \end{cases}$$

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Comprehensive Computational Cost Function

The total expected per-pair cost accounts for every stage in the master pipeline:
$$\bar{C}_{\text{total}}(\boldsymbol{\theta}) = \bar{C}_{\text{prep}} + \bar{C}_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ \bar{C}_{\text{feat-heavy}} + \bar{C}_{\text{infer-heavy}} + \bar{C}_{\text{router}} \right] + P(\text{Review} \mid \boldsymbol{\theta}) \cdot \bar{C}_{\text{explain}}$$

Where:
- $P(\text{Escalate} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}})$
- $P(\text{Review} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \left[ \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \wedge \mathcal{C}_i \in \{\{1\}, \{0, 1\}, \emptyset\}) \right]$

### 4.2 Constrained Mathematical Optimization Objective

Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with $\theta_{\text{low}} < \theta_{\text{high}}$. The objective function evaluated on $D_{\text{val}}$ is:
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
If $\left\{ \boldsymbol{\theta} \mid g_1(\boldsymbol{\theta}) \le 0 \wedge g_2(\boldsymbol{\theta}) \le 0 \wedge g_3(\boldsymbol{\theta}) \le 0 \right\} = \emptyset$, the optimizer returns:
$$\boldsymbol{\theta}^* = \text{None}, \quad \text{Status} = \text{INFEASIBLE\_BUDGET}$$

#### Pareto Dominance:
A policy $\boldsymbol{\theta}_A$ Pareto-dominates $\boldsymbol{\theta}_B$ if:
$$\text{CandidateRecall}(\boldsymbol{\theta}_A) \ge \text{CandidateRecall}(\boldsymbol{\theta}_B) \quad \text{and} \quad \bar{C}_{\text{total}}(\boldsymbol{\theta}_A) \le \bar{C}_{\text{total}}(\boldsymbol{\theta}_B)$$
with at least one strict inequality.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Text Evidence: Smith-Waterman Local Sequence Alignment

Dynamic programming recurrence with affine gap penalty:
$$H_{i, j} = \max \begin{cases}
0 \\
H_{i-1, j-1} + s(a_i, b_j) & (\text{Match / Mismatch}) \\
E_{i, j} & (\text{Gap in } \mathbf{a}) \\
F_{i, j} & (\text{Gap in } \mathbf{b})
\end{cases}$$
Where:
$$E_{i, j} = \max(H_{i-1, j} - u - v, \, E_{i-1, j} - v)$$
$$F_{i, j} = \max(H_{i, j-1} - u - v, \, F_{i, j-1} - v)$$
With match score $s(a, b) = +2.0$ for identical tokens, $-1.0$ for mismatches, gap opening $u = 2.0$, and gap extension $v = 0.5$.

### 5.2 Code Evidence: AST Subtree Isomorphism & Identifier Ledger

Two subtrees $t_A \subset T_A, t_B \subset T_B$ are isomorphic under identifier renaming if there exists a bijection $\sigma: \text{Vars}(t_A) \to \text{Vars}(t_B)$ such that:
$$\text{GrammarType}(t_A) \equiv \text{GrammarType}(t_B) \quad \text{and} \quad \sigma(\text{LeafTokens}(t_A)) = \text{LeafTokens}(t_B)$$
Identifier substitution dictionary:
$$\mathcal{M}_{\text{id}} = \left\{ (v_A, v_B) \in \text{Vars}(C_A) \times \text{Vars}(C_B) \mid \text{Support}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \right\}$$

### 5.3 Explanation Quality Metrics

1. **Evidence Faithfulness (Feature Ablation Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   $$\text{Mean Faithfulness} = \frac{1}{|D_{\text{test}}^{+, \text{heavy}}|} \sum_{i \in D_{\text{test}}^{+, \text{heavy}}} \Delta \hat{P}_{\text{faith}}(\mathbf{x}_i)$$
2. **Evidence Coverage:**
   $$\text{Coverage}(D_A, D_B) = \frac{|\mathcal{E}_{\text{tokens}}(D_B)|}{|D_B|_{\text{tokens}}}$$
3. **Transformation Alignment Precision (Code):**
   $$\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$$

