# Mathematical Formulations and Theoretical Foundations (Version 2.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 2.0 (Phase 2.1 Audit Correction Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Mathematical Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Mathematical Notation & Formal Variables

Let $\mathcal{D} = \{(D_A^{(i)}, D_B^{(i)}, y_i)\}_{i=1}^N$ denote a dataset of submission pairs, where $D_A^{(i)}$ represents a reference submission, $D_B^{(i)}$ represents a suspicious candidate submission, and $y_i \in \{0, 1\}$ represents the ground-truth integrity label ($y_i = 1$ denotes plagiarism/text reuse; $y_i = 0$ denotes independent work).

The system operates across two modalities: $\mathcal{M} \in \{\text{text}, \text{code}\}$.

### Strict Disambiguation of Decision Quantities:
1. **$s_{\text{cand}}(\mathbf{x}) \in [0, 1]$:** Multi-view candidate screening score produced by the Tier-1 fast generator on all $100\%$ candidate pairs. Governed by routing thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$.
2. **$\hat{p}(\mathbf{x}) \in [0, 1]$:** Calibrated posterior plagiarism probability $P(Y = 1 \mid z)$ produced by the heavy detector (Sentence-BERT or AST-SVM/RF) coupled with a probability calibrator on **escalated pairs only**.
3. **$\mathcal{C}_{1-\epsilon}(\mathbf{x}) \subseteq \{0, 1\}$:** Conformal prediction set constructed from $\hat{p}(\mathbf{x})$ and empirical quantile $\hat{q}_{1-\epsilon}$.

---

## 2. Contribution 1: Multi-View Candidate Generation

### 2.1 Multi-View Representation Spaces (Purely Lightweight Tier 1)

To guarantee sub-millisecond execution ($< 0.050\text{ ms/pair}$), Tier-1 feature tuples contain **zero heavy AST parsing operations**.

#### A. Natural Language Text Multi-View Vector $\mathbf{v}_{\text{text}} \in [0, 1]^4$:
1. **Word-Level Sublinear TF-IDF Cosine Similarity ($v_1$):**
   $$\text{tf-idf}(t, d) = \left( 1 + \log(\text{tf}(t, d)) \right) \cdot \log\left( \frac{1 + N_{\text{docs}}}{1 + \text{df}(t)} \right) + 1 \quad \text{for } \text{tf}(t, d) > 0$$
   $$v_1 = S_{\text{word-tfidf}}(D_A, D_B) = \frac{\langle \mathbf{t}_A, \mathbf{t}_B \rangle}{\|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2}$$
2. **Character 3-Gram Jaccard Overlap ($v_2$):**
   Let $\mathcal{G}_3(D)$ denote the multiset of character tri-grams extracted from document $D$:
   $$v_2 = S_{\text{char-3gram}}(D_A, D_B) = \frac{|\mathcal{G}_3(D_A) \cap \mathcal{G}_3(D_B)|}{|\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)|}$$
3. **Word Token Jaccard Overlap ($v_3$):**
   Let $\mathcal{W}(D)$ denote the set of whitespace- and punctuation-delimited word tokens:
   $$v_3 = S_{\text{word-jaccard}}(D_A, D_B) = \frac{|\mathcal{W}(D_A) \cap \mathcal{W}(D_B)|}{|\mathcal{W}(D_A) \cup \mathcal{W}(D_B)|}$$
4. **Length-Ratio Symmetry Metric ($v_4$):**
   $$v_4 = \Delta_{\text{len}}(D_A, D_B) = \frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}})}$$

#### B. Source Code Multi-View Vector $\mathbf{u}_{\text{code}} \in [0, 1]^4$ (Regex & Lexer Operations Only):
1. **Raw Token Multiset Jaccard ($u_1$):**
   Let $\mathcal{T}_{\text{raw}}(C)$ denote the sequence of lexical tokens after comment stripping:
   $$u_1 = S_{\text{raw-token}}(C_A, C_B) = \frac{|\mathcal{T}_{\text{raw}}(C_A) \cap \mathcal{T}_{\text{raw}}(C_B)|}{|\mathcal{T}_{\text{raw}}(C_A) \cup \mathcal{T}_{\text{raw}}(C_B)|}$$
2. **Identifier-Normalized Token Multiset Jaccard ($u_2$):**
   Let $\phi_{\text{regex}}: \mathcal{T} \to \mathcal{T}_{\text{norm}}$ be a regex-based mapping replacing all non-keyword identifiers with canonical symbol `ID` while preserving language keywords (`class`, `for`, `while`, `int`):
   $$u_2 = S_{\text{norm-token}}(C_A, C_B) = \frac{|\phi_{\text{regex}}(\mathcal{T}(C_A)) \cap \phi_{\text{regex}}(\mathcal{T}(C_B))|}{|\phi_{\text{regex}}(\mathcal{T}(C_A)) \cup \phi_{\text{regex}}(\mathcal{T}(C_B))|}$$
3. **Operator Frequency Profile Cosine ($u_3$):**
   Let $\mathbf{o}(C) \in \mathbb{R}^{K}$ denote the normalized frequency vector of Java operators (`+`, `-`, `*`, `==`, `!=`, `&&`, `||`):
   $$u_3 = S_{\text{operator}}(C_A, C_B) = \frac{\langle \mathbf{o}(C_A), \mathbf{o}(C_B) \rangle}{\|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2}$$
4. **Token Length Ratio Asymmetry ($u_4$):**
   $$u_4 = \Delta_{\text{tok-len}}(C_A, C_B) = \frac{\min(|\mathcal{T}(C_A)|, |\mathcal{T}(C_B)|)}{\max(|\mathcal{T}(C_A)|, |\mathcal{T}(C_B)|)}$$

*Methodological Note:* Full AST construction, AST node bigrams, and cyclomatic complexity differentials are strictly reserved for Tier-2 feature extraction on escalated pairs.

### 2.2 Fused Candidate Scoring & Selection Rule
We fuse the multi-view features into scalar candidate score $s_{\text{cand}} \in [0, 1]$ via a constrained logistic model fitted on $D_{\text{train}}$:
$$s_{\text{cand}}(\mathbf{x}) = \sigma\left( \mathbf{w}^T \mathbf{x} + b \right) = \frac{1}{1 + \exp\left( -(\mathbf{w}^T \mathbf{x} + b) \right)}, \quad \mathbf{w} \ge 0$$
Candidate selection rule:
$$\text{Triage}_{\text{Tier1}}(\mathbf{x}) = \begin{cases}
\text{STATE 1: LOW\_RISK\_PRUNED}, & \text{if } s_{\text{cand}}(\mathbf{x}) < \theta_{\text{low}} \\
\text{STATE 2: UNCERTAIN\_ESCALATED}, & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}) < \theta_{\text{high}} \\
\text{STATE 3: HIGH\_RISK\_FLAGGED}, & \text{if } s_{\text{cand}}(\mathbf{x}) \ge \theta_{\text{high}}
\end{cases}$$

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Probability Calibration on $D_{\text{cal-prob}}$

Let $z = f_{\text{heavy}}(\mathbf{x}) \in \mathbb{R}$ denote the uncalibrated margin/score produced by the heavy detector on escalated pairs. Calibration parameters are fitted on a dedicated split $D_{\text{cal-prob}} = \{(z_i, y_i)\}_{i=1}^{N_{\text{cal-prob}}}$ completely disjoint from training and conformal calibration.

#### A. Parametric Calibration: Platt Scaling
$$P_{\text{Platt}}(Y = 1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
$$\min_{A, B} -\sum_{i \in D_{\text{cal-prob}}} \left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log (1 - \sigma(A z_i + B)) \right]$$

#### B. Non-Parametric Calibration: Isotonic Regression
$$\min_{m} \sum_{i \in D_{\text{cal-prob}}} \left( y_i - m(z_i) \right)^2 \quad \text{subject to} \quad m(z_i) \le m(z_j) \text{ whenever } z_i \le z_j$$
Fitted via the Pool Adjacent Violators (PAV) algorithm.

### 3.2 Uncertainty Quantification Metrics
1. **Expected Calibration Error (ECE):**
   Group predictions into $M = 10$ equally spaced probability bins $B_1, \dots, B_M \subset [0, 1]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   Where $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$ and $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$.
2. **Maximum Calibration Error (MCE):**
   $$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
3. **Brier Score (Strictly Proper Scoring Rule):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
4. **Predictive Entropy ($\mathcal{H}$):**
   $$\mathcal{H}(\hat{p}) = -\hat{p} \log_2(\hat{p}) - (1 - \hat{p}) \log_2(1 - \hat{p})$$

### 3.3 Inductive Conformal Prediction on $D_{\text{cal-conf}}$

Let $s(\mathbf{x}, y) = 1 - \hat{P}(Y = y \mid \mathbf{x})$ denote the probability non-conformity score.
1. On an independent conformal calibration split $D_{\text{cal-conf}}$ ($N_{\text{cal-conf}}$ pairs), compute:
   $$\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i), \quad i \in \{1, \dots, N_{\text{cal-conf}}\}$$
2. For significance $\epsilon \in (0, 1)$, compute the empirical quantile:
   $$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (N_{\text{cal-conf}} + 1)(1 - \epsilon) \rceil}{N_{\text{cal-conf}}}, \, \{\alpha_1, \dots, \alpha_{N_{\text{cal-conf}}}\} \right)$$
3. For query instance $\mathbf{x}$, construct the prediction set:
   $$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}) \le \hat{q}_{1-\epsilon} \right\}$$

#### Theoretical Guarantee & Assumptions:
- **Exchangeability Assumption:** Assumes calibration pairs $(X_i, Y_i)_{i=1}^{N_{\text{cal-conf}}}$ and test query $(X_{N+1}, Y_{N+1})$ are exchangeable. Under exchangeability:
  $$P\left( Y_{N+1} \in \mathcal{C}_{1-\epsilon}(X_{N+1}) \right) \ge 1 - \epsilon$$
- **Coverage $\ne$ Plagiarism Recall:** Conformal coverage guarantees the true label is in $\mathcal{C}$. When uncertainty is high, $\mathcal{C} = \{0, 1\}$. Candidate recall relies on routing ambiguous sets to human review.

#### Complete Four-Set Conformal Triage Policy:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \begin{cases}
\{0\}, & \text{High-Confidence Negative} \implies \text{Cleared Non-Plagiarized} \\
\{1\}, & \text{High-Confidence Positive} \implies \text{Flagged for Human Review} \\
\{0, 1\}, & \text{High Uncertainty / Ambiguous} \implies \text{Escalated to Mandatory Review} \\
\emptyset, & \text{Out-of-Distribution Anomaly} \implies \text{OOD\_ANOMALY\_ABSTAIN (Mandatory Review)}
\end{cases}$$

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Comprehensive Computational Cost Function

The total expected per-pair cost over candidate distribution $\mathcal{D}$ is:
$$\bar{C}_{\text{total}}(\boldsymbol{\theta}) = \bar{C}_{\text{prep}} + \bar{C}_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ \bar{C}_{\text{feat-heavy}} + \bar{C}_{\text{infer-heavy}} \right] + \bar{C}_{\text{router}} + P(\text{Review} \mid \boldsymbol{\theta}) \cdot \bar{C}_{\text{explain}}$$

Where:
- $P(\text{Escalate} \mid \boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim D_{\text{val}}}\left[ \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}) < \theta_{\text{high}}) \right]$
- $P(\text{Review} \mid \boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim D_{\text{val}}}\left[ \mathbb{I}(s_{\text{cand}}(\mathbf{x}) \ge \theta_{\text{high}}) + \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}) < \theta_{\text{high}} \wedge \hat{p}(\mathbf{x}) \ge 0.50) \right]$

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

#### Infeasibility Handling:
If $\left\{ \boldsymbol{\theta} \mid g_1(\boldsymbol{\theta}) \le 0 \wedge g_2(\boldsymbol{\theta}) \le 0 \wedge g_3(\boldsymbol{\theta}) \le 0 \right\} = \emptyset$, the optimizer returns:
$$\boldsymbol{\theta}^* = \text{None}, \quad \text{Status} = \text{INFEASIBLE\_BUDGET}$$
Reporting the closest feasible recall configuration with an explicit diagnostic violation report.

#### Pareto Frontier:
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
With match bonus $s(a, b) = +2.0$ for identical tokens, $-1.0$ for mismatches, gap opening $u = 2.0$, and gap extension $v = 0.5$.

### 5.2 Code Evidence: AST Subtree Isomorphism & Identifier Ledger

Two subtrees $t_A \subset T_A, t_B \subset T_B$ are isomorphic under identifier renaming if there exists a bijection $\sigma: \text{Vars}(t_A) \to \text{Vars}(t_B)$ such that:
$$\text{GrammarType}(t_A) \equiv \text{GrammarType}(t_B) \quad \text{and} \quad \sigma(\text{LeafTokens}(t_A)) = \text{LeafTokens}(t_B)$$
Identifier substitution dictionary:
$$\mathcal{M}_{\text{id}} = \left\{ (v_A, v_B) \in \text{Vars}(C_A) \times \text{Vars}(C_B) \mid \text{Support}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \right\}$$

### 5.3 Explanation Quality Metrics

1. **Evidence Faithfulness (Feature Ablation Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   $$\text{Mean Faithfulness} = \frac{1}{|D_{\text{test}}^{+}|} \sum_{i \in D_{\text{test}}^{+}} \Delta \hat{P}_{\text{faith}}(\mathbf{x}_i)$$
2. **Evidence Coverage:**
   $$\text{Coverage}(D_A, D_B) = \frac{|\mathcal{E}_{\text{tokens}}(D_B)|}{|D_B|_{\text{tokens}}}$$
3. **Transformation Alignment Precision (Code):**
   $$\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$$

