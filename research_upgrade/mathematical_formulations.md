# Mathematical Formulations and Theoretical Foundations
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Complete Mathematical Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Mathematical Notation & Problem Setup

Let $\mathcal{D} = \{(D_A^{(i)}, D_B^{(i)}, y_i)\}_{i=1}^N$ denote a dataset of submission pairs, where $D_A^{(i)}$ represents a reference submission, $D_B^{(i)}$ represents a suspicious candidate submission, and $y_i \in \{0, 1\}$ represents the ground-truth integrity label ($y_i = 1$ denotes plagiarism/text reuse; $y_i = 0$ denotes independent work).

The universe of submissions spans two distinct modalities:
$$\mathcal{M} \in \{\text{text}, \text{code}\}$$
Our objective is to learn a decision policy $\pi(D_A, D_B) \in \{\text{PRUNED}, \text{ESCALATED}, \text{FLAGGED}\}$ and a final classification $\hat{y} \in \{0, 1\}$ accompanied by a calibrated posterior probability $\hat{p} \in [0, 1]$, an epistemic uncertainty estimate $\mathcal{U} \in [0, 1]$, and an interpretable evidence dossier $\mathcal{E}$, while minimizing total computational cost subject to explicit recall and latency budgets.

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Multi-View Representation Spaces

#### A. Natural Language Text Modality
For a pair of text documents $(D_A, D_B)$, we construct a 5-dimensional feature tuple:
$$\mathbf{v}_{\text{text}}(D_A, D_B) = \left[ v_1, v_2, v_3, v_4, v_5 \right]^T \in [0, 1]^5$$

1. **Sublinear Word TF-IDF Cosine Similarity ($v_1$):**
   Let $\mathbf{t}_A, \mathbf{t}_B \in \mathbb{R}^V$ denote TF-IDF feature vectors where term weights are scaled sublinearly:
   $$\text{tf-idf}(t, d) = \left( 1 + \log(\text{tf}(t, d)) \right) \cdot \log\left( \frac{1 + N_{\text{docs}}}{1 + \text{df}(t)} \right) + 1 \quad \text{for } \text{tf}(t, d) > 0$$
   $$v_1 = S_{\text{word-tfidf}}(D_A, D_B) = \frac{\langle \mathbf{t}_A, \mathbf{t}_B \rangle}{\|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2}$$

2. **Character 3-Gram Jaccard Similarity ($v_2$):**
   Let $\mathcal{G}_3(D)$ denote the multiset of character tri-grams extracted from document $D$:
   $$v_2 = S_{\text{char-3gram}}(D_A, D_B) = \frac{|\mathcal{G}_3(D_A) \cap \mathcal{G}_3(D_B)|}{|\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)|}$$

3. **Word Token Jaccard Overlap ($v_3$):**
   Let $\mathcal{W}(D)$ denote the set of whitespace- and punctuation-delimited word tokens:
   $$v_3 = S_{\text{word-jaccard}}(D_A, D_B) = \frac{|\mathcal{W}(D_A) \cap \mathcal{W}(D_B)|}{|\mathcal{W}(D_A) \cup \mathcal{W}(D_B)|}$$

4. **Length-Ratio Symmetry Metric ($v_4$):**
   $$v_4 = \Delta_{\text{len}}(D_A, D_B) = \frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}})}$$

5. **Fast Embedding Cosine Proxy ($v_5$):**
   Let $\mathbf{e}_A, \mathbf{e}_B \in \mathbb{R}^d$ denote average token embedding vectors obtained via an offline fast embedding dictionary (e.g., fastText or static MiniLM projection):
   $$v_5 = S_{\text{embed-proxy}}(D_A, D_B) = \frac{\langle \mathbf{e}_A, \mathbf{e}_B \rangle}{\|\mathbf{e}_A\|_2 \|\mathbf{e}_B\|_2}$$

#### B. Source Code Modality
For a pair of source code programs $(C_A, C_B)$, we construct a 5-dimensional feature tuple:
$$\mathbf{u}_{\text{code}}(C_A, C_B) = \left[ u_1, u_2, u_3, u_4, u_5 \right]^T \in [0, 1]^5$$

1. **Raw Lexical Token Jaccard ($u_1$):**
   Let $\mathcal{T}_{\text{raw}}(C)$ denote the sequence of lexical tokens after comment and docstring removal:
   $$u_1 = S_{\text{raw-token}}(C_A, C_B) = \frac{|\mathcal{T}_{\text{raw}}(C_A) \cap \mathcal{T}_{\text{raw}}(C_B)|}{|\mathcal{T}_{\text{raw}}(C_A) \cup \mathcal{T}_{\text{raw}}(C_B)|}$$

2. **Identifier-Normalized Token Jaccard ($u_2$):**
   Let $\phi_{\text{norm}}: \mathcal{T} \to \mathcal{T}_{\text{norm}}$ be a canonical mapping replacing all user-defined identifiers with generalized positional tokens $\text{ID}_k$ while preserving language keywords and operators:
   $$u_2 = S_{\text{norm-token}}(C_A, C_B) = \frac{|\phi_{\text{norm}}(\mathcal{T}(C_A)) \cap \phi_{\text{norm}}(\mathcal{T}(C_B))|}{|\phi_{\text{norm}}(\mathcal{T}(C_A)) \cup \phi_{\text{norm}}(\mathcal{T}(C_B))|}$$

3. **AST Node Type N-Gram Overlap ($u_3$):**
   Let $\mathcal{A}_{\text{type}}(C)$ denote the sequence of AST node types obtained by pre-order traversal:
   $$u_3 = S_{\text{ast-ngram}}(C_A, C_B) = \frac{|\text{Bigrams}(\mathcal{A}_{\text{type}}(C_A)) \cap \text{Bigrams}(\mathcal{A}_{\text{type}}(C_B))|}{|\text{Bigrams}(\mathcal{A}_{\text{type}}(C_A)) \cup \text{Bigrams}(\mathcal{A}_{\text{type}}(C_B))|}$$

4. **Operator Distribution Cosine Similarity ($u_4$):**
   Let $\mathbf{o}(C) \in \mathbb{R}^{K}$ denote the normalized frequency vector of Java arithmetic, relational, and bitwise operators:
   $$u_4 = S_{\text{operator}}(C_A, C_B) = \frac{\langle \mathbf{o}(C_A), \mathbf{o}(C_B) \rangle}{\|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2}$$

5. **Cyclomatic Complexity Differential Metric ($u_5$):**
   Let $CC(C) = E - N + 2P$ denote McCabe's cyclomatic complexity computed over the program control flow graph:
   $$u_5 = 1.0 - \frac{|CC(C_A) - CC(C_B)|}{\max(CC(C_A), CC(C_B), 1)}$$

### 2.2 Multi-View Projection & Candidate Decision Rule

We fuse the multi-view tuple into a scalar candidate similarity score $s_{\text{cand}} \in [0, 1]$ via a regularized logistic model fitted on $D_{\text{train}}$:
$$s_{\text{cand}}(\mathbf{x}) = \sigma\left( \mathbf{w}^T \mathbf{x} + b \right) = \frac{1}{1 + \exp\left( -(\mathbf{w}^T \mathbf{x} + b) \right)}$$
Where weights $\mathbf{w} \ge 0$ are constrained to be non-negative to enforce monotonic similarity behavior:
$$\min_{\mathbf{w} \ge 0, b} \sum_{i \in D_{\text{train}}} \ell_{\text{BCE}}\left( \sigma(\mathbf{w}^T \mathbf{x}_i + b), \, y_i \right) + \lambda \|\mathbf{w}\|_2^2$$

#### Candidate Selection Rule:
$$\text{Filter}(\mathbf{x}) = \begin{cases} \text{PRUNED}, & \text{if } s_{\text{cand}}(\mathbf{x}) < \theta_{\text{cand}} \\ \text{RETAINED}, & \text{if } s_{\text{cand}}(\mathbf{x}) \ge \theta_{\text{cand}} \end{cases}$$
The cutoff $\theta_{\text{cand}}$ is calibrated on the validation set $D_{\text{val}}$:
$$\theta_{\text{cand}} = \sup \left\{ \theta \in [0, 1] \mid \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta)}{\sum_{i \in D_{\text{val}}} y_i} \ge R_{\text{target}} \right\}$$
Where $R_{\text{target}} = 0.98$ is the target recall constraint.

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Probability Calibration

Let $z = f(\mathbf{x}) \in \mathbb{R}$ denote the raw, uncalibrated score or decision function output of the heavy classifier. We learn a calibration mapping $g: \mathbb{R} \to [0, 1]$ on a held-out calibration split $D_{\text{cal}} = \{(z_i, y_i)\}_{i=1}^{N_{\text{cal}}}$.

#### A. Parametric Calibration: Platt Scaling
Platt scaling assumes a logistic relationship between margin $z$ and class probability:
$$P_{\text{Platt}}(Y = 1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
Parameters $A, B \in \mathbb{R}$ are obtained by minimizing negative log-likelihood on $D_{\text{cal}}$:
$$\mathcal{L}_{\text{Platt}}(A, B) = -\sum_{i=1}^{N_{\text{cal}}} \left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log (1 - \sigma(A z_i + B)) \right]$$

#### B. Non-Parametric Calibration: Isotonic Regression
Isotonic regression fits a free-form non-decreasing step function $m(z)$ via the Pool Adjacent Violators (PAV) algorithm:
$$\min_{m} \sum_{i=1}^{N_{\text{cal}}} \left( y_i - m(z_i) \right)^2 \quad \text{s.t.} \quad m(z_i) \le m(z_j) \text{ for all } z_i \le z_j$$
- *Tradeoff:* Isotonic regression is non-parametric and can model arbitrary monotonic miscalibration curves, but is prone to overfitting when $N_{\text{cal}} < 1,000$. Platt scaling is robust against small sample variance but assumes a strict sigmoidal error distribution.

### 3.2 Quantitative Calibration & Uncertainty Metrics

1. **Expected Calibration Error (ECE):**
   Partition the probability interval $[0, 1]$ into $M$ disjoint bins $B_m = \left( \frac{m-1}{M}, \frac{m}{M} \right]$ for $m \in \{1, \dots, M\}$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   Where:
   $$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i, \quad \text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$$

2. **Maximum Calibration Error (MCE):**
   Measures worst-case bin deviation:
   $$\text{MCE} = \max_{m \in \{1, \dots, M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

3. **Brier Score (Strictly Proper Scoring Rule):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \left( \hat{p}_i - y_i \right)^2$$
   Decomposes into $\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$.

4. **Predictive Entropy ($\mathcal{H}$):**
   $$\mathcal{H}(\hat{p}) = -\hat{p} \log_2(\hat{p}) - (1 - \hat{p}) \log_2(1 - \hat{p})$$

### 3.3 Inductive Conformal Prediction Framework

Let $s(\mathbf{x}, y)$ denote a non-conformity function measuring how unusual label $y \in \{0, 1\}$ is for input $\mathbf{x}$. We employ the probability-based non-conformity score:
$$s(\mathbf{x}, y) = 1 - \hat{P}(Y = y \mid \mathbf{x})$$

#### Algorithm for Conformal Prediction Set Construction:
1. Compute non-conformity scores on calibration split $D_{\text{cal}}$:
   $$\alpha_i = s(\mathbf{x}_i, y_i) = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i), \quad i \in \{1, \dots, N_{\text{cal}}\}$$
2. For a specified significance level $\epsilon \in (0, 1)$ (e.g., $\epsilon = 0.05$ for $95\%$ coverage), compute the empirical $(1 - \epsilon)$-quantile:
   $$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (N_{\text{cal}} + 1)(1 - \epsilon) \rceil}{N_{\text{cal}}}, \, \{\alpha_1, \dots, \alpha_{N_{\text{cal}}}\} \right)$$
3. For a new query instance $\mathbf{x}_{N+1}$, define the prediction set $\mathcal{C}_{1-\epsilon}(\mathbf{x}_{N+1})$:
   $$\mathcal{C}_{1-\epsilon}(\mathbf{x}_{N+1}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}_{N+1}) \le \hat{q}_{1-\epsilon} \right\}$$

#### Theoretical Guarantees, Assumptions & Boundaries:
- **Theorem (Validity under Exchangeability):** If the calibration pairs $(X_i, Y_i)_{i=1}^{N_{\text{cal}}}$ and the test pair $(X_{N+1}, Y_{N+1})$ are independent and identically distributed (i.i.d.) or exchangeable, then:
  $$P\left( Y_{N+1} \in \mathcal{C}_{1-\epsilon}(X_{N+1}) \right) \ge 1 - \epsilon$$
- **Limitation 1 (Marginal vs. Conditional Coverage):** The guarantee holds *on average across the data distribution*, but does not ensure coverage for specific subgroups (e.g., Level 6 obfuscated code).
- **Limitation 2 (Plagiarism Recall vs. Conformal Coverage):** Conformal coverage guarantees that the true label is in the set. Under high uncertainty, the set returns $\mathcal{C} = \{0, 1\}$. While this satisfies coverage, it does not provide a binary decision; high candidate recall relies on escalating uncertain cases to human or deeper model review.

### 3.4 Conformal-Bounded Tri-Band Decision Policy
The conformal set output maps directly to the operational pipeline triage:
$$\pi(\mathbf{x}) = \begin{cases}
\text{STATE 1: LOW\_RISK\_PRUNED}, & \text{if } \mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{0\} \iff \hat{p} \le 1 - \hat{q}_{1-\epsilon} \\
\text{STATE 2: UNCERTAIN\_ESCALATED}, & \text{if } \mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{0, 1\} \iff 1 - \hat{q}_{1-\epsilon} < \hat{p} < \hat{q}_{1-\epsilon} \\
\text{STATE 3: HIGH\_RISK\_FLAGGED}, & \text{if } \mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{1\} \iff \hat{p} \ge \hat{q}_{1-\epsilon}
\end{cases}$$

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Comprehensive Computational Cost Function

To avoid omission or double-counting, we formulate the total per-pair computational cost:
$$C_{\text{total}}(\mathbf{x}; \boldsymbol{\theta}) = C_{\text{prep}}(\mathbf{x}) + C_{\text{Tier1}}(\mathbf{x}) + \mathbb{I}\left( \pi(\mathbf{x}; \boldsymbol{\theta}) = \text{ESCALATED} \right) \cdot \left[ C_{\text{feat-heavy}}(\mathbf{x}) + C_{\text{infer-heavy}}(\mathbf{x}) \right] + C_{\text{router}}$$

Where the expected cost per submission pair over distribution $\mathcal{D}$ is:
$$\mathbb{E}[C_{\text{total}}(\boldsymbol{\theta})] = \bar{C}_{\text{prep}} + \bar{C}_{\text{Tier1}} + P(\text{Escalated} \mid \boldsymbol{\theta}) \cdot \left( \bar{C}_{\text{feat-heavy}} + \bar{C}_{\text{infer-heavy}} \right) + \bar{C}_{\text{router}}$$

### 4.2 Constrained Optimization Formulation

Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with $\theta_{\text{low}} \le \theta_{\text{high}}$ define the routing boundary parameters. The optimization objective on validation data $D_{\text{val}}$ is:
$$\min_{\boldsymbol{\theta}} \quad \mathcal{J}(\boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i=1}^{|D_{\text{val}}|} C_{\text{total}}(\mathbf{x}_i; \boldsymbol{\theta})$$
Subject to the constraints:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \mathcal{J}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{Precision}_{\text{direct}}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

Where:
$$\text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i=1}^{|D_{\text{val}}|} y_i \cdot \mathbb{I}\left( \pi(\mathbf{x}_i; \boldsymbol{\theta}) \in \{\text{ESCALATED}, \text{FLAGGED}\} \right)}{\sum_{i=1}^{|D_{\text{val}}|} y_i}$$
$$\text{Precision}_{\text{direct}}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i=1}^{|D_{\text{val}}|} y_i \cdot \mathbb{I}\left( \pi(\mathbf{x}_i; \boldsymbol{\theta}) = \text{FLAGGED} \right)}{\sum_{i=1}^{|D_{\text{val}}|} \mathbb{I}\left( \pi(\mathbf{x}_i; \boldsymbol{\theta}) = \text{FLAGGED} \right) + \delta}$$

### 4.3 Lagrangian Relaxation & Pareto Frontier Characterization

When navigating the trade-off across varying budgets, we formulate the penalized Lagrangian:
$$\mathcal{L}(\boldsymbol{\theta}, \lambda_1, \lambda_2) = \mathcal{J}(\boldsymbol{\theta}) + \lambda_1 \max\left(0, R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta})\right) + \lambda_2 \max\left(0, \mathcal{J}(\boldsymbol{\theta}) - B_{\text{max}}\right)$$

#### Pareto Optimality:
A threshold configuration $\boldsymbol{\theta}^*$ is **Pareto optimal** if there exists no alternative $\boldsymbol{\theta}'$ such that:
$$\text{CandidateRecall}(\boldsymbol{\theta}') \ge \text{CandidateRecall}(\boldsymbol{\theta}^*) \quad \text{and} \quad \mathcal{J}(\boldsymbol{\theta}') \le \mathcal{J}(\boldsymbol{\theta}^*)$$
with at least one strict inequality.
The **Pareto Frontier** $\mathcal{PF}$ is traced by executing a fine-grained 2D grid search over $\boldsymbol{\theta} \in [0.01, 0.99]^2$ across discrete budget intervals $B \in [0.10\text{ ms}, 4.50\text{ ms}]$.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Text Evidence: Smith-Waterman Local Alignment

To identify matching contiguous and gapped text segments, we execute Smith-Waterman local sequence alignment over word token sequences $\mathbf{a} = (a_1, \dots, a_n)$ and $\mathbf{b} = (b_1, \dots, b_m)$.
The dynamic programming score matrix $H \in \mathbb{R}^{(n+1) \times (m+1)}$ is computed as:
$$H_{i, j} = \max \begin{cases}
0 \\
H_{i-1, j-1} + s(a_i, b_j) & (\text{Match / Mismatch}) \\
H_{i-k, j} - W_k & (\text{Deletion in } \mathbf{b}) \\
H_{i, j-k} - W_k & (\text{Insertion in } \mathbf{b})
\end{cases}$$
Where:
- $s(a_i, b_j) = +2.0$ if $a_i = b_j$; $s(a_i, b_j) = \cos(\mathbf{e}(a_i), \mathbf{e}(b_j)) \in [-1, 1]$ if semantic paraphrase matching is enabled.
- Affine gap penalty: $W_k = u + v \cdot k$ ($u = 2.0, v = 0.5$).
Optimal alignment segments are recovered via traceback from $\max_{i, j} H_{i, j}$ down to $H_{i', j'} = 0$.

### 5.2 Code Evidence: AST Subtree Isomorphism & Identifier Ledger

Let $T_A, T_B$ denote the ASTs of programs $C_A, C_B$.
1. **Subtree Isomorphism:** Two subtrees $t_A \subset T_A, t_B \subset T_B$ are isomorphic under identifier renaming if there exists a bijective identifier substitution $\sigma: \text{Vars}(t_A) \to \text{Vars}(t_B)$ such that:
   $$\text{TypeTree}(t_A) \equiv \text{TypeTree}(t_B) \quad \text{and} \quad \sigma(\text{Tokens}(t_A)) = \text{Tokens}(t_B)$$
2. **Identifier Renaming Ledger:**
   $$\mathcal{M}_{\text{id}} = \left\{ (v_A, v_B) \in \text{Vars}(C_A) \times \text{Vars}(C_B) \mid \text{Support}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \right\}$$

### 5.3 Quantitative Explanation Quality Metrics

1. **Evidence Faithfulness (Decision Probability Drop):**
   Let $\mathcal{E}(\mathbf{x})$ denote the set of tokens/nodes identified as plagiarism evidence. We measure the drop in predicted plagiarism probability when evidence regions are masked or ablated:
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   $$\text{Mean Faithfulness} = \frac{1}{|D_{\text{test}}^{+}|} \sum_{i \in D_{\text{test}}^{+}} \Delta \hat{P}_{\text{faith}}(\mathbf{x}_i)$$
   A high $\Delta \hat{P}$ confirms that the evidence extractor highlights tokens that the model actually relied on to predict plagiarism.

2. **Evidence Coverage:**
   $$\text{Cov}_{\text{text}}(D_A, D_B) = \frac{|\mathcal{E}_{\text{tokens}}(D_B)|}{|D_B|_{\text{tokens}}}, \quad \text{Cov}_{\text{code}}(C_A, C_B) = \frac{|\mathcal{E}_{\text{AST-nodes}}(C_B)|}{|C_B|_{\text{AST-nodes}}}$$

3. **Transformation Alignment Precision (Code Ground-Truth):**
   For test pairs with ground-truth Faidhi-Robinson obfuscations:
   $$\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$$

