# Formal Design of the Four Technical Contributions (Version 3.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.0 (Phase 2.2 Methodological Audit Revision)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Formal Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. System Overview & Selection-Aware Triage Architecture

The proposed framework replaces uncalibrated single-view heuristics with an uncertainty-aware, budget-constrained, and explainable multi-stage architecture.

### 1.1 Strict Disambiguation of Decision Quantities

To prevent methodological conflation, four distinct quantities are formalized:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Tier-1 Candidate Score    s_cand in [0, 1]     --> Computed on ALL candidate pairs (< 0.05 ms)       │
│ 2. Heavy Raw Score           z in R               --> Margin from Heavy Model on ESCALATED pairs only   │
│ 3. Calibrated Probability    p_hat in [0, 1]      --> Calibrated posterior on ESCALATED pairs only      │
│ 4. Conformal Prediction Set  C in {{0},{1},{0,1},Ø} -> Set-valued output on ESCALATED pairs only         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Selection-Aware Escalation Architecture

A fundamental critique of multi-stage inference is the **selection problem**: the heavy model and conformal router do not operate on the marginal population of all submission pairs; they operate strictly on the **conditional escalation sub-population**:
$$\mathcal{D}_{\text{esc}}(\boldsymbol{\theta}) = \left\{ (D_A, D_B, y) \mid \theta_{\text{low}} \le s_{\text{cand}}(D_A, D_B) < \theta_{\text{high}} \right\}$$

To ensure mathematical validity:
1. The Tier-1 candidate generator $g_{\text{cand}}$ is trained strictly on base training data $D_{\text{train}}$.
2. Routing thresholds $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}})$ are selected strictly on validation data $D_{\text{val}}$ and **frozen**.
3. Probability calibration is fitted on a group-disjoint population that has passed through the frozen Tier-1 filter into the escalation band:
   $$D_{\text{cal-prob}}^{\text{esc}} = \left\{ (\mathbf{x}_i, y_i) \in D_{\text{cal-prob}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$
4. Conformal quantile estimation is performed on an independent group-disjoint calibration population selected using the identical frozen policy:
   $$D_{\text{cal-conf}}^{\text{esc}} = \left\{ (\mathbf{x}_i, y_i) \in D_{\text{cal-conf}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$
5. Held-out test pairs are processed using the identical frozen cascade.

```
                                  ┌─────────────────────────────────────────┐
                                  │    Incoming Submission Pair (D_A, D_B)  │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                             ┌───────────────────────────────────────────────────┐
                             │  CONTRIBUTION 1: Lightweight Candidate Generator  │
                             │  (Purely Lightweight: Fast Token / Regex Views)   │
                             │  Text: TF-IDF, Char-3gram, Word-Jaccard, Length   │
                             │  Code: Raw Token, Norm Token, Operator, Length    │
                             └─────────────────────────┬─────────────────────────┘
                                                       │ Candidate Score: s_cand
                          ┌────────────────────────────┼────────────────────────────┐
                          ▼                            ▼                            ▼
                 [ s_cand < θ_low ]       [ θ_low <= s_cand < θ_high ]     [ s_cand >= θ_high ]
                          │                            │                            │
                          ▼                            │                            ▼
            ┌───────────────────────────┐              │              ┌───────────────────────────┐
            │ STATE 1: LOW_RISK_PRUNED  │              │              │ STATE 3: HIGH_RISK_FLAGGED│
            │ • Immediate fast negative │              │              │ • Direct Tier-1 Flag      │
            │ • Bypasses heavy detector │              │              │ • Bypasses heavy detector │
            │ • heavy_score = None      │              │              │ • heavy_score = None      │
            │ • calibrated_p = None     │              │              │ • calibrated_p = None     │
            │ • conformal_set = None    │              │              │ • conformal_set = None    │
            │ • review_required = False │              │              │ • review_required = True  │
            └───────────────────────────┘              │              └─────────────┬─────────────┘
                                                       ▼                            │
                             ┌───────────────────────────────────────────────────┐  │
                             │ STATE 2: UNCERTAIN_ESCALATED                      │  │
                             │ • Invokes Heavy Feature Extraction & Classifier   │  │
                             │ • Text: Sentence-BERT miniLM (4.0 ms)             │  │
                             │ • Code: 7-D AST Parsing (2.05 ms) + SVM / RF      │  │
                             └─────────────────────────┬─────────────────────────┘  │
                                                       │ Raw Score: z               │
                                                       ▼                            │
                             ┌───────────────────────────────────────────────────┐  │
                             │ CONTRIBUTION 2: Selection-Aware Conformal Router  │  │
                             │ • Calibrated Posterior: p_hat = g_cal(z)          │  │
                             │ • Conformal Set: C in {{0}, {1}, {0,1}, Ø}        │  │
                             │ • Predictive Entropy H(p)                         │  │
                             └─────────────────────────┬─────────────────────────┘  │
                                                       │                            │
                          ┌────────────────────────────┼────────────────────────────┤
                          ▼                            ▼                            ▼
                  C = {0} (Model Neg)          C = {0, 1} or Ø (Ambiguous)   C = {1} (Model Pos)
                          │                            │                            │
                          ▼                            ▼                            ▼
            ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
            │ Cleared Non-Plagiarized   │ │ Unresolved / Abstain      │ │ Flagged Plagiarized       │
            │ heavy_score = z           │ │ heavy_score = z           │ │ heavy_score = z           │
            │ calibrated_p = p_hat      │ │ calibrated_p = p_hat      │ │ calibrated_p = p_hat      │
            │ conformal_set = {0}       │ │ conformal_set = {0,1} or Ø│ │ conformal_set = {1}       │
            │ review_required = False   │ │ review_required = True    │ │ review_required = True    │
            └───────────────────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
                                                        │                             │
                                                        └──────────────┬──────────────┘
                                                                       ▼
                                             ┌───────────────────────────────────────────────────┐
                                             │ CONTRIBUTION 4: Cross-Modal Evidence Aggregation  │
                                             │ Generated ONLY when review_required == True       │
                                             │ • Text: Smith-Waterman Local Alignment & Spans    │
                                             │ • Code: AST Subtree Isomorphism & Identifier Map  │
                                             │ • Produces EvidenceDossier for Human Adjudicators │
                                             └───────────────────────────────────────────────────┘
```

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Technical Problem & Novelty Boundary
Single-view pre-filters (word TF-IDF alone or token Jaccard alone) exhibit two critical failure modes:
1. **The Zero-Overlap Text Ceiling:** In sentence-level text benchmarks, paraphrasing causes positive pairs to share zero vocabulary words ($S_{\text{word-tfidf}} = 0.0000$).
2. **Code Pre-Filter Computational Trap:** Full Java AST parsing requires **2.050 ms/pair**. Invoking AST construction at Tier 1 destroys the cost efficiency of the pipeline.

**Novelty Classification:** *Proposed Integration & Optimization*. Combining word TF-IDF, character n-grams, and regex tokenization is not a new fundamental machine learning algorithm; the contribution is formulating a **purely lightweight, sub-0.05 ms multi-view candidate generator** that eliminates the zero-overlap failure mode while deferring all AST construction to Tier 2.

### 2.2 Mathematical Definition of Features

#### A. Text Multi-View Features $\mathbf{v}_{\text{text}} \in [0, 1]^4$:
1. **Word-Level Sublinear TF-IDF Cosine Similarity ($v_1$):**
   $$\text{tf-idf}(t, d) = \left( 1 + \log(\text{tf}(t, d)) \right) \cdot \log\left( \frac{1 + |D_{\text{train}}|}{1 + \text{df}(t)} \right) + 1 \quad (\text{Vocabulary and IDF fitted on } D_{\text{train}} \text{ only})$$
   $$v_1 = S_{\text{word-tfidf}}(D_A, D_B) = \frac{\langle \mathbf{t}_A, \mathbf{t}_B \rangle}{\|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2} \quad (\text{Returns } 0.0 \text{ if } \|\mathbf{t}_A\|_2 \|\mathbf{t}_B\|_2 = 0)$$
2. **Character 3-Gram Set Jaccard Similarity ($v_2$):**
   Let $\mathcal{G}_3(D)$ denote the set of unique character tri-grams extracted from lowercased text $D$:
   $$v_2 = S_{\text{char-3gram}}(D_A, D_B) = \frac{|\mathcal{G}_3(D_A) \cap \mathcal{G}_3(D_B)|}{|\mathcal{G}_3(D_A) \cup \mathcal{G}_3(D_B)|} \quad (\text{Returns } 0.0 \text{ if denominator is } 0)$$
3. **Word Set Jaccard Similarity ($v_3$):**
   Let $\mathcal{W}(D)$ denote the set of unique whitespace- and punctuation-delimited words:
   $$v_3 = S_{\text{word-jaccard}}(D_A, D_B) = \frac{|\mathcal{W}(D_A) \cap \mathcal{W}(D_B)|}{|\mathcal{W}(D_A) \cup \mathcal{W}(D_B)|}$$
4. **Length-Ratio Symmetry Metric ($v_4$):**
   $$v_4 = \Delta_{\text{len}}(D_A, D_B) = \frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}}, 1)}$$

#### B. Source Code Multi-View Features $\mathbf{u}_{\text{code}} \in [0, 1]^4$ (Purely Regex/Lexer Tier 1):
1. **Raw Token Set Jaccard ($u_1$):**
   Let $\mathcal{T}_{\text{raw}}(C)$ denote unique lexical tokens after stripping line and block comments:
   $$u_1 = S_{\text{raw-token}}(C_A, C_B) = \frac{|\mathcal{T}_{\text{raw}}(C_A) \cap \mathcal{T}_{\text{raw}}(C_B)|}{|\mathcal{T}_{\text{raw}}(C_A) \cup \mathcal{T}_{\text{raw}}(C_B)|}$$
2. **Identifier-Normalized Token Set Jaccard ($u_2$):**
   Regex mapping replacing user-defined identifiers with canonical symbol `ID`, while preserving Java keywords (`public`, `class`, `for`, `while`, `int`), operators, and string literals:
   $$u_2 = S_{\text{norm-token}}(C_A, C_B) = \frac{|\phi_{\text{regex}}(\mathcal{T}(C_A)) \cap \phi_{\text{regex}}(\mathcal{T}(C_B))|}{|\phi_{\text{regex}}(\mathcal{T}(C_A)) \cup \phi_{\text{regex}}(\mathcal{T}(C_B))|}$$
3. **Operator Frequency Profile Cosine ($u_3$):**
   Normalized frequency vector $\mathbf{o}(C) \in \mathbb{R}^{14}$ of Java operators (`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`):
   $$u_3 = S_{\text{operator}}(C_A, C_B) = \frac{\langle \mathbf{o}(C_A), \mathbf{o}(C_B) \rangle}{\|\mathbf{o}(C_A)\|_2 \|\mathbf{o}(C_B)\|_2}$$
4. **Token Length Ratio Asymmetry ($u_4$):**
   $$u_4 = \Delta_{\text{tok-len}}(C_A, C_B) = \frac{\min(|\mathcal{T}(C_A)|, |\mathcal{T}(C_B)|)}{\max(|\mathcal{T}(C_A)|, |\mathcal{T}(C_B)|, 1)}$$

*Computational Complexity:* Per-submission feature caching requires $O(|D|)$ time and $O(|D|)$ space. Per-pair candidate scoring requires $O(|S_A| + |S_B|)$ time (fast hash-set intersections). Target execution latency is $< 0.050\text{ ms/pair}$, to be empirically benchmarked under warmed-up profiling.

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Selection-Aware Calibration Protocol
Because probability calibration is applied exclusively to escalated pairs, the calibrator $g_{\text{cal}}$ must be fitted on observations drawn from the conditional escalation distribution:
$$D_{\text{cal-prob}}^{\text{esc}} = \left\{ (z_i, y_i) \mid \mathbf{x}_i \in D_{\text{cal-prob}} \wedge \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \right\}$$

1. **Platt Scaling (Parametric):**
   $$P_{\text{Platt}}(Y = 1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
   Fitted via cross-entropy minimization on $D_{\text{cal-prob}}^{\text{esc}}$.
2. **Isotonic Regression (Non-Parametric):**
   Fitted via the Pool Adjacent Violators (PAV) algorithm on $D_{\text{cal-prob}}^{\text{esc}}$:
   $$\min_{m} \sum_{i \in D_{\text{cal-prob}}^{\text{esc}}} (y_i - m(z_i))^2 \quad \text{s.t.} \quad m(z_i) \le m(z_j) \text{ for } z_i \le z_j$$
3. **Independent Diagnostic Evaluation:** ECE and Brier score are evaluated on an independent validation escalation split $D_{\text{val}}^{\text{esc}}$, completely separate from $D_{\text{cal-prob}}^{\text{esc}}$ and the test split.

### 3.2 Inductive Conformal Prediction & Four-Set Operational Semantics

On the frozen conditional escalation split $D_{\text{cal-conf}}^{\text{esc}}$, compute probability non-conformity scores:
$$\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i), \quad i \in \{1, \dots, N_{\text{cal-conf}}^{\text{esc}}\}$$
For significance $\epsilon \in (0, 1)$, compute empirical quantile $\hat{q}_{1-\epsilon}$.
For escalated test query $\mathbf{x}$, the prediction set $\mathcal{C}_{1-\epsilon}(\mathbf{x})$ is:
$$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}) \le \hat{q}_{1-\epsilon} \right\}$$

#### Precise Operational Semantics for All Four Conformal Sets:
1. **$\mathcal{C} = \{0\}$ (Model-Supported Negative Prediction):**
   - *Meaning:* Label 0 conforms to calibration distribution ($p_{\text{cal}} \le \hat{q}$) and label 1 does not ($1 - p_{\text{cal}} > \hat{q}$).
   - *Action:* Cleared as non-plagiarized. Heavy model disproved initial suspicion. `human_review_required = False`.
2. **$\mathcal{C} = \{1\}$ (Model-Supported Positive Prediction):**
   - *Meaning:* Label 1 conforms ($1 - p_{\text{cal}} \le \hat{q}$) and label 0 does not ($p_{\text{cal}} > \hat{q}$).
   - *Action:* Flagged for instructor review with evidence dossier. `human_review_required = True`.
3. **$\mathcal{C} = \{0, 1\}$ (Ambiguous Prediction):**
   - *Meaning:* Both labels conform to calibration distribution; prediction margin is uncertain.
   - *Action:* Escalated to instructor review as an ambiguous boundary case. `human_review_required = True`.
4. **$\mathcal{C} = \emptyset$ (Abstention / Unresolved Prediction):**
   - *Meaning:* Neither label conforms ($p_{\text{cal}} > \hat{q}$ and $1 - p_{\text{cal}} > \hat{q}$, mathematically possible when $\hat{q} < 0.50$).
   - *Operational Rule:* **Abstention**. The system refrains from binary classification and flags the instance as unresolved due to model inconsistency. `human_review_required = True`.
   - *Scientific Boundary:* The empty set is **not** labeled as proof of out-of-distribution (OOD) data, as true OOD claims require an independent generative or density-based OOD detector.

### 3.3 Limitations of IR-Plag Code Conformal Prediction
In the IR-Plag corpus, `case-05` contains only **15 unique reference Java programs**.
- Slicing 15 groups across calibration folds yields $\approx 3$ reference programs per fold ($N \approx 20$ pairs), which violates the sample size requirements for reliable quantile estimation ($N_{\text{cal}} \ge \lceil (1 - \epsilon)/\epsilon \rceil$).
- **Methodological Verdict:** We report the source-code uncertainty mechanism as **exploratory calibrated routing**. We explicitly state that valid finite-sample conformal guarantees for code require expanding the corpus to 50+ independent assignment tasks.

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Accurate Methodological Characterization
We accurately designate this method as:
> **"Recall-Constrained Routing-Policy Optimization via Discretized Pareto Surface Search"**

It is a discrete multi-objective parameter search over a 2D scalar threshold space $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$. Because the parameter space is 2-dimensional and bounded, dense grid evaluation ($40 \times 36 = 1,440$ points) globally maps the non-convex Pareto frontier without gradient instability.

### 4.2 Comprehensive Computational Cost Model

To resolve previous inconsistencies, the cost model matches the exact execution branches of Algorithm 6:
$$\bar{C}_{\text{total}}(\boldsymbol{\theta}) = \bar{C}_{\text{prep}} + \bar{C}_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ \bar{C}_{\text{feat-heavy}} + \bar{C}_{\text{infer-heavy}} + \bar{C}_{\text{router}} \right] + P(\text{Review} \mid \boldsymbol{\theta}) \cdot \bar{C}_{\text{explain}}$$

Where:
- $P(\text{Escalate} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}})$
- $P(\text{Review} \mid \boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} \left[ \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + \mathbb{I}(\theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}} \wedge \mathcal{C}_i \in \{\{1\}, \{0, 1\}, \emptyset\}) \right]$

### 4.3 Formal Optimization Objective & Infeasibility Handling
$$\min_{\boldsymbol{\theta} \in \Theta} \quad \bar{C}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
\text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) &\ge R_{\text{target}} \quad (R_{\text{target}} = 0.98) \\
\bar{C}_{\text{total}}(\boldsymbol{\theta}) &\le B_{\text{max}} \quad (\text{Latency Budget in ms/pair}) \\
\text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) &\ge P_{\text{target}} \quad (P_{\text{target}} \ge 0.90)
\end{aligned}$$

Where:
- $\text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{low}})}{\sum_{i \in D_{\text{val}}} y_i}$
- $\text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}})}{\sum_{i \in D_{\text{val}}} \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + 10^{-7}}$

**Infeasibility Reporting:** If no grid point satisfies all three constraints on $D_{\text{val}}$, the optimizer outputs **`STATUS: INFEASIBLE_BUDGET`** and reports the constraint violation vector. Hardcoded fallback thresholds are strictly forbidden.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Evidence Extraction Scope
Evidence extraction is invoked **strictly when `human_review_required == True`** (Tier-1 direct flags, conformal $\{1\}$, $\{0, 1\}$, and $\emptyset$).

1. **Text Evidence Engine:** Smith-Waterman local sequence alignment with affine gap penalty ($u = 2.0, v = 0.5$). Recurrence matrix:
   $$H_{i, j} = \max\left( 0, \, H_{i-1, j-1} + s(a_i, b_j), \, E_{i, j}, \, F_{i, j} \right)$$
   Where $E_{i, j} = \max(H_{i-1, j} - u - v, E_{i-1, j} - v)$ and $F_{i, j} = \max(H_{i, j-1} - u - v, F_{i, j-1} - v)$.
2. **Code Evidence Engine:** Normalized AST subtree isomorphism and identifier substitution mapping:
   $$\mathcal{M}_{\text{id}} = \left\{ (v_A, v_B) \in \text{Vars}(C_A) \times \text{Vars}(C_B) \mid \text{Support}(v_A \leftrightarrow v_B) \ge \tau_{\text{id}} \right\}$$

### 5.2 Quantitative Quality Metrics
1. **Evidence Faithfulness (Feature Ablation Probability Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   Evaluated strictly on pairs where the heavy model was invoked ($\hat{p} \ne \text{None}$).
2. **Evidence Coverage:** $\text{Cov} = \frac{|\mathcal{E}_{\text{tokens}}|}{|\text{Total Tokens}|}$.
3. **Transformation Alignment Precision:** $\text{Prec}_{\text{align}} = \frac{|\mathcal{M}_{\text{id}} \cap \mathcal{M}_{\text{ground-truth}}|}{|\mathcal{M}_{\text{id}}|}$.
4. **Human Educator Usefulness Protocol:** Evaluated via a structured 5-point Likert protocol measuring clarity, span accuracy, and review-time reduction.

---

## 6. Honest Taxonomy of Claims & Contribution Structure

To maintain Q1 scientific integrity, we classify every element:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Established Techniques:   Sublinear TF-IDF, Sentence-BERT, Platt Scaling, Isotonic Regression,      │
│                              Split Conformal Prediction, Smith-Waterman Affine Alignment.              │
│ 2. Adaptations:              Gandhi SVM adapted from Python to Java AST; Mehsen RF adapted to 7-D AST; │
│                              ES-Plag reimplemented in Python for IR-Plag Java benchmark.               │
│ 3. Proposed Integrations:    Multi-stage cascade coupling fast candidate screening, calibration,       │
│                              conformal routing, and explainability under explicit compute budgets.     │
│ 4. Proposed Mechanisms:      (i) Lightweight regex-token code screening deferring AST to Tier 2;       │
│                              (ii) Selection-aware calibration and conformal routing;                   │
│                              (iii) Discretized Pareto surface search under explicit per-pair budgets;  │
│                              (iv) Unified EvidenceDossier with feature ablation faithfulness scoring.  │
│ 5. Empirical Contributions:  Resolving the 10.57% text zero-overlap failure mode; disaggregated AST    │
│                              latency audit; 6-level obfuscation resilience benchmark on Java code.     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

