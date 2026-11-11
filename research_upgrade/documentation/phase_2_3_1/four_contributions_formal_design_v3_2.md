# Formal Design of the Four Technical Contributions (Version 3.2)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.2 (Phase 2.3.1 Independent Audit Revision)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Finalized Design Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. System Overview & Selection-Aware Triage Architecture

The proposed framework replaces uncalibrated single-view heuristics with an uncertainty-aware, budget-constrained, and explainable multi-stage triage cascade.

### 1.1 Strict Disambiguation of Decision Quantities

To prevent methodological conflation across the triage cascade, four mathematically distinct quantities are defined and strictly separated:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Tier-1 Candidate Score    s_cand in [0, 1]     --> Fast fused multi-view score on 100% of pairs     │
│ 2. Heavy Raw Margin          z in R               --> Margin from Heavy Model on ESCALATED pairs only  │
│ 3. Calibrated Probability    p_hat in [0, 1]      --> Calibrated posterior on ESCALATED pairs only     │
│ 4. Conformal Prediction Set  C in {{0},{1},{0,1},Ø} -> Set-valued output on ESCALATED TEXT pairs only  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Candidate Score ($s_{\text{cand}} \in [0, 1]$):** Produced by the Tier-1 lightweight multi-view scorer $g_{\text{cand}}$ on 100% of incoming candidate pairs. It is an uncalibrated ranking score used exclusively for triage gating, not a posterior probability of misconduct.
- **Heavy Raw Margin ($z \in \mathbb{R}$):** Produced by the Tier-2 heavy detector (Sentence-BERT bi-encoder for text; 7-D AST structural classifier for source code). It is computed **strictly on escalated pairs** ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$). For bypassed pairs, $z = \text{None}$.
- **Calibrated Posterior Probability ($\hat{p} \in [0, 1]$):** Produced by a probability calibrator (Platt scaling or Isotonic regression) fitted on the conditional escalation population. For non-escalated pairs, $\hat{p} = \text{None}$.
- **Conformal Prediction Set ($\mathcal{C}_{1-\epsilon} \subseteq \{0, 1\}$):** Constructed from $\hat{p}$ and finite-sample order statistic quantile $\hat{q}_{1-\epsilon}$ on escalated pairs. **Applicable strictly to text pairs** where premise-level sample size ($n \ge \lceil (1-\epsilon)/\epsilon \rceil$) is satisfied. For source code, conformal prediction is **explicitly disclaimed** due to sample size constraints ($N_{\text{ref}}=15$), and exploratory calibrated routing is used instead. For non-escalated pairs, $\mathcal{C}_{1-\epsilon} = \text{None}$.

---

### 1.2 Four-Level Disambiguation of Adjudication Semantics (Issue 4)

To prevent ethical, legal, and operational ambiguity, the framework formally distinguishes four levels of decision-making:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUR LEVELS OF ADJUDICATION DISAMBIGUATION                               │
├────┬─────────────────────────────┬─────────────────────────────────────────────────────────────────────┤
│ Level│ Entity                      │ Operational Semantics & Scope                                       │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ 1  │ Model Algorithmic Output    │ Raw score s_cand, margin z, calibrated probability p_hat, set C     │
│ 2  │ Generated Evidence Dossier  │ Token alignments, AST isomorphic subtrees, identifier mappings      │
│ 3  │ Human Expert Adjudication   │ Independent inspection by course instructor or integrity committee  │
│ 4  │ Final Institutional Verdict │ Formal academic disciplinary finding (solely human authority)       │
└────┴─────────────────────────────┴─────────────────────────────────────────────────────────────────────┘
```

A model prediction (including a singleton conformal set $\{1\}$ or $\{0\}$) is an **algorithmic output**, not legal or institutional proof of misconduct or innocence.

### 1.3 Disambiguated Operational Output States (Issue 4)

The triage cascade outputs four standardized, strictly typed operational states:

1. **`LOW_RISK_REVIEW_DEFERRED` (Tier 1 or Tier 2):**
   - *Tier 1 Occurrence:* The pair exhibits negligible similarity across all lightweight views ($s_{\text{cand}} < \theta_{\text{low}}$). Heavy inference is bypassed ($z = \text{None}, \hat{p} = \text{None}, \mathcal{C} = \text{None}$).
   - *Tier 2 Occurrence:* The heavy model and uncertainty router evaluate the pair and determine that similarity conforms to benign independent work ($\mathcal{C} = \{0\}$ for text, or $\hat{p} < \tau_{\text{non-plag}}$ with low entropy for code).
   - *Operational Action:* Formal human review is deferred (`human_review_required = False`). No evidence extraction is charged.
2. **`HIGH_RISK_CANDIDATE_FLAGGED` (Tier 1 or Tier 2):**
   - *Tier 1 Occurrence:* The pair exhibits extreme multi-view lexical/token similarity ($s_{\text{cand}} \ge \theta_{\text{high}}$). Heavy inference is bypassed ($z = \text{None}, \hat{p} = \text{None}, \mathcal{C} = \text{None}$) to conserve compute budget.
   - *Tier 2 Occurrence:* The heavy model outputs high similarity confirmed by the router ($\mathcal{C} = \{1\}$ for text, or $\hat{p} \ge \tau_{\text{plag}}$ with low entropy for code).
   - *Operational Action:* Elevated triage signal requiring mandatory human committee review (`human_review_required = True`). An `EvidenceDossier` is generated. It does **not** constitute automated proof of plagiarism.
3. **`AMBIGUOUS_HUMAN_REVIEW` (Tier 2):**
   - *Occurrence:* The escalated pair exhibits high predictive uncertainty or conflicting signals ($\mathcal{C} = \{0, 1\}$ for text, or $\tau_{\text{non-plag}} \le \hat{p} < \tau_{\text{plag}}$ / high entropy for code).
   - *Operational Action:* The model explicitly refrains from a definitive classification and requests human adjudication (`human_review_required = True`). An `EvidenceDossier` is generated.
4. **`ABSTAIN_UNRESOLVED` (Tier 2):**
   - *Occurrence:* Conformal prediction set is empty ($\mathcal{C} = \emptyset$) for text, indicating the pair does not conform to either non-plagiarized or plagiarized calibration distributions at error rate $\epsilon$.
   - *Operational Action:* Mandatory escalation to human expert audit (`human_review_required = True`). Treated as an empirical model abstention, not automated proof of out-of-distribution data.

---

### 1.4 Selection-Aware Escalation Architecture

The heavy classifier, probability calibrator, and uncertainty router operate strictly on the **conditional escalation sub-population**:
$$\mathcal{D}_{\text{esc}}(\boldsymbol{\theta}) = \left\{ (D_A, D_B, y) \in \mathcal{D} \mid \theta_{\text{low}} \le s_{\text{cand}}(D_A, D_B) < \theta_{\text{high}} \right\}$$

To ensure statistical and experimental validity:
1. **Model Fitting ($D_{\text{train}}$):** The candidate generator $g_{\text{cand}}$ and heavy models $f_{\text{heavy}}$ are fitted strictly on base training data $D_{\text{train}}$.
2. **Threshold Selection & Freezing ($D_{\text{val}}$):** Routing thresholds $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}})$ are selected on validation data $D_{\text{val}}$ via Pareto surface grid search. Once selected, **thresholds are frozen**.
3. **Probability Calibration ($D_{\text{cal-prob}}^{\text{esc}}$):** The frozen policy is applied to $D_{\text{cal-prob}}$, yielding escalated slice $D_{\text{cal-prob}}^{\text{esc}}$. The calibrator $g_{\text{cal}}$ is fitted strictly on $D_{\text{cal-prob}}^{\text{esc}}$. Calibration quality (ECE, Brier score) is evaluated on independent validation data, never on the fitting split.
4. **Conformal Calibration ($D_{\text{cal-conf}}^{\text{esc}}$, Text Only):** The identical frozen policy is applied to $D_{\text{cal-conf}}$, yielding escalated slice $D_{\text{cal-conf}}^{\text{esc}}$. The conformal quantile $\hat{q}_{1-\epsilon}$ is estimated strictly on $D_{\text{cal-conf}}^{\text{esc}}$.
5. **Frozen Evaluation ($D_{\text{test}}$):** The entire frozen pipeline is evaluated once on the untouched held-out test split $D_{\text{test}}$. Final calibration metrics (ECE, Brier score) are computed strictly on $D_{\text{test}}^{\text{esc}}$ where probabilities are actually produced.

```
                                  ┌─────────────────────────────────────────┐
                                  │    Incoming Submission Pair (D_A, D_B)  │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                             ┌──────────────────────────────────────────────────┐
                             │  CONTRIBUTION 1: Lightweight Candidate Generator │
                             │  • Sublinear TF-IDF, Char 3-Grams, Tokens (Text) │
                             │  • Regex Tokens, Operator Vectors (Code, NO AST) │
                             │  Computes Fused Candidate Score s_cand in [0, 1] │
                             └─────────────────────────┬────────────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     │                                 │                                 │
     s_cand < theta_low                                │                     s_cand >= theta_high
                     ▼                                 │                                 ▼
         ┌───────────────────────┐                     │                     ┌───────────────────────┐
         │   LOW_RISK_REVIEW_    │                     │                     │  HIGH_RISK_CANDIDATE_ │
         │       DEFERRED        │                     │                     │        FLAGGED        │
         │   • Pruned early      │                     │                     │   • Triage signal     │
         │   • Bypasses Heavy    │                     │                     │   • Bypasses Heavy    │
         │   • Heavy = None      │                     │                     │   • Heavy = None      │
         │   • Review = False    │                     │                     │   • Review = True     │
         └───────────────────────┘                     │                     │   • Dossier Generated │
                                                       │                     └───────────────────────┘
                                   theta_low <= s_cand < theta_high
                                       UNCERTAIN_ESCALATED
                                                       │
                                                       ▼
                             ┌──────────────────────────────────────────────────┐
                             │  Tier-2 Heavy Feature Extraction & Inference     │
                             │  • Sentence-BERT Bi-Encoder (Text: ~4.0 ms)      │
                             │  • 7-D AST Parsing & Structural RF (Code: ~2.0ms)│
                             │  Produces Raw Decision Margin z in R             │
                             └─────────────────────────┬────────────────────────┘
                                                       │
                                                       ▼
                             ┌──────────────────────────────────────────────────┐
                             │  CONTRIBUTION 2: Uncertainty-Aware Routing       │
                             │  • Calibrates Probability p_hat on D_cal_prob_esc│
                             │  • TEXT: Conformal 4-Set Routing via q_hat       │
                             │  • CODE: Exploratory Calibrated Band Routing     │
                             └─────────────────────────┬────────────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     │                                 │                                 │
               Set = {0} / Low-Risk              Set = {1} / High-Risk          Set = {0,1} or Ø / Band
                     ▼                                 ▼                                 ▼
         ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
         │   LOW_RISK_REVIEW_    │         │  HIGH_RISK_CANDIDATE_ │         │  AMBIGUOUS_HUMAN_REV  │
         │       DEFERRED        │         │        FLAGGED        │         │  / ABSTAIN_UNRESOLVED │
         │ • Model negative pred │         │ • Model positive flag │         │ • High uncertainty    │
         │ • Review = False      │         │ • Review = True       │         │ • Review = True       │
         │ • No evidence charged │         │ • Dossier Generated   │         │ • Dossier Generated   │
         └───────────────────────┘         └───────────────────────┘         └───────────────────────┘
```

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Problem Definition
Traditional academic integrity pre-filters rely on single lexical similarity heuristics (e.g. TF-IDF or token Jaccard) that miss paraphrased text or renamed source code. Conversely, executing deep transformer models or full AST parsers across all $O(M^2)$ pairwise combinations in a class cohort of $M$ students creates prohibitive computational bottlenecks.

### 2.2 Multi-View Feature Representations (Purely Lightweight Tier 1)
Tier-1 extractors are strictly restricted to $O(|D|)$ string and regex operations ($\le 0.050\text{ ms/pair}$). **Heavy AST parsing is strictly prohibited at Tier 1.**

#### A. Academic Text Multi-View Vector $\mathbf{v}_{\text{text}} \in [0, 1]^4$:
1. **Sublinear TF-IDF Cosine Similarity ($v_1$):** Measures lexical keyword overlap with sublinear term-frequency dampening ($1 + \log(\text{tf})$) and inverse document frequency fitted strictly on $D_{\text{train}}$.
2. **Character 3-Gram Set Jaccard Overlap ($v_2$):** Measures substring overlap at the sub-word level. Robust against morphological alterations, typographical variations, and compounding.
3. **Word Token Set Jaccard Overlap ($v_3$):** Measures bag-of-words token co-occurrence irrespective of word order, capturing lexical overlap across sentence rearrangements.
4. **Length-Ratio Symmetry Metric ($v_4$):** Defined as $\frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}}, 1)}$. Discriminated against partial or heavily padded comparisons.

#### B. Source Code Multi-View Vector $\mathbf{u}_{\text{code}} \in [0, 1]^4$ (Pure Regex/Lexer Tier 1):
1. **Raw Token Set Jaccard Overlap ($u_1$):** Lexical token overlap after stripping Java comments and whitespace. Fast detection of direct code sharing.
2. **Normalized Identifier Token Set Jaccard ($u_2$):** All user identifiers mapped to canonical token `VAR_TOKEN` using regular expressions. Invariant to variable, method, and parameter renaming.
3. **Operator Frequency Profile Cosine ($u_3$):** Frequency vector across Java operators (`+`, `-`, `*`, `/`, `&&`, `||`, `==`, `!=`, `<`, `>`, `++`, `--`, `+=`). Captures algorithmic logic signature without parsing syntax trees.
4. **Cyclomatic Keyword Profile Differential ($u_4$):** Regex count of decision keywords (`if`, `else`, `while`, `for`, `case`, `catch`, `?`) normalized as $1.0 - \frac{|k_A - k_B|}{\max(k_A, k_B, 1)}$. Captures structural control-flow complexity.

### 2.3 Scholarly Categorization:
- **What is established:** TF-IDF (Salton & Buckley, 1988), Character 3-grams, Jaccard token overlap, Regex keyword tokenization.
- **What is adapted:** Applying regex identifier normalization and operator frequency profiles as ultra-fast code candidate views without AST generation.
- **What is integrated:** Fused logistic candidate scoring model $g_{\text{cand}}$ trained on $D_{\text{train}}$.
- **What may be novel:** Validating that multi-view candidate screening achieves high recall on zero-overlap paraphrased academic text pairs while maintaining sub-0.050 ms ingestion latency.
- **Required Experiments:** Ablation of individual views (A1–A4), Variance Inflation Factor (VIF) collinearity audit, candidate recall versus pruning rate trade-off curve on $D_{\text{val}}$.

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decisions

### 3.1 Problem Definition
Standard deep learning and ensemble classifiers output uncalibrated scores that are frequently overconfident on borderline cases. Furthermore, point predictions force binary classifications ($0$ or $1$) even when the model possesses high predictive uncertainty.

### 3.2 Probability Calibration on Conditional Escalation Slice
Because the heavy classifier evaluates only escalated instances, calibrators fitted on the unselected marginal dataset suffer from distribution shift.
- The calibrator $g_{\text{cal}}$ is fitted on raw heavy margins $z$ strictly within the conditional escalation split:
  $$D_{\text{cal-prob}}^{\text{esc}} = \{(\mathbf{x}_i, y_i) \in D_{\text{cal-prob}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}}\}$$
- **Platt Scaling (Parametric):** $\hat{P}(Y=1 \mid z) = \sigma(A \cdot z + B)$.
- **Isotonic Regression (Non-parametric):** Fits a non-decreasing step function via the Pair-Adjacent Violators (PAV) algorithm.
- **Diagnostics:** ECE, Brier score, and NLL are evaluated on independent evaluation data, never on the calibration-fitting set.

---

### 3.3 Modality-Specific Uncertainty Routing

#### Modality A: Academic Text (Formal Split Conformal Prediction)
- **Calibration Split ($D_{\text{cal-conf}}^{\text{esc}}$):** Disjoint from $D_{\text{train}}$, $D_{\text{val}}$, and $D_{\text{cal-prob}}$, partitioned at the premise group level.
- **Non-Conformity Score:** $\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$.
- **Exact Finite-Sample Order Statistic (Issue 2):** For sample size $n = |D_{\text{cal-conf}}^{\text{esc}}|$ and significance $\epsilon \in (0, 1)$:
  $$k = \lceil (n + 1)(1 - \epsilon) \rceil$$
  The conformal quantile $\hat{q}_{1-\epsilon}$ is the $k$-th smallest value in $\{\alpha_1, \dots, \alpha_n\}$.
  *Finite-Sample Condition:* If $k > n$ (which occurs when $n < \lceil (1-\epsilon)/\epsilon \rceil$), nominal coverage $1-\epsilon$ is mathematically impossible to guarantee without extrapolation. In that case, the method must abstain from claiming a conformal guarantee and fall back to exploratory uncertainty routing.
- **Conformal Prediction Set:** $\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}) \le \hat{q}_{1-\epsilon}\}$.
- **Scope of Coverage:** Guarantees marginal conditional coverage:
  $$P\left( Y \in \mathcal{C}_{1-\epsilon}(X) \mid \text{Escalated} \right) \ge 1 - \epsilon$$
  Disclaims individual-instance conditional coverage, coverage under distribution shift, and recall guarantees.

#### Modality B: Source Code (Exploratory Calibrated Uncertainty Routing)
- **Empirical Constraint:** IR-Plag has only 15 unique reference Java programs in `case-05` (all 53 plagiarized pairs share 1 original program). This sample size is mathematically insufficient for finite-sample conformal guarantees ($n < \lceil (1-\epsilon)/\epsilon \rceil = 19$ for $\epsilon=0.05$).
- **Methodological Disclaimer:** We **strictly disclaim** mathematical conformal coverage guarantees for source code.
- **Exploratory Calibrated Routing Mechanism:**
  - Evaluated via group-aware 5-fold cross-fitting on `case-05` (3 reference programs per fold).
  - Heavy code model outputs calibrated probability $\hat{p} = g_{\text{cal}}(z)$ using Platt scaling.
  - Predictive Shannon entropy: $H(\hat{p}) = -\hat{p}\log_2 \hat{p} - (1-\hat{p})\log_2(1-\hat{p})$.
  - Routing via decision bands $[\tau_{\text{non-plag}}, \tau_{\text{plag}}]$ and entropy cutoff $\tau_{\text{entropy}}$:
    - $\hat{p} < \tau_{\text{non-plag}}$ and $H \le \tau_{\text{entropy}} \implies$ `LOW_RISK_REVIEW_DEFERRED` (`human_review_required = False`).
    - $\hat{p} \ge \tau_{\text{plag}}$ and $H \le \tau_{\text{entropy}} \implies$ `HIGH_RISK_CANDIDATE_FLAGGED` (`human_review_required = True`).
    - Otherwise $\implies$ `AMBIGUOUS_HUMAN_REVIEW` (`human_review_required = True`).
  - For code records, `conformal_prediction_set` is assigned `None (Disclaimed for Code)`.

### 3.4 Scholarly Categorization:
- **What is established:** Platt scaling (Platt, 1999), Isotonic regression (Zadrozny & Elkan, 2002), Split conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021).
- **What is adapted:** Fitting calibration models conditionally on the post-screening escalation population $D^{\text{esc}}$.
- **What is integrated:** Combining probability calibration, predictive entropy, and 4-set conformal routing within a multi-stage triage cascade.
- **What may be novel:** Proving that conditional calibration prevents distribution shift in cascaded academic integrity detection.
- **Required Experiments:** Reliability diagrams, ECE/Brier score comparison between marginal and slice calibration (A8), empirical coverage verification on $D_{\text{test}}^{\text{esc}}$.

---

## 4. Contribution 3: Budget-Constrained Routing Policy Optimization

### 4.1 Problem Definition
Existing multi-stage systems choose routing cutoffs heuristically, leading to unpredictable latency spikes or candidate recall drops. We formalize threshold selection as a constrained multi-objective optimization problem evaluated over a discretized grid.

### 4.2 Mathematical Optimization Formulation

Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with $\theta_{\text{low}} < \theta_{\text{high}}$.
The optimization problem evaluated on validation data $D_{\text{val}}$ is:
$$\min_{\boldsymbol{\theta}} \quad \bar{T}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \bar{T}_{\text{total}}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

Where:
- $\bar{T}_{\text{total}}(\boldsymbol{\theta})$ is the expected per-pair wall-clock latency (milliseconds).
- $R_{\text{target}}$: Target candidate recall bound (e.g. $0.980$).
- $B_{\text{max}}$: Per-pair latency budget (e.g. $1.50\text{ ms/pair}$).
- $P_{\text{target}}$: Minimum precision required for Tier-1 direct triage flags (e.g. $0.900$).

### 4.3 Realistic Optimization Procedure: Pareto Surface Search
- **Honest Algorithmic Terminology:** The optimization procedure is a **discretized grid-search Pareto frontier extraction**, NOT an unconstrained continuous Lagrangian optimizer.
- **Infeasibility Handling:** If no policy satisfies all constraints, the optimizer returns `STATUS: INFEASIBLE_BUDGET`.
- **Strict Freezing Sequence:** The optimal policy $\boldsymbol{\theta}^*$ is chosen on $D_{\text{val}}$ and **frozen**. Changing thresholds invalidates conditional escalation calibration and requires complete re-calibration.

### 4.4 Scholarly Categorization:
- **What is established:** Multi-objective Pareto optimization, grid search.
- **What is adapted:** Applying recall, precision, and latency constraints to multi-stage triage cutoffs.
- **What is integrated:** Discretized Pareto surface search over $(\theta_{\text{low}}, \theta_{\text{high}})$ on $D_{\text{val}}$.
- **What may be novel:** Deriving the formal trade-off surface between candidate recall and per-pair inference latency under strict human review constraints.
- **Required Experiments:** Full Pareto frontier extraction across 1,440 threshold pairs on $D_{\text{val}}$, budget sensitivity analysis ($B_{\text{max}} \in [0.5, 3.0]\text{ ms}$).

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Problem Definition
In academic integrity adjudication, numerical similarity scores are legally and pedagogically insufficient. Instructors and academic committees require interpretable, verifiable evidence detailing exactly which text spans, AST subtrees, or identifier substitutions constitute the suspected misconduct.

### 5.2 Modality-Specific Evidence Engines
Evidence extraction is executed **strictly on demand** when human review is required (`human_review_required = True`), avoiding unnecessary computation on cleared pairs.

#### A. Academic Text Evidence Engine:
- **Smith-Waterman Local Sequence Alignment with Affine Gap Penalty:**
  Tokens are aligned using recurrence:
  $$H_{i, j} = \max\left(0, \, H_{i-1, j-1} + s(a_i, b_j), \, E_{i, j}, \, F_{i, j}\right)$$
  Where gap opening penalty $u = 2.0$ and extension penalty $v = 0.5$.
- **Paraphrase Substitution Detection:** Aligns semantically similar token phrases that differ lexically.

#### B. Source Code Evidence Engine:
- **AST Subtree Isomorphism Matching:** Java source code is parsed into Abstract Syntax Trees (Tier 2 only). Subtree isomorphism under variable alpha-conversion maps cloned logic blocks.
- **Identifier Substitution Ledger:** Constructs an explicit mapping dictionary $\mathcal{M}_{\text{id}} = \{v_{\text{ref}} \leftrightarrow v_{\text{susp}}\}$ detailing systematic variable and method renaming.
- **Statement Permutation Detection:** Identifies reordered independent statements within basic blocks.

### 5.3 Strongly Typed Evidence Dossier
Outputs a structured, human-auditable `EvidenceDossier` containing:
- Operational state and routing action (`LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, `ABSTAIN_UNRESOLVED`).
- Calibrated posterior probability $\hat{p}$ and predictive entropy $H(\hat{p})$ (for heavy-evaluated pairs) or candidate score $s_{\text{cand}}$ (for Tier-1 direct flags).
- Conformal prediction set $\mathcal{C}_{1-\epsilon}$ (text only) or exploratory uncertainty status (code).
- Exact character/token offset spans of aligned matching regions.
- Detected transformations (paraphrases or identifier mappings).
- Synthesized human-readable natural language justification.

### 5.4 Explanation Quality Metrics:
1. **Evidence Faithfulness (Feature Ablation Drop):** $\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y=1 \mid \mathbf{x}) - \hat{P}(Y=1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$. Evaluated strictly on positive pairs escalated to Tier 2. (Assigned `None` for Tier-1 direct flags where heavy probability is bypassed).
2. **Evidence Coverage:** Fraction of tokens in the suspicious submission covered by identified evidence spans.
3. **Transformation Alignment Precision:** Fraction of detected identifier substitutions matching verified transformation labels.

### 5.5 Scholarly Categorization:
- **What is established:** Smith-Waterman local alignment (Smith & Waterman, 1981), AST subtree isomorphism (Baxter et al., 1998).
- **What is adapted:** Applying affine gap local alignment to tokenized natural language text; extracting systematic identifier alpha-conversions.
- **What is integrated:** On-demand dossier synthesis triggered strictly by triage routing decisions.
- **What may be novel:** Formalizing decision-probability ablation drop as a quantitative faithfulness metric for cascaded academic integrity evidence.
- **Required Experiments:** Faithfulness ablation study (A12), token coverage evaluation, computational overhead profiling of evidence extraction ($T_{\text{explain}}$).

---

## 6. Comprehensive Summary Matrix of the Four Contributions (Version 3.2)

| Contribution | Scholarly Category | Core Technique | Inputs | Outputs | Primary Evaluation Metrics | Key Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1: Multi-View Candidate Gen** | Proposed Integration | Fused 4-D Logistic Scorer (Sublinear TF-IDF, Char 3-Grams, Jaccard; Regex Code Views) | Submission pair $(D_A, D_B)$, Modality $\mathcal{M}$ | $s_{\text{cand}} \in [0, 1]$, Triage State (`DEFERRED`, `ESCALATED`, `HIGH_RISK_CANDIDATE_FLAGGED`) | Candidate Recall, Pruning Rate, Ingestion Latency ($T_{\text{Tier1}} \le 0.050\text{ ms}$) | Purely lexical views miss deep semantic abstractions without heavy models |
| **C2: Uncertainty-Aware Decisions** | Methodological Adaptation & Exploratory Routing | Selection-aware calibration (Platt/Isotonic); Text: Split Conformal 4-Set; Code: Exploratory Calibrated Routing | Escalated pair, Margin $z$, Calibration data $D_{\text{cal}}^{\text{esc}}$ | Calibrated $\hat{p}$, Conformal Set $\mathcal{C}$ (text only), Entropy $H(\hat{p})$, Review Action | Empirical Coverage, Set Size, ECE, Brier Score, Selective Accuracy | Code conformal disclaimed due to 15 reference programs in `case-05` |
| **C3: Budget-Constrained Optimization** | Proposed Integration | Recall-Constrained Grid Search Pareto Frontier Extraction | $D_{\text{val}}$, Cost Model, $R_{\text{target}}, B_{\text{max}}, P_{\text{target}}$ | Optimal frozen cutoffs $(\theta_{\text{low}}^*, \theta_{\text{high}}^*)$, Pareto Frontier | Candidate Recall vs. Latency Pareto Curve, Infeasibility Detection | Discrete grid search; re-tuning requires full pipeline re-calibration |
| **C4: Explainable Evidence** | Adaptation & Integration | Affine Smith-Waterman (Text); AST Isomorphism & Identifier Ledger (Code) | Flagged/Escalated Pair, Triage State, Prob $\hat{p}$, Modality $\mathcal{M}$ | Structured `EvidenceDossier` (Offsets, Mappings, Narrative) | Faithfulness Drop, Evidence Coverage, Alignment Precision, Overhead | High complexity for large ASTs; requires human adjudication |

---
*End of Document: four_contributions_formal_design_v3_2.md*

