# Formal Design of the Four Technical Contributions (Version 3.1)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.1 (Phase 2.3 Methodological Consistency & Implementation Readiness)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Finalized Design Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. System Overview & Selection-Aware Triage Architecture

The proposed framework replaces uncalibrated single-view similarity filters and brute-force transformer evaluations with a mathematically sound, uncertainty-aware, budget-constrained, and explainable multi-stage triage cascade.

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
- **Heavy Raw Margin ($z \in \mathbb{R}$):** Produced by the Tier-2 heavy detector (Sentence-BERT bi-encoder for text; 7-D AST structural model for source code). It is computed **strictly on escalated pairs** ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$). For pruned or directly flagged pairs, $z = \text{None}$.
- **Calibrated Posterior Probability ($\hat{p} \in [0, 1]$):** Produced by a probability calibrator (Platt scaling or Isotonic regression) fitted on the conditional escalation population. For non-escalated pairs, $\hat{p} = \text{None}$.
- **Conformal Prediction Set ($\mathcal{C}_{1-\epsilon} \subseteq \{0, 1\}$):** Constructed from $\hat{p}$ and empirical quantile $\hat{q}_{1-\epsilon}$ on escalated pairs. **Applicable strictly to text pairs** where exchangeability and calibration sample size are satisfied. For source code, conformal prediction is **explicitly disclaimed** due to sample size constraints, and exploratory calibrated routing is used instead. For non-escalated pairs, $\mathcal{C}_{1-\epsilon} = \text{None}$.

---

### 1.2 Selection-Aware Escalation Architecture

A central methodological requirement of multi-stage cascades is acknowledging the **selection problem**: the heavy classifier, probability calibrator, and conformal router do not observe the marginal population of all submission pairs. They operate strictly on the **conditional escalation slice**:
$$\mathcal{D}_{\text{esc}}(\boldsymbol{\theta}) = \left\{ (D_A, D_B, y) \in \mathcal{D} \mid \theta_{\text{low}} \le s_{\text{cand}}(D_A, D_B) < \theta_{\text{high}} \right\}$$

To ensure statistical and empirical validity:
1. **Model Fitting ($D_{\text{train}}$):** The candidate generator $g_{\text{cand}}$ and heavy models $f_{\text{heavy}}$ are fitted strictly on base training data $D_{\text{train}}$.
2. **Threshold Selection & Freezing ($D_{\text{val}}$):** Routing thresholds $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}})$ are selected on validation data $D_{\text{val}}$ to minimize computational cost subject to candidate recall and direct precision constraints. Once selected, **thresholds are frozen**.
3. **Probability Calibration ($D_{\text{cal-prob}}^{\text{esc}}$):** The frozen policy is applied to $D_{\text{cal-prob}}$, yielding escalated slice $D_{\text{cal-prob}}^{\text{esc}}$. The probability calibrator $g_{\text{cal}}$ is fitted strictly on $D_{\text{cal-prob}}^{\text{esc}}$. Calibration quality (ECE, Brier score) is evaluated on independent validation data, never on the fitting split.
4. **Conformal Calibration ($D_{\text{cal-conf}}^{\text{esc}}$, Text Only):** The identical frozen policy is applied to $D_{\text{cal-conf}}$, yielding escalated slice $D_{\text{cal-conf}}^{\text{esc}}$. The conformal non-conformity quantile $\hat{q}_{1-\epsilon}$ is estimated strictly on $D_{\text{cal-conf}}^{\text{esc}}$.
5. **Frozen Evaluation ($D_{\text{test}}$):** The entire frozen pipeline is evaluated once on the untouched held-out test split $D_{\text{test}}$. Any adjustment to routing thresholds invalidates the calibration and conformal slices, requiring complete re-calibration.

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
         │   LOW_RISK_PRUNED     │                     │                     │  HIGH_RISK_CANDIDATE_ │
         │   • Pruned early      │                     │                     │        FLAGGED        │
         │   • Bypasses Heavy    │                     │                     │   • Triage signal     │
         │   • Heavy = None      │                     │                     │   • Bypasses Heavy    │
         │   • Review = False    │                     │                     │   • Heavy = None      │
         └───────────────────────┘                     │                     │   • Review = True     │
                                                       │                     │   • Dossier Generated │
                                                       ▼                     └───────────────────────┘
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
         │ CLEARED_NON_PLAGIAR   │         │ CONFIRMED_PLAGIARIZED │         │ AMBIGUOUS / ABSTAIN   │
         │ • Negative prediction │         │ • Model positive flag │         │ • Predictive conflict │
         │ • Review = False      │         │ • Review = True       │         │ • High uncertainty    │
         │ • No evidence charged │         │ • Dossier Generated   │         │ • Review = True       │
         │                       │         │                       │         │ • Dossier Generated   │
         └───────────────────────┘         └───────────────────────┘         └───────────────────────┘
```

---

### 1.3 Operational Semantics of Triage States

The triage cascade outputs strictly typed operational states with precise operational meanings:

1. **`LOW_RISK_PRUNED` (Tier 1):** The pair exhibits negligible similarity across all lightweight views ($s_{\text{cand}} < \theta_{\text{low}}$). It is pruned from further computation, bypassing Tier-2 inference. Heavy margin, calibrated probability, and conformal prediction set are assigned `None`. `human_review_required = False`. No evidence extraction is performed.
2. **`HIGH_RISK_CANDIDATE_FLAGGED` (Tier 1):** The pair exhibits extreme multi-view lexical/token similarity ($s_{\text{cand}} \ge \theta_{\text{high}}$). 
   - **Triage Signal Semantics:** This state indicates a strong preliminary similarity trigger. It is **NOT proof of academic misconduct or verbatim plagiarism**.
   - **Human Review Mandate:** Adjudication requires human inspection (`human_review_required = True`).
   - **Computation Bypass:** Heavy classifier inference is bypassed to conserve compute budget; heavy margin, calibrated probability, and conformal set are assigned `None`.
   - **Evidence Generation:** An `EvidenceDossier` is generated containing token alignments and matching spans to assist human evaluators.
3. **`UNCERTAIN_ESCALATED` (Tier 1 $\to$ Tier 2):** The pair falls into the intermediate ambiguity window ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$). It is escalated to Tier-2 heavy analysis and routed through Contribution 2.

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Problem Definition
Traditional academic integrity pipelines face a severe dilemma:
- **Lexical/string filters** (e.g. TF-IDF or character sets) fail when submissions undergo paraphrasing, synonym substitution, or structural reorganization (e.g. the 119 zero-overlap paraphrased pairs in the text corpus).
- **Deep semantic models and AST parsers** provide high recall but incur prohibitive computational costs ($2.05\text{ ms}$ for AST parsing, $4.05\text{ ms}$ for SBERT inference) when executed on all $O(M^2)$ pairwise comparisons in a cohort of $M$ student submissions.

### 2.2 Multi-View Feature Representations

To combine orthogonal similarity signals without heavy feature extraction overhead, Tier-1 extractors are restricted strictly to $O(|D|)$ string and regex operations ($\le 0.045\text{ ms/pair}$). **Heavy AST parsing is strictly prohibited at Tier 1.**

#### A. Academic Text Multi-View Vector $\mathbf{v}_{\text{text}} \in [0, 1]^4$:
1. **Sublinear TF-IDF Cosine Similarity ($v_1$):** Measures lexical keyword overlap with sublinear term-frequency dampening ($1 + \log(\text{tf})$) and inverse document frequency fitted strictly on $D_{\text{train}}$. Captures shared domain-specific terminology.
2. **Character 3-Gram Set Jaccard Overlap ($v_2$):** Measures substring overlap at the sub-word level. Robust against morphological alterations, minor spelling variations, and compounding.
3. **Word Token Set Jaccard Overlap ($v_3$):** Measures bag-of-words token co-occurrence irrespective of word order, capturing lexical overlap across sentence rearrangements.
4. **Length-Ratio Symmetry Metric ($v_4$):** Defined as $\frac{\min(|D_A|_{\text{words}}, |D_B|_{\text{words}})}{\max(|D_A|_{\text{words}}, |D_B|_{\text{words}}, 1)}$. Discriminated against partial or heavily padded comparisons.

#### B. Source Code Multi-View Vector $\mathbf{u}_{\text{code}} \in [0, 1]^4$ (Pure Regex/Lexer Tier 1):
1. **Raw Token Set Jaccard Overlap ($u_1$):** Lexical token overlap after stripping Java comments and whitespace. Fast detection of direct code sharing.
2. **Normalized Identifier Token Set Jaccard ($u_2$):** All user identifiers mapped to canonical token `VAR_TOKEN` using regular expressions. Invariant to variable, method, and parameter renaming.
3. **Operator Frequency Profile Cosine ($u_3$):** Frequency vector across Java operators (`+`, `-`, `*`, `/`, `&&`, `||`, `==`, `!=`, `<`, `>`, `++`, `--`, `+=`). Captures algorithmic logic signature without parsing syntax trees.
4. **Cyclomatic Complexity Proxy Differential ($u_4$):** Regex count of decision keywords (`if`, `else`, `while`, `for`, `case`, `catch`, `?`) normalized as $1.0 - \frac{|k_A - k_B|}{\max(k_A, k_B, 1)}$. Captures structural control-flow complexity.

### 2.3 Candidate Scorer Formulation
The multi-view representation is mapped to a candidate score $s_{\text{cand}} \in [0, 1]$ via a logistic candidate model trained strictly on $D_{\text{train}}$:
$$s_{\text{cand}}(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^T \mathbf{x} + b))}$$
- Feature collinearity is verified via Variance Inflation Factor ($\text{VIF} < 5.0$) on $D_{\text{train}}$.
- Individual feature ablation experiments isolate the marginal recall contribution of each view.

### 2.4 Evaluation Metrics
- **Candidate Recall ($R_{\text{cand}}$):** $\frac{\sum_{i} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{low}})}{\sum_{i} y_i}$ (Target: $\ge 0.980$).
- **Pruning Rate ($P_{\text{prune}}$):** Fraction of pairs classified as `LOW_RISK_PRUNED` ($s_{\text{cand}} < \theta_{\text{low}}$).
- **Direct Candidate Flag Rate ($P_{\text{flag}}$):** Fraction classified as `HIGH_RISK_CANDIDATE_FLAGGED` ($s_{\text{cand}} \ge \theta_{\text{high}}$).
- **Escalation Rate ($P_{\text{esc}}$):** Fraction requiring Tier-2 heavy analysis ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$).

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decisions

### 3.1 Problem Definition
Standard deep learning and ensemble classifiers output uncalibrated scores that are frequently overconfident on borderline cases. Furthermore, point predictions force binary classifications ($0$ or $1$) even when the model possesses high epistemic or predictive uncertainty.

### 3.2 Probability Calibration on Conditional Escalation Slice
Because the heavy classifier evaluates only escalated instances, calibrators fitted on the unselected marginal dataset suffer from severe distribution shift.
- The calibrator $g_{\text{cal}}$ is fitted on raw heavy margins $z$ strictly within the conditional escalation split:
  $$D_{\text{cal-prob}}^{\text{esc}} = \{(\mathbf{x}_i, y_i) \in D_{\text{cal-prob}} \mid \theta_{\text{low}} \le s_{\text{cand}}(\mathbf{x}_i) < \theta_{\text{high}}\}$$
- **Calibrator Candidates:**
  1. *Platt Scaling (Parametric):* $\hat{P}(Y=1 \mid z) = \sigma(A \cdot z + B)$, where parameters $A, B$ minimize cross-entropy loss.
  2. *Isotonic Regression (Non-parametric):* Fits a non-decreasing step function via the Pair-Adjacent Violators (PAV) algorithm. Preferred when sample size is large and non-sigmoid miscalibration is present.
- **Independent Calibration Diagnostics:** Expected Calibration Error (ECE) and Brier score are evaluated on an independent partition, never on $D_{\text{cal-prob}}$.

---

### 3.3 Modality-Specific Uncertainty Routing

Top-tier methodological rigor requires distinct treatment for text and source code based on available sample sizes and grouping constraints.

#### Modality A: Academic Text (Formal Split Conformal Prediction)
- **Calibration Split ($D_{\text{cal-conf}}^{\text{esc}}$):** Disjoint from $D_{\text{train}}$, $D_{\text{val}}$, and $D_{\text{cal-prob}}$, partitioned at the premise group level.
- **Non-Conformity Score:** For pair $\mathbf{x}_i$ with ground truth $y_i \in \{0, 1\}$:
  $$\alpha_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$$
- **Conformal Quantile:** For user significance level $\epsilon \in (0, 1)$ (e.g. $\epsilon = 0.05$ for $95\%$ coverage):
  $$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (N_{\text{cal-conf}}^{\text{esc}} + 1)(1 - \epsilon) \rceil}{N_{\text{cal-conf}}^{\text{esc}}}, \, \{\alpha_i\}_{i=1}^{N_{\text{cal-conf}}^{\text{esc}}} \right)$$
- **Conformal Prediction Set Construction:**
  $$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}) \le \hat{q}_{1-\epsilon}\}$$
- **Rigorous Scope of the Conformal Guarantee:**
  - *What is guaranteed:* Marginal finite-sample coverage over the conditional escalation population:
    $$P\left( Y \in \mathcal{C}_{1-\epsilon}(X) \mid \text{Escalated} \right) \ge 1 - \epsilon$$
  - *What is NOT guaranteed:* It does NOT guarantee conditional coverage for specific individual instances. It does NOT guarantee coverage under out-of-distribution domain shift. It does NOT guarantee plagiarism recall $\ge 1 - \epsilon$ (since a non-plagiarized true label $Y=0$ is covered by $\{0\}$).
- **Complete Four-Set Operational Routing:**
  - $\mathcal{C}_{1-\epsilon} = \{0\}$: Model-supported negative $\implies$ `CLEARED_NON_PLAGIARIZED` (`human_review_required = False`).
  - $\mathcal{C}_{1-\epsilon} = \{1\}$: Model-supported positive $\implies$ `CONFIRMED_PLAGIARIZED` (`human_review_required = True`).
  - $\mathcal{C}_{1-\epsilon} = \{0, 1\}$: Ambiguous / High Predictive Uncertainty $\implies$ `AMBIGUOUS_HUMAN_REVIEW` (`human_review_required = True`).
  - $\mathcal{C}_{1-\epsilon} = \emptyset$: Empty prediction set $\implies$ `ABSTAIN_UNRESOLVED` (`human_review_required = True`). (Indicates neither class conforms to calibration data; treated as mandatory abstention, not formal OOD detection).

#### Modality B: Source Code (Exploratory Calibrated Uncertainty Routing)
- **Empirical Constraint:** The IR-Plag dataset has only 15 unique reference Java programs in `case-05`. This sample size is mathematically insufficient to support a valid finite-sample conformal quantile at standard error rates ($N \ge \lceil (1-\epsilon)/\epsilon \rceil$).
- **Methodological Disclaimer:** We **strictly disclaim** mathematical conformal coverage guarantees for source code.
- **Exploratory Calibrated Routing Mechanism:**
  - The heavy code model produces calibrated probability $\hat{p} = g_{\text{cal}}(z)$ using Platt scaling fitted on exploratory code calibration data.
  - Predictive entropy is computed: $H(\hat{p}) = -\hat{p}\log_2 \hat{p} - (1-\hat{p})\log_2(1-\hat{p})$.
  - Decision bands are established:
    - $\hat{p} < \tau_{\text{non-plag}}$ (e.g. $0.20$): `CLEARED_NON_PLAGIARIZED` (`human_review_required = False`).
    - $\hat{p} \ge \tau_{\text{plag}}$ (e.g. $0.80$): `CONFIRMED_PLAGIARIZED` (`human_review_required = True`).
    - $\tau_{\text{non-plag}} \le \hat{p} < \tau_{\text{plag}}$ OR $H(\hat{p}) > \tau_{\text{entropy}}$: `AMBIGUOUS_HUMAN_REVIEW` (`human_review_required = True`).
  - For source code records, `conformal_prediction_set` is set to `None (Disclaimed for Code)`.
  - The limited number of independent calibration groups is explicitly documented as a major experimental limitation.

---

## 4. Contribution 3: Budget-Constrained Routing Policy Optimization

### 4.1 Problem Definition
Existing multi-stage systems choose routing thresholds heuristically, leading to unpredictable latency spikes or unacceptable candidate recall drops. We formalize threshold selection as a constrained multi-objective optimization problem evaluated over a discretized grid.

### 4.2 Mathematical Optimization Formulation

Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with the strict ordering constraint $\theta_{\text{low}} < \theta_{\text{high}}$.
The optimization problem evaluated on validation data $D_{\text{val}}$ is:
$$\min_{\boldsymbol{\theta}} \quad \bar{C}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \bar{C}_{\text{total}}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

Where:
- $\text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{low}})}{\sum_{i \in D_{\text{val}}} y_i}$
- $\text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) = \frac{\sum_{i \in D_{\text{val}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}})}{\sum_{i \in D_{\text{val}}} \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{high}}) + 10^{-7}}$
- $R_{\text{target}}$: User-specified candidate recall bound (e.g. $0.980$).
- $B_{\text{max}}$: Per-pair computational latency budget (e.g. $1.50\text{ ms/pair}$).
- $P_{\text{target}}$: Minimum precision required for Tier-1 direct triage flags (e.g. $0.900$).

### 4.3 Realistic Optimization Procedure: Pareto Surface Search
- **Honest Algorithmic Terminology:** The optimization procedure is a **discretized grid-search Pareto frontier extraction**, NOT an unconstrained continuous Lagrangian optimizer. We evaluate a 2D grid ($\theta_{\text{low}} \in [0.01, 0.40]$, $\theta_{\text{high}} \in [0.60, 0.95]$ with step size $0.01$).
- **Infeasibility Handling:** If no policy satisfies all three constraints simultaneously, the optimizer returns `STATUS: INFEASIBLE_BUDGET`. Arbitrary fallback thresholds are prohibited.
- **Strict Freezing Sequence:** The optimal policy $\boldsymbol{\theta}^*$ is chosen on $D_{\text{val}}$ and **frozen**. Changing thresholds invalidates conditional escalation calibration and requires complete re-calibration.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Problem Definition
In high-stakes academic integrity adjudication, opaque numerical scores are legally and pedagogically insufficient. Instructors and academic committees require interpretable, verifiable evidence detailing exactly which text spans, AST subtrees, or identifier substitutions constitute the suspected misconduct.

### 5.2 Modality-Specific Evidence Engines

Evidence extraction is executed **strictly on demand** when human review is required (`human_review_required = True`), avoiding unnecessary computation on cleared pairs.

#### A. Academic Text Evidence Engine:
- **Smith-Waterman Local Sequence Alignment with Affine Gap Penalty:**
  Tokens are aligned using recurrence:
  $$H_{i, j} = \max\left(0, \, H_{i-1, j-1} + s(a_i, b_j), \, E_{i, j}, \, F_{i, j}\right)$$
  Where $E_{i, j}$ and $F_{i, j}$ track gap opening ($u = 2.0$) and gap extension ($v = 0.5$).
- **Paraphrase Substitution Detection:** Aligns semantically similar token phrases that differ lexically, identifying synonym replacements.

#### B. Source Code Evidence Engine:
- **AST Subtree Isomorphism Matching:** Java source code is parsed into Abstract Syntax Trees (Tier 2 only). Subtree isomorphism under variable alpha-conversion maps cloned logic blocks.
- **Identifier Substitution Ledger:** Constructs an explicit mapping dictionary $\mathcal{M}_{\text{id}} = \{v_{\text{ref}} \leftrightarrow v_{\text{susp}}\}$ detailing systematic variable and method renaming.
- **Statement Permutation Detection:** Identifies reordered independent statements within basic blocks.

### 5.3 Strongly Typed Evidence Dossier

Outputs a structured, human-auditable `EvidenceDossier` containing:
- Operational state and routing action.
- Calibrated posterior probability $\hat{p}$ and predictive entropy $H(\hat{p})$ (for heavy-evaluated pairs) or candidate score $s_{\text{cand}}$ (for Tier-1 direct flags).
- Conformal prediction set $\mathcal{C}_{1-\epsilon}$ (text only) or exploratory uncertainty status (code).
- Exact character/token offset spans of aligned matching regions.
- Detected transformations (paraphrases or identifier mappings).
- Synthesized human-readable natural language justification.

### 5.4 Explanation Quality Metrics
1. **Evidence Faithfulness (Feature Ablation Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y=1 \mid \mathbf{x}) - \hat{P}(Y=1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   Evaluated strictly on positive pairs escalated to Tier 2. (Not applicable to Tier-1 direct flags where heavy probability is bypassed).
2. **Evidence Coverage:** Fraction of tokens in the suspicious submission covered by identified evidence spans.
3. **Transformation Alignment Precision:** Fraction of detected identifier substitutions matching verified transformation labels.
4. **Human Review Mandate:** Evidence dossiers are designed to assist human decision-makers. They do not constitute automated verdicts of academic guilt.

---

## 6. Comprehensive Cross-Contribution Summary Matrix

| Contribution | Methodological Category | Core Technique | Inputs | Outputs | Primary Evaluation Metrics | Key Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1: Robust Multi-View Candidate Gen** | Proposed Integration | Fused 4-D Logistic Scorer (Sublinear TF-IDF, Char 3-Grams, Jaccard; Regex Code Views) | Raw submission pair $(D_A, D_B)$, Modality $\mathcal{M}$ | $s_{\text{cand}} \in [0, 1]$, Triage State (`PRUNED`, `ESCALATED`, `HIGH_RISK_CANDIDATE_FLAGGED`) | Candidate Recall, Pruning Rate, Ingestion Latency ($< 0.05\text{ ms}$) | Lexical views miss novel structural abstractions |
| **C2: Uncertainty-Aware Adaptive Decisions** | Methodological Adaptation & Exploratory Routing | Selection-aware calibration (Platt/Isotonic); Text: Split Conformal 4-Set; Code: Exploratory Calibrated Routing | Escalated pair, Raw margin $z$, Calibration data $D_{\text{cal}}^{\text{esc}}$ | Calibrated $\hat{p}$, Conformal Set $\mathcal{C}$ (text only), Entropy $H(\hat{p})$, Review Action | Empirical Coverage, Set Size, ECE, Brier Score, Selective Accuracy | Code conformal disclaimed due to 15-group constraint |
| **C3: Budget-Constrained Routing Optimization** | Proposed Integration | Recall-Constrained Grid Search Pareto Frontier Extraction | $D_{\text{val}}$, Cost Model, $R_{\text{target}}, B_{\text{max}}, P_{\text{target}}$ | Optimal frozen thresholds $(\theta_{\text{low}}^*, \theta_{\text{high}}^*)$, Pareto Frontier | Candidate Recall vs. Latency Pareto Curve, Infeasibility Detection | Discrete grid search; re-tuning requires full pipeline re-calibration |
| **C4: Explainable Cross-Modal Evidence** | Adaptation & Integration | Affine Smith-Waterman (Text); AST Isomorphism & Identifier Ledger (Code) | Flagged/Escalated Pair, Triage State, Prob $\hat{p}$, Modality $\mathcal{M}$ | Structured `EvidenceDossier` (Offsets, Mappings, Narrative) | Faithfulness Drop, Evidence Coverage, Alignment Precision, Overhead | High complexity for large ASTs; requires human adjudication |

---
*End of Document: four_contributions_formal_design_v3_1.md*

