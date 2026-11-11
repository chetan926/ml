# Formal Design of the Four Technical Contributions (Version 3.3)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.3 (Phase 2.3.2 Methodological Audit & Certification)  
**Author:** Senior Machine Learning Researcher, Statistical Methodologist & Independent Research Auditor  
**Date:** November 2026  
**Status:** Certified Final Architectural & Methodological Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals  
**Source Baseline:** Audited from Version 3.2 (`research_upgrade/documentation/phase_2_3_1/`)

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
- **Heavy Raw Margin ($z \in \mathbb{R}$):** Produced by the Tier-2 heavy detector (Sentence-BERT bi-encoder for text; 7-D structural classifier for source code). It is computed **strictly on escalated pairs** ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$). For bypassed pairs, $z = \text{None}$.
- **Calibrated Posterior Probability ($\hat{p} \in [0, 1]$):** Produced by a probability calibrator (Platt scaling or Isotonic regression) fitted on the conditional escalation population. For non-escalated pairs, $\hat{p} = \text{None}$.
- **Conformal Prediction Set ($\mathcal{C}_{1-\epsilon} \subseteq \{0, 1\}$):** Constructed from $\hat{p}$ and finite-sample order statistic quantile $\hat{q}_{1-\epsilon}$ on escalated pairs. **Applicable strictly to text pairs** where group-level sample size ($n \ge \lceil (1-\epsilon)/\epsilon \rceil$) is satisfied. For source code, conformal prediction is **explicitly disclaimed** due to sample size constraints ($N_{\text{ref}}=15$), and exploratory raw score routing is used instead. For non-escalated pairs, $\mathcal{C}_{1-\epsilon} = \text{None}$.

---

### 1.2 Four-Level Adjudication Semantics (Issue 4)

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

An algorithmic prediction (including a singleton conformal set $\{1\}$ or $\{0\}$) is an **algorithmic output**, not legal or institutional proof of misconduct or innocence.

---

### 1.3 Standardized Operational Output States and Modality Availability (Issue 5)

The triage cascade defines a common output schema across four standardized states, while transparently documenting modality-specific availability:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               STANDARDIZED OPERATIONAL OUTPUT STATES                                   │
├────┬───────────────────────────────┬───────────────────────────┬───────────────────────────────────────┤
│ #  │ Standardized State Name       │ Academic Text Modality    │ Source Code Modality                  │
├────┼───────────────────────────────┼───────────────────────────┼───────────────────────────────────────┤
│ 1  │ LOW_RISK_REVIEW_DEFERRED      │ Available (Tier 1 & 2)    │ Available (Tier 1 & 2)                │
│ 2  │ HIGH_RISK_CANDIDATE_FLAGGED   │ Available (Tier 1 & 2)    │ Available (Tier 1 & 2)                │
│ 3  │ AMBIGUOUS_HUMAN_REVIEW        │ Available (Tier 2, C={0,1})│ Available (Tier 2, Intermediate band) │
│ 4  │ ABSTAIN_UNRESOLVED            │ Available (Tier 2, C=Ø)   │ NOT APPLICABLE (Code uses 3 states)   │
└────┴───────────────────────────────┴───────────────────────────┴───────────────────────────────────────┘
```

1. **`LOW_RISK_REVIEW_DEFERRED` (Both Modalities):**
   - *Tier 1:* Similarity across all lightweight views is below cutoff ($s_{\text{cand}} < \theta_{\text{low}}$). Heavy inference is bypassed ($z = \text{None}, \hat{p} = \text{None}, \mathcal{C} = \text{None}$).
   - *Tier 2:* The heavy model and router determine similarity conforms to benign independent work ($\mathcal{C} = \{0\}$ for text, or raw score $z < \tau_{\text{low}}^{\text{raw}}$ for code).
   - *Action:* Formal human review is deferred (`human_review_required = False`). No evidence extraction is charged.
2. **`HIGH_RISK_CANDIDATE_FLAGGED` (Both Modalities):**
   - *Tier 1:* Multi-view lexical/token similarity exceeds cutoff ($s_{\text{cand}} \ge \theta_{\text{high}}$). Heavy inference is bypassed ($z = \text{None}, \hat{p} = \text{None}, \mathcal{C} = \text{None}$) to conserve compute budget.
   - *Tier 2:* Heavy model confirms high similarity ($\mathcal{C} = \{1\}$ for text, or $z \ge \tau_{\text{high}}^{\text{raw}}$ for code).
   - *Action:* Elevated triage signal requiring mandatory human committee review (`human_review_required = True`). An `EvidenceDossier` is generated. It does **not** constitute automated proof of plagiarism.
3. **`AMBIGUOUS_HUMAN_REVIEW` (Both Modalities):**
   - *Occurrence:* Conflicting or borderline signals ($\mathcal{C} = \{0, 1\}$ for text, or $\tau_{\text{low}}^{\text{raw}} \le z < \tau_{\text{high}}^{\text{raw}}$ for code).
   - *Action:* The model explicitly refrains from classification and requests human adjudication (`human_review_required = True`). An `EvidenceDossier` is generated.
4. **`ABSTAIN_UNRESOLVED` (Text Only):**
   - *Occurrence:* Conformal prediction set is empty ($\mathcal{C} = \emptyset$) for text, indicating the pair does not conform to either non-plagiarized or plagiarized calibration distributions at error rate $\epsilon$.
   - *Action:* Mandatory escalation to human audit (`human_review_required = True`).
   - *Modality Note:* Source code uncertainty routing operates on 3-band score thresholding and **does not produce `ABSTAIN_UNRESOLVED`**. We strictly refrain from inventing a synthetic fourth code state.

---

### 1.4 Selection-Aware Lifecycle Architecture (Issue 2)

All components operate under a single, non-contradictory experimental lifecycle:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE UNIFIED SEVEN-STEP EXECUTION LIFECYCLE                                │
├──────┬───────────────────────────────────┬───────────────────────────┬─────────────────────────────────┤
│ Step │ Operation                         │ Target Partition          │ Dataset Role                    │
├──────┼───────────────────────────────────┼───────────────────────────┼─────────────────────────────────┤
│ 1    │ Fit Candidate & Heavy Models      │ D_train                   │ Model fitting & representation  │
│ 2    │ Select Routing Thresholds         │ D_val                     │ Threshold optimization (Pareto) │
│ 3    │ FREEZE Routing Thresholds         │ (theta_low, theta_high)   │ Threshold policy locked         │
│ 4    │ Fit Probability Calibrators       │ D_cal_prob^esc            │ Calibration fitting (Text only) │
│ 5    │ Fit Conformal Quantile            │ D_cal_conf^esc            │ Conformal calibration (Text)    │
│ 6    │ FREEZE Complete Pipeline          │ All components            │ Entire cascade locked           │
│ 7    │ Final Independent Evaluation      │ D_test                    │ Final test & ECE on D_test^esc  │
└──────┴───────────────────────────────────┴───────────────────────────┴─────────────────────────────────┘
```

The untouched test set $D_{\text{test}}$ **never influences threshold selection, model fitting, calibration fitting, or hyperparameter tuning**. Calibration diagnostics (ECE, Brier score, NLL) are computed strictly on the untouched test escalation slice $D_{\text{test}}^{\text{esc}}$, where heavy predictions genuinely exist.

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Problem Definition
Traditional pre-filters rely on single similarity heuristics (e.g. TF-IDF or token Jaccard) that fail under paraphrasing or identifier renaming. Conversely, running heavy models across all $O(M^2)$ pairs in a cohort of $M$ students creates severe computational bottlenecks.

### 2.2 Multi-View Feature Representations (Purely Lightweight Tier 1)
Restricted strictly to $O(|D|)$ string and regex operations ($\le 0.050\text{ ms/pair}$). **Heavy AST parsing is strictly prohibited at Tier 1.**

- **Text Views ($\mathbf{v}_{\text{text}} \in [0, 1]^4$):**
  1. Sublinear TF-IDF cosine ($v_1$).
  2. Character 3-gram set Jaccard ($v_2$).
  3. Word token set Jaccard ($v_3$).
  4. Length-ratio symmetry metric ($v_4$).
- **Code Views ($\mathbf{u}_{\text{code}} \in [0, 1]^4$):**
  1. Raw token set Jaccard ($u_1$).
  2. Normalized identifier token set Jaccard ($u_2$).
  3. Operator frequency profile cosine ($u_3$).
  4. Regex cyclomatic keyword differential ($u_4$).

### 2.3 Scholarly Categorization:
- **What is established:** TF-IDF (Salton & Buckley, 1988), Character 3-grams, Jaccard token overlap, Regex keyword tokenization.
- **What is adapted:** Applying regex identifier normalization and operator frequency profiles as ultra-fast code candidate views without AST generation.
- **What is integrated:** Fused logistic candidate scoring model $g_{\text{cand}}$ trained on $D_{\text{train}}$.
- **What may be novel:** Empirical validation that a 4-D multi-view candidate generator retains $\ge 0.980$ recall on zero-overlap paraphrased academic text pairs while maintaining sub-0.050 ms ingestion latency.
- **Required Experiments:** View ablation studies (A1–A4), VIF redundancy audit, recall-pruning trade-off curves.

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Problem Definition
Standard classifiers output uncalibrated scores that are overconfident on borderline cases, and point predictions force binary decisions even under high epistemic or predictive uncertainty.

### 3.2 Text Modality: Group-Level Conformal Formulation (Issue 1)
- **Prediction Unit:** Individual student submission pair $(D_A^{(i)}, D_B^{(i)})$.
- **Grouping Unit:** Reference premise / source document identity $g \in \mathcal{G}$.
- **Group Non-Conformity Score:** For group $g$ containing escalated pairs:
  $$\alpha_g = \max_{i \in g, \, \text{escalated}} \left( 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i) \right)$$
- **Exact Finite-Sample Order Statistic:** For $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ independent calibration groups:
  $$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$
  If $k > n$ (i.e. $n < \lceil (1-\epsilon)/\epsilon \rceil$), the quantile is set to $\infty$, and the system explicitly abstains from making a finite-sample coverage guarantee.
- **Guaranteed Coverage Scope:** Under group-exchangeability between calibration and test groups:
  $$\mathbb{P}\left( \forall i \in g_{N+1}, \, y_i \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_i) \mid \text{Escalated} \right) \ge 1 - \epsilon$$
  Since $\alpha_i \le \alpha_g$, pair-level coverage $\mathbb{P}(y_i \in \mathcal{C}_{1-\epsilon}(\mathbf{x}_i) \mid \text{Escalated}) \ge 1 - \epsilon$ is also guaranteed, but is **conservative** (over-coverage for individual pairs).
- **Disclaimers:** Does not guarantee individual-conditional coverage, coverage under distribution shift, or plagiarism recall.

### 3.3 Code Modality: Exploratory Score Thresholding (Issue 3)
- **Audit Finding:** In IR-Plag `case-05`, all 53 plagiarized pairs originate from 1 base program (`original`). A 5-fold group-aware split leaves Fold 1 with **0 positive training pairs** and Folds 2–5 with **0 positive held-out pairs**.
- **Correction:** Forced 5-fold Platt scaling is **strictly rejected**. Code probability calibration is downgraded to **exploratory score thresholding** on raw decision margins $z$.
- **Operational Rules:**
  - $z < \tau_{\text{low}}^{\text{raw}} \implies$ `LOW_RISK_REVIEW_DEFERRED` (`review = False`).
  - $z \ge \tau_{\text{high}}^{\text{raw}} \implies$ `HIGH_RISK_CANDIDATE_FLAGGED` (`review = True`).
  - $\tau_{\text{low}}^{\text{raw}} \le z < \tau_{\text{high}}^{\text{raw}} \implies$ `AMBIGUOUS_HUMAN_REVIEW` (`review = True`).
- **Disclaimers:** Conformal guarantees are formally disclaimed. Raw margins or RF vote fractions are never described as calibrated probabilities.

---

## 4. Contribution 3: Budget-Constrained Routing Policy Optimization

### 4.1 Problem Definition
Heuristic routing cutoffs cause latency spikes or recall drops. We formalize threshold selection as a constrained multi-objective optimization problem evaluated over a discretized grid on $D_{\text{val}}$.

### 4.2 Mathematical Optimization Formulation
Let $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$ with $\theta_{\text{low}} < \theta_{\text{high}}$.
$$\min_{\boldsymbol{\theta}} \quad \bar{T}_{\text{total}}(\boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
g_1(\boldsymbol{\theta}) &:= R_{\text{target}} - \text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) \le 0 \\
g_2(\boldsymbol{\theta}) &:= \bar{T}_{\text{total}}(\boldsymbol{\theta}) - B_{\text{max}} \le 0 \\
g_3(\boldsymbol{\theta}) &:= P_{\text{target}} - \text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) \le 0
\end{aligned}$$

- **Honest Terminology:** Discretized grid-search Pareto frontier extraction, NOT a novel continuous Lagrangian optimizer.
- **Infeasibility State:** Returns `STATUS: INFEASIBLE_BUDGET` if no thresholds satisfy constraints.
- **Freezing Invariant:** Thresholds are frozen on $D_{\text{val}}$ before calibration; re-tuning requires full pipeline re-calibration.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Problem Definition
Numerical similarity scores do not provide legally or pedagogically defensible evidence. Committees require verifiable token spans, AST subtrees, and identifier mapping ledgers.

### 5.2 Modality-Specific Evidence Extraction (On-Demand Only)
Evidence extraction is charged **strictly when human review is required** (`human_review_required = True`). Cleared pairs bypass evidence extraction.
- **Text:** Smith-Waterman local alignment with affine gap penalty ($u = 2.0, v = 0.5$).
- **Code:** AST subtree isomorphism under variable alpha-conversion and identifier substitution mapping dictionary $\mathcal{M}_{\text{id}}$.
- **Faithfulness Metric:** Feature ablation probability drop $\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y=1 \mid \mathbf{x}) - \hat{P}(Y=1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$ (evaluated strictly on escalated true positive pairs; `None` for Tier-1 direct flags).

---

## 6. Comprehensive Summary Matrix of the Four Contributions (Version 3.3)

| Contribution | Scholarly Category | Core Technique | Inputs | Outputs | Primary Evaluation Metrics | Key Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1: Multi-View Candidate Gen** | Proposed Integration | Fused 4-D Logistic Scorer (Sublinear TF-IDF, Char 3-Grams, Jaccard; Regex Code Views) | Submission pair $(D_A, D_B)$, Modality $\mathcal{M}$ | $s_{\text{cand}} \in [0, 1]$, Triage State (`DEFERRED`, `ESCALATED`, `HIGH_RISK_CANDIDATE_FLAGGED`) | Candidate Recall, Pruning Rate, Ingestion Latency ($T_{\text{Tier1}} \le 0.050\text{ ms}$) | Purely lexical views miss deep semantic abstractions without heavy models |
| **C2: Uncertainty-Aware Decisions** | Methodological Adaptation & Exploratory Routing | Selection-aware calibration (Platt/Isotonic); Text: Group Conformal 4-Set; Code: Exploratory Raw Score Routing | Escalated pair, Margin $z$, Calibration data $D_{\text{cal}}^{\text{esc}}$ | Calibrated $\hat{p}$ (Text), Conformal Set $\mathcal{C}$ (Text), Raw Margin $z$ (Code), Review Action | Empirical Coverage, Set Size, ECE, Brier Score, Selective Accuracy | Code conformal disclaimed due to 15 reference programs; Platt cross-fitting rejected |
| **C3: Budget-Constrained Optimization** | Proposed Integration | Recall-Constrained Grid Search Pareto Frontier Extraction | $D_{\text{val}}$, Cost Model, $R_{\text{target}}, B_{\text{max}}, P_{\text{target}}$ | Optimal frozen cutoffs $(\theta_{\text{low}}^*, \theta_{\text{high}}^*)$, Pareto Frontier | Candidate Recall vs. Latency Pareto Curve, Infeasibility Detection | Discrete grid search; re-tuning requires full pipeline re-calibration |
| **C4: Explainable Evidence** | Adaptation & Integration | Affine Smith-Waterman (Text); AST Isomorphism & Identifier Ledger (Code) | Flagged/Escalated Pair, Triage State, Prob $\hat{p}$, Modality $\mathcal{M}$ | Structured `EvidenceDossier` (Offsets, Mappings, Narrative) | Faithfulness Drop, Evidence Coverage, Alignment Precision, Overhead | High complexity for large ASTs; requires human adjudication |

---
*End of Document: four_contributions_formal_design_v3_3.md*

