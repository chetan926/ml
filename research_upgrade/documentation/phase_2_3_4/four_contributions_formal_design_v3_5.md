# Four Research Contributions: Formal Architectural & Methodological Specification (Version 3.5)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.4 (Final Independent Certification and Phase 3 Readiness Gate)  
**Document Version:** Version 3.5  
**Date:** November 10, 2026  
**Status:** INDEPENDENTLY AUDITED & CERTIFIED (Reproducible Artifacts Verified)

---

## 1. Overview and Core Research Objective

The objective of this research is to resolve the fundamental trade-off between computational cost and detection reliability in large-scale academic integrity systems. Modern academic cohorts generate pairwise comparison spaces scaling quadratically ($O(M^2)$ for $M$ student submissions). Existing systems either rely on brittle single-view heuristic filters that suffer severe candidate recall degradation on paraphrased text or transformed source code, or deploy expensive deep transformers uniformly across all pairs, resulting in prohibitive computational latency and energy waste.

This framework introduces a principled, two-stage architecture that integrates:
1. **Multi-View Candidate Screening** across lexical, semantic, syntactic, and structural dimensions.
2. **Uncertainty-Aware Adaptive Escalation** using probability calibration and group-level split conformal prediction.
3. **Multi-Objective Budget-Constrained Optimization** operating over a strict Pareto frontier.
4. **Faithful Multi-Modal Attribution Evidence** separating automated statistical signals from institutional adjudication.

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 The Paraphrase and Obfuscation Failure Modes
Single-view similarity metrics suffer catastrophic false-negative rates when confronting semantic paraphrase or structural code obfuscation:
- **Academic Text:** Empirical analysis of `datasets/processed/text_plagiarism_benchmark.csv` reveals that **119 true positive candidate pairs** ($y=1$) share **zero lexical word overlap** (Jaccard similarity $= 0.0$). A purely lexical filter (such as BM25 or word n-grams) set to prune candidates at similarity $> 0$ misses 100% of these positive instances, causing candidate recall to drop by $\sim 9.0\%$ on the positive slice.
- **Academic Source Code:** In the IR-Plag corpus (`datasets/raw/ir_plag/extracted/IR-Plag-Dataset/`), obfuscation levels L1–L6 systematically neutralize token-level matching through identifier renaming, loop restructuring, statement reordering, and dead-code injection.

### 2.2 Text Multi-View Architecture
For any text candidate pair $(D_A, D_B)$, Stage 1 extracts four complementary representation views:
1. **Character n-gram TF-IDF View ($\mathbf{v}_{\text{char}}$):** Substring n-grams ($n \in \{3, 4, 5\}$) capturing morphological stem overlap and sub-word borrowings robust to inflectional shifts.
2. **Word n-gram TF-IDF View ($\mathbf{v}_{\text{word}}$):** Lexical token n-grams ($n \in \{1, 2\}$) weighted via BM25/TF-IDF scoring capturing verbatim phrase preservation.
3. **Dense Semantic Embedding View ($\mathbf{v}_{\text{sem}}$):** 384-dimensional dense vectors generated via `all-MiniLM-L6-v2` capturing latent propositional semantics invariant to lexical substitution.
4. **Linguistic Surface Complexity View ($\mathbf{v}_{\text{ling}}$):** Difference vectors over sentence length, type-token ratio (TTR), and readability metrics.

The multi-view similarity score is computed via a linear fusion model:
$$s_{\text{cand}}^{\text{text}}(D_A, D_B) = \sigma\left( \mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(D_A, D_B) + b \right)$$
where fusion weights $\mathbf{w}$ are optimized on the validation partition $D_{\text{val}}$ to maximize candidate recall under the budget constraint.

### 2.3 Source Code Multi-View Architecture
For Java source code pairs $(C_A, C_B)$, Stage 1 extracts three structural views:
1. **Token Levenshtein & RKR-GST View:** Normalized token edit distance and greedy string tiling over normalized token streams (stripping comments and canonicalizing identifiers).
2. **Normalized AST Subtree View:** Subtree isomorphism and normalized tree edit distance computed over Java Abstract Syntax Trees (generated via `javalang`).
3. **Software Metric Vector View:** Cosine similarity over 8-dimensional software metric vectors (Halstead volume, Halstead difficulty, Cyclomatic complexity, loop nesting depth, method count, parameter count).

Linear fusion yields the code candidate score:
$$s_{\text{cand}}^{\text{code}}(C_A, C_B) = \sum_{m=1}^3 \beta_m s_m(C_A, C_B), \quad \sum \beta_m = 1$$

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Framework

### 3.1 Strict Separation of Prediction Unit and Grouping Unit
To prevent methodological conflation:
- **Prediction Unit:** An individual candidate pair $(D_A^{(i)}, D_B^{(i)})$ with ground-truth binary label $y_i \in \{0, 1\}$.
- **Grouping Unit (Text):** The shared reference premise sentence $g \in \mathcal{G}$. Candidate pairs evaluated against premise $g$ form cluster $\mathcal{P}_g$.
- **Grouping Unit (Code):** The underlying programming assignment / base program identity.

### 3.2 Stage-1 Escalation and Direct Bypass Routing
The candidate screening routing is formulated with lower and upper thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$:

$$\text{Route}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \\
\text{ESCALATE\_TO\_STAGE2} & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \\
\text{HIGH\_RISK\_DIRECT\_BYPASS} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}}
\end{cases}$$

- **Low-Risk Deferral ($s_{\text{cand}} < \theta_{\text{low}}$):** Computational budget conserved; Stage 2 is bypassed; candidate deferred from review.
- **Escalation Slice ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$):** Ambiguous candidate pairs are escalated to the heavy Stage-2 cross-encoder detector (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and enter the conformal calibration set.
- **Direct High-Risk Bypass ($s_{\text{cand}} \ge \theta_{\text{high}}$):** Obvious verbatim plagiarism pairs bypass Stage 2 to conserve heavy model FLOPs and are routed directly to human audit with high-risk candidate flags.

> [!IMPORTANT]
> **Conformal Population Consistency:** Direct high-risk bypass pairs **do not receive heavy-model forward passes** and therefore **must NOT enter the conformal calibration population**. Conformal non-conformity scores are calibrated strictly on pairs satisfying $\theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}}$.

### 3.3 Text Group Conformal Prediction
For escalated text pairs in premise group $g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}$, the group non-conformity score is defined by the worst-case pair non-conformity:
$$\alpha_g = \max_{i \in \mathcal{P}_g \cap \mathcal{E}} \left( 1 - \hat{P}_{\text{cal}}(Y = y_i \mid \mathbf{x}_i) \right)$$
where $\mathcal{E} = \{ i : \theta_{\text{low}} \le s_{\text{cand}}(i) < \theta_{\text{high}} \}$.

Let $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ be the number of escalated premise groups. The conformal quantile cutoff is:
$$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (n+1)(1-\epsilon) \rceil}{n}; \, \{\alpha_g\}_{g=1}^n \right)$$
**Finite-Sample Attainability Bound:** If $n < \lceil (1-\epsilon)/\epsilon \rceil$, then $\lceil (n+1)(1-\epsilon) \rceil = n + 1$, yielding $\hat{q}_{1-\epsilon} = \infty$. In this scenario, the system formally abstains from claiming a finite-sample guarantee on the escalated slice.

### 3.4 Operational 4-State Output Schema
Every pair processed through the pipeline is routed into exactly one of four standardized output states:

| Output State Name | Modality | Trigger Condition | Stage-2 Executed? | Human Review Required? | Semantic Definition |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **`LOW_RISK_REVIEW_DEFERRED`** | Text & Code | $s_{\text{cand}} < \theta_{\text{low}}$ OR ($\hat{P}_{\text{cal}} \le \gamma_{\text{lower}}$ and $\mathcal{C} = \{0\}$) | No (if pruned) / Yes (if escalated) | No | Pair exhibits minimal lexical/semantic overlap. Automated review deferred; computational budget conserved. |
| **`HIGH_RISK_CANDIDATE_FLAGGED`** | Text & Code | $s_{\text{cand}} \ge \theta_{\text{high}}$ (direct bypass) OR ($\hat{P}_{\text{cal}} \ge \gamma_{\text{upper}}$ and $\mathcal{C} = \{1\}$) | No (if bypass) / Yes (if escalated) | Yes | High-confidence statistical evidence of reuse. Dossier synthesized for committee review. |
| **`AMBIGUOUS_HUMAN_REVIEW`** | Text & Code | $\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$ AND ($\gamma_{\text{lower}} < \hat{P}_{\text{cal}} < \gamma_{\text{upper}}$ OR $\mathcal{C} = \{0, 1\}$) | Yes | Yes | Statistical ambiguity. Classifier abstains from binary decision; evidence routed to human reviewer. |
| **`ABSTAIN_UNRESOLVED`** | Text Only | Escalated with empty conformal prediction set $\mathcal{C} = \emptyset$ | Yes | Yes (Auditor) | Model abstention; observation violates calibration distribution. Treated as uncalibrated anomaly. |

---

## 4. Source Code Modality Boundaries and Limitations

### 4.1 Impossibility of Group Cross-Fitting on IR-Plag `case-05`
The IR-Plag benchmark consists of 7 tasks (`case-01` to `case-07`). Empirical inspection of `case-05` establishes that:
- Total pairs: 107 (54 non-plagiarized, 53 plagiarized).
- **All 53 plagiarized pairs originate from a single base program (`original/Main.java`).**
- In a 5-fold group split across the 16 program identities (1 original + 15 non-plagiarized):
  - **Fold 1:** Holds out `original/Main.java`. The training set contains **0 positive pairs** ($N_{\text{pos}}^{\text{train}} = 0$). Logistic regression (Platt scaling) maximum likelihood diverges to $-\infty$.
  - **Folds 2–5:** Hold out non-plagiarized programs. Held-out test sets contain **0 positive pairs** ($N_{\text{pos}}^{\text{test}} = 0$), rendering ROC-AUC, PR-AUC, and calibration error undefined.

### 4.2 Code Modality Downgrade
Because group cross-fitting Platt scaling is mathematically impossible on `case-05`, and the remaining tasks ($N=7$) cannot satisfy conformal exchangeability:
1. Probability calibration and conformal prediction guarantees are **formally disclaimed for source code**.
2. Source code pairs are evaluated using **exploratory raw decision margin thresholding** $z(\mathbf{x})$.
3. Code evaluation emits a **3-state schema** (`LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`). State `ABSTAIN_UNRESOLVED` is not emitted for code.

---

## 5. Contribution 3: Budget-Constrained Multi-Stage Optimization

### 5.1 Global Cost Formulation
The end-to-end computational expenditure across an evaluation batch of $N$ pairs is:
$$C_{\text{total}} = N \cdot c_{\text{stage1}} + |\mathcal{E}| \cdot c_{\text{stage2}} + |\mathcal{R}| \cdot c_{\text{evidence}} \le B$$
where:
- $c_{\text{stage1}}$ is the unit cost of multi-view screening ($0.045\text{ ms/pair}$).
- $c_{\text{stage2}}$ is the unit cost of cross-encoder inference ($4.046\text{ ms/pair}$).
- $c_{\text{evidence}}$ is the unit cost of attribution generation ($1.850\text{ ms/pair}$).
- $|\mathcal{E}|$ is the count of escalated pairs ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$).
- $|\mathcal{R}|$ is the count of pairs routed to human review (`FLAGGED` + `AMBIGUOUS` + `ABSTAIN`).

### 5.2 Multi-Objective Pareto Formulation
The screening thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ are jointly optimized on $D_{\text{val}}$:
$$\max_{\theta_{\text{low}}, \theta_{\text{high}}} R_{\text{cand}}(\theta_{\text{low}}) \quad \text{s.t.} \quad C_{\text{total}}(\theta_{\text{low}}, \theta_{\text{high}}) \le B, \quad R_{\text{cand}} \ge 0.98$$

---

## 6. Contribution 4: Faithful Attribution and Ethical Governance

### 6.1 Explainable Attribution Evidence
When a candidate pair is flagged or routed to human review, the framework synthesizes an interpretable dossier:
- **Text:** Exact character-offset span alignment via Myers diff, coupled with cross-attention saliency maps.
- **Code:** Normalized AST subtree diffs highlighting statement reordering, dead-code insertion, and an $\alpha$-conversion ledger mapping renamed identifiers.

### 6.2 Strict Four-Level Institutional Adjudication Architecture
To prevent ethical overreach and algorithmic bias:
- **Level 1 (Feature Extraction & Screening):** Algorithmic computation of lexical, semantic, and structural similarity.
- **Level 2 (Calibrated Statistical Inference):** Computation of calibrated probabilities and conformal sets; emission of the 4 model states.
- **Level 3 (Evidence Dossier Synthesis):** Automated generation of side-by-side alignment diffs and attribution weights.
- **Level 4 (Human Institutional Adjudication):** Independent evaluation by human course instructors or academic integrity committees.

> [!CAUTION]
> **ETHICAL NON-DELEGATION PRINCIPLE:** The automated machine learning model **NEVER** issues verdicts of plagiarism, academic dishonesty, or disciplinary sanctions. The system outputs strictly empirical statistical indicators of text or code reuse to aid human adjudication.

