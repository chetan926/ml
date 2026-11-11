# The Four Research Contributions: Final Methodological Design (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 02 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** FULLY SPECIFIED & CERTIFIED

---

## 1. Contribution 1: Robust Multi-View Candidate Generation

### 1.1 Text Multi-View Screening Architecture
To overcome the failure mode where **119 true positive text pairs** share zero lexical overlap in [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv), Stage 1 extracts four complementary representation views:

1. **Character n-gram TF-IDF View ($\mathbf{v}_{\text{char}}$):**  
   Extracts sub-word character $n$-grams ($n \in \{3, 4, 5\}$). Captures morphological stem overlaps, compounding, and minor typographical perturbations invariant to inflectional endings.
2. **Word-Level Lexical View ($\mathbf{v}_{\text{word}}$):**  
   Extracts token unigrams and bigrams weighted by sublinear TF-IDF. Measures verbatim phrase retention and shared keyword vocabulary.
3. **Dense Contextual Semantic View ($\mathbf{v}_{\text{sem}}$):**  
   Extracts 384-dimensional dense sequence embeddings using `all-MiniLM-L6-v2`. Captures latent semantic entailment and propositional equivalence even under complete synonym substitution.
4. **Linguistic Surface Complexity View ($\mathbf{v}_{\text{ling}}$):**  
   Extracts length ratio, type-token ratio (TTR) difference, and readability delta. Captures stylistic consistency.

#### Mathematical Feature Transformation & Bounded Domains:
To avoid undefined negative similarities, raw cosine similarities are transformed into strictly bounded $[0, 1]$ domains:
$$s_{\text{view}}(A, B) = \frac{1 + \cos(\mathbf{e}_A, \mathbf{e}_B)}{2} \in [0, 1]$$

#### Zero-Denominator and Empty-Input Behavior:
- **Empty text sequence:** If $|D| = 0$, TF-IDF vector is the zero vector; $s_{\text{char}} = 0.0$, $s_{\text{word}} = 0.0$.
- **Zero-norm embedding:** If $\|\mathbf{e}\| = 0$, semantic similarity defaults to $0.0$.
- **Missing surface feature:** Missing length/TTR values default to maximum difference ($s_{\text{ling}} = 0.0$).

The fused Stage-1 text candidate score is computed via validation-optimized linear combination:
$$s_{\text{cand}}^{\text{text}}(D_A, D_B) = \sigma\left( \mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(D_A, D_B) + b \right)$$

---

### 1.2 Source Code Multi-View Screening Architecture
For Java source code pairs $(C_A, C_B)$ in the IR-Plag corpus ([datasets/raw/ir_plag/extracted/IR-Plag-Dataset/](file:///D:/ml%20rearcch%20project%20ad/datasets/raw/ir_plag/extracted/IR-Plag-Dataset/)), Stage 1 extracts three structural views:

1. **Canonicalized Token Stream View ($s_{\text{tok}}$):**  
   Stripping comments, normalising variable identifiers to generic tokens (`VAR_1`, `VAR_2`), and computing normalized token edit similarity:
   $$s_{\text{tok}}(C_A, C_B) = 1 - \frac{\text{Levenshtein}(\tau(C_A), \tau(C_B))}{\max(|\tau(C_A)|, |\tau(C_B)|)}$$
2. **Normalized AST Subtree View ($s_{\text{ast}}$):**  
   Constructs Abstract Syntax Trees via `javalang`, extracts canonicalized subtree hashes $\mathcal{T}(C)$, and evaluates subtree Jaccard overlap:
   $$s_{\text{ast}}(C_A, C_B) = \frac{2 \cdot |\mathcal{T}(C_A) \cap \mathcal{T}(C_B)|}{|\mathcal{T}(C_A)| + |\mathcal{T}(C_B)|}$$
3. **Software Metric Vector View ($s_{\text{metric}}$):**  
   8-dimensional feature vector (Halstead volume, Halstead difficulty, McCabe cyclomatic complexity, loop nesting depth, method count, parameter count, LOC, comment density):
   $$s_{\text{metric}}(C_A, C_B) = \frac{1 + \cos(\mathbf{m}_A, \mathbf{m}_B)}{2} \in [0, 1]$$

#### Empty-Input Handling for Code:
- If a Java file fails to parse into an AST (syntax error), $s_{\text{ast}}$ defaults to $0.0$, and the parser error is logged.
- If token count is 0, $s_{\text{tok}} = 0.0$.

The fused code candidate score is:
$$s_{\text{cand}}^{\text{code}}(C_A, C_B) = \beta_1 s_{\text{tok}} + \beta_2 s_{\text{ast}} + \beta_3 s_{\text{metric}}, \quad \sum \beta_i = 1$$

---

## 2. Contribution 2: Uncertainty-Aware Adaptive Decision Framework

### 2.1 Three-Way Adaptive Routing Rule
Using screening thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$, candidate pairs are routed into three operational paths:

$$\text{Route}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \\
\text{ESCALATE\_TO\_STAGE2} & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \\
\text{HIGH\_RISK\_DIRECT\_BYPASS} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}}
\end{cases}$$

- **Direct Bypasses:** Pairs with $s_{\text{cand}} < \theta_{\text{low}}$ (benign) and $s_{\text{cand}} \ge \theta_{\text{high}}$ (verbatim copies) bypass Stage 2 entirely, saving heavy model forward passes.
- **Escalation Slice:** Ambiguous candidate pairs ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$) receive Stage-2 evaluation and enter the calibration population.

### 2.2 Text Probability Calibration & Group Conformal Prediction
1. **Platt Probability Calibration (Text Modality Only):**  
   For escalated text pairs, raw cross-encoder logits $z \in \mathbb{R}$ are calibrated on $D_{\text{cal-prob}}^{\text{esc}}$ via:
   $$\hat{P}_{\text{cal}}(Y = 1 \mid z) = \sigma(A z + B)$$
2. **Exact Group Conformal Quantile:**  
   For escalated text premise groups $g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}$, the group non-conformity is $\alpha_g = \max_{i \in \mathcal{P}_g^{\text{esc}}} (1 - \hat{P}_{\text{cal}}(y_i \mid z_i))$.  
   Using sorted scores $s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$, the exact order statistic is:
   $$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$
   If $k > n$, then $\hat{q}_{1-\epsilon} = \infty$ (finite-sample guarantee unattainable); otherwise $\hat{q}_{1-\epsilon} = s_{(k)}$.
3. **Empty Prediction Set Semantics ($\mathcal{C}(x) = \emptyset$):**  
   An empty prediction set indicates model inconsistency or extreme predictive ambiguity. It is formally mapped to the operational state **`ABSTAIN_UNRESOLVED`** and routed to an auditor queue. It is NOT claimed as mathematical proof of out-of-distribution data.

### 2.3 Code Modality Fallback: Exploratory Raw Decision Margins
As proven in [case_05_fold_feasibility_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/case_05_fold_feasibility_report.json), IR-Plag `case-05` exhibits a structural single-class topology where all 53 positive pairs share `prog_original`. Group-disjoint cross-fitting leaves training folds with zero positives (causing Platt scaling MLE divergence / scikit-learn `ValueError`) or test folds with zero positives (rendering Precision, Recall, and ROC-AUC undefined).  
**Protocol Mandate:** Probability calibration and conformal prediction are NOT applied to code. Code evaluation uses **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$).

### 2.4 The Four Operational System-Level Output States

| State Name | Modality | Condition | Stage-2 Ran? | Human Review? | Action & Semantics |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **`LOW_RISK_REVIEW_DEFERRED`** | Text & Code | $s_{\text{cand}} < \theta_{\text{low}}$ OR ($\hat{P}_{\text{cal}} \le \gamma_{\text{lower}}$ and $\mathcal{C} = \{0\}$) | Pruned: No / Escalated: Yes | No | Computational budget conserved. Automated review deferred. |
| **`HIGH_RISK_CANDIDATE_FLAGGED`** | Text & Code | $s_{\text{cand}} \ge \theta_{\text{high}}$ OR ($\hat{P}_{\text{cal}} \ge \gamma_{\text{upper}}$ and $\mathcal{C} = \{1\}$) | Bypass: No / Escalated: Yes | Yes | High-confidence statistical overlap. Evidence dossier generated for review. |
| **`AMBIGUOUS_HUMAN_REVIEW`** | Text & Code | Escalated with $\gamma_{\text{lower}} < \hat{P}_{\text{cal}} < \gamma_{\text{upper}}$ OR $\mathcal{C} = \{0, 1\}$ | Yes | Yes | Statistical ambiguity. Model abstains from binary decision; evidence routed to reviewer. |
| **`ABSTAIN_UNRESOLVED`** | Text Only | Escalated with empty conformal set $\mathcal{C} = \emptyset$ | Yes | Yes (Auditor) | Model abstention; observation violates calibration distribution. Isolated for manual audit. |

---

## 3. Contribution 3: Budget-Constrained Routing Optimization

### 3.1 Pareto-Frontier Threshold Search
Threshold selection is formulated as a multi-objective optimization problem over validation data $D_{\text{val}}$:
$$\min_{\boldsymbol{\theta} \in \Theta} \begin{bmatrix} 1 - \text{Recall}(D_{\text{val}}^{\text{pos}}; \boldsymbol{\theta}) \\ \mathbb{E}[\text{Latency}(x; \boldsymbol{\theta})] \\ \text{Workload}(D_{\text{val}}; \boldsymbol{\theta}) \end{bmatrix}$$
subject to:
1. **Candidate Recall Constraint:** $\text{Recall}(D_{\text{val}}^{\text{pos}}; \boldsymbol{\theta}) \ge 0.95$ (Design target: $\ge 0.98$).
2. **Review Workload Bound:** $\text{Workload} \le 0.15$ (At most $15\%$ of cohort escalated to human review).

To maintain scientific accuracy:
- The optimization is executed via a dense grid search over $\Theta = (\theta_{\text{low}}, \theta_{\text{high}}, \gamma_{\text{lower}}, \gamma_{\text{upper}})$.
- It is strictly characterized as a **Pareto-frontier analysis**, NOT a Lagrangian optimizer.

### 3.2 Resource Dimension Separation
Computational costs are reported across distinct, un-mixed resource dimensions:
1. **Latency:** CPU wall-clock milliseconds per pair.
2. **Memory:** Peak resident set size (RAM/VRAM in megabytes).
3. **Compute:** Floating-point operations (FLOPs) and heavy model invocation counts.

---

## 4. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 4.1 Modality-Specific Evidence Artifacts
For any pair flagged for review (`HIGH_RISK_CANDIDATE_FLAGGED` or `AMBIGUOUS_HUMAN_REVIEW`), the framework synthesizes an interpretable evidence dossier:
- **Text Modality Evidence:**
  - Token-level diff highlights computed via Myers' greedy $\mathcal{O}(ND)$ shortest edit script algorithm.
  - Sentence-level cross-attention heatmaps extracted from the Stage-2 cross-encoder.
  - Conformal prediction set $\mathcal{C}(x)$ and calibrated posterior $\hat{P}_{\text{cal}}$.
- **Code Modality Evidence:**
  - Token-level aligned diffs highlighting renamed variables and reordered statements.
  - Normalized AST subtree overlap highlighting isomorphic syntax blocks.
  - Software complexity radar plots comparing Halstead metrics, McCabe cyclomatic complexity, and nesting depth.

### 4.2 Legal and Administrative Separation
The framework enforces a strict institutional boundary:
- Automated outputs are designated as **Statistical Evidence Dossiers**.
- A similarity explanation is **never presented as proof of academic misconduct**.
- Administrative adjudication remains the sole prerogative of the university academic integrity committee.

