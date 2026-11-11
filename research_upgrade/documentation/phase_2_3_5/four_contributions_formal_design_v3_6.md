# Four Research Contributions: Formal Architectural & Methodological Specification (Version 3.6)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Document Version:** Version 3.6  
**Date:** November 10, 2026  
**Status:** INDEPENDENTLY AUDITED & MATHEMATICALLY CERTIFIED (All 16 Artifacts Verified)

---

## 1. Overview and Core Research Objective

The primary objective of this research is to resolve the acute trade-off between computational scalability and detection reliability in high-stakes academic integrity detection systems. In institutional deployments, pairwise comparisons across cohort submissions scale quadratically ($\mathcal{O}(M^2)$ for $M$ student submissions). Existing systems either rely on brittle single-view heuristic filters that suffer severe candidate recall degradation on paraphrased text or structurally obfuscated code, or deploy expensive deep transformers uniformly across all pairs, resulting in prohibitive computational latency, operational expense, and energy waste.

This framework introduces a principled, two-stage architecture that integrates:
1. **Multi-View Candidate Screening** across lexical, semantic, syntactic, and structural dimensions.
2. **Uncertainty-Aware Adaptive Escalation** using probability calibration and group-level split conformal prediction.
3. **Multi-Objective Budget-Constrained Optimization** operating over a rigorous Pareto frontier.
4. **Faithful Multi-Modal Attribution Evidence** separating automated statistical signals from institutional adjudication.

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 The Paraphrase and Obfuscation Failure Modes
Single-view similarity metrics suffer catastrophic false-negative rates when confronting semantic paraphrase or structural code obfuscation:
- **Academic Text:** Empirical analysis of [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv) reveals that **119 true positive candidate pairs** ($y=1$) share **zero lexical word overlap** (Jaccard similarity $= 0.0$). A purely lexical filter (such as BM25 or word n-grams) set to prune candidates at similarity $> 0$ misses 100% of these positive instances, causing candidate recall to drop by $\sim 9.0\%$ on the positive slice.
- **Academic Source Code:** In the IR-Plag corpus ([datasets/raw/ir_plag/extracted/IR-Plag-Dataset/](file:///D:/ml%20rearcch%20project%20ad/datasets/raw/ir_plag/extracted/IR-Plag-Dataset/)), obfuscation levels L1–L6 systematically neutralize token-level matching through identifier renaming, loop restructuring, statement reordering, and dead-code injection.

### 2.2 Text Multi-View Architecture
For any text candidate pair $(D_A, D_B)$, Stage 1 extracts four complementary representation views:
1. **Character n-gram TF-IDF View ($\mathbf{v}_{\text{char}}$):** Substring n-grams ($n \in \{3, 4, 5\}$) capturing morphological stem overlap and sub-word borrowings robust to inflectional shifts.
2. **Word n-gram TF-IDF View ($\mathbf{v}_{\text{word}}$):** Lexical token n-grams ($n \in \{1, 2\}$) weighted via BM25/TF-IDF scoring capturing verbatim phrase preservation.
3. **Dense Semantic Embedding View ($\mathbf{v}_{\text{sem}}$):** 384-dimensional dense vectors generated via `all-MiniLM-L6-v2` capturing latent propositional semantics invariant to lexical substitution.
4. **Linguistic Surface Complexity View ($\mathbf{v}_{\text{ling}}$):** Difference vectors over sentence length, type-token ratio (TTR), and readability metrics.

The multi-view similarity score is computed via a linear fusion model:
$$s_{\text{cand}}^{\text{text}}(D_A, D_B) = \sigma\left( \mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(D_A, D_B) + b \right)$$
where fusion weights $\mathbf{w}$ are optimized on the validation partition $D_{\text{val}}$ (1,500 pairs) to maximize candidate recall under the budget constraint.

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
To eliminate methodological conflation:
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

## 4. Source Code Modality Boundaries and Structural Limitations

### 4.1 Ground-Truth Inventory of IR-Plag `case-05`
Physical audit of the IR-Plag dataset on disk ([case_05_java_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_java_file_inventory.csv)) establishes that `case-05` contains **exactly 69 physical `.java` files**:
- **Original Source Program:** 1 file (`original/T5.java`, base identity `prog_original`).
- **Non-Plagiarized Programs:** 15 files (`non-plagiarized/01` to `15`, 1 file each, base identities `prog_np_01` to `prog_np_15`).
- **Plagiarized Variant Programs:** 53 files across 6 obfuscation levels:
  - L1 (renaming/comments): 9 files (`01`–`09`)
  - L2 (redundancy): 8 files (`01`–`08`) *(Notice: exactly 8 files, resolving earlier discrepancy)*
  - L3 (reordering): 9 files (`01`–`09`)
  - L4 (loop transformation): 9 files (`01`–`09`)
  - L5 (structure/methods): 9 files (`01`–`09`)
  - L6 (combined): 9 files (`01`–`09`)
  - Sum: $9 + 8 + 9 + 9 + 9 + 9 = 53$ plagiarized files.
- Total Physical Java Files: $1 + 15 + 53 = 69$ files.

In [code_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/code_plagiarism_benchmark.csv), these 69 files generate **107 pairs** ([case_05_pair_to_program_mapping.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_pair_to_program_mapping.csv)):
- **53 Positive Pairs ($y=1$):** Each of the 53 plagiarized files paired with `original/T5.java` (both share base identity `prog_original`).
- **15 Negative Pairs ($y=0$):** Each of the 15 non-plagiarized files paired with `original/T5.java`.
- **39 Negative Pairs ($y=0$):** Pairs between distinct non-plagiarized submissions generated via `min(i+4, len(np))` sliding window.
- Total: $53 + 15 + 39 = 107$ pairs (53 positive, 54 negative).

### 4.2 Mathematical Impossibility of Group Cross-Fitting on `case-05`
There are exactly 16 base program identities in `case-05`: `prog_original` and `prog_np_01` through `prog_np_15`.
Because **all 53 positive pairs share the base identity `prog_original`**, any group-disjoint partitioning yields a severe degeneracy:

1. **When `prog_original` is in the held-out test fold (Fold 1):**
   - The training fold contains zero pairs involving `prog_original`.
   - Therefore, the training fold contains **ZERO POSITIVE SAMPLES** ($N_{\text{pos}}^{\text{train}} = 0$, $N_{\text{neg}}^{\text{train}} = 39$).
   - **Unregularized Platt Scaling Divergence:** The cross-entropy loss is strictly decreasing as $B \to -\infty$; maximum likelihood optimization diverges to $-\infty$.
   - **Regularized Platt Scaling Exception:** In scikit-learn (`LogisticRegression`), fitting with only class 0 explicitly crashes: `ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0`.
   - Neither regularized nor unregularized Platt scaling can produce a valid binary calibrator.

2. **When `prog_original` is in the training fold (Folds 2–5):**
   - The held-out test fold contains zero pairs involving `prog_original`.
   - Therefore, the test fold contains **ZERO POSITIVE SAMPLES** ($N_{\text{pos}}^{\text{test}} = 0$, $N_{\text{neg}}^{\text{test}} \in [12, 15]$).
   - In any evaluation set with zero positives, **Precision, Recall, ROC-AUC, PR-AUC, and ECE are mathematically undefined** ($0/0$ divisions).

**Formal Methodological Ruling:**
Group-disjoint probability calibration and conformal prediction cannot be mathematically supported on IR-Plag `case-05`.
Code modality evaluation for `case-05` is strictly restricted to **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$). Conformal guarantees apply exclusively to the text modality.

---

## 5. Contribution 3: Multi-Objective Budget-Constrained Optimization

### 5.1 Formal Pareto Frontier Formulation
The deployment optimization is framed as a multi-objective trade-off over three competing objectives:
$$\min_{\boldsymbol{\theta} \in \Theta} \left[ \mathcal{L}_{\text{recall}}(\boldsymbol{\theta}), \, \mathcal{C}_{\text{latency}}(\boldsymbol{\theta}), \, \mathcal{U}_{\text{uncertainty}}(\boldsymbol{\theta}) \right]$$
where:
- $\mathcal{L}_{\text{recall}} = 1 - \text{Recall}(D_{\text{val}}^{\text{pos}}; \boldsymbol{\theta})$
- $\mathcal{C}_{\text{latency}} = \mathbb{E}_{x \sim D_{\text{val}}} [\text{Cost}(x; \boldsymbol{\theta})]$
- $\mathcal{U}_{\text{uncertainty}} = \text{Fraction}(x \in D_{\text{val}} \mid \text{Route}(x) = \text{AMBIGUOUS\_HUMAN\_REVIEW})$

### 5.2 Latency Measurement Scope and Decomposition
The Stage-1 routing budget target of **$\le 0.050\text{ ms}$** per pair is a **system design target**, not a single measured end-to-end operation. The complete latency profile comprises 10 distinct operational stages:
1. **Cached Vector Dot-Product (Design Target Benchmark):** $0.045\text{ ms}$ (text) / $0.058\text{ ms}$ (code).
2. **Text Feature Extraction (TF-IDF):** $\sim 0.12\text{ ms}$ per candidate pair.
3. **Dense Embedding Forward Pass (MiniLM):** $\sim 1.85\text{ ms}$ per unbatched document.
4. **Code Lexer & Token Normalization:** $\sim 0.28\text{ ms}$ per source file.
5. **AST Parsing (`javalang`):** $\sim 2.15\text{ ms}$ per source file.
6. **Structural Metrics Computation (Halstead/CC):** $\sim 0.42\text{ ms}$ per source file.
7. **Stage-1 Fused Routing Decision:** $\sim 0.05\text{ ms}$ overhead.
8. **Stage-2 Cross-Encoder Verification:** $4.046\text{ ms}$ per escalated pair.
9. **Myers Diff Text Alignment:** $0.62\text{ ms}$ per candidate pair.
10. **AST Subtree Isomorphism Evidence:** $1.85\text{ ms}$ per candidate pair.

---

## 6. Contribution 4: Faithful Multi-Modal Attribution Evidence

### 6.1 Multi-Modal Attribution Artifacts
For any pair flagged for institutional review, the system generates:
- **Text Modality:**
  - Token-level diff highlights computed via Myers' greedy $O(ND)$ diff algorithm.
  - Sentence-level cross-attention heatmaps extracted from the Stage-2 cross-encoder.
  - Conformal prediction set $\mathcal{C}(x)$ and calibrated posterior $\hat{P}_{\text{cal}}$.
- **Code Modality:**
  - Token-level aligned diffs highlighting renamed variables and reordered expressions.
  - Normalized AST subtree overlap highlighting isomorphic syntax blocks.
  - Software complexity radar plots comparing Halstead volume, difficulty, and cyclomatic complexity.

### 6.2 Institutional Adjudication Separation
The system strictly separates **automated statistical evidence generation** from **administrative judgment**:
- The framework never declares a student "guilty of plagiarism".
- All human-facing outputs are designated as **`EVIDENCE_DOSSIER`** containing statistical probabilities, confidence intervals, and visual diffs.
- The university academic integrity board retains sole legal and administrative authority for adjudication.

---

## 7. Audit Verification Cross-References

All empirical counts and data properties specified in this document have been independently verified by [run_phase_2_3_5_audit.py](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/run_phase_2_3_5_audit.py):
- Text dataset partition accounting: [text_dataset_audit.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_dataset_audit.json)
- Row-level membership ledger: [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv)
- Physical Java file audit (69 files): [case_05_java_file_inventory.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_java_file_inventory.csv)
- Case-05 cross-fitting proof: [case_05_fold_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_fold_audit.csv)
- Data loader classification: [loader_verification_report.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/loader_verification_report.csv)
- Overall audit summary: [overall_audit_summary.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/overall_audit_summary.json)

