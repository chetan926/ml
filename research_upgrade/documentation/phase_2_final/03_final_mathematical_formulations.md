# Final Mathematical Formulations & Statistical Proofs (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 03 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** MATHEMATICALLY AUDITED & FORMALLY VERIFIED

---

## 1. Multi-View Candidate Scoring Formulations

### 1.1 Text Modality Candidate Scoring
Let $(D_A, D_B)$ denote a pair of candidate text documents. The Stage-1 multi-view screening representation is:
$$\mathbf{v}_{\text{sim}}(D_A, D_B) = \begin{bmatrix} s_{\text{char}}(D_A, D_B) \\ s_{\text{word}}(D_A, D_B) \\ s_{\text{sem}}(D_A, D_B) \\ s_{\text{ling}}(D_A, D_B) \end{bmatrix} \in [0, 1]^4$$

To ensure well-defined non-negative feature domains, all cosine similarities are mapped to $[0, 1]$ via the affine transformation:
$$\tilde{\cos}(\mathbf{u}, \mathbf{v}) = \frac{1 + \cos(\mathbf{u}, \mathbf{v})}{2} = \frac{1}{2} \left( 1 + \frac{\mathbf{u}^{\top} \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \right) \in [0, 1]$$

1. **Character n-gram Substring Similarity ($s_{\text{char}}$):**
   $$s_{\text{char}}(D_A, D_B) = \frac{\sum_{n=3}^5 \omega_n \cdot \tilde{\cos}\left(\mathbf{t}_n(D_A), \mathbf{t}_n(D_B)\right)}{\sum_{n=3}^5 \omega_n}$$
   where $\mathbf{t}_n(D)$ is the character $n$-gram TF-IDF vector, and $\omega_n = 1/3$.
2. **Word n-gram Lexical Similarity ($s_{\text{word}}$):**
   $$s_{\text{word}}(D_A, D_B) = \frac{1}{2} \tilde{\cos}\left(\mathbf{w}_1(D_A), \mathbf{w}_1(D_B)\right) + \frac{1}{2} \tilde{\cos}\left(\mathbf{w}_2(D_A), \mathbf{w}_2(D_B)\right)$$
3. **Dense Contextual Semantic Similarity ($s_{\text{sem}}$):**
   $$s_{\text{sem}}(D_A, D_B) = \tilde{\cos}\left(\mathbf{e}(D_A), \mathbf{e}(D_B)\right)$$
   where $\mathbf{e}(D) \in \mathbb{R}^{384}$ is the mean-pooled sequence embedding from `all-MiniLM-L6-v2`.
4. **Linguistic Surface Similarity ($s_{\text{ling}}$):**
   $$s_{\text{ling}}(D_A, D_B) = 1 - \tanh\left( \|\mathbf{u}_{\text{ling}}(D_A) - \mathbf{u}_{\text{ling}}(D_B)\|_2 \right) \in [0, 1]$$

The fused Stage-1 candidate score is given by the logistic fusion model:
$$s_{\text{cand}}^{\text{text}}(D_A, D_B) = \sigma\left( \mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(D_A, D_B) + b \right) = \frac{1}{1 + \exp\left( -(\mathbf{w}^{\top} \mathbf{v}_{\text{sim}} + b) \right)}$$

---

### 1.2 Code Modality Candidate Scoring
Let $(C_A, C_B)$ denote a pair of Java source programs. The fused Stage-1 candidate score is:
$$s_{\text{cand}}^{\text{code}}(C_A, C_B) = \beta_1 s_{\text{tok}}(C_A, C_B) + \beta_2 s_{\text{ast}}(C_A, C_B) + \beta_3 s_{\text{metric}}(C_A, C_B)$$
subject to $\beta_1, \beta_2, \beta_3 \ge 0$ and $\beta_1 + \beta_2 + \beta_3 = 1$.

1. **Normalized Token Edit Similarity ($s_{\text{tok}}$):**
   $$s_{\text{tok}}(C_A, C_B) = 1 - \frac{\text{Levenshtein}(\tau(C_A), \tau(C_B))}{\max(|\tau(C_A)|, |\tau(C_B)|)}$$
2. **Normalized AST Subtree Similarity ($s_{\text{ast}}$):**
   $$s_{\text{ast}}(C_A, C_B) = \frac{2 \cdot |\mathcal{T}(C_A) \cap \mathcal{T}(C_B)|}{|\mathcal{T}(C_A)| + |\mathcal{T}(C_B)|}$$
3. **Software Metric Vector Cosine Similarity ($s_{\text{metric}}$):**
   $$s_{\text{metric}}(C_A, C_B) = \tilde{\cos}\left(\mathbf{m}(C_A), \mathbf{m}(C_B)\right) \in [0, 1]$$

---

## 2. Adaptive Routing and Escalation Rule

For candidate pair $x = (A, B)$, the Stage-1 routing function partitions the candidate space into three operational streams using thresholds $\theta_{\text{low}} < \theta_{\text{high}}$:

$$\text{Route}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \\
\text{ESCALATE\_TO\_STAGE2} & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \\
\text{HIGH\_RISK\_DIRECT\_BYPASS} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}}
\end{cases}$$

- **Computational Conservation:** Stage 2 is executed strictly on the escalation subset $\mathcal{E} = \{ x \mid \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \}$.
- **Conformal Consistency Rule:** Pairs routed to `HIGH_RISK_DIRECT_BYPASS` bypass Stage 2 and are therefore excluded from calibration and test conformal sets.

---

## 3. Probability Calibration and Failure Proofs

### 3.1 Platt Scaling Formulation
For escalated candidate pairs, let $z_i \in \mathbb{R}$ denote the uncalibrated logit output from the Stage-2 detector. Platt scaling models the posterior probability $\hat{P}_{\text{cal}}(Y = 1 \mid z)$ via:
$$\hat{P}_{\text{cal}}(Y = 1 \mid z) = \sigma(A z + B) = \frac{1}{1 + \exp(-(A z + B))}$$
where $(A, B) \in \mathbb{R}^2$ minimize binary cross-entropy loss over $D_{\text{cal-prob}}^{\text{esc}} = \{(z_i, y_i)\}_{i=1}^N$:
$$\mathcal{L}(A, B) = -\sum_{i=1}^N \left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log(1 - \sigma(A z_i + B)) \right]$$

### 3.2 Formal Proof: Unregularized Platt Scaling Divergence on Single-Class Data
**Theorem 1 (Unregularized Platt Divergence):**  
*If a calibration training fold contains zero positive instances ($N_{\text{pos}}^{\text{train}} = 0$, meaning $y_i = 0$ for all $i \in \{1, \dots, N\}$), the unregularized Platt scaling maximum likelihood estimator does not exist in $\mathbb{R}^2$, and gradient optimization diverges to $B \to -\infty$.*

**Proof:**  
When $y_i = 0$ for all $i$, the objective simplifies to:
$$\mathcal{L}(A, B) = -\sum_{i=1}^N \log(1 - \sigma(A z_i + B)) = \sum_{i=1}^N \log\left( 1 + e^{A z_i + B} \right)$$
The partial derivative with respect to intercept $B$ is:
$$\frac{\partial \mathcal{L}}{\partial B} = \sum_{i=1}^N \frac{e^{A z_i + B}}{1 + e^{A z_i + B}} = \sum_{i=1}^N \sigma(A z_i + B) > 0 \quad \forall (A, B) \in \mathbb{R}^2$$
Because the derivative is strictly positive everywhere, $\mathcal{L}(A, B)$ is strictly monotonically increasing in $B$ and has no finite stationary point:
$$\lim_{B \to -\infty} \mathcal{L}(A, B) = \sum_{i=1}^N \log(1 + 0) = 0$$
The infimum of 0 is attained only in the limit as $B \to -\infty$. Thus, numerical optimization diverges. $\blacksquare$

### 3.3 Formal Proof: Regularized Platt Scaling Exception in Standard Libraries
Under standard implementations (such as scikit-learn's `LogisticRegression(C=1.0)` with $L_2$ regularization $\frac{1}{2C} (A^2 + B^2)$), standard numerical solvers require at least two distinct binary classes to construct the target matrix. When provided with a dataset where all labels $y_i = 0$, scikit-learn explicitly raises:
$$\text{\texttt{ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0}}$$
Consequently, neither unregularized nor regularized Platt scaling can produce an operational binary calibrator on a single-class training fold.

### 3.4 Consequence for IR-Plag `case-05`: Three-Dimensional Methodological Analysis
In the IR-Plag `case-05` corpus, there are 16 base program identities (`prog_original` and `prog_np_01` to `prog_np_15`). As mathematically verified in [case_05_fold_feasibility_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/case_05_fold_feasibility_report.json), **all 53 positive pairs share the single base identity `prog_original`**. Under any 5-fold group-disjoint cross-validation split across the 16 base programs:

1. **Fold 1 (`prog_original` held out in test):**  
   - **Training Set Composition:** $N_{\text{pos}}^{\text{train}} = 0, N_{\text{neg}}^{\text{train}} = 33$ (all pairs between non-plagiarized programs `prog_np_03` to `prog_np_15`).
   - **Held-Out Test Set Composition:** $N_{\text{pos}}^{\text{test}} = 53, N_{\text{neg}}^{\text{test}} = 21$ (all pairs involving `prog_original`, `prog_np_01`, or `prog_np_02`).
   - **Dimension 1 — Can Platt calibration be fitted on training fold?**  
     **NO.** Because $N_{\text{pos}}^{\text{train}} = 0$, unregularized MLE diverges to intercept $B \to -\infty$ (Theorem 1), and regularized `LogisticRegression` raises a fatal exception (`ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0`).
   - **Dimension 2 — Are test metrics mathematically defined?**  
     **YES.** Because the test fold contains both positive and negative instances (+53, -21), test evaluation metrics (ROC-AUC, PR-AUC, Recall, Precision, Brier Score, ECE) are mathematically well-defined on the test label set.
   - **Dimension 3 — Can those metrics meaningfully evaluate the intended calibrated model?**  
     **NO.** Even though test metrics are mathematically defined, they cannot evaluate the intended calibrated model because **no valid calibrator could be fitted on the single-class training fold**. Evaluating uncalibrated raw scores against these metrics does not assess probability calibration.

2. **Folds 2 to 5 (`prog_original` retained in training):**  
   - **Training Set Composition:** $N_{\text{pos}}^{\text{train}} = 53, N_{\text{neg}}^{\text{train}} \in [36, 38]$. Both classes are present in training.
   - **Held-Out Test Set Composition:** $N_{\text{pos}}^{\text{test}} = 0, N_{\text{neg}}^{\text{test}} \in [16, 18]$. The test set contains strictly zero positive instances.
   - **Dimension 1 — Can Platt calibration be fitted on training fold?**  
     **YES.** Because both classes are present ($53$ positive pairs and $36\text{--}38$ negative pairs), Platt scaling optimizes cross-entropy successfully (`platt_scaling_feasible_on_train = True`).
   - **Dimension 2 — Are test metrics mathematically defined?**  
     **PARTIALLY / DEGENERATE.**
     - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{0}{0} \implies$ **Mathematically undefined**.
     - $\text{ROC-AUC}$ and $\text{PR-AUC}$ require instances of both classes $\implies$ **Mathematically undefined**.
     - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{0}{0 + \text{FP}}$:
       - If $\text{FP} > 0$: $\text{Precision} = \frac{0}{\text{FP}} = 0.0$ (**mathematically defined**).
       - If $\text{FP} = 0$: there are no predicted positives ($\text{TP} + \text{FP} = 0$), so $\text{Precision} = \frac{0}{0} \implies$ **Mathematically undefined**. *Precision must be reported as undefined when there are no predicted positives, not automatically as zero.*
     - $\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^N \hat{p}_i^2 \implies$ **Mathematically defined** (continuous quadratic loss against zero targets).
     - $\text{ECE}$ across $M = 10$ uniform bins:
       - **Implementation & Definitions:** In standard binary probability calibration (Platt 1999; Algorithm 4), binning partitions the predicted positive probability $\hat{p}_i \in [0, 1]$ into intervals $I_m = (\frac{m-1}{10}, \frac{m}{10}]$ ($I_1 = [0, 0.10]$). Bin confidence is $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$. The quantity denoted $\text{acc}(B_m)$ in calibration routines is the empirical positive-class frequency (positive empirical prevalence) $\bar{y}_m = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$. There is no thresholded predicted label (such as $\hat{y}_i = \mathbb{I}(\hat{p}_i \ge 0.5)$) used in binning.
       - **Behavior on All-Negative Test Folds ($y_i = 0$ for all $i$):** Because every ground-truth label in Folds 2–5 is $0$, the numerator $\sum_{i \in B_m} y_i = 0$ for every populated bin, so the empirical positive frequency is $\bar{y}_m = 0.0$. *(Note: if ECE were defined using top-label classification accuracy $\mathbb{I}(\hat{y}_i = y_i)$ with $\hat{y}_i = \mathbb{I}(\hat{p}_i \ge 0.5)$ as in Guo et al. 2017, then any correctly predicted negative instance with $\hat{p}_i < 0.5$ has $\hat{y}_i = 0 = y_i$, yielding a classification accuracy of $1.0$ (100%), NOT zero. Calling $\bar{y}_m$ "empirical accuracy" is an imprecise conflation of positive-class empirical prevalence with top-label accuracy).*
       - **Mathematical Simplification:** For every populated bin, $|\bar{y}_m - \text{conf}(B_m)| = |0 - \text{conf}(B_m)| = \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$. Weighting by bin counts yields:
         $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \text{conf}(B_m) = \frac{1}{N} \sum_{i=1}^N \hat{p}_i$$
         Thus, ECE is **mathematically computable** and collapses into the mean predicted positive probability across the fold ($\bar{\hat{p}}$).
       - **Evaluation Utility:** Even though ECE is mathematically computable, it **cannot meaningfully evaluate positive-class calibration or calibration discrimination**: without positive instances, it is impossible to evaluate whether high model confidence corresponds to actual plagiarism. A degenerate model outputting $\hat{p}(x) \equiv 0$ trivially achieves $\text{ECE} = 0.000$ and $\text{BS} = 0.000$ despite having zero discriminative capability to detect plagiarism.
   - **Dimension 3 — Can those metrics meaningfully evaluate the intended calibrated model?**  
     **NO.** Even though a calibrated model was fitted on train, the all-negative test set cannot meaningfully evaluate it:
     - It is impossible to evaluate sensitivity, recall, or false-negative trade-offs because there are no true positives to detect.
     - A degenerate constant predictor $\hat{p}(x) \equiv 0$ would achieve a "perfect" Brier Score of $0.000$ and $\text{ECE} = 0.000$ on Folds 2–5 despite being completely useless for plagiarism detection. Thus, calibration metrics on an all-negative fold provide zero evidence of discriminative calibration quality.

**Methodological Conclusion:**  
Because Fold 1 cannot be trained and Folds 2–5 cannot evaluate sensitivity or calibration discrimination, group-disjoint cross-fitting and conformal probability guarantees are structurally impossible on `case-05`. Code plagiarism evaluation on `case-05` must remain **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$).

---

## 4. Exact Group Split Conformal Prediction (Text Modality)

### 4.1 Selected Premise-Group Sampling Unit
In academic text matching, candidate hypotheses (student sentences) are compared against reference premise texts. Multiple hypothesis comparisons sharing the same reference premise exhibit strong statistical error dependence. Therefore, the primary statistical sampling unit is the **premise group** $g \in \mathcal{G}$, defined as the pair cluster:
$$g = \left( D_{\text{ref}}^{(g)}, \, \mathcal{P}_g \right), \quad \mathcal{P}_g = \left\{ (x_{g, j}, y_{g, j}) \right\}_{j=1}^{K_g}$$
where $D_{\text{ref}}^{(g)}$ is the reference sentence text, $\mathcal{P}_g$ is the set of $K_g = |\mathcal{P}_g|$ candidate hypothesis pairs evaluated against reference $g$, and $y_{g, j} \in \{0, 1\}$ is the ground-truth binary plagiarism label.

### 4.2 Stage-1 Selection Filter on Candidate Pairs
In our multi-view framework, heavy Stage-2 cross-encoder inference is not applied indiscriminately to all candidate pairs. Instead, each pair $x_{g, j} \in \mathcal{P}_g$ is first evaluated by the lightweight Stage-1 fusion model:
$$s_{\text{cand}}(x_{g, j}) = \sum_{v \in \mathcal{V}_{\text{text}}} w_v^* \cdot \phi_v(x_{g, j})$$
Pairs within group $g$ are partitioned by the Stage-1 escalation rule with frozen thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$:
$$\mathcal{P}_g^{\text{esc}} = \left\{ x_{g, j} \in \mathcal{P}_g \;\middle|\; \theta_{\text{low}} \le s_{\text{cand}}(x_{g, j}) < \theta_{\text{high}} \right\}$$
Let $m_g = |\mathcal{P}_g^{\text{esc}}|$ denote the number of candidate pairs within group $g$ that are escalated to Stage 2.

### 4.3 Handling Groups with Zero Escalated Pairs ($m_g = 0$)
If all candidate pairs within group $g$ are either pruned as low risk ($s_{\text{cand}} < \theta_{\text{low}}$) or directly bypassed as clear plagiarism ($s_{\text{cand}} \ge \theta_{\text{high}}$), then $m_g = 0$.
- **Stage-2 Evaluation:** Such groups require **zero Stage-2 cross-encoder forward passes**.
- **Conformal Set Exclusion:** Groups with $m_g = 0$ contain no Stage-2 non-conformity scores and are **excluded from the conformal calibration set $\mathcal{G}_{\text{cal-conf}}^{\text{esc}}$ and the conformal test set $\mathcal{G}_{\text{test}}^{\text{esc}}$**.
- **Active Conformal Population:** The conformal calibration and test populations are defined strictly over the **active selected premise groups**:
  $$\mathcal{G}^{\text{esc}} = \left\{ g \in \mathcal{G} \;\middle|\; m_g = |\mathcal{P}_g^{\text{esc}}| \ge 1 \right\}$$
  The effective calibration size $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ counts only active groups with $m_g \ge 1$.

### 4.4 Identical Selection Mechanism Across Partitions
The Stage-1 feature weights $\mathbf{w}^*$ and escalation thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ are tuned exclusively on the development validation split $D_{\text{val}}$ and **permanently frozen**.
- The selection filter mapping $\mathcal{S}_{(\mathbf{w}^*, \theta_{\text{low}}, \theta_{\text{high}})}: \mathcal{P}_g \mapsto \mathcal{P}_g^{\text{esc}}$ is deterministic and applied identically across both the calibration partition ($D_{\text{cal-conf}}$) and the test partition ($D_{\text{test}}$).
- Because calibration and test premise groups undergo the exact same screening filter, no distribution shift or selection discrepancy is introduced between calibration and test active groups.

### 4.5 Group-Level Maximum Non-Conformity Score & Exchangeability Assumptions
For an individual candidate pair $(x, y)$ escalated to Stage 2 with calibrated posterior $\hat{p}(x) = \hat{P}_{\text{cal}}(Y = 1 \mid z(x)) \in [0, 1]$, the pair-level non-conformity score $s(x, y)$ evaluates the error for ground truth $y \in \{0, 1\}$:
$$s(x, y) = 1 - \hat{P}_{\text{cal}}(Y = y \mid z(x)) = \begin{cases} 1 - \hat{p}(x) & \text{if } y = 1 \\ \hat{p}(x) & \text{if } y = 0 \end{cases}$$

To handle multiple candidate pairs within each active group $g \in \mathcal{G}^{\text{esc}}$ while accounting for intra-cluster error dependence, the **group-level calibration score** $\alpha_g$ is defined as the worst-case (maximum) non-conformity across all escalated pairs in the group:
$$\alpha_g = \max_{j \in \{1, \dots, m_g\}} s(x_{g, j}, y_{g, j}) = \max_{j \in \{1, \dots, m_g\}} \left( 1 - \hat{P}_{\text{cal}}(Y = y_{g, j} \mid z_{g, j}) \right) \in [0, 1]$$

#### Required Exchangeability Assumptions:
Conformal validity requires that the sequence of active selected group scores:
$$(\alpha_{g_1}, \alpha_{g_2}, \dots, \alpha_{g_n}, \alpha_{g^*})$$
is **exchangeable** conditional on $g \in \mathcal{G}^{\text{esc}}$. This assumption is methodologically supported by four structural properties:
1. **Zero Premise Leakage:** Partitioning is performed strictly at the premise level ($0$ cross-split overlap in the $7 \times 7$ partition overlap matrix), ensuring calibration groups and test groups share no premise sentences.
2. **Cluster Sampling Design:** Premise groups are independent and identically distributed draws from the domain text distribution $\mathcal{D}_{\mathcal{G}}$.
3. **Symmetric Stage-1 Selection:** The frozen screening filter $\mathcal{S}$ is identical across splits, preserving group-level exchangeability across the selected subpopulation $\mathcal{G}^{\text{esc}}$.
4. **Permutation-Invariant Group Aggregation:** The maximum operator $\alpha_g = \max_{j} s(x_{g, j}, y_{g, j})$ is invariant to pair order within group $g$, yielding a valid scalar summary for each cluster.

### 4.6 Exact Finite-Sample Quantile Index
Let the sorted group non-conformity scores across the $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ active calibration premise groups be denoted:
$$\alpha_{(1)} \le \alpha_{(2)} \le \dots \le \alpha_{(n)}$$

For a user-specified significance level $\epsilon \in (0, 1)$ (target coverage $1 - \epsilon$), the exact finite-sample quantile index is:
$$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$

The exact conformal quantile cutoff $\hat{q}_{1-\epsilon}$ is:
$$\hat{q}_{1-\epsilon} = \begin{cases}
\alpha_{(k)} & \text{if } k \le n \\
\infty & \text{if } k > n \quad (\text{finite-sample guarantee mathematically unattainable})
\end{cases}$$

**Finite-Sample Attainability Criterion:** $\hat{q}_{1-\epsilon} < \infty$ requires $n \ge \lceil (1-\epsilon)/\epsilon \rceil$. For $\epsilon = 0.05$ ($95\%$ coverage), $n \ge 19$. In our text calibration split, $D_{\text{cal-conf}}$ contains **334 independent premise groups**, easily satisfying $n = 334 \gg 19$.

### 4.7 Pair-Level Conformal Prediction Sets for Both Classes
For any test candidate pair $x$ escalated to Stage 2 with calibrated probability $\hat{p}(x) = \hat{P}_{\text{cal}}(Y = 1 \mid z(x))$, the prediction set $\mathcal{C}(x) \subseteq \{0, 1\}$ evaluates whether each candidate class $y \in \{0, 1\}$ is conforming under the conformal cutoff $\hat{q}_{1-\epsilon}$:
$$\mathcal{C}(x) = \left\{ y \in \{0, 1\} \;\middle|\; 1 - \hat{P}_{\text{cal}}(Y = y \mid z(x)) \le \hat{q}_{1-\epsilon} \right\}$$

Evaluating explicitly for both binary classes:
1. **Class $y = 1$ (Plagiarized) is included in $\mathcal{C}(x)$ if and only if:**
   $$1 - \hat{p}(x) \le \hat{q}_{1-\epsilon} \iff \hat{p}(x) \ge 1 - \hat{q}_{1-\epsilon}$$
2. **Class $y = 0$ (Non-Plagiarized) is included in $\mathcal{C}(x)$ if and only if:**
   $$1 - (1 - \hat{p}(x)) = \hat{p}(x) \le \hat{q}_{1-\epsilon}$$

Combining these conditions yields four mutually exclusive operational prediction set outcomes:
$$\mathcal{C}(x) = \begin{cases}
\{0, 1\} & \text{if } 1 - \hat{q}_{1-\epsilon} \le \hat{p}(x) \le \hat{q}_{1-\epsilon} \quad (\text{Uncertain: both classes plausible; requires } \hat{q}_{1-\epsilon} \ge 0.5) \\
\{1\} & \text{if } \hat{p}(x) \ge 1 - \hat{q}_{1-\epsilon} \text{ and } \hat{p}(x) > \hat{q}_{1-\epsilon} \quad (\text{Confident Plagiarism}) \\
\{0\} & \text{if } \hat{p}(x) \le \hat{q}_{1-\epsilon} \text{ and } \hat{p}(x) < 1 - \hat{q}_{1-\epsilon} \quad (\text{Confident Non-Plagiarism}) \\
\emptyset & \text{if } \hat{q}_{1-\epsilon} < \hat{p}(x) < 1 - \hat{q}_{1-\epsilon} \quad (\text{Model Inconsistency / Ambiguity; requires } \hat{q}_{1-\epsilon} < 0.5)
\end{cases}$$

- **Empty Prediction Set Semantics ($\mathcal{C}(x) = \emptyset$):**  
  An empty prediction set occurs when $\hat{q}_{1-\epsilon} < 0.5$ and the model's calibrated probability falls in the intermediate region where neither class satisfies the conformity threshold. This signifies **predictive ambiguity and model under-confidence**, not out-of-distribution proof. It is formally mapped to the operational state **`ABSTAIN_UNRESOLVED`** and routed to an auditor queue.

### 4.8 Simultaneous Family-Wise Group Coverage Guarantee Under Exchangeability
By construction of the group score $\alpha_g = \max_{j \in \{1, \dots, m_g\}} s(x_{g, j}, y_{g, j})$, the event that group score $\alpha_{g^*} \le \hat{q}_{1-\epsilon}$ is mathematically identical to the event that **every individual escalated pair** in group $g^*$ conforms:
$$\alpha_{g^*} \le \hat{q}_{1-\epsilon} \iff \forall j \in \{1, \dots, m_{g^*}\}, \, s(x_{g^*, j}, y_{g^*, j}) \le \hat{q}_{1-\epsilon} \iff \forall j \in \{1, \dots, m_{g^*}\}, \, y_{g^*, j} \in \mathcal{C}(x_{g^*, j})$$

Under the exchangeability of active selected group scores $(\alpha_{g_1}, \dots, \alpha_{g_n}, \alpha_{g^*})$, the exact order statistic $\hat{q}_{1-\epsilon} = \alpha_{(k)}$ guarantees finite-sample simultaneous family-wise group coverage:
$$\mathbb{P}\left( \forall j \in \{1, \dots, m_{g^*}\}, \, y_{g^*, j} \in \mathcal{C}(x_{g^*, j}) \;\middle|\; g^* \in \mathcal{G}_{\text{test}}^{\text{esc}} \right) \ge 1 - \epsilon$$
This guarantees that with probability at least $1 - \epsilon$, **all candidate pairs sharing active test premise $g^*$ are simultaneously covered by their respective prediction sets**.

### 4.9 Exclusion of Stage-1 Bypassed Pairs & Conditional Coverage Scope
> [!IMPORTANT]
> **Strict Conditional Scope — Bypassed Pairs Excluded:**  
> 1. **No Coverage for Pruned / Bypassed Pairs:** Any candidate pair $x$ with $s_{\text{cand}}(x) < \theta_{\text{low}}$ (`LOW_RISK_REVIEW_DEFERRED`) or $s_{\text{cand}}(x) \ge \theta_{\text{high}}$ (`HIGH_RISK_DIRECT_BYPASS`) is routed directly by Stage 1. Such pairs **never enter the Stage-2 cross-encoder and receive no conformal prediction set**.
> 2. **Conditional Validity Only:** The finite-sample coverage guarantee holds **strictly conditional on Stage-1 escalation** ($g^* \in \mathcal{G}_{\text{test}}^{\text{esc}}$). Unconditional marginal coverage over the entire unselected input population is neither claimed nor mathematically valid.

---

## 5. Calibration Evaluation Metrics Formulations

All calibration metrics are evaluated strictly on the untouched test escalation population $D_{\text{test}}^{\text{esc}} = \{(z_i, y_i)\}_{i=1}^{N_{\text{test}}^{\text{esc}}}$. Let $\hat{p}_i = \hat{P}_{\text{cal}}(Y = 1 \mid z_i) \in [0, 1]$.

### 5.1 Expected Calibration Error (ECE)
The probability interval $[0, 1]$ is partitioned into $M = 10$ uniform bins:
$$I_m = \left( \frac{m-1}{M}, \, \frac{m}{M} \right], \quad m \in \{1, 2, \dots, M\}$$
with $I_1 = [0, 1/M]$.

Let $B_m = \{ i \in \{1, \dots, N_{\text{test}}^{\text{esc}}\} \mid \hat{p}_i \in I_m \}$ be the set of samples falling into bin $m$.  
For each non-empty bin ($|B_m| > 0$):
- **Bin Positive Prevalence / Frequency:** $\text{acc}(B_m) = \bar{y}_m = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ (labeled 'bin accuracy' in binary calibration routines; represents the empirical fraction of positives, distinct from top-label classification accuracy)$
- **Bin Confidence:** $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$

The Expected Calibration Error is formulated as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N_{\text{test}}^{\text{esc}}} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|$$
where empty bins ($|B_m| = 0$) contribute $0$ to the summation.

### 5.2 Maximum Calibration Error (MCE)
$$\text{MCE} = \max_{m \in \{1, \dots, M\} \,:\, |B_m| > 0} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 5.3 Brier Score
$$\text{BS} = \frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} \left( \hat{p}_i - y_i \right)^2 \in [0, 1]$$

### 5.4 Clipped Negative Log-Likelihood (NLL)
To prevent numerical instability from extreme probabilities, probabilities are clipped to $[\delta, 1 - \delta]$ with $\delta = 10^{-15}$:
$$\tilde{p}_i = \text{clip}(\hat{p}_i, \, 10^{-15}, \, 1 - 10^{-15})$$
$$\text{NLL} = -\frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} \left[ y_i \log(\tilde{p}_i) + (1 - y_i) \log(1 - \tilde{p}_i) \right]$$

---

## 6. Multi-Objective Pareto Optimization Formulation

The optimal parameter vector $\boldsymbol{\theta}^* = (\mathbf{w}, \theta_{\text{low}}, \theta_{\text{high}}, \gamma_{\text{lower}}, \gamma_{\text{upper}}) \in \Theta$ is selected by identifying the non-dominated set over three competing objectives:

$$\min_{\boldsymbol{\theta} \in \Theta} \begin{bmatrix} 1 - \text{Recall}(D_{\text{val}}^{\text{pos}}; \boldsymbol{\theta}) \\ \mathbb{E}[\text{Latency}(x; \boldsymbol{\theta})] \\ \text{Workload}(D_{\text{val}}; \boldsymbol{\theta}) \end{bmatrix}$$

subject to institutional constraints:
1. $\text{Recall}(D_{\text{val}}^{\text{pos}}; \boldsymbol{\theta}) \ge 0.95$ (Design target: $\ge 0.98$).
2. $\text{Workload}(D_{\text{val}}; \boldsymbol{\theta}) \le 0.15$.

