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

### 3.4 Consequence for IR-Plag `case-05`
In the IR-Plag `case-05` corpus, there are 16 base program identities (`prog_original` and `prog_np_01` to `prog_np_15`). As proven by [case_05_fold_feasibility_report.json](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/case_05_fold_feasibility_report.json), **all 53 positive pairs share the base identity `prog_original`**. Under any group-disjoint cross-validation split:
1. In the fold holding out `prog_original` (Fold 1):  
   $N_{\text{pos}}^{\text{train}} = 0$. The training set contains only negative pairs. Platt calibration fails by Theorem 1 / scikit-learn `ValueError`.
2. In all other folds (Folds 2 to 5):  
   `prog_original` is retained in training, meaning the held-out test fold contains zero positive pairs ($N_{\text{pos}}^{\text{test}} = 0$).

When $N_{\text{pos}}^{\text{test}} = 0$:
- $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{0}{0}$ (undefined).
- $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{0}{\text{FP}}$ ($0/0$ if no false alarms).
- $\text{ROC-AUC}$ and $\text{PR-AUC}$ require both positive and negative evaluation instances; with 0 positives, rank statistics are mathematically undefined.

**Methodological Conclusion:**  
Group-disjoint cross-fitting and probability calibration are structurally impossible on `case-05`.
Code plagiarism evaluation on `case-05` must remain **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$).

---

## 4. Exact Group Split Conformal Prediction (Text Modality)

### 4.1 Group-Level Conformal Prediction
Let $\mathcal{G}_{\text{cal-conf}}^{\text{esc}}$ denote the set of premise groups in the escalated conformal calibration split $D_{\text{cal-conf}}^{\text{esc}}$. For each premise group $g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}$, let $\mathcal{P}_g^{\text{esc}}$ denote the escalated candidate pairs associated with premise $g$.

The group non-conformity score $\alpha_g$ is defined as the worst-case pair non-conformity:
$$\alpha_g = \max_{i \in \mathcal{P}_g^{\text{esc}}} \left( 1 - \hat{P}_{\text{cal}}(Y = y_i \mid z_i) \right)$$
where $y_i \in \{0, 1\}$ is the ground-truth label.

Let $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ be the number of calibration premise groups. Let the sorted group non-conformity scores be denoted:
$$s_{(1)} \le s_{(2)} \le \dots \le s_{(n)}$$
For a user-specified significance level $\epsilon \in (0, 1)$ (target coverage $1 - \epsilon$), the exact finite-sample quantile index is:
$$k = \left\lceil (n + 1)(1 - \epsilon) \right\rceil$$

The exact conformal quantile cutoff is:
$$\hat{q}_{1-\epsilon} = \begin{cases}
s_{(k)} & \text{if } k \le n \\
\infty & \text{if } k > n
\end{cases}$$

### 4.2 Finite-Sample Attainability Criterion
**Proposition 1 (Finite-Sample Attainability):**  
*The conformal quantile cutoff $\hat{q}_{1-\epsilon}$ is finite if and only if:*
$$n \ge \left\lceil \frac{1-\epsilon}{\epsilon} \right\rceil$$
For $\epsilon = 0.05$ ($95\%$ coverage), the finite-sample threshold requires $n \ge \lceil 0.95/0.05 \rceil = 19$ groups. In our text calibration partition [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv), $D_{\text{cal-conf}}$ contains **334 independent premise groups**, easily satisfying $n \gg 19$.

### 4.3 Test Group Prediction Sets
For a new escalated test premise group $g^* \in \mathcal{G}_{\text{test}}^{\text{esc}}$ and any candidate pair $j \in \mathcal{P}_{g^*}^{\text{esc}}$, the conformal prediction set is:
$$\mathcal{C}(x_j) = \left\{ y \in \{0, 1\} \;\middle|\; 1 - \hat{P}_{\text{cal}}(Y = y \mid z_j) \le \hat{q}_{1-\epsilon} \right\}$$

Under group exchangeability, the marginal group coverage guarantee holds:
$$\mathbb{P}\left( \forall j \in \mathcal{P}_{g^*}^{\text{esc}}, \, y_j \in \mathcal{C}(x_j) \right) \ge 1 - \epsilon$$

---

## 5. Calibration Evaluation Metrics Formulations

All calibration metrics are evaluated strictly on the untouched test escalation population $D_{\text{test}}^{\text{esc}} = \{(z_i, y_i)\}_{i=1}^{N_{\text{test}}^{\text{esc}}}$. Let $\hat{p}_i = \hat{P}_{\text{cal}}(Y = 1 \mid z_i) \in [0, 1]$.

### 5.1 Expected Calibration Error (ECE)
The probability interval $[0, 1]$ is partitioned into $M = 10$ uniform bins:
$$I_m = \left( \frac{m-1}{M}, \, \frac{m}{M} \right], \quad m \in \{1, 2, \dots, M\}$$
with $I_1 = [0, 1/M]$.

Let $B_m = \{ i \in \{1, \dots, N_{\text{test}}^{\text{esc}}\} \mid \hat{p}_i \in I_m \}$ be the set of samples falling into bin $m$.  
For each non-empty bin ($|B_m| > 0$):
- **Bin Accuracy:** $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$
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

