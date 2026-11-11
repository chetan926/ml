# Mathematical Formulations & Statistical Proofs (Version 3.6)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Document Version:** Version 3.6  
**Date:** November 10, 2026  
**Status:** MATHEMATICALLY AUDITED & FORMALLY VERIFIED

---

## 1. Multi-View Candidate Scoring Formulations

### 1.1 Text Modality Candidate Scoring
Let $(D_A, D_B)$ denote a pair of academic text documents. The Stage-1 multi-view screening representation is a vector of four similarity scores:
$$\mathbf{v}_{\text{sim}}(D_A, D_B) = \begin{bmatrix} s_{\text{char}}(D_A, D_B) \\ s_{\text{word}}(D_A, D_B) \\ s_{\text{sem}}(D_A, D_B) \\ s_{\text{ling}}(D_A, D_B) \end{bmatrix} \in [0, 1]^4$$

1. **Character n-gram Substring Similarity ($s_{\text{char}}$):**
   $$s_{\text{char}}(D_A, D_B) = \frac{\sum_{n=3}^5 \omega_n \cdot \cos\left(\mathbf{t}_n(D_A), \mathbf{t}_n(D_B)\right)}{\sum_{n=3}^5 \omega_n}$$
   where $\mathbf{t}_n(D)$ is the character $n$-gram TF-IDF vector of document $D$.

2. **Word n-gram Lexical Similarity ($s_{\text{word}}$):**
   $$s_{\text{word}}(D_A, D_B) = \frac{1}{2} \cos\left(\mathbf{w}_1(D_A), \mathbf{w}_1(D_B)\right) + \frac{1}{2} \cos\left(\mathbf{w}_2(D_A), \mathbf{w}_2(D_B)\right)$$

3. **Dense Semantic Embedding Similarity ($s_{\text{sem}}$):**
   $$s_{\text{sem}}(D_A, D_B) = \frac{\mathbf{e}(D_A)^{\top} \mathbf{e}(D_B)}{\|\mathbf{e}(D_A)\|_2 \|\mathbf{e}(D_B)\|_2}$$
   where $\mathbf{e}(D) \in \mathbb{R}^{384}$ is the mean-pooled token embedding from `all-MiniLM-L6-v2`.

4. **Linguistic Surface Similarity ($s_{\text{ling}}$):**
   $$s_{\text{ling}}(D_A, D_B) = 1 - \tanh\left( \|\mathbf{u}_{\text{ling}}(D_A) - \mathbf{u}_{\text{ling}}(D_B)\|_2 \right)$$

The fused Stage-1 candidate score is given by the logistic combination:
$$s_{\text{cand}}^{\text{text}}(D_A, D_B) = \sigma\left( \mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(D_A, D_B) + b \right) = \frac{1}{1 + e^{-(\mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(D_A, D_B) + b)}}$$

### 1.2 Code Modality Candidate Scoring
Let $(C_A, C_B)$ denote a pair of Java source programs. The fused Stage-1 candidate score is:
$$s_{\text{cand}}^{\text{code}}(C_A, C_B) = \beta_1 s_{\text{tok}}(C_A, C_B) + \beta_2 s_{\text{ast}}(C_A, C_B) + \beta_3 s_{\text{metric}}(C_A, C_B)$$
subject to $\beta_1, \beta_2, \beta_3 \ge 0$ and $\beta_1 + \beta_2 + \beta_3 = 1$.

1. **Normalized Token Edit Similarity ($s_{\text{tok}}$):**
   $$s_{\text{tok}}(C_A, C_B) = 1 - \frac{\text{Levenshtein}(\tau(C_A), \tau(C_B))}{\max(|\tau(C_A)|, |\tau(C_B)|)}$$
   where $\tau(C)$ denotes the canonicalized token stream.

2. **Normalized AST Subtree Similarity ($s_{\text{ast}}$):**
   $$s_{\text{ast}}(C_A, C_B) = \frac{2 \cdot |\mathcal{T}(C_A) \cap \mathcal{T}(C_B)|}{|\mathcal{T}(C_A)| + |\mathcal{T}(C_B)|}$$
   where $\mathcal{T}(C)$ is the set of canonicalized subtree hashes extracted from the AST.

3. **Software Metric Vector Cosine Similarity ($s_{\text{metric}}$):**
   $$s_{\text{metric}}(C_A, C_B) = \frac{\mathbf{m}(C_A)^{\top} \mathbf{m}(C_B)}{\|\mathbf{m}(C_A)\|_2 \|\mathbf{m}(C_B)\|_2}$$
   where $\mathbf{m}(C) \in \mathbb{R}^8$ captures Halstead Volume, Halstead Difficulty, Cyclomatic Complexity, loop nesting depth, method count, parameter count, LOC, and comment density.

---

## 2. Adaptive Routing and Escalation Decision Rule

For any candidate pair $x = (A, B)$, the Stage-1 routing function partitions the candidate space into three operational streams using thresholds $\theta_{\text{low}} < \theta_{\text{high}}$:

$$\text{Route}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \\
\text{ESCALATE\_TO\_STAGE2} & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \\
\text{HIGH\_RISK\_DIRECT\_BYPASS} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}}
\end{cases}$$

- **Computational Conservation:** Stage 2 (cross-encoder) is evaluated strictly on the escalation subset $\mathcal{E} = \{ x \mid \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \}$.
- **Conformal Consistency Rule:** Pairs routed to `HIGH_RISK_DIRECT_BYPASS` never receive Stage-2 scores; hence, they are excluded from the conformal calibration and test sets.

---

## 3. Probability Calibration and Failure Proofs

### 3.1 Platt Scaling Mathematical Formulation
For escalated candidate pairs, let $z_i \in \mathbb{R}$ denote the uncalibrated logit output from the Stage-2 cross-encoder detector. Platt scaling models the calibrated posterior probability $\hat{P}_{\text{cal}}(Y = 1 \mid z)$ via a two-parameter sigmoid:
$$\hat{P}_{\text{cal}}(Y = 1 \mid z) = \sigma(A z + B) = \frac{1}{1 + \exp(-(A z + B))}$$
where $(A, B) \in \mathbb{R}^2$ are estimated by minimizing the binary cross-entropy loss over a probability calibration partition $D_{\text{cal-prob}}^{\text{esc}} = \{(z_i, y_i)\}_{i=1}^N$:
$$\mathcal{L}_{\text{Platt}}(A, B) = -\sum_{i=1}^N \left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log(1 - \sigma(A z_i + B)) \right]$$

### 3.2 Formal Proof: Unregularized Platt Scaling Divergence on Zero-Positive Slices
**Theorem 1 (Unregularized Platt Divergence):**  
*If a calibration training fold contains zero positive instances ($N_{\text{pos}}^{\text{train}} = 0$, meaning $y_i = 0$ for all $i \in \{1, \dots, N\}$), the unregularized Platt scaling maximum likelihood estimator does not exist in $\mathbb{R}^2$, and gradient optimization diverges to $B \to -\infty$.*

**Proof:**  
When $y_i = 0$ for all $i$, the objective function simplifies to:
$$\mathcal{L}(A, B) = -\sum_{i=1}^N \log(1 - \sigma(A z_i + B)) = \sum_{i=1}^N \log\left( 1 + e^{A z_i + B} \right)$$
Compute the partial derivative with respect to intercept $B$:
$$\frac{\partial \mathcal{L}}{\partial B} = \sum_{i=1}^N \frac{e^{A z_i + B}}{1 + e^{A z_i + B}} = \sum_{i=1}^N \sigma(A z_i + B)$$
Because $\sigma(u) > 0$ for all finite $u \in \mathbb{R}$, we have $\frac{\partial \mathcal{L}}{\partial B} > 0$ everywhere on $\mathbb{R}^2$.  
Therefore, $\mathcal{L}(A, B)$ is strictly increasing in $B$ and has no stationary point:
$$\lim_{B \to -\infty} \mathcal{L}(A, B) = \sum_{i=1}^N \log(1 + 0) = 0$$
The infimum of the loss is 0, which is attained only in the limit as $B \to -\infty$. Thus, numerical maximum likelihood optimization diverges to $-\infty$. $\blacksquare$

### 3.3 Formal Proof: Regularized Platt Scaling Exception in Standard Libraries
Under standard implementations (such as scikit-learn's `LogisticRegression(C=1.0)` with $L_2$ regularization $\frac{1}{2C} (A^2 + B^2)$):
$$\min_{A, B} \left[ \sum_{i=1}^N \log(1 + e^{A z_i + B}) + \frac{1}{2C} (A^2 + B^2) \right]$$
While mathematically the $L_2$ penalty forces a finite minimizer $(A^*, B^*)$ with $B^* \ll 0$, standard numerical solvers require at least two distinct binary classes to construct the target matrix.
Specifically, when provided with a dataset where all labels $y_i = 0$, scikit-learn raises a fatal runtime exception:
$$\text{\texttt{ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0}}$$
Consequently, neither unregularized nor regularized Platt scaling can produce an operational binary calibrator on a single-class training fold.

### 3.4 Consequence for IR-Plag `case-05`
In the IR-Plag `case-05` corpus, there are 16 base program identities (`prog_original` and 15 non-plagiarized programs `prog_np_01` to `prog_np_15`).  
As proven by [case_05_pair_to_program_mapping.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_pair_to_program_mapping.csv), **all 53 positive pairs share the base identity `prog_original`**.
Under any group-disjoint cross-validation partition across base identities:
1. In the fold holding out `prog_original` (Fold 1 in [case_05_fold_audit.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/case_05_fold_audit.csv)):  
   $N_{\text{pos}}^{\text{train}} = 0$. The training set contains only negative pairs. Platt calibration fails by Theorem 1 / scikit-learn `ValueError`.
2. In all other folds (Folds 2 to 5):  
   `prog_original` is retained in training, meaning the held-out test fold contains zero positive pairs ($N_{\text{pos}}^{\text{test}} = 0$).

When $N_{\text{pos}}^{\text{test}} = 0$:
- $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{0}{0}$ (undefined).
- $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{0}{\text{FP}}$ ($0/0$ if no false alarms).
- $\text{ROC-AUC}$ and $\text{PR-AUC}$ require both positive and negative evaluation instances; with 0 positives, the rank statistics are mathematically undefined.
- $\text{ECE}$ on the positive class is trivial/undefined.

**Methodological Conclusion:**  
Group-disjoint cross-fitting and probability calibration are structurally impossible on `case-05`.
Code plagiarism evaluation on `case-05` must remain **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$).

---

## 4. Split Conformal Prediction Formulations

### 4.1 Group-Level Conformal Prediction (Text Modality)
Let $\mathcal{G}_{\text{cal-conf}}^{\text{esc}}$ denote the set of premise groups in the escalated conformal calibration split $D_{\text{cal-conf}}^{\text{esc}}$.  
For each premise group $g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}$, let $\mathcal{P}_g^{\text{esc}}$ denote the set of candidate pairs associated with premise $g$ that were escalated by Stage 1.

The group non-conformity score $\alpha_g$ is defined as the worst-case pair non-conformity across all pairs in group $g$:
$$\alpha_g = \max_{i \in \mathcal{P}_g^{\text{esc}}} \left( 1 - \hat{P}_{\text{cal}}(Y = y_i \mid z_i) \right)$$
where $y_i \in \{0, 1\}$ is the ground-truth label, and $\hat{P}_{\text{cal}}(Y = 0 \mid z_i) = 1 - \hat{P}_{\text{cal}}(Y = 1 \mid z_i)$.

Let $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ be the number of calibration premise groups. For a user-specified significance level $\epsilon \in (0, 1)$ (target coverage $1 - \epsilon$, e.g., $\epsilon = 0.05$ for $95\%$ coverage), the conformal quantile cutoff is:
$$\hat{q}_{1-\epsilon} = \text{Quantile}\left( \frac{\lceil (n+1)(1-\epsilon) \rceil}{n}; \, \{\alpha_g\}_{g=1}^n \right)$$

### 4.2 Finite-Sample Attainability Criterion
**Proposition 1 (Finite-Sample Attainability):**  
*The conformal quantile cutoff $\hat{q}_{1-\epsilon}$ is finite if and only if:*
$$n \ge \left\lceil \frac{1-\epsilon}{\epsilon} \right\rceil$$
*Proof:*  
The empirical quantile index is $\frac{\lceil (n+1)(1-\epsilon) \rceil}{n}$. If $\lceil (n+1)(1-\epsilon) \rceil > n$, the required order statistic exceeds the sample size $n$, yielding an empirical quantile of $\infty$.  
$$\lceil (n+1)(1-\epsilon) \rceil \le n \iff (n+1)(1-\epsilon) \le n \iff 1 - \epsilon \le n\epsilon \iff n \ge \frac{1-\epsilon}{\epsilon}$$
Because $n$ is an integer, this condition is exactly $n \ge \lceil (1-\epsilon)/\epsilon \rceil$. $\blacksquare$

- For $\epsilon = 0.05$ ($95\%$ coverage), the finite-sample threshold requires:
  $$n \ge \left\lceil \frac{0.95}{0.05} \right\rceil = 19 \text{ groups}$$
In our text calibration partition [text_partition_membership.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv), $D_{\text{cal-conf}}$ contains **334 independent premise groups**, easily satisfying $n \gg 19$.

### 4.3 Test Group Prediction Sets
For a new escalated test premise group $g^* \in \mathcal{G}_{\text{test}}^{\text{esc}}$ and any candidate pair $j \in \mathcal{P}_{g^*}^{\text{esc}}$, the conformal prediction set is constructed as:
$$\mathcal{C}(x_j) = \left\{ y \in \{0, 1\} \;\middle|\; 1 - \hat{P}_{\text{cal}}(Y = y \mid z_j) \le \hat{q}_{1-\epsilon} \right\}$$

Under group exchangeability, the marginal group coverage guarantee holds:
$$\mathbb{P}\left( \forall j \in \mathcal{P}_{g^*}^{\text{esc}}, \, y_j \in \mathcal{C}(x_j) \right) \ge 1 - \epsilon$$

---

## 5. Calibration Evaluation Metrics Formulations

All calibration metrics are evaluated strictly on the untouched test escalation population $D_{\text{test}}^{\text{esc}} = \{(z_i, y_i)\}_{i=1}^{N_{\text{test}}^{\text{esc}}}$.

Let $\hat{p}_i = \hat{P}_{\text{cal}}(Y = 1 \mid z_i) \in [0, 1]$ be the calibrated predicted probability of plagiarism for pair $i$.

### 5.1 Expected Calibration Error (ECE)
The probability interval $[0, 1]$ is partitioned into $M = 10$ uniform bins:
$$I_m = \left( \frac{m-1}{M}, \, \frac{m}{M} \right], \quad m \in \{1, 2, \dots, M\}$$
with the first bin $I_1$ closed at both ends: $I_1 = [0, 1/M]$.

Let $B_m = \{ i \in \{1, \dots, N_{\text{test}}^{\text{esc}}\} \mid \hat{p}_i \in I_m \}$ be the set of samples falling into bin $m$.  
For each non-empty bin ($|B_m| > 0$), define:
- **Bin Accuracy:** $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$
- **Bin Confidence:** $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$

The Expected Calibration Error is formulated as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N_{\text{test}}^{\text{esc}}} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|$$
where empty bins ($|B_m| = 0$) contribute $0$ to the summation.

### 5.2 Maximum Calibration Error (MCE)
The Maximum Calibration Error measures the worst-case divergence across all occupied confidence bins:
$$\text{MCE} = \max_{m \in \{1, \dots, M\} \,:\, |B_m| > 0} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 5.3 Brier Score
The Brier Score measures the mean squared difference between predicted probabilities and ground-truth binary labels:
$$\text{BS} = \frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} \left( \hat{p}_i - y_i \right)^2$$
The Brier score strictly lies in $[0, 1]$, where $\text{BS} = 0$ represents perfect probabilistic calibration and classification.

### 5.4 Negative Log-Likelihood (NLL) with Probability Clipping
To prevent numerical overflow or undefined $-\log(0)$ evaluations from extreme probabilities, probabilities are clipped to $[\delta, 1 - \delta]$ with $\delta = 10^{-15}$:
$$\tilde{p}_i = \text{clip}(\hat{p}_i, \, 10^{-15}, \, 1 - 10^{-15})$$

The clipped Negative Log-Likelihood is formulated as:
$$\text{NLL} = -\frac{1}{N_{\text{test}}^{\text{esc}}} \sum_{i=1}^{N_{\text{test}}^{\text{esc}}} \left[ y_i \log(\tilde{p}_i) + (1 - y_i) \log(1 - \tilde{p}_i) \right]$$

---

## 6. Multi-Objective Pareto Optimization Formulation

The optimal parameter vector $\boldsymbol{\theta}^* = (\mathbf{w}, \theta_{\text{low}}, \theta_{\text{high}}, \gamma_{\text{lower}}, \gamma_{\text{upper}}) \in \Theta$ is selected by finding the non-dominated set (Pareto frontier) over three conflicting objectives:

$$\min_{\boldsymbol{\theta} \in \Theta} \begin{bmatrix} \mathcal{L}_{\text{recall}}(\boldsymbol{\theta}) \\ \mathcal{C}_{\text{latency}}(\boldsymbol{\theta}) \\ \mathcal{U}_{\text{uncertainty}}(\boldsymbol{\theta}) \end{bmatrix}$$

subject to institutional deployment constraints:
1. **Recall Guarantee:** $\text{Recall}(D_{\text{val}}^{\text{pos}}; \boldsymbol{\theta}) \ge 0.95$ (target: $\ge 0.98$).
2. **Computational Budget:** $\mathbb{E}_{x \sim D_{\text{val}}} [\text{Cost}(x; \boldsymbol{\theta})] \le B_{\text{max}}$.
3. **Audit Workload Bound:** $\mathcal{U}_{\text{uncertainty}}(\boldsymbol{\theta}) \le 0.15$ (at most $15\%$ of cohort escalated to human review).

