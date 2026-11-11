# Mathematical Formulations and Statistical Framework (Version 3.4)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.3 (Final Dataset, Routing, and Integrity Verification)  
**Document Version:** Version 3.4  
**Date:** March 2026  
**Status:** FULL AUDIT CERTIFIED (Empirical Ground-Truth Verified)

---

## 1. Multi-View Candidate Generation Formulations

### 1.1 Text Multi-View Extraction & Fusion
Let $x = (D_A, D_B)$ denote a candidate pair of academic text documents. The Stage-1 multi-view feature vector is defined as:
$$\mathbf{v}_{\text{sim}}(x) = \begin{bmatrix} s_{\text{char}}(D_A, D_B) \\ s_{\text{word}}(D_A, D_B) \\ s_{\text{sem}}(D_A, D_B) \\ s_{\text{ling}}(D_A, D_B) \end{bmatrix} \in [0, 1]^4$$
where:
1. **Character n-gram similarity ($s_{\text{char}}$):** Substring TF-IDF cosine similarity over character 3-grams, 4-grams, and 5-grams:
   $$s_{\text{char}}(D_A, D_B) = \frac{\mathbf{u}_{\text{char}}(D_A)^{\top} \mathbf{u}_{\text{char}}(D_B)}{\|\mathbf{u}_{\text{char}}(D_A)\|_2 \|\mathbf{u}_{\text{char}}(D_B)\|_2}$$
2. **Word n-gram similarity ($s_{\text{word}}$):** Token-level unigram and bigram TF-IDF cosine similarity:
   $$s_{\text{word}}(D_A, D_B) = \frac{\mathbf{u}_{\text{word}}(D_A)^{\top} \mathbf{u}_{\text{word}}(D_B)}{\|\mathbf{u}_{\text{word}}(D_A)\|_2 \|\mathbf{u}_{\text{word}}(D_B)\|_2}$$
3. **Dense semantic embedding similarity ($s_{\text{sem}}$):** Cosine similarity between `all-MiniLM-L6-v2` dense embeddings:
   $$s_{\text{sem}}(D_A, D_B) = \frac{\mathbf{e}(D_A)^{\top} \mathbf{e}(D_B)}{\|\mathbf{e}(D_A)\|_2 \|\mathbf{e}(D_B)\|_2}$$
4. **Linguistic surface complexity divergence ($s_{\text{ling}}$):** Gaussian similarity over normalized length and type-token ratio divergence:
   $$s_{\text{ling}}(D_A, D_B) = \exp\left( -\frac{|L(D_A) - L(D_B)|}{\sigma_L^2} - \frac{|\text{TTR}(D_A) - \text{TTR}(D_B)|}{\sigma_{\text{TTR}}^2} \right)$$

The fused candidate screening score $s_{\text{cand}}^{\text{text}}(x)$ is:
$$s_{\text{cand}}^{\text{text}}(x) = \sigma\left( \mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(x) + b \right) = \frac{1}{1 + \exp\left( -(\mathbf{w}^{\top} \mathbf{v}_{\text{sim}}(x) + b) \right)}$$

---

## 2. Stage-1 Routing and Escalation Condition

To guarantee that the conformal calibration population matches the deployed execution pipeline, the candidate pair space is partitioned using lower and upper screening thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$:

$$\text{Routing}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \\
\text{ESCALATE\_TO\_STAGE2} & \text{if } \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \\
\text{HIGH\_RISK\_DIRECT\_BYPASS} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}}
\end{cases}$$

### 2.1 Escalation Slice Definition
The escalation set $\mathcal{E}$ consists strictly of pairs falling in the ambiguous band:
$$\mathcal{E} = \left\{ x : \theta_{\text{low}} \le s_{\text{cand}}(x) < \theta_{\text{high}} \right\}$$

Pairs with $s_{\text{cand}}(x) \ge \theta_{\text{high}}$ represent obvious lexical/semantic matches. They bypass expensive Stage-2 forward passes and are directly routed to the evidence synthesizer with the state `HIGH_RISK_CANDIDATE_FLAGGED`.

> [!IMPORTANT]
> **Mathematical Escalation Invariant:** Pairs in the bypass slice ($s_{\text{cand}} \ge \theta_{\text{high}}$) **never receive a Stage-2 prediction**. Consequently, the calibration populations $D_{\text{cal-prob}}^{\text{esc}}$ and $D_{\text{cal-conf}}^{\text{esc}}$ **must strictly exclude bypass pairs**, ensuring that every calibrated instance possesses a valid Stage-2 cross-encoder prediction.

---

## 3. Probability Calibration Formulations (Stage 2)

For escalated candidate pairs ($x \in \mathcal{E}$), the Stage-2 cross-encoder computes raw logit $z(x) = f_{\text{cross}}(x) \in \mathbb{R}$.

### 3.1 Platt Scaling (Parametric Logistic Calibration)
The calibrated posterior probability of reuse $\hat{P}_{\text{cal}}(Y = 1 \mid x)$ is modeled via logistic transformation:
$$\hat{P}_{\text{cal}}(Y = 1 \mid x) = \sigma(A z(x) + B) = \frac{1}{1 + \exp(-(A z(x) + B))}$$
The parameters $(A^*, B^*)$ are optimized strictly on the escalated calibration slice:
$$D_{\text{cal-prob}}^{\text{esc}} = \left\{ (x_i, y_i) \in D_{\text{cal-prob}} : \theta_{\text{low}} \le s_{\text{cand}}(x_i) < \theta_{\text{high}} \right\}$$
by minimizing the regularized negative log-likelihood:
$$(A^*, B^*) = \arg\min_{A, B} \sum_{(x_i, y_i) \in D_{\text{cal-prob}}^{\text{esc}}} -\left[ y_i \log \sigma(A z_i + B) + (1 - y_i) \log(1 - \sigma(A z_i + B)) \right]$$

### 3.2 Evaluation of Calibration Quality
Calibration metrics are evaluated strictly on the escalated test slice $D_{\text{test}}^{\text{esc}} = \{ (x_i, y_i) \in D_{\text{test}} : \theta_{\text{low}} \le s_{\text{cand}}(x_i) < \theta_{\text{high}} \}$:
- **Expected Calibration Error (ECE):** Partitioning $[0, 1]$ into $M = 10$ bins $B_m$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{|D_{\text{test}}^{\text{esc}}|} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Brier Score:**
  $$\text{BS} = \frac{1}{|D_{\text{test}}^{\text{esc}}|} \sum_{i=1}^{|D_{\text{test}}^{\text{esc}}|} (\hat{P}_{\text{cal}}(Y = 1 \mid x_i) - y_i)^2$$

---

## 4. Group Conformal Prediction Formulations (Text Modality)

### 4.1 Group Non-Conformity Scoring
Let $\mathcal{G}_{\text{cal-conf}}^{\text{esc}}$ denote the set of unique reference premises appearing in the escalated calibration set:
$$\mathcal{G}_{\text{cal-conf}}^{\text{esc}} = \left\{ g \in \mathcal{G}_{\text{cal-conf}} : \exists i \in \mathcal{P}_g, \, \theta_{\text{low}} \le s_{\text{cand}}(i) < \theta_{\text{high}} \right\}$$

For each premise group $g \in \mathcal{G}_{\text{cal-conf}}^{\text{esc}}$, the cluster non-conformity score $\alpha_g$ is the worst-case pair error:
$$\alpha_g = \max_{i \in \mathcal{P}_g \cap \mathcal{E}} \left( 1 - \hat{P}_{\text{cal}}(Y = y_i \mid \mathbf{x}_i) \right)$$

### 4.2 Finite-Sample Quantile and Attainability Lower Bound
Let $n = |\mathcal{G}_{\text{cal-conf}}^{\text{esc}}|$ be the number of escalated premise groups. The conformal cutoff $\hat{q}_{1-\epsilon}$ is the $k$-th smallest element of the augmented set $\{\alpha_g\}_{g=1}^n \cup \{\infty\}$, where:
$$k = \lceil (n + 1)(1 - \epsilon) \rceil$$

**Sample Size Attainability Condition:**
The $(1-\epsilon)$ quantile is finite if and only if $k \le n$. This requires:
$$n \ge \left\lceil \frac{1 - \epsilon}{\epsilon} \right\rceil$$
For $\epsilon = 0.05$ (95% coverage guarantee), $n \ge \lceil 0.95 / 0.05 \rceil = 19$ unique premise groups.  
If $n < 19$, then $k = n + 1$, yielding $\hat{q}_{1-\epsilon} = \infty$. The system formally abstains from claiming a finite-sample guarantee when this condition is violated.

### 4.3 Prediction Sets and Coverage Guarantees
For an escalated test pair $i \in \mathcal{P}_{g^*} \cap \mathcal{E}$ evaluated against test premise $g^*$, the prediction set is:
$$\mathcal{C}(\mathbf{x}_i) = \left\{ y \in \{0, 1\} : 1 - \hat{P}_{\text{cal}}(Y = y \mid \mathbf{x}_i) \le \hat{q}_{1-\epsilon} \right\}$$

Under premise-level exchangeability of groups in $\mathcal{G}_{\text{cal-conf}}^{\text{esc}} \cup \{g^*\}$, the framework guarantees **simultaneous group-level coverage**:
$$\mathbb{P}\left( \forall i \in \mathcal{P}_{g^*} \cap \mathcal{E}, \, y_i \in \mathcal{C}(\mathbf{x}_i) \right) \ge 1 - \epsilon$$
and conservative **marginal pair-level coverage**:
$$\mathbb{P}\left( y_i \in \mathcal{C}(\mathbf{x}_i) \right) \ge 1 - \epsilon$$

---

## 5. Decision Tree Logic & Output State Mapping

The end-to-end classification function maps every pair $x$ into exactly one of four standardized states:

$$\hat{Y}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \\
\text{HIGH\_RISK\_CANDIDATE\_FLAGGED} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}} \\
\text{HIGH\_RISK\_CANDIDATE\_FLAGGED} & \text{if } x \in \mathcal{E} \land \hat{P}_{\text{cal}}(x) \ge \gamma_{\text{upper}} \land \mathcal{C}(x) = \{1\} \\
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } x \in \mathcal{E} \land \hat{P}_{\text{cal}}(x) \le \gamma_{\text{lower}} \land \mathcal{C}(x) = \{0\} \\
\text{ABSTAIN\_UNRESOLVED} & \text{if } x \in \mathcal{E} \land \mathcal{C}(x) = \emptyset \\
\text{AMBIGUOUS\_HUMAN\_REVIEW} & \text{otherwise} \quad (\text{i.e., } \mathcal{C}(x) = \{0, 1\} \lor \gamma_{\text{lower}} < \hat{P}_{\text{cal}} < \gamma_{\text{upper}})
\end{cases}$$

---

## 6. Source Code Modality: Raw Score Thresholding & Conformal Disclaimer

### 6.1 Formal Proof of Platt Scaling Divergence on IR-Plag `case-05`
In IR-Plag `case-05` (58 Java files, 107 pairs), all 53 plagiarized pairs originate from a single base program (`original/Main.java`). In any 5-fold cross-validation split grouped by base program identity:
- **Fold 1:** Holds out `original/Main.java`. The training fold contains $N_{\text{pos}}^{\text{train}} = 0$ positive pairs.
- The negative log-likelihood objective of Platt scaling simplifies to:
  $$\mathcal{L}(A, B) = \sum_{i=1}^{N^{\text{train}}} \log\left(1 + \exp(A z_i + B)\right)$$
  Because $y_i = 0$ for all instances, $\mathcal{L}(A, B)$ is strictly monotonically decreasing as $B \to -\infty$. Solvers diverge, producing infinite parameters.
- **Folds 2–5:** Training fold contains all 53 positive pairs, but the held-out test fold contains **0 positive pairs** ($N_{\text{pos}}^{\text{test}} = 0$), rendering ROC-AUC and ECE undefined.

### 6.2 Code Decision Function (Exploratory Raw Margin)
Because group cross-fitting fails and IR-Plag contains only 7 tasks, conformal coverage guarantees and probability calibration are **formally disclaimed for code**. Code pairs are evaluated via raw decision margin $z(x)$:
$$\hat{Y}_{\text{code}}(x) = \begin{cases}
\text{LOW\_RISK\_REVIEW\_DEFERRED} & \text{if } s_{\text{cand}}(x) < \theta_{\text{low}} \lor z(x) \le \gamma_{\text{lower}}^{\text{code}} \\
\text{HIGH\_RISK\_CANDIDATE\_FLAGGED} & \text{if } s_{\text{cand}}(x) \ge \theta_{\text{high}} \lor z(x) \ge \gamma_{\text{upper}}^{\text{code}} \\
\text{AMBIGUOUS\_HUMAN\_REVIEW} & \text{otherwise}
\end{cases}$$

---

## 7. Statistical Testing Formulations

### 7.1 Exact Two-Sided Binomial McNemar Test
To evaluate paired discordance between the proposed model $M_1$ and baseline $M_2$ without asymptotic approximation error:
Let:
- $b = \sum_{i=1}^N \mathbb{I}(M_1(x_i) = y_i \land M_2(x_i) \neq y_i)$
- $c = \sum_{i=1}^N \mathbb{I}(M_1(x_i) \neq y_i \land M_2(x_i) = y_i)$

The exact two-sided p-value under the null hypothesis $H_0: p_b = p_c$ is:
$$p = \min\left(1, \, 2 \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} 0.5^{b + c}\right)$$

> [!WARNING]
> **Intracluster Correlation Boundary:** The exact Binomial test assumes independence across pairs. In text benchmarks where multiple pairs share a reference premise $g$, pairs within the same cluster exhibit intracluster correlation. Therefore, McNemar p-values on text benchmarks must be interpreted as **descriptive measures of sample discordance rather than formal proofs of cluster-robust statistical hypothesis testing**.

### 7.2 Clustered Bootstrap Confidence Intervals
To obtain valid confidence intervals accounting for premise clustering in text, resampling is conducted at the **premise group level**:
1. For $b = 1, \dots, 1000$:
   - Sample $G$ premise groups $\mathcal{G}^{(b)}$ with replacement from $\mathcal{G}_{\text{test}}$.
   - Form bootstrap dataset $D_{\text{test}}^{(b)} = \bigcup_{g \in \mathcal{G}^{(b)}} \mathcal{P}_g$.
   - Compute metric $\theta^{(b)} = \mathcal{M}(D_{\text{test}}^{(b)})$.
2. Compute the 95% percentile confidence interval: $[\theta_{0.025}^*, \, \theta_{0.975}^*]$.

