# Formal Design of the Four Technical Contributions (Version 2.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 2.0 (Phase 2.1 Audit Correction Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Formal Design  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. System Overview and Operational Triage Flow

The proposed framework replaces uncalibrated single-view heuristics with an uncertainty-aware, budget-constrained, and explainable multi-stage architecture. To resolve previous architectural ambiguities, we strictly distinguish three separate operational quantities:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Multi-View Candidate Score  s_cand in [0, 1]    --> Computed by Tier-1 Fast Screening on 100% pairs  │
│ 2. Calibrated Probability       p_hat in [0, 1]    --> Computed by Heavy Detector on ESCALATED pairs only │
│ 3. Conformal Prediction Set    C in {{0},{1},{0,1},Ø} -> Constructed from Calibrated Probability p_hat   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Complete End-to-End Decision & Triage Flow:

```
                                  ┌─────────────────────────────────────────┐
                                  │    Incoming Submission Pair (D_A, D_B)  │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                             ┌───────────────────────────────────────────────────┐
                             │  CONTRIBUTION 1: Multi-View Candidate Generator   │
                             │  (Purely Lightweight: Fast Token / Regex Views)   │
                             │  Text: TF-IDF, Char-3gram, Word-Jaccard, Length   │
                             │  Code: Raw Token, Norm Token, Operator, Length    │
                             │  Total Ingestion Latency: < 0.050 ms / pair       │
                             └─────────────────────────┬─────────────────────────┘
                                                       │ Candidate Score: s_cand
                          ┌────────────────────────────┼────────────────────────────┐
                          ▼                            ▼                            ▼
                 [ s_cand < θ_low ]       [ θ_low <= s_cand < θ_high ]     [ s_cand >= θ_high ]
                          │                            │                            │
                          ▼                            │                            ▼
            ┌───────────────────────────┐              │              ┌───────────────────────────┐
            │ STATE 1: LOW_RISK_PRUNED  │              │              │ STATE 3: HIGH_RISK_FLAGGED│
            │ • Immediate fast negative │              │              │ • Direct Tier-1 Flag      │
            │ • Bypasses heavy models   │              │              │ • Bypasses heavy models   │
            │ • Zero GPU/Heavy Compute  │              │              │ • Assigned Tier-1 rule    │
            │ • Latency: < 0.050 ms     │              │              │ • Routed to HUMAN REVIEW  │
            └───────────────────────────┘              │              └─────────────┬─────────────┘
                                                       ▼                            │
                             ┌───────────────────────────────────────────────────┐  │
                             │ STATE 2: UNCERTAIN_ESCALATED                      │  │
                             │ • Invokes Heavy Feature Extraction & Detector     │  │
                             │ • Text: Sentence-BERT miniLM forward pass         │  │
                             │ • Code: 7-D AST Parsing (2.05ms) + SVM / RF       │  │
                             └─────────────────────────┬─────────────────────────┘  │
                                                       │ Raw Score: z               │
                                                       ▼                            │
                             ┌───────────────────────────────────────────────────┐  │
                             │ CONTRIBUTION 2: Uncertainty-Aware Conformal Router│  │
                             │ • Calibrated Posterior: p_hat = g_cal(z)          │  │
                             │ • Conformal Set: C_{1-eps} in {{0}, {1}, {0,1}, Ø}│  │
                             │ • Predictive Entropy H(p)                         │  │
                             └─────────────────────────┬─────────────────────────┘  │
                                                       │                            │
                          ┌────────────────────────────┼────────────────────────────┤
                          ▼                            ▼                            ▼
                  C = {0} (Resolved Neg)       C = {0, 1} or Ø (Ambiguous)   C = {1} (Resolved Plag)
                          │                            │                            │
                          ▼                            ▼                            ▼
            ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
            │ Confirmed Non-Plagiarized │ │ Unresolved / OOD Anomaly  │ │ Confirmed Plagiarized     │
            │ Heavy cleared false alarm │ │ Mandatory Human Review    │ │ Routed to Human Review    │
            └─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
                          │                             │                             │
                          └─────────────────────────────┼─────────────────────────────┘
                                                        ▼
                             ┌───────────────────────────────────────────────────┐
                             │ CONTRIBUTION 4: Cross-Modal Evidence Aggregation  │
                             │ • Text: Smith-Waterman Local Alignment & Spans    │
                             │ • Code: AST Subtree Isomorphism & Identifier Map  │
                             │ • Generates Formal EvidenceDossier for Reviewers  │
                             └───────────────────────────────────────────────────┘
```

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Research Gap
Single-view pre-filtering relies on a solitary scalar threshold (e.g., word-level TF-IDF alone or token Jaccard alone):
1. **The Zero-Overlap Text Dilemma:** In sentence-level text reuse benchmarks, **10.57% of true positive pairs share zero common vocabulary ($\text{TF-IDF} = 0.0000$)** due to synonym substitution. Any scalar lexical filter $\theta_{low} > 0$ automatically drops them, capping candidate recall at $88.81\%$.
2. **Code Boilerplate Distortion & Computational Trap:** In source code, student submissions share language boilerplate (`public static void main`, scanner imports). If a pre-filter naively extracts full AST features and cyclomatic complexity at Tier 1, it incurs **2.050 ms/pair** of Java parsing on 100% of pairs, destroying the cost advantage of pre-filtering!

### 2.2 Mathematical Role of Every Feature in the Multi-View Tuple

To guarantee sub-millisecond Tier-1 execution ($< 0.050\text{ ms/pair}$), all Tier-1 candidate features are strictly restricted to **fast string, token, and regex operations**. Full AST subtree parsing is strictly reserved for Tier 2!

#### A. Natural Language Text Multi-View Vector $\mathbf{v}_{\text{text}}(D_A, D_B) \in [0, 1]^4$:
1. **Word-Level Sublinear TF-IDF Cosine Similarity ($v_1 = S_{\text{word-tfidf}}$):**
   - *Role:* Captures exact lexical reuse and dominant topic term preservation. Sublinear scaling ($1 + \log(\text{tf})$) dampens repeated words.
   - *Computation:* Precomputed sparse inverted index lookup ($0.015\text{ ms}$).
2. **Character 3-Gram Jaccard Similarity ($v_2 = S_{\text{char-3gram}}$):**
   - *Role:* Sub-word lexical capture. Robust to typographical errors, morphological inflections (`running` vs. `ran`), and hyphenation. Crucially, provides non-zero similarity for stem-sharing synonyms where word-level TF-IDF produces $0$.
   - *Computation:* Hash-set intersection of 3-character shingles ($0.012\text{ ms}$).
3. **Word Multiset Jaccard Similarity ($v_3 = S_{\text{word-jaccard}}$):**
   - *Role:* Symmetric, non-parametric lexical overlap unweighted by corpus frequency; acts as an orthogonal sanity check against inverted-index skew.
   - *Computation:* Hash-multiset intersection ($0.008\text{ ms}$).
4. **Length-Ratio Symmetry Metric ($v_4 = \Delta_{\text{len}}$):**
   - *Formula:* $\Delta_{\text{len}} = \frac{\min(|D_A|, |D_B|)}{\max(|D_A|, |D_B|)} \in (0, 1]$.
   - *Role:* Penalizes extreme length disparities; authentic text reuse involves substantive volume relative to document length.
   - *Computation:* Integer length comparison ($< 0.001\text{ ms}$).

#### B. Source Code Multi-View Vector $\mathbf{u}_{\text{code}}(C_A, C_B) \in [0, 1]^4$ (Purely Lightweight Tier 1):
1. **Raw Token Multiset Jaccard ($u_1 = S_{\text{raw-token}}$):**
   - *Role:* Detects verbatim duplication and comment/layout stripping (Level 1 obfuscation). High values ($u_1 \ge 0.80$) enable instant direct flagging.
   - *Computation:* Fast regex lexer tokenizing on whitespace, punctuation, and delimiters ($0.020\text{ ms}$).
2. **Identifier-Normalized Token Multiset Jaccard ($u_2 = S_{\text{norm-token}}$):**
   - *Role:* All non-reserved identifiers are replaced with canonical tokens (`ID`), while Java keywords (`for`, `while`, `int`, `return`) and operators are preserved. This neutralizes identifier renaming (Level 2 obfuscation), restoring overlap from $< 0.40$ to $> 0.90$.
   - *Computation:* Regex token stream normalization without building an AST ($0.015\text{ ms}$).
3. **Operator Frequency Profile Cosine ($u_3 = S_{\text{operator}}$):**
   - *Role:* Counts arithmetic, relational, and logical operators (`+`, `-`, `*`, `==`, `&&`). Preserves underlying computational logic even when loops are restructured (Level 4 obfuscation: `for` to `while`).
   - *Computation:* Character-level operator tallying ($0.008\text{ ms}$).
4. **Token Length Ratio Asymmetry ($u_4 = \Delta_{\text{tok-len}}$):**
   - *Role:* Structural volume proxy. Detects code expansion or dead code injection attacks.
   - *Computation:* Ratio of token sequence lengths ($< 0.001\text{ ms}$).

*Important Architectural Correction:* AST node bigrams and cyclomatic complexity differentials are **excluded from Tier 1**. They require full Java AST construction ($2.050\text{ ms}$) and are executed strictly at Tier 2 on escalated pairs. This guarantees Tier-1 candidate screening executes in **$\le 0.045\text{ ms/pair}$**.

### 2.3 Candidate Decision Rule & Investigation of Adversarial Subsets
- **Scoring Function:** $s_{\text{cand}}(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$ trained on $D_{\text{train}}$.
- **Cutoff Selection:** $\theta_{\text{cand}}$ is calibrated strictly on $D_{\text{val}}$ to satisfy:
  $$\theta_{\text{cand}} = \max \left\{ \theta \in [0, 1] \mid \text{CandidateRecall}(\theta, D_{\text{val}}) \ge R_{\text{target}} \right\}$$
- **Target Constraint:** $R_{\text{target}} = 0.98$ (treated strictly as an **empirical target**, not an asserted guarantee).
- **Adversarial Subsets:**
  1. *Zero-Overlap Text Subset:* By incorporating character 3-grams ($v_2$), pairs with zero word-TF-IDF ($v_1 = 0$) obtain scores above $\theta_{\text{cand}}$, breaking the $88.81\%$ ceiling.
  2. *Code Renaming & Obfuscations:* By incorporating identifier normalization ($u_2$), Level 2 obfuscated code is retained with near 100% recall at Tier 1.

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Resolving the High-Risk Heavy Score Contradiction
In the previous draft, high-risk cases bypassed the heavy detector but were paradoxically assigned probabilities from that heavy detector. We resolve this contradiction with strict operational semantics:
- **Tier-1 Direct Flags (`HIGH_RISK_FLAGGED`):** Triggered when $s_{\text{cand}} \ge \theta_{\text{high}}$ (e.g., verbatim copy with Jaccard $\ge 0.85$). These pairs **completely bypass the heavy detector** (saving $4.5\text{ ms}$ or $2.05\text{ ms}$). They are assigned a Tier-1 rule label (`Direct High Risk`) and carry no heavy-model probability ($\hat{p} = \text{N/A* (Tier-1 Rule Trigger)}$). They are routed directly to human review.
- **Heavy Detector Invocations:** The heavy detector (Sentence-BERT on text; 7-D AST + SVM/RF on code) is executed **exclusively on pairs escalated to Tier 2** ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$). Heavy detector output $z$ is then calibrated into posterior probability $\hat{p} = g_{\text{cal}}(z)$.

### 3.2 Probability Calibration: Platt Scaling vs. Isotonic Regression
To prevent optimistic calibration bias, the probability calibrator is fitted on an independent calibration split ($D_{\text{cal-prob}}$) completely disjoint from the base-model training set ($D_{\text{train}}$) and the conformal calibration set ($D_{\text{cal-conf}}$).
1. **Platt Scaling (Parametric Logistic Sigmoid):**
   $$P_{\text{Platt}}(Y = 1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
   Fitted via cross-entropy minimization on $D_{\text{cal-prob}}$. Low variance, resilient on small datasets, but assumes a sigmoidal distortion.
2. **Isotonic Regression (Non-Parametric Piecewise Step Function):**
   Fitted via the Pool Adjacent Violators (PAV) algorithm on $D_{\text{cal-prob}}$. Capable of fitting arbitrary monotonic miscalibration, but prone to overfitting on small datasets ($N < 500$).
3. **Selection Criterion:** Evaluated out-of-sample on $D_{\text{val}}$ using **Expected Calibration Error (ECE)** and **Brier Score**. The superior calibrator is selected.

### 3.3 Inductive Conformal Prediction & Complete Four-Set Policy
Conformal non-conformity scores are computed on a separate split $D_{\text{cal-conf}}$:
$$s_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$$
For significance $\epsilon \in (0, 1)$, compute empirical quantile $\hat{q}_{1-\epsilon}$.
For query instance $\mathbf{x}$, the prediction set $\mathcal{C}_{1-\epsilon}(\mathbf{x})$ can take **all four possible states**:

$$\mathcal{C}_{1-\epsilon}(\mathbf{x}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}) \le \hat{q}_{1-\epsilon} \right\}$$

#### Comprehensive Conformal Set Triage Mapping:
1. **Case 1: $\mathcal{C}_{1-\epsilon} = \{0\}$ (High-Confidence Negative):**
   - *Condition:* $\hat{p} \le 1 - \hat{q}_{1-\epsilon}$ and $1 - \hat{p} > 1 - \hat{q}_{1-\epsilon}$.
   - *Action:* Cleared as non-plagiarized. Heavy model disproved initial suspicion.
2. **Case 2: $\mathcal{C}_{1-\epsilon} = \{1\}$ (High-Confidence Positive):**
   - *Condition:* $\hat{p} \ge \hat{q}_{1-\epsilon}$ and $1 - \hat{p} < \hat{q}_{1-\epsilon}$.
   - *Action:* Flagged as high-confidence plagiarism; routed to instructor review with evidence dossier.
3. **Case 3: $\mathcal{C}_{1-\epsilon} = \{0, 1\}$ (High Uncertainty / Ambiguous):**
   - *Condition:* Both labels conform to calibration distribution.
   - *Action:* Routed to **Mandatory Human Review** as an unresolved boundary case.
4. **Case 4: $\mathcal{C}_{1-\epsilon} = \emptyset$ (Empty Set / Out-of-Distribution Anomaly):**
   - *Condition:* Neither label conforms ($1 - \hat{p} > \hat{q}_{1-\epsilon}$ and $\hat{p} < 1 - \hat{q}_{1-\epsilon}$, possible when $\hat{q}_{1-\epsilon} < 0.50$).
   - *Scientific Meaning:* The query is an extreme out-of-distribution anomaly (e.g., corrupt text encoding, foreign language snippet, or malformed Java AST).
   - *Safe Abstention Policy:* The empty set MUST NOT be treated as $\{0, 1\}$. It is classified as **`OOD_ANOMALY_ABSTAIN`** with an explicit diagnostic warning in the evidence dossier, triggering senior instructor review.

### 3.4 Four Critical Scientific Disclosures
1. **Conformal Marginal Coverage $\ne$ Plagiarism Recall:** Conformal guarantees $P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$ ensure that the true label is in the set. Under extreme uncertainty, $\mathcal{C} = \{0, 1\}$, which satisfies coverage but yields an uninformative set. Candidate recall depends on escalating uncertain pairs.
2. **No Conditional Coverage Guarantee:** Marginal coverage does not guarantee conditional coverage for rare subgroups (e.g., Level 6 compound attacks).
3. **Decision Support, Not Conviction:** A high-risk prediction is a triage signal for human review, not automated proof of misconduct.
4. **Predictive Entropy Limits:** Predictive entropy $\mathcal{H}(\hat{p})$ conflates aleatoric uncertainty (inherent ambiguity) with epistemic uncertainty (lack of training data). It is an uncertainty proxy, not a pure epistemic estimator.

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Accurate Methodological Characterization
We do not make inflated claims of a "novel Lagrangian optimizer" for a 2D parameter space. We accurately describe the method as:
> **"Recall-Constrained Routing-Policy Optimization via Discretized Pareto Surface Search"**

For the 2D threshold vector $\boldsymbol{\theta} = (\theta_{\text{low}}, \theta_{\text{high}}) \in [0, 1]^2$, evaluating the empirical validation objective across a dense $40 \times 36$ grid is mathematically guaranteed to find the **global Pareto-optimal solution** without non-convex gradient instability or saddle-point divergence.

### 4.2 Comprehensive Computational Cost Model
To avoid double-counting, the total expected per-pair cost accounts for every stage:
$$\bar{C}_{\text{total}}(\boldsymbol{\theta}) = \bar{C}_{\text{prep}} + \bar{C}_{\text{Tier1}} + P(\text{Escalate} \mid \boldsymbol{\theta}) \cdot \left[ \bar{C}_{\text{feat-heavy}} + \bar{C}_{\text{infer-heavy}} \right] + \bar{C}_{\text{router}} + P(\text{Review} \mid \boldsymbol{\theta}) \cdot \bar{C}_{\text{explain}}$$

| Processing Stage | Text Implementation Cost | Code Implementation Cost | Execution Frequency |
| :--- | :--- | :--- | :--- |
| **$C_{\text{prep}}$: Preprocessing** | Tokenization & lowercasing ($0.005\text{ ms}$) | Comment stripping & regex lexing ($0.008\text{ ms}$) | 100% of candidate pairs |
| **$C_{\text{Tier1}}$: Screening** | 4-view lexical computation ($0.035\text{ ms}$) | 4-view lightweight code computation ($0.038\text{ ms}$) | 100% of candidate pairs |
| **$C_{\text{feat-heavy}}$: Deep Features** | None (embeddings computed on-the-fly) | 7-D AST parsing & grammar walk ($2.050\text{ ms}$) | Escalated pairs only ($P_{\text{esc}}$) |
| **$C_{\text{infer-heavy}}$: Inference** | Sentence-BERT forward pass ($4.000\text{ ms}$) | RBF SVM / RF inference ($0.013 - 0.025\text{ ms}$) | Escalated pairs only ($P_{\text{esc}}$) |
| **$C_{\text{router}}$: Calibration** | Probability mapping & quantile lookup ($0.002\text{ ms}$) | Probability mapping & quantile lookup ($0.002\text{ ms}$) | Escalated pairs only ($P_{\text{esc}}$) |
| **$C_{\text{explain}}$: Evidence** | Smith-Waterman text alignment ($0.450\text{ ms}$) | AST subtree isomorphism ($0.380\text{ ms}$) | Flagged/Escalated pairs ($P_{\text{rev}}$) |

### 4.3 Formal Optimization Objective & Infeasibility Handling
$$\min_{\boldsymbol{\theta}} \quad \bar{C}_{\text{total}}(\boldsymbol{\theta}) = \frac{1}{|D_{\text{val}}|} \sum_{i \in D_{\text{val}}} C_{\text{total}}(\mathbf{x}_i; \boldsymbol{\theta})$$
Subject to:
$$\begin{aligned}
\text{CandidateRecall}(\boldsymbol{\theta}; D_{\text{val}}) &\ge R_{\text{target}} \quad (R_{\text{target}} = 0.98) \\
\bar{C}_{\text{total}}(\boldsymbol{\theta}) &\le B_{\text{max}} \quad (\text{Latency Budget in ms/pair}) \\
\text{DirectPrecision}(\boldsymbol{\theta}; D_{\text{val}}) &\ge P_{\text{target}} \quad (P_{\text{target}} \ge 0.90)
\end{aligned}$$

- **Removal of Hard-Coded Fallback:** If no threshold configuration satisfies all constraints on $D_{\text{val}}$, the optimizer **explicitly outputs `STATUS: INFEASIBLE_BUDGET`** and reports the closest Pareto configuration with an explicit warning, rather than silently substituting an arbitrary fallback policy.
- **Fair Comparator Baselines:** Evaluated against Full Heavy Model Execution (100% call rate), Fixed-Threshold Single Cascade (Condition B), and Random Pruning at matched candidate retention rates.

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Modality-Specific Evidence Engines
1. **Text Evidence Engine:**
   - **Smith-Waterman Local Sequence Alignment:** Dynamic programming matrix with affine gap penalty ($W_k = u + v \cdot k$ with $u=2.0, v=0.5$), extracting matching contiguous and gapped phrase spans.
   - **Paraphrase Span Attribution:** Identifies sentence segments with high contextual similarity but low lexical overlap, flagging synonym substitutions.
2. **Code Evidence Engine:**
   - **AST Subtree Isomorphism Matching:** Traverses ASTs of escalated pairs to find structurally identical subtrees invariant to variable names.
   - **Identifier Renaming Ledger:** Extracts an explicit substitution dictionary:
     $$\mathcal{M}_{\text{id}} = \left\{ \text{"i"} \mapsto \text{"counter"}, \, \text{"total"} \mapsto \text{"sumTotal"} \right\}$$
   - **Statement Permutation Trace:** Tracks rearrangement of basic blocks across methods.

### 5.2 Common Evidence Schema (`EvidenceDossier`)
Outputs a polymorphic dataclass containing calibrated probability, operational triage state, conformal set, matching spans, transformation ledgers, faithfulness scores, and narrative summaries.

### 5.3 Quantitative Explanation Quality Metrics
1. **Evidence Faithfulness (Feature Ablation Drop):**
   $$\Delta \hat{P}_{\text{faith}}(\mathbf{x}) = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$$
   Measures whether removing identified evidence causes the model's confidence to drop.
2. **Evidence Coverage:** Ratio of matched evidence tokens/nodes to total document length.
3. **Transformation Alignment Precision:** Precision of extracted identifier mappings against ground-truth Faidhi-Robinson labels.
4. **Ethical Due-Process Mandate:** Dossiers provide **decision support for educators**, never automated proof of misconduct.

---

## 6. Component-Level Specification Matrix

| Dimension | Contribution 1 (Multi-View) | Contribution 2 (Uncertainty) | Contribution 3 (Budget) | Contribution 4 (Explainability) |
| :--- | :--- | :--- | :--- | :--- |
| **Research Gap** | Single-view lexical filters drop paraphrased text (10.57% zero-overlap) | Uncalibrated scores; no statistical coverage guarantees | Heuristic sweeps ignore hard compute budgets | Scalar scores are uninterpretable and fail academic review |
| **Input Data** | Raw text / code pair | Heavy model score $z$ on escalated pairs | Validation cost profiles & recall constraints | Classified candidate pair & model artifacts |
| **Primary Output** | $s_{\text{cand}} \in [0, 1]$, retention status | Calibrated prob $\hat{p}$, entropy $\mathcal{H}$, conformal set $\mathcal{C}$ | Optimal threshold policy $\boldsymbol{\theta}^*$ on Pareto frontier | Structured `EvidenceDossier` (JSON/Markdown) |
| **Key Metric** | Candidate Recall @ Pruning Ratio | ECE, Brier Score, Conformal Coverage | Latency (ms), Workload Red., Constrained F1 | Faithfulness ($\Delta P$), Evidence Coverage |
| **Ablation Study** | Remove each view individually (drop char n-grams, drop norm-tokens) | Compare Uncalibrated vs. Platt vs. Isotonic vs. Conformal | Compare Pareto-Optimal vs. Fixed vs. Random Pruning | Ablate evidence masks to measure decision drop |
| **Key Failure Mode** | AST parsing at Tier 1 destroys speed advantage (resolved in v2) | Empty set $\emptyset$ on OOD data (resolved via abstain policy in v2) | Infeasible budget where target recall is mathematically impossible | Over-attribution of standard idiom boilerplate |
| **Novelty Claim** | Lightweight multi-view screening with zero-overlap handling | Inductive conformal prediction with complete 4-set triage | Pareto surface search under explicit per-pair compute budgets | Cross-modal dual-engine evidence aggregation dossiers |

