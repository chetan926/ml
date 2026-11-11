# Formal Design of the Four Technical Contributions
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Complete Formal Design  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Overview of the Four Contributions

The proposed framework replaces brittle single-metric heuristics and uncalibrated decision thresholds with a mathematically grounded, cost-bounded, and explainable multi-stage pipeline. The system operates across natural language text and source code via four technically distinct, modular contributions:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT SUBMISSION PAIR (D_A, D_B)                                     │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CONTRIBUTION 1: Robust Multi-View Candidate Generation                                                 │
│ • Text Views: Word TF-IDF, Character 3-Gram Overlap, Jaccard Lexical, Length-Ratio Proxy               │
│ • Code Views: Raw Token Jaccard, Identifier-Normalized Token Multiset, AST Node Histogram, Operators   │
│ • Objective: Filter out clear non-plagiarized pairs while preserving >= 98% Candidate Recall Target   │
└───────────────────────────────────┬────────────────────────────────────────────────┬───────────────────┘
                                    │ Retained Candidate Pairs                       │ Pruned Pairs
                                    ▼                                                ▼
┌──────────────────────────────────────────────────────────────────────┐  ┌──────────────────────────────┐
│ CONTRIBUTION 2: Uncertainty-Aware Adaptive Decisions                 │  │ STATE 1: LOW_RISK_PRUNED     │
│ • Calibrated Probabilities via Platt Scaling / Isotonic Regression   │  │ Instant Negative (No Heavy)  │
│ • Expected Calibration Error (ECE) & Predictive Entropy H(p)         │  └──────────────────────────────┘
│ • Inductive Conformal Prediction Set Guarantees: P(Y in C) >= 1 - eps│
│ • Tri-Band State Routing: Low Risk, Uncertain, High Risk             │
└───────────────────┬──────────────────────────────────────────────┬───┘
                    │ High-Confidence Verbatim                     │ Uncertain Pairs (High Entropy)
                    ▼                                              ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────────────────────────────┐
│ STATE 3: HIGH_RISK_FLAGGED           │  │ CONTRIBUTION 3: Budget-Constrained Routing Optimization      │
│ Instant Direct Flag (Bypasses Heavy) │  │ • Formal Lagrangian: min Cost s.t. Recall >= R_target       │
│                                      │  │ • Latency Budget B (ms/pair) & Memory Constraints            │
│                                      │  │ • Selective Invocation of Heavy Detectors (SBERT / SVM / RF) │
└───────────────────┬──────────────────┘  └──────────────────────────────┬───────────────────────────────┘
                    │                                                    │
                    └───────────────────────┬────────────────────────────┘
                                            │ Final Predictions & Classification Confidences
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CONTRIBUTION 4: Explainable Cross-Modal Evidence Aggregation                                           │
│ • Text Evidence: Smith-Waterman Local Alignment, Paraphrase Span Attribution, Semantic Heatmaps        │
│ • Code Evidence: AST Subtree Isomorphism, Identifier Substitution Maps, Statement Reordering Traces     │
│ • Structured Output: Human-Readable Evidence Dossier with Faithfulness & Coverage Metrics              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Contribution 1: Robust Multi-View Candidate Generation

### 2.1 Research Gap
Traditional pre-filtering relies on unimodal scalar thresholds (e.g., TF-IDF alone or Token Jaccard alone). As proven in the Phase 1 audit:
1. **The Zero-Overlap Text Dilemma:** In the SNLI text benchmark, **10.57% of true positive pairs share zero common vocabulary** ($\text{TF-IDF} = 0.0000$) due to heavy paraphrasing and synonym substitution. Any scalar cutoff $\theta_{low} > 0$ automatically drops these pairs, creating an empirical ceiling of $88.81\%$ on candidate recall.
2. **Code Boilerplate Distortion:** In introductory source code assignments, shared framework templates (`public static void main`, scanner imports) cluster token overlap into a narrow band (0.40–0.70), causing single-token filters to prune almost nothing (0.48% pruning ratio in existing baseline) or catastrophically prune true positives when thresholds are tightened.

### 2.2 Mathematical Role of Every Feature in the Multi-View Tuple

#### A. Natural Language Text Multi-View Vector $\mathbf{v}_{\text{text}}(D_A, D_B) \in \mathbb{R}^5$:
1. **Word-Level Sublinear TF-IDF Cosine Similarity ($v_1 = S_{\text{word-tfidf}}$):**
   - *Role:* Captures exact lexical reuse and dominant topic term preservation. Sublinear scaling ($1 + \log(\text{tf})$) dampens repeated buzzwords.
2. **Character 3-Gram Jaccard Similarity ($v_2 = S_{\text{char-3gram}}$):**
   - *Role:* Sub-word lexical capture. Robust to typographical errors, morphological inflections (`running` vs. `ran`), and hyphenation. Crucially, provides non-zero similarity for stem-sharing synonyms where word-level TF-IDF produces 0.
3. **Word Multiset Jaccard Similarity ($v_3 = S_{\text{word-jaccard}}$):**
   - *Role:* Symmetric, non-parametric lexical overlap unweighted by corpus frequency; acts as an orthogonal sanity check against inverted-index skew.
4. **Length-Ratio Asymmetry Penalty ($v_4 = \Delta_{\text{len}}$):**
   - *Formula:* $\Delta_{\text{len}} = \frac{\min(|D_A|, |D_B|)}{\max(|D_A|, |D_B|)} \in (0, 1]$.
   - *Role:* Penalizes extreme length disparities; authentic plagiarism involves substantive text reuse relative to document length.
5. **Lightweight Embedding Proxy / Semantic Anchor ($v_5 = S_{\text{embed-proxy}}$):**
   - *Formula:* Cosine similarity of mean-pooled token vectors using a lightweight offline lookup or fast projection (or MiniLM fast pass).
   - *Role:* Detects semantic equivalence when vocabulary overlap is exactly zero ($v_1 = 0, v_3 = 0$), breaking the $88.81\%$ recall ceiling.

#### B. Source Code Multi-View Vector $\mathbf{v}_{\text{code}}(C_A, C_B) \in \mathbb{R}^5$:
1. **Raw Token Multiset Jaccard ($u_1 = S_{\text{raw-token}}$):**
   - *Role:* Detects verbatim cut-and-paste and layout/comment changes (Level 1 obfuscation). High values ($u_1 \ge 0.80$) enable instant direct flagging.
2. **Identifier-Normalized Token Jaccard ($u_2 = S_{\text{norm-token}}$):**
   - *Role:* All user-defined variable, method, and class names are mapped to generic symbols (`VAR_1`, `METHOD_1`). Language keywords and operators are preserved. This neutralizes identifier renaming (Level 2 obfuscation), restoring overlap from $< 0.40$ to $> 0.90$.
3. **AST Node Type N-Gram Overlap ($u_3 = S_{\text{ast-ngram}}$):**
   - *Role:* Structural invariance. Computes Jaccard overlap of linearized AST node type bigrams (e.g., `ForStmt-BlockStmt`, `IfStmt-BinaryExpr`). Invariant to statement reordering and identifier renaming.
4. **Operator Frequency Profile Cosine ($u_4 = S_{\text{operator}}$):**
   - *Role:* Counts arithmetic, relational, and logical operators (`+`, `-`, `*`, `==`, `&&`). Preserves underlying computational logic even when control structures are rewritten (Level 4 obfuscation: `for` to `while`).
5. **Cyclomatic Complexity Ratio ($u_5 = \Delta_{\text{complexity}}$):**
   - *Formula:* $1.0 - \frac{|CC(C_A) - CC(C_B)|}{\max(CC(C_A), CC(C_B))}$.
   - *Role:* Structural complexity proxy. Independent student solutions to simple tasks typically possess distinct branching profiles.

### 2.3 Validation-Trained Candidate Generation Mechanism
Rather than taking a simple minimum or maximum across features, the multi-view candidate generator learns a non-linear scoring function $g(\mathbf{v}) \in [0, 1]$ trained strictly on the training partition ($D_{\text{train}}$) and calibrated on validation data ($D_{\text{val}}$):
$$g(\mathbf{v}) = \sigma\left(\mathbf{w}^T \mathbf{v} + b\right) \quad \text{or} \quad g(\mathbf{v}) = \text{GBDT}_{\text{shallow}}(\mathbf{v})$$
The candidate filter decision rule is:
$$\text{Filter}(D_A, D_B) = \begin{cases} \text{PRUNED (Drop)}, & \text{if } g(\mathbf{v}) < \theta_{\text{cand}} \\ \text{RETAINED (Candidate)}, & \text{if } g(\mathbf{v}) \ge \theta_{\text{cand}} \end{cases}$$
Where $\theta_{\text{cand}}$ is selected on $D_{\text{val}}$ to satisfy:
$$\theta_{\text{cand}} = \max \left\{ \theta \in [0, 1] \mid \text{CandidateRecall}(\theta, D_{\text{val}}) \ge R_{\text{target}} \right\}$$
**Target Constraint:** $R_{\text{target}} = 0.98$ (treated strictly as an optimization target, not an asserted empirical guarantee).

### 2.4 Feature Redundancy & Complementarity Analysis
To verify that features are truly complementary rather than redundant:
- Variance Inflation Factor (VIF) and Pearson/Spearman cross-correlation matrices will be computed on $D_{\text{train}}$.
- Any feature pair with correlation $r > 0.92$ will undergo greedy backward elimination to minimize computational overhead.

### 2.5 Investigation of Specific Adversarial Subsets
1. **The 119 Zero-Overlap Positive Text Pairs:** The multi-view generator will be audited specifically on this subset to verify whether $S_{\text{char-3gram}}$ and $S_{\text{embed-proxy}}$ retain these pairs above $\theta_{\text{cand}}$.
2. **Java Faidhi-Robinson Obfuscation Levels:** The code generator will be audited across L1 (comments), L2 (renaming), L3 (reordering), L4 (control flow), L5 (method abstraction), and L6 (compound multi-transform).

---

## 3. Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism

### 3.1 Research Gap
Existing plagiarism detectors output uncalibrated classification scores or arbitrary similarity percentages. In high-stakes academic integrity contexts:
- A raw SVM output probability of $0.85$ does not mean there is an $85\%$ true probability of plagiarism.
- Fixed decision boundaries cannot distinguish between high confidence (clear copying) and high epistemic uncertainty (shared domain jargon, textbook starter templates).
- Standard classification provides no statistical coverage guarantees, leading to either excessive false accusations or missed misconduct.

### 3.2 Mathematical Formulation of Probability Calibration

Let $z = f(\mathbf{x}) \in \mathbb{R}$ denote the uncalibrated margin/score produced by the heavy detector (SVM decision function or Random Forest probability). We map $z$ to a calibrated posterior probability $P(Y = 1 \mid z) \in [0, 1]$.

#### A. Parametric Calibration: Platt Scaling (Logistic Sigmoid)
Fit scalar parameters $A, B \in \mathbb{R}$ on the independent calibration split $D_{\text{cal}}$:
$$P_{\text{Platt}}(Y = 1 \mid z) = \frac{1}{1 + \exp(A \cdot z + B)}$$
Optimized via negative log-likelihood (binary cross-entropy):
$$\min_{A, B} -\sum_{i=1}^{N_{\text{cal}}} \left[ y_i \log P_{\text{Platt}}(Y=1 \mid z_i) + (1 - y_i) \log (1 - P_{\text{Platt}}(Y=1 \mid z_i)) \right]$$

#### B. Non-Parametric Calibration: Isotonic Regression
Fits a piecewise-constant, non-decreasing step function $m(z)$ on $D_{\text{cal}}$ using the Pair-Adjacent Violators (PAV) algorithm:
$$\min_{m} \sum_{i=1}^{N_{\text{cal}}} \left( y_i - m(z_i) \right)^2 \quad \text{subject to} \quad m(z_i) \le m(z_j) \text{ whenever } z_i \le z_j$$

#### Comparison Protocol:
Both Platt scaling and Isotonic regression will be fitted on $D_{\text{cal}}$. The model achieving lower **Expected Calibration Error (ECE)** and lower **Brier Score** on out-of-sample validation data will be selected as the calibration engine.

### 3.3 Uncertainty Quantification Metrics
1. **Expected Calibration Error (ECE):**
   Group predictions into $M = 10$ equally spaced probability bins $B_1, \dots, B_M \subset [0, 1]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   Where $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$ and $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$.
2. **Brier Score (Strictly Proper Scoring Rule):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
3. **Predictive Entropy (Epistemic Uncertainty Proxy):**
   $$\mathcal{H}(\hat{p}) = -\hat{p} \log_2(\hat{p}) - (1 - \hat{p}) \log_2(1 - \hat{p})$$
   Maximal uncertainty occurs at $\hat{p} = 0.50$ ($\mathcal{H} = 1.0$), while certainty approaches $0.0$ as $\hat{p} \to 0$ or $\hat{p} \to 1$.

### 3.4 Conformal Prediction with Inductive Calibration Set
To provide distribution-free uncertainty guarantees, we apply **Inductive Split Conformal Prediction**:
1. Partition data into training $D_{\text{train}}$ and calibration $D_{\text{cal}}$ ($N_{\text{cal}}$ pairs).
2. Train model and probability calibrator on $D_{\text{train}}$.
3. Compute non-conformity scores on $D_{\text{cal}}$:
   $$s_i = 1 - \hat{P}(Y = y_i \mid \mathbf{x}_i)$$
4. For a user-specified significance level $\epsilon \in (0, 1)$ (e.g., $\epsilon = 0.05$ for $95\%$ marginal coverage), compute the empirical quantile:
   $$\hat{q} = \text{Quantile}\left( \frac{\lceil (N_{\text{cal}} + 1)(1 - \epsilon) \rceil}{N_{\text{cal}}}, \, \{s_1, \dots, s_{N_{\text{cal}}}\} \right)$$
5. For a new test candidate $\mathbf{x}_{N+1}$, construct the prediction set:
   $$\mathcal{C}_{1-\epsilon}(\mathbf{x}_{N+1}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{x}_{N+1}) \le \hat{q} \right\}$$

#### Theoretical Assumptions and Explicit Limitations:
- **Exchangeability Assumption:** Conformal validity assumes test pairs $(X_{N+1}, Y_{N+1})$ are exchangeable with calibration pairs $(X_i, Y_i)_{i=1}^{N_{\text{cal}}}$.
- **Marginal vs. Conditional Coverage:** The guarantee $P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$ is *marginal* over the dataset. It does **not** guarantee conditional coverage for every specific obfuscation level or topic domain.
- **Plagiarism Recall Disclaimer:** Conformal set coverage ensures that the true label is contained in the set $\mathcal{C}(X)$ with probability $1 - \epsilon$. **This does not automatically guarantee high plagiarism recall**, because when uncertainty is high, $\mathcal{C}(X) = \{0, 1\}$ (both labels are returned). High recall depends on the subsequent escalation routing mechanism.

### 3.5 Adaptive Three-Way Decision Routing
The prediction set $\mathcal{C}_{1-\epsilon}(\mathbf{x})$ directly maps to the three-way operational states:
$$\text{State}(\mathbf{x}) = \begin{cases}
\text{STATE 1: LOW\_RISK\_PRUNED}, & \text{if } \mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{0\} \quad (\hat{p} < \theta_{\text{low}}^{\text{cal}}) \\
\text{STATE 2: UNCERTAIN\_ESCALATED}, & \text{if } \mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{0, 1\} \quad (\theta_{\text{low}}^{\text{cal}} \le \hat{p} < \theta_{\text{high}}^{\text{cal}}) \\
\text{STATE 3: HIGH\_RISK\_FLAGGED}, & \text{if } \mathcal{C}_{1-\epsilon}(\mathbf{x}) = \{1\} \quad (\hat{p} \ge \theta_{\text{high}}^{\text{cal}})
\end{cases}$$

---

## 4. Contribution 3: Budget-Constrained Routing Optimization

### 4.1 Research Gap
Existing cascaded detection models evaluate thresholds heuristically without an explicit formulation of institutional compute budgets.
- Running heavy Sentence-BERT transformers ($4.5\text{ ms/pair}$) or AST graph parsers ($2.05\text{ ms/pair}$) on all submissions is computationally prohibitive for university cohorts ($O(N^2)$ pairs).
- Conversely, arbitrary aggressive pruning leads to catastrophic candidate loss.
- What institutions require is a **principled Pareto-optimal threshold optimizer** that minimizes computational cost subject to a rigorous candidate recall constraint under a fixed per-pair latency budget.

### 4.2 Comprehensive Computational Cost Model

To prevent double-counting or omission of pipeline stages, we formulate the complete end-to-end computational cost per candidate pair:
$$C_{\text{total}}(\mathbf{x}; \boldsymbol{\theta}) = C_{\text{prep}}(\mathbf{x}) + C_{\text{Tier1}}(\mathbf{x}) + P\left(\text{Escalate} \mid \boldsymbol{\theta}\right) \cdot \left[ C_{\text{feat-heavy}}(\mathbf{x}) + C_{\text{infer-heavy}}(\mathbf{x}) \right] + C_{\text{router}}(\boldsymbol{\theta})$$

#### Parameter Definitions & Empirical Baselines:
1. $C_{\text{prep}}$: Text tokenization / Code comment stripping ($0.005\text{ ms}$).
2. $C_{\text{Tier1}}$: Multi-view feature computation ($0.035\text{ ms}$ text; $0.050\text{ ms}$ code).
3. $C_{\text{feat-heavy}}$: Deep feature extraction (Text: none / cached; Code: 7-D AST parsing = $2.050\text{ ms}$).
4. $C_{\text{infer-heavy}}$: Heavy classifier forward pass (Text SBERT: $4.000\text{ ms}$; Code SVM/RF: $0.013 - 0.025\text{ ms}$).
5. $C_{\text{router}}$: Calibrated uncertainty lookup ($0.002\text{ ms}$).
6. $P(\text{Escalate} \mid \boldsymbol{\theta})$: The proportion of candidate pairs assigned to `STATE 2: UNCERTAIN_ESCALATED`.

### 4.3 Constrained Mathematical Optimization Objective

We formalize the routing policy optimization as a budget-constrained minimization problem:
$$\min_{\boldsymbol{\theta} \in \Theta} \quad \mathcal{J}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim D_{\text{val}}}\left[ C_{\text{total}}(\mathbf{x}; \boldsymbol{\theta}) \right]$$
Subject to the operational constraints:
$$\begin{aligned}
\text{Constraint 1 (Candidate Recall):} \quad & \mathcal{R}_{\text{cand}}(\boldsymbol{\theta}; D_{\text{val}}) \ge R_{\text{target}} \quad (\text{e.g., } R_{\text{target}} = 0.98) \\
\text{Constraint 2 (Latency Budget):} \quad & \mathcal{J}(\boldsymbol{\theta}) \le B_{\text{max}} \quad (\text{e.g., } B_{\text{max}} = 1.50\text{ ms/pair}) \\
\text{Constraint 3 (Direct Precision):} \quad & \mathcal{P}_{\text{direct}}(\boldsymbol{\theta}; D_{\text{val}}) \ge P_{\text{target}} \quad (\text{e.g., } P_{\text{target}} \ge 0.90)
\end{aligned}$$

Where:
- $\mathcal{R}_{\text{cand}}(\boldsymbol{\theta}) = \frac{\sum_{i=1}^N y_i \cdot \mathbb{I}(\text{State}(\mathbf{x}_i; \boldsymbol{\theta}) \in \{\text{ESCALATED}, \text{FLAGGED}\})}{\sum_{i=1}^N y_i}$.
- $\mathcal{P}_{\text{direct}}(\boldsymbol{\theta}) = \frac{\sum_{i=1}^N y_i \cdot \mathbb{I}(\text{State}(\mathbf{x}_i; \boldsymbol{\theta}) = \text{FLAGGED})}{\sum_{i=1}^N \mathbb{I}(\text{State}(\mathbf{x}_i; \boldsymbol{\theta}) = \text{FLAGGED})}$.

### 4.4 Lagrangian Formulation & Pareto Optimization
When hard constraints are relaxed for multi-objective exploration, we solve the Lagrangian dual:
$$\mathcal{L}(\boldsymbol{\theta}, \lambda_1, \lambda_2) = \mathcal{J}(\boldsymbol{\theta}) + \lambda_1 \max(0, R_{\text{target}} - \mathcal{R}_{\text{cand}}(\boldsymbol{\theta})) + \lambda_2 \max(0, \mathcal{J}(\boldsymbol{\theta}) - B_{\text{max}})$$
- **Pareto Frontier Generation:** Sweep budget constraints $B \in [0.10\text{ ms}, 4.50\text{ ms}]$ to trace the empirical Pareto curve between Candidate Recall, F1-Score, and Latency.
- **Fair Comparison Control:** The Pareto frontier of the proposed uncertainty-aware budget optimizer will be compared against:
  1. Full Expensive Execution (100% call rate).
  2. Fixed-Threshold Single Cascade (Condition B baseline).
  3. Random Pruning Control (dropout at matched pruning ratios).
  4. Original Typed-Decision Prototype (Condition C baseline).

---

## 5. Contribution 4: Explainable Cross-Modal Evidence Aggregation

### 5.1 Research Gap
Automated academic integrity tools face fierce legal and institutional scrutiny. A single similarity score ($s = 0.74$) provides no verifiable provenance and cannot distinguish between deliberate plagiarism and template reuse. Instructors require **interpretable evidence dossiers** detailing exact matching passages, transformed variables, and aligned syntax trees.

### 5.2 Text Evidence Extraction Architecture
For a suspicious text pair $(D_A, D_B)$:
1. **Smith-Waterman Local Sequence Alignment:** Finds optimal local matching token subsequences with affine gap penalties, identifying verbatim copying and localized insertion/deletion.
2. **Contextual Paraphrase Span Attribution:** Using SBERT token-level cross-attention or token cosine similarities, identifies high-similarity sentence pairs where lexical overlap is low, highlighting synonym substitutions.
3. **Evidence Extraction Output:**
   - Exact matching substring spans with character offsets.
   - Paraphrase mapping pairs: `[("An old man with a package", "A man"), ("advertisement", "ad")]`.
   - Lexical overlap proportion vs. contextual equivalence score.

### 5.3 Source Code Evidence Extraction Architecture
For a suspicious code pair $(C_A, C_B)$:
1. **AST Subtree Isomorphism Matching:** Identifies isomorphic AST statement subtrees (e.g., equivalent loop structures, arithmetic calculation blocks) even when identifiers differ.
2. **Identifier Renaming Mapping Dictionary:** Analyzes token-aligned variables to construct an explicit substitution ledger:
   $$\mathcal{M}_{\text{id}} = \left\{ \text{"i"} \mapsto \text{"counter"}, \, \text{"total"} \mapsto \text{"sumTotal"}, \, \text{"print"} \mapsto \text{"displayOutput"} \right\}$$
3. **Statement Reordering Permutation Trace:** Tracks displacement of independent basic blocks (e.g., Block 1 moved below Block 3).

### 5.4 Unified Cross-Modal Evidence Schema (`EvidenceDossier`)
To avoid forcing text and code into artificial identical features while maintaining a consistent review API, we define a polymorphic evidence contract:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EVIDENCE DOSSIER CONTRACT                       │
├────────────────────────────────────────────────────────────────────────┤
│ • Pair_ID: Unique identifier (String)                                  │
│ • Modality: 'text' | 'code'                                            │
│ • Calibrated_Plagiarism_Probability: float in [0.0, 1.0]               │
│ • Uncertainty_Metric: Predictive Entropy H(p) in [0.0, 1.0]            │
│ • Conformal_Prediction_Set: {'Non-Plagiarized'} | {'Plagiarized'} | both│
│ • Operational_Decision: 'LOW_RISK_PRUNED' | 'ESCALATED' | 'FLAGGED'   │
│ • Alignment_Spans: List of {ref_start, ref_end, sus_start, sus_end}   │
│ • Transformation_Ledger: Dict of {category, details, confidence}       │
│ • Faithfulness_Score: float in [0.0, 1.0]                              │
│ • Coverage_Score: float in [0.0, 1.0]                                  │
│ • Actionable_Summary: Human-readable narrative explanation             │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Quantitative Explanation-Quality Metrics
1. **Evidence Faithfulness (Feature Ablation Drop):**
   Measures whether masked evidence regions actually drive the model's decision. If identified matching spans are replaced with neutral text/code, the model's plagiarism probability must drop significantly:
   $$\text{Faithfulness} = \hat{P}(Y = 1 \mid \mathbf{x}) - \hat{P}(Y = 1 \mid \mathbf{x} \setminus \text{EvidenceSpans})$$
   A higher drop confirms that the extracted evidence is faithful to the classifier's internal reasoning.
2. **Evidence Coverage:**
   $$\text{Coverage} = \frac{|\text{Matched Tokens in Evidence}|}{\min(|D_A|, |D_B|)}$$
3. **Evidence Precision (Code Transformation Alignment):**
   On the IR-Plag benchmark where ground-truth obfuscation levels (L1–L6) are known, we verify whether the extracted evidence matches the injected transformation:
   - For L2: Does $\mathcal{M}_{\text{id}}$ contain the renamed variables?
   - For L3: Does the statement permutation trace match the rearranged lines?

### 5.6 Ethical & Institutional Due-Process Boundaries
- **Strict Scientific Principle:** The evidence dossier is formulated as **decision-support evidence for human educators**, never as an automated prosecutorial instrument.
- The dossier explicitly states the uncertainty interval and highlights potential benign explanations (e.g., instructor starter code, shared library templates).

---

## 6. Component-Level Specification Matrix

| Dimension | Contribution 1 (Multi-View) | Contribution 2 (Uncertainty) | Contribution 3 (Budget) | Contribution 4 (Explainability) |
| :--- | :--- | :--- | :--- | :--- |
| **Research Gap** | Single-view lexical filters drop paraphrased text (10.57% zero-overlap) | Uncalibrated scores; no statistical coverage guarantees | Heuristic threshold sweeps ignore hard compute budgets | Scalar scores are uninterpretable and fail academic review |
| **Input Data** | Raw text / code pair | Feature vectors & candidate scores | Validation cost profiles & recall constraints | Classified candidate pair & model artifacts |
| **Primary Output** | Multi-view vector & candidate retention | Calibrated prob $\hat{p}$, entropy $\mathcal{H}$, conformal set $\mathcal{C}$ | Optimal threshold policy $\boldsymbol{\theta}^*$ on Pareto frontier | Structured `EvidenceDossier` (JSON/Markdown) |
| **Key Metric** | Candidate Recall @ Pruning Ratio | ECE, Brier Score, Conformal Coverage | Latency (ms), Workload Red., Constrained F1 | Faithfulness ($\Delta P$), Evidence Coverage |
| **Ablation Study** | Remove each view individually (drop char n-grams, drop AST) | Compare Uncalibrated vs. Platt vs. Isotonic vs. Conformal | Compare Budget-Optimal vs. Fixed vs. Random Pruning | Ablate evidence masks to measure decision drop |
| **Key Failure Mode** | High multi-view compute cancels Tier-1 speed advantage | Violation of exchangeability under severe domain shift | Infeasible budget where target recall is mathematically impossible | Over-attribution of standard idiom boilerplate |
| **Novelty Claim** | Principled multi-representation candidate filtering | Distribution-free conformal routing for academic triage | Constrained Lagrangian budget optimization | Cross-modal dual-engine evidence aggregation |

---

## 7. Next Implementation Steps in Phase 2
Following this formal design, five accompanying technical documents will be generated:
1. `research_upgrade/mathematical_formulations.md` (KaTeX equations and proofs)
2. `research_upgrade/algorithm_pseudocode.md` (Executable pseudocode for all 6 core algorithms)
3. `research_upgrade/system_architecture.md` (Software components, class diagrams, data contracts)
4. `research_upgrade/experimental_protocol.md` (4-way data splitting, validation, and testing rules)
5. `research_upgrade/novelty_and_literature_analysis.md` (Focused literature review and claim boundary audit)

