# An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Authors:** Advanced ML & Academic Integrity Research Group  
**Status:** Certified Phase 4 Methodological Correction Package  
**Repository Scope:** `research_upgrade/phase_4_corrections/`  
**Date:** November 11, 2026  

---

## Abstract

Automated academic integrity detection across natural language essays and source code submissions is fundamentally constrained by the quadratic computational complexity of deep neural cross-encoders and graph-based verifiers. Deployed systems confront an unfavorable trilemma: relying on brittle lexical heuristics that collapse under paraphrasing and identifier renaming, executing computationally prohibitive deep neural verifiers across all pairwise comparisons, or outputting uncalibrated confidence scores that lack statistical guarantees. In this paper, we present an uncertainty-aware, budget-constrained multi-view detection framework that addresses this operational trilemma through four tightly integrated contributions: (1) **Robust Multi-View Candidate Generation** combining lexical, character n-gram, dense semantic, structural, and complexity representations to filter unambiguous non-matches while maximizing candidate recall; (2) **Uncertainty-Aware Probability Calibration and Group Conformal Prediction** employing regularized Platt scaling and exact finite-sample group conformal inference over exchangeable premise sampling units; (3) **Budget-Constrained Adaptive Routing** optimizing a three-way decision taxonomy (`LOW_RISK_PRUNED`, `UNCERTAIN_ESCALATED`, `HIGH_RISK_FLAGGED`) that allocates expensive Stage-2 verification exclusively to high-uncertainty boundary candidates; and (4) **Authentic Cross-Modal Evidence Aggregation** generating audit-compliant evidence dossiers containing verbatim matching character spans, token ranges, Java line numbers, and procedural disclaimers directly from genuine submissions.

Rigorous empirical evaluation on a 3,000-pair held-out text test set (1,015 premise groups) demonstrates that the proposed framework achieves **MCC = 0.5288**, **Accuracy = 0.7920**, and **Precision = 0.7157** (compared to 0.4549 MCC and 0.5369 precision for dense semantic similarity, $p < 10^{-16}$ via McNemar's test), while achieving **95.67% empirical group coverage** under a nominal 95% conformal guarantee and safely eliminating **36.13%** of expensive Stage-2 verifier calls. On the IR-Plag Java benchmark, rigorous Leave-One-Group-Out cross-validation across the 7 assignment tasks establishes a zero-program-leakage baseline of **MCC = 0.6892** and **Accuracy = 0.8431**, directly exposing that standard stratified cross-validation artificially inflated historical performance by **+0.1772 MCC** due to reference program overlap. We transparently report that the framework prioritizes high precision and cost reduction by trading off marginal recall on diffuse paraphrases.

**Keywords:** Academic integrity detection, multi-view learning, probability calibration, conformal prediction, budget-constrained routing, explainable AI, plagiarism detection.

---

## 1. Introduction & Problem Formulation

Institutions evaluating student cohorts face an $O(N^2)$ comparison explosion: evaluating pairwise similarity across $N = 1,000$ student submissions requires 499,500 document comparisons. While recent transformer cross-encoders achieve state-of-the-art accuracy, their substantial latency (30–100 ms per pair) renders exhaustive universal deployment economically and computationally infeasible.

Conversely, legacy fingerprinting tools (e.g., Turnitin, JPlag, MOSS) rely predominantly on raw lexical overlap. As demonstrated in our experiments, single-view lexical thresholds collapse under semantic paraphrasing and identifier obfuscation, requiring thresholds to drop to zero to capture non-verbatim borrowing, which eliminates all filtering efficiency. Furthermore, uncalibrated similarity scores provide no statistical validity, introducing substantial procedural fairness risks when scores are interpreted as posterior probabilities of misconduct.

To resolve these compounding challenges, this research introduces a principled, budget-adaptive architecture combining multi-view candidate screening, calibrated uncertainty estimation, conformal prediction sets, and authentic explainable evidence dossiers.

---

## 2. Methodology & Proposed Architecture

The proposed system enforces a sequential decision pipeline:
$$\text{Input Pair } (x_a, x_b) \xrightarrow{\text{C1: Multi-View}} s_1 \xrightarrow{\text{C2: Calibration}} (p, H(p), C_\alpha) \xrightarrow{\text{C3: Routing}} \text{Decision} \xrightarrow{\text{Escalated Only}} \text{Stage 2} \xrightarrow{\text{C4: Evidence}} \text{Dossier}$$

### 2.1 C1: Robust Multi-View Candidate Generation
Candidate generation projects document pairs into complementary feature spaces:
- **Text Modality (11 Views):** Word TF-IDF cosine, word Jaccard, token overlap ratio, character TF-IDF, character 3-gram Jaccard, longest common subsequence (LCS) ratio, dense MiniLM semantic cosine similarity, structural length ratio, token count delta, sentence count delta, and Type-Token Ratio (TTR) similarity.
- **Code Modality (13 Views):** Raw token Jaccard, token frequency overlap, Running-Karp-Rabin Greedy-String-Tiling (RKR-GST), identifier-normalized token Jaccard and frequency overlap, normalized RKR-GST, AST nesting depth similarity, control block similarity, method structure similarity, operator distribution divergence, control-flow similarity, cyclomatic complexity delta, and SLOC ratio.

### 2.2 C2: Uncertainty-Aware Calibration & Group Conformal Prediction
1. **Regularized Platt Calibration:** Raw Stage-1 scores $s_1 \in [0, 1]$ are mapped to posterior probabilities:
   $$P(Y = 1 \mid s_1) = \frac{1}{1 + \exp(-(A \cdot s_1 + B))}$$
   where parameters $A, B$ are fitted on validation cohorts with $L_2$ regularization.
2. **Predictive Entropy:** Epistemic and aleatoric uncertainty are quantified in bits via Shannon predictive entropy:
   $$H(p) = - [p \log_2(p) + (1 - p) \log_2(1 - p)]$$
3. **Exact Finite-Sample Group Conformal Prediction:**
   Premise groups $G_j$ act as exchangeable sampling units. Nonconformity for pair $i \in G_j$ is defined as $R_i = 1 - P(Y_i = y_i \mid x_i)$. Group-level nonconformity is the group supremum:
   $$R(G_j) = \max_{i \in G_j} (1 - P(Y_i = y_i \mid x_i))$$
   Given $m$ calibration groups, the exact conformal quantile rank is $k = \lceil (m + 1)(1 - \alpha) \rceil$. If $k > m$, $\hat{q} = \infty$; otherwise $\hat{q} = R_{(k)}$. Prediction sets are constructed as:
   $$C_\alpha(x) = \{y \in \{0, 1\} : (1 - P(Y = y \mid x)) \le \hat{q}\}$$

### 2.3 C3: Budget-Constrained Adaptive Routing
Candidates are partitioned into three operational states:
- $\text{LOW\_RISK\_PRUNED}$: $s_1 < \tau_{\text{prune}} \land p < 0.25 \land H(p) < \eta$ (pruned early as non-plagiarized; Stage-2 calls avoided).
- $\text{HIGH\_RISK\_FLAGGED}$: $s_1 \ge \tau_{\text{flag}} \land p \ge 0.75 \land H(p) < \eta$ (directly flagged; Stage-2 calls avoided).
- $\text{UNCERTAIN\_ESCALATED}$: intermediate risk or high predictive entropy ($H(p) \ge \eta$), escalated to Stage-2 heavy verification.

### 2.4 C4: Authentic Explainable Evidence Aggregation
Evidence is extracted directly from genuine source document pairs. Verbatim matching character spans, token ranges, and 1-indexed Java line numbers are localized and categorized by feature view into immutable JSON Evidence Dossiers.

---

## 3. Experimental Protocol & Reconciliations

### 3.1 Dataset Partitioning & Leakage Safeguards
- **Text Corpus:** 8,500 clean pairs partitioned into strictly disjoint premise groups: $D_{\text{train}}$ (4,000 pairs, 1,349 premises), $D_{\text{val}}$ (1,500 pairs, 508 premises), and $D_{\text{test}}$ (3,000 pairs, 1,015 premises). Validation premise groups were further partitioned into disjoint calibration sets for probability scaling (254 groups) and conformal prediction (254 groups).
- **Code Corpus:** 720 pairs under Policy A and 709 pairs under Policy B from IR-Plag. Evaluated using Leave-One-Group-Out (LOGO) across the 7 assignment cases to eliminate reference program leakage.
- **Reconciliation of Zero-Overlap Positives:**
  - `119`: Historical exploratory count from Phase 2 on an unpartitioned development subset.
  - `33`: Exact count of zero raw whitespace-separated word-overlap positive pairs in active $D_{\text{test}}$.
  - `25`: Exact count in $D_{\text{test}}$ under regex alphanumeric word-token normalization (`\w+`).

---

## 4. Empirical Results & Discussion

### 4.1 Text Modality Results

| Model / Framework | Accuracy | Precision | Recall | F1-Score | MCC | Stage-2 Calls Avoided |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Framework** | **0.7920** | **0.7157** | **0.6501** | **0.6813** | **0.5288** | **36.13%** |
| Dense Semantic Baseline | 0.6987 | 0.5369 | 0.8645 | 0.6624 | 0.4549 | 0.00% |
| TF-IDF Lexical Baseline | 0.6143 | 0.4572 | 0.6823 | 0.5475 | 0.2480 | 0.00% |
| **Paired Difference $\Delta$** | **+0.0932** | **+0.1787** | **-0.2146** | **+0.0187** | **+0.0737** | **+36.13%** |
| 95% Bootstrap CI | [0.076, 0.111] | [0.155, 0.202] | [-0.245, -0.185] | [-0.004, 0.041] | [0.039, 0.108] | — |
| Significance (McNemar) | — | — | — | — | $\chi^2 = 95.16, p < 10^{-16}$ | Statistically Significant |

### 4.2 Conformal Prediction Coverage
On 1,015 test premise groups, the exact finite-sample conformal predictor achieved **95.67% empirical group coverage** (nominal 95% target met) and **98.53% empirical pair coverage**, with an average prediction set size of **1.6757** (32.4% singletons, 67.6% ambiguous pairs, 0.0% empty sets).

### 4.3 Code Modality & Leakage Analysis

| Evaluation Protocol | Accuracy | Precision | Recall | F1-Score | MCC | Leakage Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group-Disjoint LOGO (Policy A)** | **0.8431** | **0.8070** | **0.8801** | **0.8420** | **0.6892** | **ZERO_LEAKAGE** |
| **Group-Disjoint LOGO (Policy B)** | **0.8533** | **0.8127** | **0.8912** | **0.8501** | **0.7100** | **ZERO_LEAKAGE** |
| Historical Stratified 5-Fold | 0.9333 | 0.9249 | 0.9357 | 0.9302 | 0.8665 | Leakage Present |
| **Leakage Inflation ($\Delta$)** | **-0.0902** | **-0.1179** | **-0.0556** | **-0.0882** | **-0.1772** | Measured Bias |
| Token Jaccard Baseline | 0.4750 | 0.4750 | 1.0000 | 0.6441 | 0.0000 | Baseline |

---

## 5. Limitations & Threats to Validity

1. **Declared Precision-Recall Trade-Off:** The framework prioritizes high precision and false-positive elimination over exhaustive recall. In applications where catching every marginal violation outweighs false accusation harm, single-view semantic thresholds remain competitive.
2. **Proxy Text Benchmark Scope:** Evaluated on sentence-level semantic inference pairs (SNLI/Pereira). Generalization to multi-page essays requires future benchmarking on long-form corpora (e.g. PAN).
3. **Heuristic Code AST Parsing:** Due to execution environment constraints, code structural features utilize regex- and indentation-based heuristic blocks rather than compiler AST graph embeddings.
4. **Case-05 Topology:** All plagiarized pairs in Case-05 depend on a single reference program (`prog_original`). When `prog_original` is held out, the training fold contains 0 positive pairs. Cross-case training is mathematically required for valid evaluation.

---

## 6. Conclusion

The Phase 4 corrections establish a rigorous, reproducible, and methodologically sound academic integrity detection framework. By integrating multi-view screening, regularized calibration, finite-sample valid group conformal prediction, budget-adaptive routing, and authentic evidence dossiers, the framework achieves statistically superior precision and efficiency without relying on inflated cross-validation metrics.

