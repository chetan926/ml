# PHASE 3.3 FEATURE CONTRIBUTION REPORT: MULTI-VIEW FEATURE SIGNAL ANALYSIS

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Document:** Phase 3.3 Feature Contribution & Attribution Analysis  
**Date:** 2026-11-11  
**Author:** Independent Research Architect & Methodologist  

---

## 1. Objective and Overview

This report provides a formal, evidence-driven attribution of feature contributions across the **11 text features** and **13 code features** utilized in the Phase 3.3 multi-view candidate-generation architecture.

The central research question answered by this analysis is:
> *Which similarity views provide the strongest discriminative power for candidate escalation, which features provide complementary coverage of edge cases, and which features exhibit redundancy or minimal marginal utility?*

---

## 2. Text Modality Feature Attribution Analysis

### 2.1 Univariate Correlation & Linear Attribution ($D_{\text{train}}$, 4,000 pairs)

| View | Feature Name | Pearson $r$ with Label | Logistic Regression Weight $\beta_j$ | Random Forest Feature Importance | Primary Signal / Attribution Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **View 3: Semantic** | `text_semantic_minilm_sim` | **0.5482** | **+2.4180** | **44.82%** | **Primary anchor signal.** Captures high-level semantic meaning and paraphrases that share few or no common tokens. |
| **View 1: Lexical** | `text_word_tfidf_sim` | **0.4912** | **+1.8540** | **18.35%** | **Strong verbatim indicator.** High precision on direct copying; discriminates exact lexical overlap. |
| **View 2: Character** | `text_char_tfidf_sim` | 0.4421 | +1.1205 | 11.20% | **Sub-word similarity.** Resilient to minor misspellings and word order permutations. |
| **View 2: Character** | `text_char_jaccard_3gram` | 0.4305 | +0.7850 | 7.64% | Character n-gram coverage; reinforces character-level alignment. |
| **View 1: Lexical** | `text_word_jaccard` | 0.4120 | +0.5410 | 5.80% | Simple vocabulary overlap; partially redundant with word TF-IDF. |
| **View 1: Lexical** | `text_word_overlap_ratio` | 0.3890 | +0.4120 | 4.10% | Multiset token frequency overlap. |
| **View 2: Character** | `text_char_lcs_ratio` | 0.3540 | +0.3980 | 3.25% | Sequential character alignment (Longest Common Subsequence). |
| **View 4: Structural** | `text_token_count_delta` | 0.1850 | +0.3120 | 2.10% | Length asymmetry penalty; filters pairs with wildly different token counts. |
| **View 4: Structural** | `text_length_ratio` | 0.1620 | +0.2150 | 1.25% | Character-level length parity. |
| **View 4: Structural** | `text_sentence_delta` | 0.0840 | +0.0980 | 0.85% | Sentence count alignment. |
| **View 4: Structural** | `text_ttr_sim` | 0.0410 | +0.0520 | 0.64% | Lexical diversity similarity (Type-Token Ratio); minimal standalone signal. |

### 2.2 Complementarity and Cross-View Synergy
1. **The Semantic Rescue Mechanism:**
   In $D_{\text{val}}$, there are 18 positive pairs with zero word overlap (`text_word_jaccard == 0.0`). The single-view lexical model assigned scores $\le 0.02$ to all of them, causing 100% false negatives at standard thresholds. The multi-view model leveraged `text_semantic_minilm_sim` (mean: 0.741) to escalate **16 of the 18 (88.9%)** without forcing lexical thresholds to zero.
2. **The Structural Triage Filter:**
   While structural features have low standalone correlation ($r < 0.20$), their presence in the 11-feature model increased test workload reduction from **20.77%** (semantic-only) to **27.37%** (all views). They provide an effective negative filter: non-plagiarized pairs that coincidentally discuss similar topics often have asymmetric lengths, which the structural weights penalize.

---

## 3. Code Modality Feature Attribution Analysis

### 3.1 Univariate Correlation & Linear Attribution (Policy A, 720 pairs)

| View | Feature Name | Pearson $r$ with Label | Random Forest Importance | Gradient Boosting Importance | Primary Signal / Attribution Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **View 2: Id-Norm** | `code_norm_rkrgst_sim` | **0.6120** | **31.45%** | **34.20%** | **Dominant structural sequence signal.** Resilient to variable renaming; matches maximal structural token tiles. |
| **View 2: Id-Norm** | `code_norm_token_jaccard` | **0.5840** | **22.10%** | **24.50%** | Normalized vocabulary overlap; provides fast renaming-invariant screening. |
| **View 1: Raw Token** | `code_rkrgst_sim` | 0.5210 | 14.80% | 12.40% | Exact sequence tiling; definitive on unmodified source code. |
| **View 1: Raw Token** | `code_token_jaccard` | 0.4980 | 9.60% | 8.90% | Fast raw token Jaccard similarity. |
| **View 2: Id-Norm** | `code_norm_freq_overlap` | 0.4620 | 6.20% | 5.80% | Normalized token multiset intersection. |
| **View 3: Struct Fallback** | `code_control_block_sim` | 0.3810 | 4.80% | 4.50% | Distribution of control-flow keywords (`for`, `while`, `if`, `try`). |
| **View 3: Struct Fallback** | `code_nesting_depth_sim` | 0.3420 | 3.50% | 3.20% | Brace nesting depth profile similarity. |
| **View 4: Complexity** | `code_cyclomatic_sim` | 0.2980 | 2.40% | 2.10% | McCabe cyclomatic complexity ratio. |
| **View 4: Complexity** | `code_control_flow_sim` | 0.2850 | 1.90% | 1.80% | Branching keyword profile cosine similarity. |
| **View 4: Complexity** | `code_operator_dist_sim` | 0.2740 | 1.45% | 1.30% | Operator pattern cosine similarity. |
| **View 4: Complexity** | `code_sloc_ratio` | 0.2410 | 1.10% | 0.90% | Source lines of code ratio. |
| **View 3: Struct Fallback** | `code_method_structure_sim` | 0.2120 | 0.70% | 0.40% | Method modifier distribution. |

### 3.2 The Identifier Normalization Advantage
On obfuscated Java submissions (levels L1 to L4 in IR-Plag):
- Raw token Jaccard similarity drops by an average of **0.341** when variable and function names are systematically renamed.
- Identifier-normalized token similarity (`code_norm_token_jaccard`) recovers **+0.0765** mean similarity over raw token Jaccard, directly preventing false negatives on renamed code.
- Tree ensembles effectively combine normalized RKR-GST and structural block similarities to safely filter **43.5% of un-plagiarized pairs** while preserving **98.8% candidate recall**.

---

## 4. Key Recommendations for Hierarchical Adaptive Routing (Phase 3.5 Preview)

Based on the empirical latency and attribution findings of Phase 3.2 and Phase 3.3:
1. **Stage 1 Fast Lexical Screening:** Structural features (0.12 ms) and Lexical TF-IDF (2.26 ms) can instantly eliminate obvious negative pairs that share neither length nor vocabulary before invoking dense MiniLM embeddings (26.6 ms).
2. **Stage 1 Fast Code Screening:** Complexity (0.46 ms) and Structural fallback (0.49 ms) can triage distractor pairs at ~1,000 pairs/sec before running computationally expensive RKR-GST sequence tiling (2.42 ms).
3. **Multi-View Decision Fusion:** Candidate generation must retain dense semantic and identifier-normalized signals to avoid catastrophic recall drops on paraphrased and renamed plagiarism cases.

