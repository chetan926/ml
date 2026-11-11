# Performance Comparison Report: Original vs. Modified Solutions & Dataset Analysis

**Project Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Document Type:** Comparative Performance & Dataset Characterization Report  
**Date:** November 2026  
**Workspace:** `D:\ml rearcch project ad`

---

## 1. Executive Summary: Before vs. After Performance Comparison

This report provides a head-to-head comparison between the **Original Baseline Methodologies** ("Before" / standalone monolithic systems) and the **Modified / Proposed Staged Pipeline Solutions** ("After" / typed-decision pre-filtering).

### 1.1 Natural Language Text Modality ($N = 3,000$ Test Pairs)

| Pipeline Stage / Solution | Operating Condition | Accuracy | Precision | Recall | F1-Score | Workload Reduction | Latency/Pair |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Before)** | Traditional TF-IDF Lexical (Sajid et al.) | 0.5723 | 0.4417 | 0.7846 | 0.5652 | 0.00% (100% eval) | **0.020 ms** |
| **Original Paper (Before)** | Standalone SBERT Contextual (Setu et al. 2025) | 0.7200 | 0.5697 | **0.8570** | 0.6844 | 0.00% (100% eval) | 4.046–4.505 ms |
| **Original Paper (Before)** | Setu et al. Full Hybrid E1 ($\max(\text{TF-IDF}, \text{SBERT})$) | **0.7373** | **0.5950** | 0.8100 | **0.6861** | 0.00% (100% eval) | 4.505 ms |
| **Modified Extension (After)**| **Setu Staged Hybrid E2 (Proposed Extension)** | 0.7303 | 0.5934 | 0.7592 | 0.6661 | **25.70% saved** | **3.484 ms** |
| **Proposed Pipeline (After)** | **Typed-Decision Pre-Filter (Condition C)** | 0.7180 | 0.5751 | 0.7817 | 0.6627 | **29.97% saved** | **3.138–3.167 ms** |

*Key Text Takeaway:* The modified solutions achieve comparable semantic accuracy and F1 ($\Delta \text{F1} \le 0.023$) while avoiding **25.70% to 29.97% of heavy neural transformer calls**, accelerating inference by up to **23.5%**.

---

### 1.2 Source Code Modality ($N = 210$ Test Pairs)

| Pipeline Stage / Solution | Operating Condition | Accuracy | Precision | Recall | F1-Score | Workload Reduction | Triage/Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Before)** | Token Jaccard Lexical Pre-Filter Alone | 0.5286 | 0.5083 | 0.9020 | 0.6502 | N/A (0.00%) | 0.050 ms |
| **Original Paper (Before)** | Standalone Gandhi et al. (2024) SVM | 0.7000 | 0.6258 | 0.9510 | 0.7549 | 0.00% (100% eval) | **0.013 ms\*** |
| **Original Paper (Before)** | Standalone Mehsen & Joshi (2024) Random Forest | **0.7333** | **0.6643** | 0.9118 | **0.7686** | 0.00% (100% eval) | 0.025 ms\* |
| **Reference Paper (Before)**| Standalone Full RKRGST String Tiling (Sulistiani 2019)| 0.4905 | 0.4878 | **0.9804** | 0.6515 | 0.00% (100% eval) | 11.236–13.831 ms |
| **Reference Paper (Before)**| ES-Plag Dynamic Filter + RKRGST ($\text{init\_th}=0.50$) | 0.4524 | 0.4656 | 0.8627 | 0.6048 | 10.00% pruned | 9.914–16.970 ms |
| **Modified / Proposed (After)**| **Typed-Decision Code Pipeline (Condition C)** | 0.6762 | 0.6049 | **0.9608** | 0.7424 | **36.67% saved** | 0.058 ms\* |

*\*Note on Code Latency Scopes:* 0.013 ms, 0.025 ms, and 0.058 ms reflect classifier inference / routing over pre-extracted vectors. Offline 7-D AST feature extraction requires an additional 2.050 ms/pair.  
*Key Code Takeaway:* The proposed typed-decision pipeline increases true-positive detection sensitivity to **96.08%** (higher than Gandhi SVM at 95.10%), achieves **100.00% candidate recall** (0 dropped true positives), and delivers a **36.67% reduction in expensive classifier calls** (36.19% direct Tier-1 flags + 0.48% low-risk pruning).

---

## 2. Detailed "Before vs. After" Dissection by Research Paper

### 2.1 Setu et al. (2025) — Contextual Text Methodology

#### Original Solution ("Before")
- **Architecture:** Monolithic hybrid similarity score evaluating 100% of candidate pairs:
  $$S_{\text{hybrid}}(D_A, D_B) = \max\left(S_{\text{TF-IDF}}(D_A, D_B), \, S_{\text{SBERT}}(D_A, D_B)\right)$$
- **Performance (Condition E1, Calibrated $\tau = 0.5400$):**
  - **Accuracy:** 0.7373
  - **Precision:** 0.5950
  - **Recall:** 0.8100
  - **F1-Score:** 0.6861
  - **Confusion Matrix:** $[TN=1,351, FP=586, FN=202, TP=861]$
  - **Compute Overhead:** Evaluates heavy Sentence-BERT transformer across all 3,000 candidate pairs (100.0% call rate, 4.505 ms/pair).

#### Modified Solution ("After")
- **Condition E2 (Cost-Efficient Staged Hybrid):** Applies validation-calibrated dual-threshold routing $[\theta_{low}=0.05, \theta_{high}=0.80]$ before invoking SBERT.
- **Performance:**
  - **Accuracy:** 0.7303
  - **Precision:** 0.5934
  - **Recall:** 0.7592
  - **F1-Score:** 0.6661
  - **Confusion Matrix:** $[TN=1,384, FP=553, FN=256, TP=807]$
  - **SBERT Invocations:** Reduced to 2,229 pairs (**74.30% call rate**).
  - **Workload Avoided:** **25.70%** (771 transformer calls eliminated).
  - **Execution Latency:** Dropped from 4.505 ms to **3.484 ms/pair** (**22.66% latency reduction** vs. E1).

---

### 2.2 Gandhi et al. (2024) — AST/Syntactic Support Vector Machine

#### Original Solution ("Before")
- **Architecture:** Standalone RBF-kernel SVM using a 7-dimensional feature space (Jaccard, Cosine, Levenshtein, Keyword overlap, Operator overlap, Token-length delta, Cyclomatic complexity delta).
- **Performance (Tuned on `case-05`, $\tau = 0.3700$):**
  - **Accuracy:** 0.7000
  - **Precision:** 0.6258
  - **Recall:** 0.9510
  - **F1-Score:** 0.7549
  - **Confusion Matrix:** $[TN=50, FP=58, FN=5, TP=97]$
  - **Workload:** 100.0% of pairs undergo full 7-D AST extraction and SVM classification.

#### Modified Solution ("After")
- **Architecture:** Token Jaccard triage pre-filter $[\theta_{low}=0.20, \theta_{high}=0.80]$ wrapping the Gandhi SVM.
- **Performance (Proposed Condition C):**
  - **Accuracy:** 0.6762
  - **Precision:** 0.6049
  - **Recall:** **0.9608** (98 of 102 detected — higher sensitivity than standalone SVM)
  - **F1-Score:** 0.7424
  - **Confusion Matrix:** $[TN=44, FP=64, FN=4, TP=98]$
  - **Candidate Recall:** **1.0000 (100.00%)** — perfect preservation of plagiarized pairs.
  - **Workload Avoided:** **36.67%** (76 pairs directly flagged at Tier 1 + 1 pair pruned; only 133 pairs escalated to SVM).

---

### 2.3 Mehsen & Joshi (2024) — Random Forest Ensemble

#### Original Solution ("Before")
- **Architecture:** Ensemble of 100 de-correlated decision trees ($\text{max\_depth}=6$) evaluated on the 7-D feature space.
- **Performance ($\tau = 0.5000$):**
  - **Accuracy:** 0.7333 (Highest among standalone code models)
  - **Precision:** 0.6643 (Highest specificity: 56.48%)
  - **Recall:** 0.9118
  - **F1-Score:** 0.7686
  - **Confusion Matrix:** $[TN=61, FP=47, FN=9, TP=93]$
  - **Obfuscation Hallmark:** Achieved **83.33% detection on Level 6 compound attacks** (15 of 18 detected), outperforming SVM (77.78%).

---

### 2.4 Sulistiani & Karnalim (2019) — ES-Plag RKRGST

#### Original Solution ("Before")
- **Architecture:** Dynamic length-based cosine pre-filter + Running-Karp-Rabin Greedy String Tiling (RKRGST) with tile penalty normalization $(k - 1)$.
- **Performance (Nominal Baseline $\text{init\_th}=0.50$):**
  - **Accuracy:** 0.4524
  - **Precision:** 0.4656
  - **Recall:** 0.8627
  - **F1-Score:** 0.6048
  - **Candidate Recall:** 0.8627 (14 false negatives pruned prematurely by dynamic filter)
  - **Pruning Ratio:** 10.00%
  - **Runtime:** 9.914–16.970 ms/pair (extremely compute-intensive string tiling).

---

## 3. Dataset Limitations & Threats to External Validity

### 3.1 Text Modality: SNLI / Pereira Proxy Benchmark

1. **Synthetic NLI Task Proxy vs. Authentic Academic Submissions:**
   - The dataset hosted on Kaggle (`ruvelpereira/mit-plagairism-detection-dataset`) is derived directly from the **Stanford Natural Language Inference (SNLI)** corpus (Bowman et al., 2015), based on Flickr30k image captions.
   - It represents sentence-level logical entailment, **not** genuine student plagiarism, academic essays, or dissertation chapters.
   - Logical entailment ("A woman playing guitar" $\to$ "A woman making music") tests conceptual subsumption, whereas academic plagiarism involves unauthorized copying, patchwriting, and paraphrasing without citation.

2. **Sentence-Level Granularity vs. Macro Document Structure:**
   - In SNLI, premise sentences average **13.92 words** and hypothesis sentences average **7.50 words**.
   - Real academic integrity pipelines inspect multi-paragraph essays, thesis chapters, or lab reports ($500$ to $5,000+$ words).
   - Paragraph-length documents provide extensive lexical and structural context that mitigates short-sentence ambiguity.

3. **The Zero-Overlap Theoretical Ceiling (88.81% Candidate Recall Limit):**
   - **10.57% of true positive pairs (119 of 1,063 test positives) share exactly zero common words** ($\text{TF-IDF} = 0.0000$).
   - *Consequence:* Any lexical pre-filter with a threshold $\theta_{low} > 0$ will drop these pairs. This imposes an intrinsic theoretical ceiling of $\approx 88.81\%$ on lexical candidate recall for this dataset. Full-length essays rarely exhibit zero lexical overlap.

4. **Class Balance Inconsistency with Production:**
   - Positive prevalence in the test set is **35.43%** (1,063 positives out of 3,000 pairs).
   - In university deployments, the true prevalence of plagiarized submissions is typically $< 3\% - 5\%$. Evaluating at 35% prevalence artificially inflates precision compared to operational environments.

---

### 3.2 Source Code Modality: IR-Plag Java Corpus

1. **Introductory Curriculum Scope:**
   - IR-Plag consists of 7 introductory undergraduate programming tasks (`case-01` to `case-07`): print statements, matrix multiplication, triangle area, prime sieve, calendar date calculation, string palindrome, and bubble sort.
   - File lengths range from **20 to 50 lines of code**.
   - *Limitation:* Does not represent upper-division computer science codebases involving object-oriented architectures, design patterns, inheritance hierarchies, multi-threading, or third-party libraries ($500$ to $5,000+$ lines).

2. **Low Algorithmic Entropy & Shared Boilerplate:**
   - Elementary programming tasks have limited syntactic diversity. For example, in `case-06` (Palindrome), almost all independent student solutions utilize identical two-pointer `while` loops.
   - This creates extensive incidental overlap among completely independent students, driving high false positive rates:
     - Gandhi SVM: $\text{FPR} = 53.70\%$ (58 false alarms)
     - Mehsen RF: $\text{FPR} = 43.52\%$ (47 false alarms)
     - ES-Plag Full RKRGST: $\text{FPR} = 97.22\%$ (105 false alarms)

3. **Language Specificity:**
   - The AST parser and feature extraction pipeline are tailored to the **Java language** (50 reserved keywords, standard delimiters, and Java cyclomatic rules).
   - It cannot be directly applied to Python, C++, Go, or Rust without developing language-specific AST grammar extractors.
   - Cross-language plagiarism (e.g., translating a Java submission into Python) was not evaluated.

4. **Resilience Attenuation Under Compound Attacks (Level 6):**
   - While detectors achieve near-perfect detection on single transformations (100% on Levels 1–3), detection drops significantly when students combine renaming, loop restructuring, and method extraction simultaneously (Gandhi SVM: 77.78%, Mehsen RF: 83.33%, Proposed Pipeline: 88.89%).

---

## 4. Key Empirical Observations on the Datasets

### 4.1 Natural Language Text Dataset (SNLI / Pereira Proxy)

```
Text Dataset Partitioning & Descriptive Statistics:
┌──────────────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Feature / Attribute          │ Validation Split │ Test Split       │ Total / Scope    │
├──────────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Total Evaluated Pairs        │ 3,000 pairs      │ 3,000 pairs      │ 6,000 pairs      │
│ Plagiarized Pairs (Positive) │ 1,031 (34.37%)   │ 1,063 (35.43%)   │ 2,094 (34.90%)   │
│ Non-Plagiarized (Negative)   │ 1,969 (65.63%)   │ 1,937 (64.57%)   │ 3,906 (65.10%)   │
│ Premise Mean Word Length     │ 13.92 words      │ 13.88 words      │ 13.90 words      │
│ Hypothesis Mean Word Length  │ 7.50 words       │ 7.48 words       │ 7.49 words       │
│ Zero-Overlap Positive Pairs  │ 109 (10.57%)     │ 119 (11.19%)     │ 228 (10.89%)     │
│ Cross-Split Duplicate Pairs  │ 0 pairs (0.00%)  │ 0 pairs (0.00%)  │ 0 pairs (0.00%)  │
└──────────────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

#### Observations:
1. **Extreme Brevity:** Hypotheses average only 7.5 words, severely limiting the vocabulary available for statistical frequency models like TF-IDF.
2. **Distinct Semantic Cluster:** Scatter analysis (Figure 15) shows a dense cluster of pairs where $\text{TF-IDF} \approx 0.0$ but $\text{SBERT} > 0.70$. These represent pure synonym replacements that fool lexical models but are caught by dense neural embeddings.
3. **High Negative Specificity of Lexical Match:** When $\text{TF-IDF} \ge 0.80$, the empirical precision is **96.55%** (only 4 false positives out of 116 pairs). This proves that high lexical similarity is an extraordinarily reliable fast trigger for direct high-risk flagging.

---

### 4.2 Source Code Dataset (IR-Plag Java Corpus)

```
IR-Plag Java Dataset Task-Disjoint Architecture:
┌──────────────┬────────────────────────────────┬─────────────┬──────────┬──────────┬─────────────────┐
│ Split Role   │ Assessment Tasks Included      │ Total Pairs │ Positive │ Negative │ Isolation Level │
├──────────────┼────────────────────────────────┼─────────────┼──────────┼──────────┼─────────────────┤
│ Training     │ case-01, case-02, case-03, 04  │ 416 pairs   │ 200      │ 216      │ Train Only      │
│ Validation   │ case-05 (Day of Week Calc)     │ 107 pairs   │ 53       │ 54       │ Val Only (Tune) │
│ Test Split   │ case-06 (Palin), case-07 (Sort)│ 210 pairs   │ 102      │ 108      │ Out-of-Sample   │
├──────────────┼────────────────────────────────┼─────────────┼──────────┼──────────┼─────────────────┤
│ Total Corpus │ 7 Introductory Tasks           │ 733 pairs   │ 355      │ 378      │ Zero Leakage    │
└──────────────┴────────────────────────────────┴─────────────┴──────────┴──────────┴─────────────────┘
```

#### Obfuscation Breakdown across Test Partition ($N = 210$ Pairs):
- **Level 1 (Comments & Layout):** 18 pairs $\to$ SVM: 100.0%, RF: 100.0%, Proposed: 100.0%
- **Level 2 (Identifier Renaming):** 16 pairs $\to$ SVM: 100.0%, RF: 93.75%, Proposed: 100.0%
- **Level 3 (Statement Reordering):** 17 pairs $\to$ SVM: 100.0%, RF: 100.0%, Proposed: 100.0%
- **Level 4 (Control Flow Replacement):** 17 pairs $\to$ SVM: 94.12%, RF: 88.24%, Proposed: 94.12%
- **Level 5 (Method Abstraction):** 16 pairs $\to$ SVM: 93.75%, RF: 87.50%, Proposed: 93.75%
- **Level 6 (Compound Attack):** 18 pairs $\to$ SVM: 77.78%, RF: 83.33%, Proposed: 88.89%

#### Observations:
1. **Pre-Filter Asymmetry:** In code, token Jaccard similarity is highly effective at identifying verbatim and lightly modified submissions ($\theta_{high} \ge 0.80$ captures 36.19% of pairs immediately). However, low-similarity pruning ($\theta_{low} \le 0.20$) only prunes 0.48% because student solutions to the same problem almost always share basic syntax (`class`, `public static void main`, loop variables).
2. **Workload Reduction Shift:** In text, workload reduction is driven primarily by **low-risk pruning** (21.83% to 26.53%). In code, workload reduction is driven almost entirely by **high-risk direct flagging** (36.19%), demonstrating that typed-decision architectures adapt naturally to the distinct structural properties of different media.
3. **Compound Attack Vulnerability:** Single-metric pre-filters and even standalone SVMs struggle with multi-transform attacks (L6), confirming that structural AST metrics and non-linear ensembles (Random Forest / GNNs) are necessary for resilient code defense.

---

## 5. Methodological Summary & Recommendations

1. **Academic Honesty in Reporting:** Treat the text benchmark strictly as a sentence-level semantic equivalence proxy (SNLI) rather than genuine essay plagiarism.
2. **Dual-Threshold Necessity:** Single-threshold pre-filters fail catastrophic recall constraints (dropping 22%–26% of true positives). The tri-band formulation $[\theta_{low}, \theta_{high}]$ is mathematically required to preserve candidate sensitivity.
3. **Future Research Directions:**
   - Evaluate text pre-filtering on authentic document-level corpora (e.g., PAN Plagiarism Corpus).
   - Expand code pre-filtering to Program Dependence Graph (PDG) embeddings and Graph Neural Networks for L6 compound obfuscation.
   - Incorporate sub-quadratic inverted index clustering ($O(N \log N)$ or LSH) prior to pairwise pre-filtering for large cohorts ($N > 1,000$).

