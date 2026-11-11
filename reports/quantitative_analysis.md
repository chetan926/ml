# Quantitative Analysis and Experimental Findings

**Project Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Document:** Quantitative Evaluation Report  
**Date:** November 2026  

---

## 1. Executive Quantitative Summary

All numerical results reported in this document were generated from executed experiments on authentic benchmark datasets:
- **Text:** MIT / SNLI Plagiarism Benchmark ($N = 3,000$ validation, $N = 3,000$ test pairs).
- **Source Code:** IR-Plag Java Corpus ($N = 416$ train, $N = 107$ validation, $N = 210$ test pairs across tasks `case-01` to `case-07`).

---

## 2. Text Modality Results

### 2.1 Benchmark Performance Metrics

| Experimental Condition | Accuracy | Precision | Recall | F1-Score | Specificity | False Positive Rate | Candidate Recall | Pruning Ratio (%) | Call Rate (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Method 1: TF-IDF Baseline (Sajid et al. 2025)** | 0.5723 | 0.4417 | 0.7846 | 0.5652 | 0.4559 | 0.5441 | N/A | 0.00% | 100.0% | **0.020 ms** |
| **Method 2: SBERT Semantic (Setu et al. 2025)** | **0.7200** | 0.5697 | **0.8570** | **0.6844** | 0.6448 | 0.3552 | N/A | 0.00% | 100.0% | 4.143 ms |
| **Condition A: Full Analysis** | 0.7200 | 0.5697 | 0.8570 | 0.6844 | 0.6448 | 0.3552 | 1.0000 | 0.00% | 100.0% | 4.143 ms |
| **Condition B: Fixed Threshold (50th %ile)** | 0.7090 | 0.5750 | 0.6849 | 0.6252 | **0.7223** | **0.2777** | 0.7366 | **42.87%** | 57.13% | 2.904 ms |
| **Condition C: Proposed Typed Pipeline** | **0.7180** | **0.5751** | **0.7817** | **0.6627** | 0.6830 | 0.3170 | **0.8598** | **26.53%** | **70.03%** | **3.167 ms** |

### 2.2 Key Analytical Observations for Text
1. **Computational Speedup vs. Semantic Fidelity:** Method 1 executes in 0.020 ms per pair but suffers from poor specificity (0.4559) and elevated FPR (0.5441). Method 2 increases F1 to 0.6844 and accuracy to 0.7200, but is ~207x slower (4.143 ms).
2. **Failure of Single-Threshold Filtering (Condition B):** A fixed lexical threshold reduces heavy model calls to 57.13% but causes candidate recall to collapse from 1.0000 to 0.7366—discarding over 26% of true plagiarized submissions before deep analysis.
3. **Proposed Pipeline Pareto Efficiency (Condition C):** Condition C successfully prunes 796 non-plagiarized pairs (26.53%) and directly flags 103 high-risk pairs, requiring only 2,101 SBERT invocations (70.03% call rate), while preserving **0.8598 Candidate Recall** (retaining 914 of 1,063 true positives).

---

## 3. Source Code Modality Results

### 3.1 Benchmark Performance Metrics

| Experimental Condition | Accuracy | Precision | Recall | F1-Score | Specificity | False Positive Rate | Candidate Recall | Pruning Ratio (%) | Call Rate (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Token Jaccard Filter** | 0.5286 | 0.5083 | 0.9020 | 0.6502 | 0.1759 | 0.8241 | N/A | 0.00% | 0.00% | 0.050 ms |
| **Method 3: SVM (Gandhi et al. 2024)** | **0.7000** | **0.6258** | 0.9510 | **0.7549** | **0.4630** | **0.5370** | N/A | 0.00% | 100.0% | **0.013 ms** |
| **Condition A: Full SVM Analysis** | 0.7000 | 0.6258 | 0.9510 | 0.7549 | 0.4630 | 0.5370 | 1.0000 | 0.00% | 100.0% | 0.013 ms |
| **Condition B: Fixed Threshold Filtering** | 0.6381 | 0.6000 | 0.7647 | 0.6724 | 0.5185 | 0.4815 | 0.7745 | 24.76% | 75.24% | 0.060 ms |
| **Condition C: Proposed Typed Pipeline** | **0.6762** | 0.6049 | **0.9608** | 0.7424 | 0.4074 | 0.5926 | **1.0000** | **0.48%** | **63.33%** | 0.058 ms |

### 3.2 Obfuscation Resilience Breakdown across Faidhi-Robinson Levels

| Level | Attack Category | Total Positives | Correctly Flagged | Detection Recall |
| :---: | :--- | :---: | :---: | :---: |
| **L1** | Exact code duplication (comments and indentation modified) | 16 | 15 | **93.75%** |
| **L2** | Identifier renaming (systematic student variable renaming) | 18 | 18 | **100.00%** |
| **L3** | Statement reordering & declaration order rearrangement | 17 | 17 | **100.00%** |
| **L4** | Control flow transformation (`for` to `while`, boolean logic) | 18 | 18 | **100.00%** |
| **L5** | Method abstraction & procedural restructuring | 16 | 16 | **100.00%** |
| **L6** | Compound multi-transform obfuscation attacks | 18 | 14 | **77.78%** |

---

## 4. Workload and Compute Reduction Summary

$$\text{Workload Reduction Factor} = 1.0 - \text{Expensive Call Rate}$$

- **Natural Language Text:** $1.0 - 0.7003 = \mathbf{29.97\% \text{ reduction}}$ in deep contextual neural forward passes.
- **Java Source Code:** $1.0 - 0.6333 = \mathbf{36.67\% \text{ reduction}}$ in multi-dimensional feature extractions and SVM inference calls.
- **Direct Triage Rate:** The pipeline safely flagged 103 text pairs and 76 Java code pairs directly for human review, completely bypassing unnecessary heavy neural processing.

