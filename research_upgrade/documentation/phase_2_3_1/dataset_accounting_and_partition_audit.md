# Forensic Dataset Accounting and Data Partition Audit (Version 3.2)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.2 (Phase 2.3.1 Deliverable)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited Forensic Dataset Reconciliation Ledger  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Executive Summary

This document provides a comprehensive, ground-truth accounting of all text and source-code datasets available in the repository. Every numerical count in this document has been verified against the physical disk files in:
- `datasets/processed/text_plagiarism_benchmark.csv`
- `datasets/processed/code_plagiarism_benchmark.csv`
- `datasets/raw/text_mit_plagiarism/` (`snli_val.parquet`, `snli_test.parquet`)
- `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` (467 raw `.java` files)

---

## 2. Text Modality Forensic Accounting (SNLI / Pereira Benchmark)

### 2.1 File Provenance & Raw Corpus Dimensions
The text benchmark is derived from the Stanford Natural Language Inference (SNLI) corpus, adapted for academic text integrity and paraphrasing detection:
- `snli_val.parquet` (Development Pool): **9,842 pairs**, containing **3,304 unique reference premises**.
- `snli_test.parquet` (Evaluation Pool): **9,824 pairs**, containing **3,261 unique reference premises**.
- **Total Available Text Pairs in Processed CSV:** **19,666 pairs**.

### 2.2 Cross-File Premise Intersection Audit
- A pairwise string comparison between the 3,304 development premises and 3,261 evaluation premises revealed **exactly one shared premise sentence**:
  - *Premise text:* `"A woman in a green jacket is walking down a crowded street."`
  - *Isolation Rate:* $1.0 - \frac{1}{3304} = 99.9697\% \approx 99.98\%$.
- **Handling Protocol:** Rather than asserting "100% zero premise overlap" without disclosure, the data loader explicitly detects and removes this single shared premise sentence during partitioning, guaranteeing complete group isolation across all 5 splits.

### 2.3 Full Text Partition Reconciliation (The 9,166 Reserve Pairs Explained)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               COMPLETE TEXT DATASET PARTITION RECONCILIATION                           │
├────┬─────────────────────────────┬────────────┬─────────────┬──────────────┬──────────────────────────┤
│ #  │ Partition Name              │ Pair Count │ Premise Cnt │ Source Pool  │ Operational Role         │
├────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼──────────────────────────┤
│ 1  │ Base Train (D_train)        │ 4,000      │ 1,340       │ snli_val     │ Vocab extraction & Tier-1│
│ 2  │ Threshold Val (D_val)       │ 1,500      │ 505         │ snli_val     │ Pareto grid search       │
│ 3  │ Prob Calibration (D_cal_pr) │ 1,000      │ 335         │ snli_val     │ Fit Platt / Isotonic     │
│ 4  │ Conformal Cal (D_cal_conf)  │ 1,000      │ 335         │ snli_val     │ Estimate quantile q_hat  │
│ 5  │ Held-Out Test (D_test)      │ 3,000      │ 1,010       │ snli_test    │ Final single-pass test   │
├────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼──────────────────────────┤
│ ΣA │ ACTIVE EXPERIMENTAL SPLITS  │ 10,500     │ 3,525       │ Both Pools   │ Active Pipeline Total    │
├────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼──────────────────────────┤
│ 6  │ Development Reserve Split   │ 2,342      │ 789         │ snli_val     │ Unselected Dev Reserve   │
│ 7  │ Test Reserve Split          │ 6,824      │ 2,251       │ snli_test    │ Unselected Test Reserve  │
├────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼──────────────────────────┤
│ ΣB │ UNSELECTED RESERVE PAIRS    │ 9,166      │ 3,040       │ Both Pools   │ Reserved for Sensitivity │
├────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼──────────────────────────┤
│ ΣT │ TOTAL CORPUS PAIRS          │ 19,666     │ 6,564       │ Both Pools   │ Complete Raw Corpus      │
└────┴─────────────────────────────┴────────────┴─────────────┴──────────────┴──────────────────────────┘
```

#### Status of the Remaining 9,166 Pairs:
- **2,342 pairs** from `snli_val.parquet` remain unselected after forming the $4,000 + 1,500 + 1,000 + 1,000 = 7,500$ development splits.
- **6,824 pairs** from `snli_test.parquet` remain unselected after subsampling the standardized $3,000$-pair held-out test split.
- **Total unselected observations:** $2,342 + 6,824 = 9,166$ pairs.
- **Scientific Role:** These pairs are not lost or missing; they are held in reserve to evaluate sample size sensitivity curves and cross-validation stability.

---

## 3. Source Code Modality Forensic Accounting (IR-Plag Java Benchmark)

### 3.1 Raw Java Source Files on Disk
The extracted IR-Plag corpus in `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` contains **467 Java files** distributed across 7 programming assignment tasks:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 IR-PLAG RAW JAVA SOURCE FILES ON DISK                                  │
├──────────┬─────────────────────────────┬────────────────────────────────┬──────────────────────────────┤
│ Case ID  │ Programming Assignment Task │ Total Java Files on Disk       │ Subdirectory Structure       │
├──────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ case-01  │ Task 1 (Matrix Operations)  │ 56 files                       │ 1 original, 15 non, 40 plag  │
│ case-02  │ Task 2 (String Processing)  │ 70 files                       │ 1 original, 15 non, 54 plag  │
│ case-03  │ Task 3 (Data Structures)    │ 68 files                       │ 1 original, 15 non, 52 plag  │
│ case-04  │ Task 4 (Math & Primes)      │ 70 files                       │ 1 original, 15 non, 54 plag  │
│ case-05  │ Task 5 (Array Permutations) │ 69 files                       │ 1 original, 15 non, 53 plag  │
│ case-06  │ Task 6 (Palindrome Checks)  │ 67 files                       │ 1 original, 15 non, 51 plag  │
│ case-07  │ Task 7 (Sorting Algorithms) │ 67 files                       │ 1 original, 15 non, 51 plag  │
├──────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ TOTAL    │ 7 Independent Java Tasks    │ 467 Raw Java Source Files      │ Standardized Benchmark       │
└──────────┴─────────────────────────────┴────────────────────────────────┴──────────────────────────────┘
```

---

### 3.2 Processed Code Pair Distribution (`code_plagiarism_benchmark.csv`)

The processed dataset contains **733 evaluated pairs** (378 non-plagiarized, 355 plagiarized):

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CODE PAIR DISTRIBUTION ACROSS ALL SEVEN CASES                             │
├──────────┬─────────────────────────────┬────────────┬─────────────┬──────────────┬─────────────────────┤
│ Case ID  │ Assignment Task Description │ Total Pairs│ Non-Plag (0)│ Plagiarized (│ Partition Role      │
├──────────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────────────┤
│ case-01  │ Matrix Operations           │ 94         │ 54          │ 40           │ Training Split      │
│ case-02  │ String Processing           │ 108        │ 54          │ 54           │ Training Split      │
│ case-03  │ Data Structures             │ 106        │ 54          │ 52           │ Training Split      │
│ case-04  │ Math & Primes               │ 108        │ 54          │ 54           │ Training Split      │
├──────────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────────────┤
│ Σ Train  │ Training Tasks Total        │ 416        │ 216         │ 200          │ Base Model Fitting  │
├──────────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────────────┤
│ case-05  │ Array Permutations          │ 107        │ 54          │ 53           │ Val & Exploratory Cal│
├──────────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────────────┤
│ case-06  │ Palindrome Detection        │ 105        │ 54          │ 51           │ Held-Out Test Task 1│
│ case-07  │ Sorting Algorithms          │ 105        │ 54          │ 51           │ Held-Out Test Task 2│
├──────────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────────────┤
│ Σ Test   │ Held-Out Test Tasks Total   │ 210        │ 108         │ 102          │ Disaggregated Test  │
├──────────┼─────────────────────────────┼────────────┼─────────────┼──────────────┼─────────────────────┤
│ TOTAL    │ Complete IR-Plag Corpus     │ 733        │ 378         │ 355          │ Complete Dataset    │
└──────────┴─────────────────────────────┴────────────┴─────────────┴──────────────┴─────────────────────┘
```

---

### 3.3 Deep Forensic Audit of `case-05` (The 15-Group Constraint)

A fine-grained inspection of `case-05` ($N = 107$ pairs) reveals the underlying file structure:
- **1 Original Reference Program:** `original/Main.java`.
- **53 Plagiarized Variants:** All 53 pairs are paired against the exact same original reference program across 6 obfuscation tiers:
  - Level 1 (Format/Comment Alteration): 9 pairs
  - Level 2 (Identifier Renaming): 8 pairs
  - Level 3 (Declaration Restructuring): 9 pairs
  - Level 4 (Control Flow Replacement): 9 pairs
  - Level 5 (Statement Reordering): 9 pairs
  - Level 6 (Compound Obfuscation): 9 pairs
  - Total Plagiarized Pairs = $9 + 8 + 9 + 9 + 9 + 9 = 53$ pairs.
- **15 Non-Plagiarized Files:** `non-plag-01.java` through `non-plag-15.java`.
  - 15 pairs of `original` vs. `non-plag-XX`.
  - 39 pairs of `non-plag-XX` vs. `non-plag-YY`.
  - Total Non-Plagiarized Pairs = $15 + 39 = 54$ pairs.

#### Why This Prevents Formal Conformal Coverage:
Because all 53 plagiarized variants originate from a **single base program**, the true number of independent reference programs is only **15**.
For split conformal prediction at $\epsilon = 0.05$:
$$N_{\text{cal}} \ge \left\lceil \frac{1 - \epsilon}{\epsilon} \right\rceil = 19 \text{ independent groups}$$
With only 15 groups, the order statistic index is $k = \lceil 16 \times 0.95 \rceil = 16 > 15$, rendering exact finite-sample conformal guarantees mathematically unattainable.
Consequently, `case-05` is evaluated via **group-aware 5-fold cross-fitting** (3 reference programs per fold) for exploratory uncertainty routing, and conformal guarantees are formally disclaimed.

---

## 4. Summary of Partition Boundaries and Methodological Safeguards

1. **Text Grouping:** Grouped strictly by `reference_text` (Premise ID). All hypotheses for a premise stay in the same split. The 1 shared cross-file premise is removed.
2. **Code Grouping:** Grouped strictly by task / reference program. No tasks cross partitions.
3. **Disaggregated Testing:** `case-06` ($N=105$) and `case-07` ($N=105$) are evaluated and reported separately.

---
*End of Document: dataset_accounting_and_partition_audit.md*

