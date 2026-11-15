# Final Research Synthesis — Progressive Model Development

This document synthesizes the complete progressive model development study:
5 configurations (Sajid 2025 baseline → +C1 → +C1+C2 → +C1+C2+C3 → +C1+C2+C3+C4)
evaluated on two datasets under a common experimental protocol.

---

## Executive Summary

| Aspect | Finding |
|--------|---------|
| Best overall model | Model 5 (complete proposed model) on both datasets |
| Best D1 MCC | Model 5: 0.4705 (vs baseline Model 1: 0.3573) |
| Best D2 MCC | Model 5: 0.3958 (vs baseline Model 1: 0.1990) |
| Largest single contribution gain | C1 on D2: +0.1486 MCC |
| Second-largest gain | C4 on D1: +0.0670 MCC |
| Contributions meeting ≥0.05 threshold | C1 on D2, C4 on D1 only |
| Datasets/splits shared | Yes — identical across all 5 models |

---

## Study Design

**Baseline:** Sajid et al. 2025 (PLoS ONE, DOI 10.1371/journal.pone.0319551)
— the original research-paper baseline against which the progressive models
are compared.

**Contributions (C1–C4):** Original contributions defined in
`reports/paper_wise_comparison/proposed_typed_solution.py`:
- C1: Word-Overlap Count Features (8 dims)
- C2: Character N-Gram Features (5 dims)
- C3: Edit Distance Features (4 dims)
- C4: Semantic, Structural & Phonetic Features (58 dims: LSA + question structure + Soundex)

**Protocol (identical across all 5 models):**
- Same train/test splits for all configurations
- Same preprocessing (clean_text: lowercase, strip non-alphanumeric, collapse whitespace)
- Same classifier (LogisticRegression, class_weight="balanced", max_iter=2000)
- Same decision rule (MCC-optimized threshold)
- Same metrics (Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC)

**Datasets:**
- Dataset 1: Repository text plagiarism benchmark — 9,824 test pairs
- Dataset 2: Quora Question Pairs — 8,000 test pairs, 5,000-pair training subset

**Abstracts:** All 5 model abstracts are labeled "proposed, not published"
(see PROGRESSIVE_ABSTRACTS.md).

---

## Quantitative Results

### Dataset 1 (9,824 test pairs)

| Model | Configuration | Features | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|-------|--------------|----------|----------|-----------|--------|----|-----|---------|--------|
| Model 1 | Baseline (Sajid 2025) | 8 | 0.7149 | 0.5903 | 0.5502 | 0.5695 | 0.3573 | 0.7375 | 0.5939 |
| Model 2 | + C1 | 16 | 0.7218 | 0.5951 | 0.5900 | 0.5925 | 0.3813 | 0.7512 | 0.6431 |
| Model 3 | + C1 + C2 | 21 | 0.7392 | 0.6459 | 0.5297 | 0.5821 | 0.3993 | 0.7608 | 0.6601 |
| Model 4 | + C1 + C2 + C3 | 25 | 0.7387 | 0.6368 | 0.5534 | 0.5922 | 0.4035 | 0.7619 | 0.6644 |
| Model 5 | + C1 + C2 + C3 + C4 | 83 | 0.7641 | 0.6644 | 0.6306 | 0.6471 | 0.4705 | 0.8007 | 0.7109 |

**D1 trend:** Monotonic improvement in MCC from Model 1 → Model 5.
Largest jump: Model 4 → Model 5 (+0.0670, from C4's LSA/Soundex/question-structure features).

### Dataset 2 (8,000 test pairs, 5K training subset)

| Model | Configuration | Features | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|-------|--------------|----------|----------|-----------|--------|----|-----|---------|--------|
| Model 1 | Baseline (Sajid 2025) | 8 | 0.6009 | 0.5445 | 0.5936 | 0.5680 | 0.1990 | 0.6367 | 0.5760 |
| Model 2 | + C1 | 16 | 0.6804 | 0.6503 | 0.5990 | 0.6236 | 0.3476 | 0.7273 | 0.6737 |
| Model 3 | + C1 + C2 | 21 | 0.6824 | 0.6528 | 0.6010 | 0.6258 | 0.3516 | 0.7272 | 0.6750 |
| Model 4 | + C1 + C2 + C3 | 25 | 0.6976 | 0.7404 | 0.4864 | 0.5871 | 0.3844 | 0.7552 | 0.7237 |
| Model 5 | + C1 + C2 + C3 + C4 | 83 | 0.7046 | 0.6897 | 0.6029 | 0.6434 | 0.3958 | 0.7585 | 0.7263 |

**D2 trend:** Monotonic improvement in MCC from Model 1 → Model 5.
Largest jump: Model 1 → Model 2 (+0.1486, from C1's word-overlap count features).

---

## Contribution-Wise Impact Summary

| Contribution | D1 Incremental MCC | D1 ≥0.05? | D2 Incremental MCC | D2 ≥0.05? |
|--------------|-------------------|-----------|-------------------|-----------|
| C1 (Word-Overlap Count) | +0.0241 | No | +0.1486 | Yes |
| C2 (Char N-Gram) | +0.0180 | No | +0.0041 | No |
| C3 (Edit Distance) | +0.0042 | No | +0.0328 | No |
| C4 (Semantic/Structural/Phonetic) | +0.0670 | Yes | +0.0115 | No |

**Research-integrity threshold (≥0.05 absolute MCC per dataset per contributor):**
- Met by: C1 on D2 (+0.1486), C4 on D1 (+0.0670)
- Not met by: C1 on D1 (+0.0241), C2 on both datasets (+0.0180, +0.0041),
  C3 on both datasets (+0.0042, +0.0328), C4 on D2 (+0.0115)

All values are computed from actual experiments — no fabricated results.
Negative and small-impact results are preserved without modification.

---

## Key Qualitative Findings

1. **Model 5 (complete proposed model) outperforms the Sajid 2025 baseline on both datasets.**
   - D1: 0.3573 → 0.4705 MCC (+31.7% relative improvement)
   - D2: 0.1990 → 0.3958 MCC (+98.9% relative improvement)

2. **Contribution impact is dataset-dependent.** No single contribution
   is uniformly beneficial across both datasets:
   - C1 dominates on D2 (question pairs) but is moderate on D1.
   - C4 dominates on D1 (text benchmark) but is small on D2.
   - C2 and C3 have modest impact on both datasets.

3. **C4 delivers the largest single gain on D1** (+0.0670) through LSA topic
   vectors, question-structure features, and Soundex phonetic similarity —
   capturing semantically related but orthographically distant pairs.

4. **C1 delivers the largest single gain on D2** (+0.1486) through normalized
   overlap and bigram features — highly effective for question paraphrases
   differing mainly in word order.

5. **C2 and C3 add limited incremental value** beyond C1 and baseline
   features, suggesting partial redundancy with existing word-overlap signals.

---

## Limitations

- D2 results use a 5,000-pair training subset (from 49,401 available pairs)
  for computational tractability. Test set uses all 8,000 pairs.
- Not all contributions meet the ≥0.05 MCC threshold on all datasets.
- C4 on D2 (+0.0115) and C3 on D1 (+0.0042) show small but real gains.
- The baseline's word-overlap features partially overlap with C1, reducing
  C1's incremental impact on D1.

---

## Research Integrity Statement

- All 10 model results (5 configurations × 2 datasets) were computed from
  actual experiments — no fabricated or synthetic results.
- All results are reported exactly as computed, including small and negative
  impacts.
- No experiments were omitted; all configurations were executed and reported.
- Identical datasets, train/test splits, preprocessing, classifier, and
  evaluation protocol were used across all 5 models.
- All abstracts are labeled as "proposed, not published."

---

## Deliverables

| File | Status | Description |
|------|--------|-------------|
| PROGRESSIVE_MODEL_DEVELOPMENT_REPORT.md | Written | Full experimental report with methods, results, analysis |
| PROGRESSIVE_ABSTRACTS.md | Written | 5 abstracts (all proposed, not published) |
| PROGRESSIVE_ARCHITECTURES.md | Written | Architecture diagrams for Models 1–5 |
| PROGRESSIVE_QUANTITATIVE_COMPARISON.csv | Written | 10 results (5 models × 2 datasets) with all metrics |
| PROGRESSIVE_QUALITATIVE_ANALYSIS.md | Written | Qualitative analysis of framework evolution |
| CONTRIBUTION_WISE_ANALYSIS.md | Written | Per-contribution impact analysis with threshold assessment |
| FINAL_RESEARCH_SYNTHESIS.md | Written | This document — complete research synthesis |

All files are located in `/home/user/ml/reports/progressive_model_development/`.