# Progressive Model Development Report

**Branch:** `arena/50adecf3-ml` (from main @ `85b162ec`)
**Date:** 2026-11-12
**Experiment:** 5 progressive configurations × 2 datasets
**Status:** All experiments completed

---

## 1. Overview

This report documents the progressive model development study: five configurations
implemented and evaluated on two datasets under a common experimental protocol.

**Progressive configurations:**
- Model 1 — Sajid 2025 baseline (original research paper)
- Model 2 — Baseline + C1 (Word-Overlap Count Features)
- Model 3 — Baseline + C1 + C2 (Character N-Gram Features)
- Model 4 — Baseline + C1 + C2 + C3 (Edit Distance Features)
- Model 5 — Baseline + C1 + C2 + C3 + C4 (Complete proposed model)

**Datasets:**
- Dataset 1: Repository text plagiarism benchmark (9,824 test pairs)
- Dataset 2: Quora Question Pairs (8,000 test pairs, 5,000-pair training subset)

**Research integrity:** All results computed from actual experiments. No fabricated
results. NOT RUN reported where applicable (none — all 10 configurations executed).
Negative and small-impact results preserved. Identical datasets, train/test splits,
preprocessing, classifier, and evaluation protocol across all 5 models.

---

## 2. Experimental Protocol

### 2.1 Datasets

| Dataset | Description | Train Pairs | Test Pairs |
|---------|-------------|-------------|------------|
| D1 | Repository text plagiarism benchmark | n/a (val/test split) | 9,824 |
| D2 | Quora Question Pairs | 5,000 (subset of 49,401) | 8,000 |

### 2.2 Preprocessing (identical for all models)

```
clean_text():
  1. Lowercase the text
  2. Remove non-alphanumeric characters (except spaces)
  3. Collapse multiple whitespace into single space
  4. Trim leading/trailing whitespace
```

### 2.3 Classifier (identical for all models)

- Algorithm: LogisticRegression
- class_weight: "balanced"
- max_iter: 2000
- No regularization tuning (same C across all models)
- Decision threshold: optimized for MCC on the validation/train split

### 2.4 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC) — primary metric
- ROC-AUC
- PR-AUC

---

## 3. Feature Blocks

Feature block definitions verified against
`/home/user/ml/reports/paper_wise_comparison/proposed_typed_solution.py`.

### Baseline Features (8 dims) — Sajid 2025

| Feature | Description |
|---------|-------------|
| jaccard_word | Word-set Jaccard similarity |
| jaccard_char | Character-trigram Jaccard similarity |
| overlap_count | Raw word overlap count |
| len_diff_ratio | Normalized length difference |
| shorter_overlap | Overlap / min length |
| exact_match | Exact string equality (0/1) |
| tfidf_cos | TF-IDF cosine similarity |
| lsa_cos | LSA (TruncatedSVD) cosine similarity |

### C1 — Word-Overlap Count Features (8 dims)

| Feature | Description |
|---------|-------------|
| jw | Word Jaccard similarity |
| oc | Overlap count |
| ldr | Length-difference ratio |
| so | Shorter-overlap ratio |
| bigram_overlap | Bigram overlap count |
| exact_match | Exact string equality (0/1) |
| oc_norm | Normalized overlap count |
| bigram_containment | Bigram containment score |

### C2 — Character N-Gram Features (5 dims)

| Feature | Description |
|---------|-------------|
| char2_jac | Bigram character Jaccard similarity |
| char3_jac | Trigram character Jaccard similarity |
| char4_jac | 4-gram character Jaccard similarity |
| char5_jac | 5-gram character Jaccard similarity |
| char3_cont | Character-trigram containment score |

### C3 — Edit Distance Features (4 dims)

| Feature | Description |
|---------|-------------|
| w_levenshtein | Word-level Levenshtein similarity |
| w_jaro_winkler | Word-level Jaro-Winkler similarity |
| c_levenshtein | Character-level Levenshtein similarity |
| c_jaro_winkler | Character-level Jaro-Winkler similarity |

### C4 — Semantic, Structural & Phonetic Features (58 dims)

| Sub-block | Count | Features |
|-----------|-------|----------|
| LSA topic vectors | 50 | 25 dims per sentence via TruncatedSVD |
| Question structure | 5 | first-word match, last-word match, both wh-questions, both aux-verb questions, wh-word overlap |
| Soundex | 3 | Soundex Jaccard, Soundex overlap count, Soundex containment |

---

## 4. Model Configurations

| Model | Configuration | Total Features | Feature Blocks |
|-------|--------------|----------------|----------------|
| Model 1 | Baseline (Sajid 2025) | 8 | Baseline |
| Model 2 | Baseline + C1 | 16 | Baseline + C1 |
| Model 3 | Baseline + C1 + C2 | 21 | Baseline + C1 + C2 |
| Model 4 | Baseline + C1 + C2 + C3 | 25 | Baseline + C1 + C2 + C3 |
| Model 5 | Baseline + C1 + C2 + C3 + C4 | 83 | Baseline + C1 + C2 + C3 + C4 |

---

## 5. Quantitative Results

Full results in:
`/home/user/ml/reports/progressive_model_development/PROGRESSIVE_QUANTITATIVE_COMPARISON.csv`

### 5.1 Dataset 1 — Repository Text Plagiarism Benchmark (9,824 test pairs)

| Model | Configuration | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|-------|--------------|----------|-----------|--------|----|-----|---------|--------|
| Model 1 | Baseline | 0.7149 | 0.5903 | 0.5502 | 0.5695 | 0.3573 | 0.7375 | 0.5939 |
| Model 2 | + C1 | 0.7218 | 0.5951 | 0.5900 | 0.5925 | 0.3813 | 0.7512 | 0.6431 |
| Model 3 | + C1 + C2 | 0.7392 | 0.6459 | 0.5297 | 0.5821 | 0.3993 | 0.7608 | 0.6601 |
| Model 4 | + C1 + C2 + C3 | 0.7387 | 0.6368 | 0.5534 | 0.5922 | 0.4035 | 0.7619 | 0.6644 |
| Model 5 | + C1 + C2 + C3 + C4 | 0.7641 | 0.6644 | 0.6306 | 0.6471 | 0.4705 | 0.8007 | 0.7109 |

**D1 MCC trend (Model 1 → Model 5):** 0.3573 → 0.3813 → 0.3993 → 0.4035 → 0.4705
(Mono-tonically increasing)

**Largest single jump:** Model 4 → Model 5 (+0.0670 MCC, from C4)

### 5.2 Dataset 2 — Quora Question Pairs (8,000 test pairs, 5K training subset)

| Model | Configuration | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|-------|--------------|----------|-----------|--------|----|-----|---------|--------|
| Model 1 | Baseline | 0.6009 | 0.5445 | 0.5936 | 0.5680 | 0.1990 | 0.6367 | 0.5760 |
| Model 2 | + C1 | 0.6804 | 0.6503 | 0.5990 | 0.6236 | 0.3476 | 0.7273 | 0.6737 |
| Model 3 | + C1 + C2 | 0.6824 | 0.6528 | 0.6010 | 0.6258 | 0.3516 | 0.7272 | 0.6750 |
| Model 4 | + C1 + C2 + C3 | 0.6976 | 0.7404 | 0.4864 | 0.5871 | 0.3844 | 0.7552 | 0.7237 |
| Model 5 | + C1 + C2 + C3 + C4 | 0.7046 | 0.6897 | 0.6029 | 0.6434 | 0.3958 | 0.7585 | 0.7263 |

**D2 MCC trend (Model 1 → Model 5):** 0.1990 → 0.3476 → 0.3516 → 0.3844 → 0.3958
(Mono-tonically increasing)

**Largest single jump:** Model 1 → Model 2 (+0.1486 MCC, from C1)

**Important caveat:** D2 results use a 5,000-pair training subset for computational
tractability. The test set uses all 8,000 pairs.

---

## 6. Contribution-Wise Incremental Impact

| Contribution | D1 Incremental MCC | D2 Incremental MCC | Meets ≥0.05 (D1)? | Meets ≥0.05 (D2)? |
|--------------|-------------------|-------------------|-------------------|-------------------|
| C1 (Word-Overlap Count) | +0.0241 | +0.1486 | No | Yes |
| C2 (Char N-Gram) | +0.0180 | +0.0041 | No | No |
| C3 (Edit Distance) | +0.0042 | +0.0328 | No | No |
| C4 (Semantic/Structural/Phonetic) | +0.0670 | +0.0115 | Yes | No |

**Threshold (≥0.05 absolute MCC per dataset per contributor):**
- Met: C1 on D2 (+0.1486), C4 on D1 (+0.0670)
- Not met: C1 on D1 (+0.0241), C2 on both datasets (+0.0180, +0.0041),
  C3 on both datasets (+0.0042, +0.0328), C4 on D2 (+0.0115)

---

## 7. Key Findings

1. **Model 5 (complete proposed model) outperforms the Sajid 2025 baseline on both datasets.**
   - D1: MCC 0.3573 → 0.4705 (+31.7% relative)
   - D2: MCC 0.1990 → 0.3958 (+98.9% relative)

2. **Contribution impact is dataset-dependent.** No single contribution is
   uniformly beneficial across both datasets:
   - C1 dominates on D2 (question pairs) but is moderate on D1.
   - C4 dominates on D1 (text benchmark) but is small on D2.
   - C2 and C3 have modest impact on both datasets.

3. **C4 delivers the largest single gain on D1** (+0.0670) through LSA topic
   vectors, question-structure features, and Soundex phonetic similarity.

4. **C1 delivers the largest single gain on D2** (+0.1486) through normalized
   overlap and bigram features — highly effective for question paraphrases.

5. **C2 and C3 add limited incremental value** beyond C1 and baseline features,
   suggesting partial redundancy with existing word-overlap signals.

6. **MCC improves monotonically** across all 5 models on both datasets — each
   added contribution provides non-negative incremental value.

---

## 8. Research Integrity

- All 10 model results (5 configurations × 2 datasets) computed from actual
  experiments — no fabricated or synthetic results.
- All results reported exactly as computed, including small and negative impacts.
- No experiments omitted; all configurations executed and reported.
- Identical datasets, train/test splits, preprocessing, classifier, and
  MCC-optimized threshold used across all 5 models.
- All abstracts labeled as "proposed, not published" (see PROGRESSIVE_ABSTRACTS.md).
- Feature block definitions verified against
  `reports/paper_wise_comparison/proposed_typed_solution.py`.

---

## 9. Deliverables

| File | Path | Description |
|------|------|-------------|
| PROGRESSIVE_MODEL_DEVELOPMENT_REPORT.md | reports/progressive_model_development/ | This full report |
| PROGRESSIVE_ABSTRACTS.md | reports/progressive_model_development/ | 5 abstracts (all proposed, not published) |
| PROGRESSIVE_ARCHITECTURES.md | reports/progressive_model_development/ | Architecture diagrams for Models 1–5 |
| PROGRESSIVE_QUANTITATIVE_COMPARISON.csv | reports/progressive_model_development/ | 10 results with all metrics |
| PROGRESSIVE_QUALITATIVE_ANALYSIS.md | reports/progressive_model_development/ | Qualitative analysis of framework evolution |
| CONTRIBUTION_WISE_ANALYSIS.md | reports/progressive_model_development/ | Per-contribution impact analysis |
| FINAL_RESEARCH_SYNTHESIS.md | reports/progressive_model_development/ | Complete research synthesis |

---

## 10. Conclusion

The progressive model development study demonstrates that the complete proposed
model (Model 5, with all four contributions C1–C4) outperforms the Sajid 2025
baseline on both datasets under a common experimental protocol. The improvement
is most pronounced on Dataset 2 (Quora question pairs, +98.9% relative MCC
improvement), driven primarily by C1's word-overlap count features. On Dataset 1
(text plagiarism benchmark), C4's semantic, structural, and phonetic features
deliver the largest single contribution gain (+0.0670 MCC).

Two of eight contribution-dataset cells meet the ≥0.05 MCC threshold (C1 on D2,
C4 on D1). The remaining contributions show smaller but real incremental value.
All results are computed from actual experiments with no fabrication, and
negative/small-impact results are preserved as required by research integrity
standards.