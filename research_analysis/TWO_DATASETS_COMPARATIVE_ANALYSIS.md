# Phase 7 — Comparative Analysis of Both Datasets
*Companion CSV: `DATASET_COMPARISON_STATISTICS.csv`; figures: `fig01–fig04`.*

## 7.1 Qualitative comparison

| Dimension | Dataset 1 (text) | Dataset 2 (code) |
|---|---|---|
| Research purpose | Sentence-level semantic equivalence / paraphrase-reuse proxy | Authentic code-plagiarism detection under graded obfuscation |
| Data characteristics | 19.7k short sentence pairs; SNLI-derived | 733 Java program pairs; 7 assignment cases |
| Feature complexity | Shallow: short strings; signal concentrated in lexical/semantic overlap | Rich: tokens, AST shape, identifier system, complexity; structurally heavier |
| Label quality | Crowd consensus re-mapped (conceptual proxy caveat) | Procedural transformation labels (unambiguous by construction) |
| Domain differences | Everyday image-caption sentences | Student introductory Java programs |
| Data quality | 0 missing; 8 purged rows; mild duplication | 378 structural NaNs (by design); 24 duplicate rows (policy-handled) |
| Annotation challenges | Entailment ≁ plagiarism semantics | Independent-work similarity on same tasks |
| Generalization challenges | Single genre; sentence-scale only | 7 cases; Java-only; small corpus |
| Suitability for framework | Ideal for C2/C3 calibration+routing economics | Ideal for C1/C3 resilience + leak-free LOGO demonstration |

## 7.2 Quantitative comparison

| Measure | Dataset 1 (text) | Dataset 2 (code) |
|---|---|---|
| Raw rows | 19,666 | 733 |
| Analysis rows used | 8,500 core (4,000/1,500/3,000) | 720 (Policy A) |
| Positives (eval split) | 34.20% (test) | 47.50% (Policy A) |
| Majority:minority ratio | 1.92 : 1 (mild) | 1.10 : 1 (near-balanced) |
| Missing % | 0.0 | 51.6% of cells in one structural column only (0 elsewhere) |
| Duplicate % | ≈0.01% (2 rows) | 3.3% of rows inside collision groups |
| Median suspicious length | 34 chars / 7 words | 28 lines / 79 whitespace tokens (~678 chars) |
| Length distribution | right-skewed, heavy-tailed (kurt 5.5) | near-symmetric (kurt ≈ 0) |
| Framework F1 (frozen 4.1) | 0.6688 | 0.8467 |
| Framework MCC (frozen) | 0.5188 | 0.6996 |
| Framework F1 (upgrade 5) | 0.7273 | 0.8481 (β=1) |
| Error rate (frozen) | FNR 0.378, FPR 0.124 | FNR 0.120, FPR 0.180 |
| Performance stability | test CI ±0.015 (grouped bootstrap) | case F1 spread 0.49–0.98 (high variance) |

## 7.3 Research interpretation

1. **Which dataset is more challenging and why?** Dataset 1 (text). Its positives require *semantic*
   discrimination on 7-word hypotheses where lexical evidence is deliberately absent; the framework's
   text F1 (≈0.67–0.73) trails its code F1 (≈0.83–0.85) by ~0.12, and its error mass is recall-side —
   the signature of genuinely ambiguous short paraphrases.
2. **Which dataset exhibits greater class imbalance?** Dataset 1 (1.92:1 mild imbalance) vs Dataset 2
   (≈1.1:1 near balance).
3. **Which has more difficult classification characteristics?** Text, on two axes: (i) best
   single-view text feature reaches only MCC 0.42 (semantic view) while code fusion reaches 0.70;
   (ii) the text ROC-AUC (0.864) indicates a a hard minority of paraphrase-only positives.
4. **Distribution differences.** Text is heavy-tailed and asymmetric in length (suspicious ≪ reference);
   code is symmetric with suspicious **longer** than reference (+131 chars) — opposite asymmetries,
   consistent with paraphrase compression in text and transformation accretion in code.
5. **Does the framework generalize across both?** **Within-dataset, yes — cross-modal, yes:** the same
   architecture (multi-view fusion → calibrated routing → heavy verification → dossier) tops the
   baseline field in both modalities under the respective leak-free protocols (text holdout;
   code nested LOGO). This is *cross-domain instantiation* of one framework, not cross-dataset
   transfer of one trained model.
6. **Remaining limitations.** Ecological validity (text is an SNLI proxy); corpus scale and case
   count (code); no overlap in *subjects* across datasets, so "cross-dataset generalization" in the
   strict transfer-learning sense is not claimable (see Phase 20).
