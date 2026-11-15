# Progressive Qualitative Analysis — Model 1 → Model 5

This analysis examines how the framework evolves qualitatively as each
contribution is added, based on the actual experimental results from
`reports/progressive_model_development/PROGRESSIVE_QUANTITATIVE_COMPARISON.csv`.

---

## Dataset 1 — Repository Text Plagiarism Benchmark (9,824 test pairs)

| Model | Configuration | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|-------|--------------|----------|-----------|--------|----|-----|---------|--------|
| Model 1 | Baseline (Sajid 2025) | 0.7149 | 0.5903 | 0.5502 | 0.5695 | 0.3573 | 0.7375 | 0.5939 |
| Model 2 | + C1 | 0.7218 | 0.5951 | 0.5900 | 0.5925 | 0.3813 | 0.7512 | 0.6431 |
| Model 3 | + C1 + C2 | 0.7392 | 0.6459 | 0.5297 | 0.5821 | 0.3993 | 0.7608 | 0.6601 |
| Model 4 | + C1 + C2 + C3 | 0.7387 | 0.6368 | 0.5534 | 0.5922 | 0.4035 | 0.7619 | 0.6644 |
| Model 5 | + C1 + C2 + C3 + C4 | 0.7641 | 0.6644 | 0.6306 | 0.6471 | 0.4705 | 0.8007 | 0.7109 |

### Qualitative observations — Dataset 1:

**Model 1 → Model 2 (adding C1):** Recall improves substantially (0.5502 → 0.5900, +0.0398) while precision stays nearly flat (0.5903 → 0.5951). This indicates C1's bigram overlap and normalized overlap features help identify more true plagiarized pairs without adding many false positives. MCC improves by +0.0241.

**Model 2 → Model 3 (adding C2):** Precision jumps significantly (0.5951 → 0.6459, +0.0508) while recall drops (0.5900 → 0.5297, −0.0603). Character n-gram features are more selective — they correctly filter out some false positives that C1 caught, but miss some true positives. This trade-off reflects character n-grams' sensitivity to exact character sequences, which can be brittle under heavy paraphrasing.

**Model 3 → Model 4 (adding C3):** Recall recovers (0.5297 → 0.5534, +0.0237) while precision stays strong (0.6459 → 0.6368). Edit distance features capture near-duplicate pairs with minor insertions/deletions that character n-gram Jaccard underweights. The modest MCC gain (+0.0042) suggests partial overlap with C1/C2 information.

**Model 4 → Model 5 (adding C4):** The largest single improvement — MCC jumps +0.0670, accuracy +0.0254, recall +0.0772. C4's LSA topic vectors, question-structure features, and Soundex phonetic similarity capture paraphrases that are semantically related but orthographically distant — pairs where surface overlap is low but topical and phonetic similarity is high.

---

## Dataset 2 — Quora Question Pairs (8,000 test pairs, 5K training subset)

| Model | Configuration | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|-------|--------------|----------|-----------|--------|----|-----|---------|--------|
| Model 1 | Baseline (Sajid 2025) | 0.6009 | 0.5445 | 0.5936 | 0.5680 | 0.1990 | 0.6367 | 0.5760 |
| Model 2 | + C1 | 0.6804 | 0.6503 | 0.5990 | 0.6236 | 0.3476 | 0.7273 | 0.6737 |
| Model 3 | + C1 + C2 | 0.6824 | 0.6528 | 0.6010 | 0.6258 | 0.3516 | 0.7272 | 0.6750 |
| Model 4 | + C1 + C2 + C3 | 0.6976 | 0.7404 | 0.4864 | 0.5871 | 0.3844 | 0.7552 | 0.7237 |
| Model 5 | + C1 + C2 + C3 + C4 | 0.7046 | 0.6897 | 0.6029 | 0.6434 | 0.3958 | 0.7585 | 0.7263 |

**Important caveat:** D2 results use a 5,000-pair training subset (from 49,401 available) for computational tractability. The test set uses all 8,000 pairs.

### Qualitative observations — Dataset 2:

**Model 1 → Model 2 (adding C1):** The most dramatic improvement on D2 — MCC jumps +0.1486 (from 0.1990 to 0.3476). C1's word-overlap count features (bigram overlap, normalized overlap, bigram containment) are highly effective for Quora question pairs, which often differ mainly in word order and minor lexical substitutions. Precision improves significantly (0.5445 → 0.6503).

**Model 2 → Model 3 (adding C2):** Minimal change — MCC +0.0041. Character n-grams add little beyond word-overlap counts on D2, suggesting Quora paraphrases are primarily distinguished by word-level rather than character-level overlap.

**Model 3 → Model 4 (adding C3):** MCC improves +0.0328, but with a notable trade-off: precision increases (0.6528 → 0.7404, +0.0876) while recall drops sharply (0.6010 → 0.4864, −0.1146). Edit distance features are highly selective — they correctly identify true negatives but miss many true positives, indicating that many Quora paraphrase pairs have significant word-level divergence that edit distance cannot bridge.

**Model 4 → Model 5 (adding C4):** Recall recovers substantially (0.4864 → 0.6029, +0.1165) while precision stays strong (0.7404 → 0.6897). C4's question-structure features (wh-question patterns, first/last word match) and Soundex phonetic similarity help identify paraphrased question pairs that share intent but differ in surface form. MCC improves +0.0115.

---

## Cross-Dataset Comparison

| Aspect | Dataset 1 | Dataset 2 |
|--------|-----------|-----------|
| Best model | Model 5 (MCC 0.4705) | Model 5 (MCC 0.3958) |
| Largest single gain | Model 4→5 (+0.0670 MCC) | Model 1→2 (+0.1486 MCC) |
| C1 impact | Moderate (+0.0241) | Very large (+0.1486) |
| C2 impact | Moderate (+0.0180) | Negligible (+0.0041) |
| C3 impact | Negligible (+0.0042) | Moderate (+0.0328) |
| C4 impact | Large (+0.0670) | Small (+0.0115) |

**Dataset-dependent contribution patterns:**
- **C1** is most impactful on D2 (question pairs), where normalized overlap and bigram features capture paraphrase patterns.
- **C2** has moderate impact on D1, negligible on D2 — character n-grams help on the more varied text benchmark but not on question pairs.
- **C3** has negligible impact on D1, moderate on D2 — edit distance helps distinguish question paraphrases but adds little to the text benchmark where C1/C2 already capture overlap.
- **C4** has large impact on D1, small on D2 — latent topic semantics and question structure help on the diverse text benchmark; on D2, word-overlap already captures most signal.

---

## Contribution Complementarity Analysis

- **Complementary:** C3 and C4 show the strongest complementarity — C3 catches near-duplicates via edit distance, C4 catches semantically related but orthographically distant pairs via topics and phonetics. The combined Model 5 outperforms any individual configuration.
- **Partially redundant:** C1 partially overlaps with baseline features (word Jaccard, overlap count), but adds bigram variants and normalization that provide incremental value.
- **Dataset-dependent:** No contribution is uniformly beneficial across both datasets at all stages. C2 adds little on D2, C3 adds little on D1, C4 adds little on D2.