# Progressive Abstracts — Model 1 → Model 5

**Important integrity note:** The published abstracts of the three research papers (Amirzhanov et al. 2025, Sajid et al. 2025, Singh & Gupta 2026) are **not available in this repository**. The repository contains only re-implementations of the three papers' methodologies (see `reports/paper_wise_comparison/implement_three_papers.py` and `reports/paper_wise_comparison/four_new_contributions.py`). Therefore, no published abstract is reproduced verbatim below. All abstracts for Models 1–5 are **proposed abstracts written for this analysis** based on the actual source code implementations and experimental results. None of them claim to be the published abstract of any research paper.

---

## Model 1 — Original Research-Paper Baseline (Sajid et al. 2025)

**Newly introduced:** None — this is the baseline configuration.

**Abstract (proposed):**

We present a multi-feature text plagiarism detection model based on Sajid et al. (2025), PLoS ONE. The model extracts eight similarity features from sentence pairs: word Jaccard, character Jaccard, overlap count, length-difference ratio, shorter-overlap ratio, exact match, TF-IDF cosine similarity, and LSA (latent semantic analysis) cosine similarity. These features are fed into a Logistic Regression classifier with balanced class weights and a validation-optimized decision threshold. The model is evaluated on two datasets: a repository text plagiarism benchmark (Dataset 1) and the Quora Question Pairs dataset (Dataset 2). On Dataset 1, the model achieves MCC = 0.3573 and accuracy = 0.7149. On Dataset 2, the model achieves MCC = 0.1990 and accuracy = 0.6009. The baseline relies on surface-level lexical overlap and latent semantic similarity but lacks edit-distance, character-level granularity beyond trigram Jaccard, question-structure, and phonetic features, limiting its ability to detect paraphrases that preserve meaning while altering surface form.

---

## Model 2 — Baseline + C1 (Word-Overlap Count Features)

**Newly introduced:** C1 — Word-overlap COUNT features [jw, oc, ldr, so, bigram_overlap, exact_match, oc_norm, bigram_containment] — 8 dimensions.

**Abstract (proposed):**

We extend the Sajid et al. (2025) baseline with C1, a set of word-overlap count features that augment the baseline's lexical feature set with bigram overlap, normalized overlap count, and bigram containment. C1 captures surface-level sharing at the bigram level and normalizes overlap counts by sequence length, providing additional discrimination for near-duplicate pairs. On Dataset 1, Model 2 achieves MCC = 0.3813 (ΔMCC = +0.0241 vs. Model 1) and accuracy = 0.7218. On Dataset 2, Model 2 achieves MCC = 0.3476 (ΔMCC = +0.1486 vs. Model 1) and accuracy = 0.6804. C1 provides the largest single-contribution gain on Dataset 2, demonstrating that normalized overlap counts and bigram containment are particularly effective for question-pair paraphrase detection. However, C1 features partially overlap with the baseline's existing word Jaccard and overlap count features, limiting incremental gains on Dataset 1.

---

## Model 3 — Baseline + C1 + C2 (Char N-Gram Features)

**Newly introduced:** C2 — Character n-gram JACCARD + containment [char2_jac, char3_jac, char4_jac, char5_jac, char3_containment] — 5 dimensions.

**Abstract (proposed):**

We extend Model 2 with C2, a set of character-level n-gram Jaccard similarity features spanning n-gram sizes 2 through 5, plus character-3 containment. While the baseline includes only character-trigram Jaccard, C2 extends coverage to bigram, 4-gram, and 5-gram character overlap, capturing orthographic similarity at multiple granularities. On Dataset 1, Model 3 achieves MCC = 0.3993 (ΔMCC = +0.0180 vs. Model 2) and accuracy = 0.7392. On Dataset 2, Model 3 achieves MCC = 0.3516 (ΔMCC = +0.0041 vs. Model 2) and accuracy = 0.6824. C2 provides modest incremental gains on both datasets, with character n-grams complementing word-level overlap by detecting typo-level and character-transposition paraphrases that word-based features miss.

---

## Model 4 — Baseline + C1 + C2 + C3 (Edit Distance Features)

**Newly introduced:** C3 — Edit DISTANCE features [word-level Levenshtein, word-level Jaro-Winkler, character-level Levenshtein, character-level Jaro-Winkler] — 4 dimensions.

**Abstract (proposed):**

We extend Model 3 with C3, a family of edit-distance features measuring string similarity at both word and character levels using Levenshtein distance and Jaro-Winkler similarity. C3 is designed to detect paraphrases involving insertions, deletions, substitutions, and transpositions that character n-gram Jaccard may underweight. On Dataset 1, Model 4 achieves MCC = 0.4035 (ΔMCC = +0.0042 vs. Model 3) and accuracy = 0.7387. On Dataset 2, Model 4 achieves MCC = 0.3844 (ΔMCC = +0.0328 vs. Model 3) and accuracy = 0.6976. C3 provides its largest gain on Dataset 2, where character-level edit distance helps distinguish near-paraphrases from unrelated question pairs. On Dataset 1, the incremental gain is modest, suggesting edit-distance features partially overlap with information already captured by C1 and C2.

---

## Model 5 — Baseline + C1 + C2 + C3 + C4 (Complete Proposed Model)

**Newly introduced:** C4 — LSA TOPIC VECTORS (50 dims) + QUESTION STRUCTURE (5 dims) + SOUNDEX (3 dims) — 58 dimensions total.

**Abstract (proposed):**

We present the complete proposed model, extending Model 4 with C4, a high-dimensional feature block combining latent topic semantics (LSA topic vectors extracted separately for each sentence), question-structure features (first/last word match, wh-question and auxiliary-verb question patterns, wh-word overlap), and phonetic similarity (Soundex Jaccard, overlap count, containment). C4 captures discourse-level and phonetic signal completely absent from word-overlap, character n-gram, and edit-distance features. On Dataset 1, Model 5 achieves MCC = 0.4705 (ΔMCC = +0.0670 vs. Model 4) and accuracy = 0.7641. On Dataset 2, Model 5 achieves MCC = 0.3958 (ΔMCC = +0.0115 vs. Model 4) and accuracy = 0.7046. C4 provides its largest gain on Dataset 1, where latent topic semantics and question-structure features meaningfully improve detection. On Dataset 2, the incremental gain is smaller, consistent with Dataset 2 being dominated by word-overlap signal already captured by C1 and C3.

---

## Abstract Comparison Summary

| Model | Configuration | New Contribution | D1 MCC | D2 MCC |
|-------|--------------|------------------|--------|--------|
| Model 1 | Sajid et al. (2025) baseline | None | 0.3573 | 0.1990 |
| Model 2 | Baseline + C1 | Word-overlap counts | 0.3813 | 0.3476 |
| Model 3 | Baseline + C1 + C2 | Char n-gram Jaccard | 0.3993 | 0.3516 |
| Model 4 | Baseline + C1 + C2 + C3 | Edit distance | 0.4035 | 0.3844 |
| Model 5 | Baseline + C1 + C2 + C3 + C4 | LSA topics + QS + Soundex | 0.4705 | 0.3958 |

**Note:** All results are experimentally measured on the repository's actual datasets using identical train/test splits, preprocessing, Logistic Regression with class_weight="balanced", and MCC-optimized thresholds. D2 results use a 5,000-pair training subset (from 49,401 available) for computational tractability; the test set uses all 8,000 pairs.