# Progressive Architecture Comparison — Model 1 → Model 5

All five architectures share the same input datasets, preprocessing, classifier
(LogisticRegression with class_weight="balanced"), and MCC-optimized threshold
procedure. Each successive model adds exactly one contribution's feature block.

---

## Model 1 — Original Research-Paper Baseline (Sajid et al. 2025)

**Newly added component:** None (baseline only)

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Dataset 1 (val/test split) or Dataset 2 (train/test split)   │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: clean_text() — lowercase, remove non-alphanumeric,   │
│                collapse whitespace, trim                            │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BASELINE FEATURE EXTRACTION (8 dims):                               │
│  • jaccard_word        — word-set Jaccard                           │
│  • jaccard_char        — char-trigram Jaccard                       │
│  • overlap_count       — raw word overlap count                     │
│  • len_diff_ratio      — normalized length difference               │
│  • shorter_overlap     — overlap / min length                       │
│  • exact_match         — exact string equality (0/1)                │
│  • tfidf_cos           — TF-IDF cosine similarity                   │
│  • lsa_cos             — LSA (TruncatedSVD) cosine similarity       │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CLASSIFIER: LogisticRegression (balanced weights, max_iter=2000)    │
│ DECISION:   threshold optimized for MCC on validation/train split   │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: binary prediction (plagiarized / not plagiarized)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model 2 — Baseline + C1 (Word-Overlap Count Features)

**Newly added component:** C1 feature block (highlighted)

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Dataset 1 (val/test split) or Dataset 2 (train/test split)   │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: clean_text()                                         │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BASELINE FEATURES (8 dims): jaccard_word, jaccard_char, overlap_    │
│ count, len_diff_ratio, shorter_overlap, exact_match, tfidf_cos,     │
│ lsa_cos                                                               │
│                                                                       │
│ ████ C1 FEATURES (8 dims) ████                                       │
│  • jw (word Jaccard)         • oc (overlap count)                   │
│  • ldr (length-diff ratio)   • so (shorter-overlap ratio)           │
│  • bigram_overlap            • exact_match                          │
│  • oc_norm (norm. overlap)   • bigram_containment                   │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FEATURE CONCATENATION: 16 dims (baseline 8 + C1 8)                  │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CLASSIFIER: LogisticRegression + MCC-optimized threshold            │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: binary prediction                                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Note:** C1 partially overlaps with baseline features (jw ≈ jaccard_word, oc ≈ overlap_count, ldr ≈ len_diff_ratio, so ≈ shorter_overlap, em ≈ exact_match) but adds bigram_overlap, oc_norm, and bigram_containment.

---

## Model 3 — Baseline + C1 + C2 (Char N-Gram Features)

**Newly added component:** C2 feature block (highlighted)

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Dataset 1 or Dataset 2                                       │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: clean_text()                                         │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BASELINE FEATURES (8 dims)                                            │
│ ████ C1 FEATURES (8 dims) ████                                        │
│                                                                       │
│ ████ C2 FEATURES (5 dims) ████  ← NEW                                 │
│  • char2_jac    — bigram character Jaccard                          │
│  • char3_jac    — trigram character Jaccard                         │
│  • char4_jac    — 4-gram character Jaccard                          │
│  • char5_jac    — 5-gram character Jaccard                          │
│  • char3_cont   — char-trigram containment                          │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FEATURE CONCATENATION: 21 dims (8 + 8 + 5)                          │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CLASSIFIER: LogisticRegression + MCC-optimized threshold            │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: binary prediction                                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Note:** C2 extends the baseline's single char-trigram Jaccard feature to 5 dimensions spanning bigram through 5-gram character overlap plus containment.

---

## Model 4 — Baseline + C1 + C2 + C3 (Edit Distance Features)

**Newly added component:** C3 feature block (highlighted)

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Dataset 1 or Dataset 2                                       │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: clean_text()                                         │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BASELINE FEATURES (8 dims)                                            │
│ ████ C1 FEATURES (8 dims) ████                                        │
│ ████ C2 FEATURES (5 dims) ████                                        │
│                                                                       │
│ ████ C3 FEATURES (4 dims) ████  ← NEW                                 │
│  • w_levenshtein   — word-level Levenshtein similarity              │
│  • w_jaro_winkler  — word-level Jaro-Winkler similarity             │
│  • c_levenshtein   — character-level Levenshtein similarity         │
│  • c_jaro_winkler  — character-level Jaro-Winkler similarity        │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FEATURE CONCATENATION: 25 dims (8 + 8 + 5 + 4)                      │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CLASSIFIER: LogisticRegression + MCC-optimized threshold            │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: binary prediction                                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Note:** C3 introduces a completely new feature family (edit distance) with no overlap with baseline, C1, or C2 features.

---

## Model 5 — Baseline + C1 + C2 + C3 + C4 (Complete Proposed Model)

**Newly added component:** C4 feature block (highlighted)

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: Dataset 1 or Dataset 2                                       │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: clean_text()                                         │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BASELINE FEATURES (8 dims)                                            │
│ ████ C1 FEATURES (8 dims) ████                                        │
│ ████ C2 FEATURES (5 dims) ████                                        │
│ ████ C3 FEATURES (4 dims) ████                                        │
│                                                                       │
│ ████ C4 FEATURES (58 dims) ████  ← NEW                                │
│  • LSA topic vectors — 50 dims (25 per sentence via TruncatedSVD)   │
│  • Question structure — 5 dims:                                     │
│      - first word match                                             │
│      - last word match                                              │
│      - both wh-questions                                            │
│      - both auxiliary-verb questions                                │
│      - wh-word overlap                                              │
│  • Soundex — 3 dims:                                                │
│      - Soundex Jaccard                                              │
│      - Soundex overlap count                                        │
│      - Soundex containment                                          │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FEATURE CONCATENATION: 83 dims (8 + 8 + 5 + 4 + 58)                 │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CLASSIFIER: LogisticRegression + MCC-optimized threshold            │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: binary prediction                                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Note:** C4 introduces latent topic semantics (LSA), discourse structure (question patterns), and phonetic similarity (Soundex) — feature types completely absent from Models 1–4.

---

## Architecture Evolution Summary

| Model | Total Features | New Feature Dims | Cumulative Dims |
|-------|---------------|------------------|-----------------|
| Model 1 | 8 | 0 (baseline) | 8 |
| Model 2 | 16 | +8 (C1) | 16 |
| Model 3 | 21 | +5 (C2) | 21 |
| Model 4 | 25 | +4 (C3) | 25 |
| Model 5 | 83 | +58 (C4) | 83 |

All models use identical: datasets, train/test splits, preprocessing, LogisticRegression classifier, class_weight="balanced", MCC-optimized threshold, and evaluation metrics.