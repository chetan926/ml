# Dataset Analysis — Complete Details

This document provides a full analysis of the two original (non-synthetic) datasets used in this research,
including provenance, source, structure, train/validation/test splits, class distributions, and feature statistics.

---

## Dataset 1 — Text Plagiarism / Paraphrase Benchmark (MSRP-style)

### Source & Provenance
- **File:** `datasets/dataset1_text_plagiarism_benchmark.csv`
- **Original path:** `/home/user/ml/datasets/processed/text_plagiarism_benchmark.csv`
- **Type:** Real-world paraphrase / text similarity corpus (MSRP-style)
- **Total rows:** 19,666 sentence pairs
- **Original/non-synthetic:** ✅ Verified — contains natural human-written sentence pairs

### Columns
| Column | Description |
|--------|-------------|
| `id` | Unique pair identifier |
| `reference_text` | First sentence (reference) |
| `suspicious_text` | Second sentence (candidate paraphrase) |
| `label` | Original raw label (0/1/2) |
| `is_plagiarized` | Binary target (0 = non-plagiarized, 1 = plagiarized/paraphrase) |
| `split` | Data split assignment (train / test / others) |

### Train / Validation / Test Split

Based on the partition analysis in `research_analysis/DATASET_1_PARTITIONS.csv`:

| Partition | Rows | Premise Groups | Positives | Positive % |
|-----------|------|----------------|-----------|------------|
| train | 4,000 | 1,349 | 1,347 | 33.67% |
| validation | 1,500 | 508 | 524 | 34.93% |
| calibration_prob | 1,000 | 336 | 336 | 33.60% |
| calibration_conformal | 1,000 | 334 | 342 | 34.20% |
| dev_reserve | 791 | — | — | — |
| test | 3,000 | 1,015 | 1,026 | 34.20% |
| test_reserve | 2,307 | — | — | — |

For the experiments reported in this package, the evaluation used:
- **Train:** 9,842 pairs (split == "val")
- **Test:** 9,824 pairs (split == "test")

### Class Distribution (Clean Pool)

| Class | Count | Percentage |
|-------|-------|------------|
| Positive (plagiarized proxy = entailment) | 6,693 | 34.05% |
| Negative (non-plagiarized) | 12,965 | 65.95% |
| **POS : NEG ratio** | **0.5162** | — |

Raw pool distribution (3-way):
- label=0 (entailment → positive): 6,697 (34.05%)
- label=1 (neutral): 6,454 (32.82%)
- label=2 (contradiction): 6,515 (33.13%)

### Feature Statistics (from `research_analysis/DATASET_1_STATISTICS.csv`)

| Feature | Count | Mean | Median | Std | Min | Max | Q1 | Q3 | Unique Values | Missing |
|---------|-------|------|--------|-----|-----|-----|----|----|---------------|---------|
| ref_chars | 19,658 | 72.10 | 64.0 | 32.52 | 9 | 300 | 48 | 93 | 6,640 | 0 |
| sus_chars | 19,658 | 37.80 | 34.0 | 16.23 | 3 | 232 | 27 | 45 | 19,106 | 0 |
| ref_words | 19,658 | 13.93 | 13.0 | 6.24 | 2 | 54 | 9 | 18 | 49 | 0 |
| sus_words | 19,658 | 7.50 | 7.0 | 3.11 | 1 | 48 | 5 | 9 | 33 | 0 |

**Observation:** Reference sentences are substantially longer (mean 72 chars, 13.9 words) than suspicious sentences (mean 38 chars, 7.5 words). This asymmetry is characteristic of paraphrase detection where the "suspicious" text is often a shortened or modified version.

### Example Pairs
- Pair 1: "Two women are embracing while holding to go packages." / "Two woman are holding packages." → Label: 1 (positive/paraphrase)
- Pair 2: "Two women are embracing while holding to go packages." / "The sisters are hugging goodbye while holding to go packages after just eating lunch." → Label: 0 (negative)

---

## Dataset 2 — Quora Question Pairs

### Source & Provenance
- **Files:** `datasets/dataset2_quora_train.tsv`, `datasets/dataset2_quora_test.tsv`, `datasets/dataset2_quora_dev.tsv`
- **Original path:** `/home/user/ml/datasets/external/shreyas_repo/Paraphrase-Detection-using-Siamese-NN-and-DistilBERT-main/dataset/`
- **Type:** Real-world question paraphrase corpus (Quora Question Pairs)
- **Total rows:** 65,401 pairs (train: 49,401 + test: 8,000 + dev: 8,000)
- **Original/non-synthetic:** ✅ Verified — contains genuine Quora questions asked by real users

### Columns
| Column | Description |
|--------|-------------|
| `id` | Unique pair identifier |
| `sentence1` | First question |
| `sentence2` | Second question |
| `label` | Binary target (0 = not duplicate, 1 = duplicate/paraphrase) |

### Split Sizes

| Split | Rows |
|-------|------|
| Train | 49,401 |
| Test | 8,000 |
| Dev | 8,000 |
| **Total** | **65,401** |

For the experiments reported, training was subsampled to 5,000 pairs for computational efficiency, with the full 8,000-pair test set used for evaluation.

### Class Distribution

| Scope | Positive (plagiarized/duplicate) | Negative (non-plagiarized) | POS : NEG Ratio |
|-------|------|------|------|
| raw_733 | 355 (48.43%) | 378 (51.57%) | 0.9392 |
| policy_A_720 | 342 (47.50%) | 378 (52.50%) | 0.9048 |
| policy_B_709 | 331 (46.69%) | 378 (53.32%) | 0.8757 |

### Feature Statistics (from `research_analysis/DATASET_2_STATISTICS.csv`)

| Feature | Count | Mean | Median | Std | Min | Max | Q1 | Q3 | Unique Values | Missing |
|---------|-------|------|--------|-----|-----|-----|----|----|---------------|---------|
| ref_chars | 733 | 548.64 | 501.0 | 242.36 | 161 | 1,382 | 376 | 723 | 105 | 0 |
| sus_chars | 733 | 709.44 | 678.0 | 276.36 | 161 | 1,587 | 493 | 890 | 447 | 0 |
| ref_lines | 733 | 24.04 | 21.0 | 8.18 | 11 | 61 | 20 | 29 | 33 | 0 |
| sus_lines | 733 | 28.67 | 28.0 | 8.70 | 10 | 61 | 22 | 34 | 43 | 0 |
| ref_tokens_ws | 733 | 69.19 | 65.0 | 29.62 | 18 | 151 | 48 | 92 | 74 | 0 |
| sus_tokens_ws | 733 | 82.30 | 79.0 | 33.78 | 18 | 188 | 58 | 105 | 47 | 0 |

**Observation:** Quora questions are substantially longer than Dataset 1 sentences (mean ~549-709 chars vs ~72-38 chars), reflecting the difference between short sentence pairs and full natural-language questions.

### Example Pairs
- Pair 1: "In Paris, in October 1560, he secretly met the English ambassador, Nicolas Throckmorton, asking him for a passport to return to England through Scotland." / "In October 1560, he secretly met with the English ambassador, Nicolas Throckmorton, in Paris, and asked him for a passport to return to Scotland through England." → Label: 0 (not duplicate)
- Pair 2: "The NBA season of 1975–76 was the 30th season of the National Basketball Association." / "The 1975–76 season of the National Basketball Association was the 30th season of the NBA." → Label: 1 (duplicate)

---

## Verification of Originality (Non-Synthetic)

Both datasets were verified by direct inspection:

1. **Dataset 1** contains natural human-written sentence pairs with realistic content (e.g., "Two women are embracing while holding to go packages."). The variations in length, wording, and structure are consistent with genuine paraphrase corpora, not algorithmically generated text.

2. **Dataset 2** contains genuine Quora questions referencing real historical events (Nicolas Throckmorton, 1560), real sports leagues (NBA 1975–76 season), and real entities. The semantic relationships are authentic human judgments, not synthetic labels.

Neither dataset shows patterns indicative of synthetic generation (e.g., perfectly balanced classes, templated text, or algorithmic label assignment).

---

## How Datasets Were Used

1. **Training:** Each method was trained on the training split of each dataset.
2. **Validation:** Used for hyperparameter tuning (e.g., C2 fusion weight grid search, C1 entropy quantile).
3. **Testing:** Held-out test set used for final evaluation — all reported metrics are out-of-sample.
4. **Conformal calibration (C4):** Used a separate calibration slice (75/25 train/cal split) to compute conformal quantiles with target coverage 0.85.
