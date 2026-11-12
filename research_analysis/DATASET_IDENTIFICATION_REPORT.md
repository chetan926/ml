# Phase 2 — Dataset Identification Report

Both datasets were identified **from repository files** (config constants, loaders, provenance audit,
processed CSVs) and verified against the actual bytes on disk. No replacement dataset is introduced.

---

## Dataset 1 — Text modality

| Field | Verified value |
|---|---|
| Exact dataset name (as used by project) | SNLI/Pereira **Sentence-Level Semantic Equivalence & Text Reuse Benchmark** — hosted on Kaggle as *"MIT Plagiarism Detection Dataset"* (`ruvelpereira/mit-plagairism-detection-dataset`), provenance-traced to the **SNLI corpus** (Bowman et al. 2015) |
| Original source | Stanford Natural Language Inference corpus (premises from Flickr30k captions; hypotheses crowd-authored); Kaggle re-export by R. Pereira |
| Repository location | Raw: `datasets/raw/text_mit_plagiarism/snli_val.parquet` (10,010 rows → 9,842 after consensus filter), `snli_test.parquet` (→9,824). Processed: `datasets/processed/text_plagiarism_benchmark.csv` |
| File format | Parquet (raw), CSV (processed: `id, reference_text, suspicious_text, label, is_plagiarized, split`) |
| Total records | **19,666 raw pair rows** → 8 purged → **19,658 clean** (`N_RAW_TEXT_ROWS=19666`, `N_TOTAL_PURGED_ROWS=8`, `N_TOTAL_CLEAN_ROWS=19658` in `research_upgrade/src/config.py`; independently re-computed in this analysis) |
| Features | Raw text fields only; all model features are derived by `research_upgrade/src/features/text_*.py` |
| Target labels | `label` ∈ {0 entailment, 1 neutral, 2 contradiction}; research target `is_plagiarized = (label==0)` |
| Class definitions | Positive = semantic-equivalence/paraphrase-reuse proxy (entailment); Negative = neutral ∪ contradiction |
| Dataset versions | Single in-repo version; schema-guarded by `validate_text_schema` and asserts in `text_loader.py` |
| Existing preprocessing | Whitespace normalization upstream; premise-group purging (cross-split premise contamination: 6 rows from 1 shared premise; ordered redundant duplicates: 2 rows) |
| Partitioning strategy | **Deterministic premise-group allocation** (`TextBenchmarkDataLoader`): premise SHA-256 ordering + non-overshooting allocator over the SNLI-derived `val` pool and `test` pool |

### Verified partition sizes (re-computed in this analysis, `DATASET_1_PARTITIONS.csv`)

| Partition | Rows | Premise groups | Positives | Pos % |
|---|---|---|---|---|
| train | 4,000 | 1,349 | 1,347 | 33.67 |
| validation | 1,500 | 508 | 524 | 34.93 |
| calibration_prob | 1,000 | 336 | 336 | 33.60 |
| calibration_conformal | 1,000 | 334 | 342 | 34.20 |
| dev_reserve | 791 | — | — | — |
| test | 3,000 | 1,015 | 1,026 | 34.20 |
| test_reserve | 2,307 | — | — | — |

Zero premise overlap between any pair of partitions by construction (allocator is disjoint);
consistent with README §7 and `partition_validation.py`.

---

## Dataset 2 — Code modality

| Field | Verified value |
|---|---|
| Exact dataset name | **IR-Plag Java Corpus** (Oscar Karnalim, 2019) |
| Original source | Authentic introductory-programming student submissions; transformations graded into 6 Faidhi–Robinson obfuscation levels (L1–L6) |
| Repository location | `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` — cases `case-01 … case-07`, each with `original/`, `non-plagiarized/01..15`, `plagiarized/L1..L6`; archive copy `ir_plag.zip` (repo root) |
| File format | 467 physical `.java` files (count verified on disk, matches `N_IR_PLAG_PHYSICAL_JAVA_FILES`) + processed pair table |
| Processed artifact | `datasets/processed/code_plagiarism_benchmark.csv` — `id, case, reference_code, suspicious_code, ref_type, sus_type, obfuscation_level, is_plagiarized` |
| Total records | **733 raw pairs**; **24 rows in 11 duplicate collision groups** (verified) → **Policy A: 720 pairs** (drop subsequent occurrences); **Policy B: 709 pairs** (drop entire collision groups) |
| Target labels | `is_plagiarized` ∈ {0,1}; positives: 355 (48.43% of raw), negatives: 378 |
| Class definitions | Positive = plagiarized submission vs. original; positives carry `obfuscation_level` ∈ {L1..L6} (63/56/57/60/59/63 pairs); negatives have structural NaN in `obfuscation_level` |
| Existing preprocessing | Comment/whitespace stripping for canonical forms upstream; `javalang` AST parsing in feature pipeline |
| Splits / validation | **Nested Leave-One-Case-Out (LOGO)** over the 7 assignment cases (inner folds for calibration/OP tuning); Case-05 singular cluster (53 positives from one base program) handled by audited cross-case fitting (`CASE_05_*` constants) |

---

## Verification evidence

- Row counts, class counts, purge accounting, and Policy A/B sizes above were **re-computed from the
  CSVs in this phase** and match the asserted constants in `research_upgrade/src/config.py` exactly.
- The two processed CSVs are the entry points declared in `config.py` (`TEXT_DATASET_PATH`,
  `CODE_DATASET_PATH`) and are consumed by both the legacy `src/` pipeline and the upgrade loaders.
- IR-Plag physical file count (467) verified by `find ... -name "*.java" | wc -l`.
- SNLI row counts verified by reading the raw parquet files (10,010 → 9,842/9,824 with consensus
  filtering, per provenance audit).
