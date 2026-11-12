# Dataset qualitative analysis

- Dataset 1 (text): SNLI-derived premise/hypothesis pairs; paraphrase-entailment is a proxy for text reuse; legitimate pairs include topical near-misses — hard negative regime.
- Dataset 2 (code): IR-Plag Java with Faidhi-Robinson obfuscation L1-L6; positives at L6 are globally restructured programs — the hardest detection regime.
Counts in 01_DATASET_ANALYSIS/*.csv; leakage checks PASS in DATASET_QUALITY_REPORT.json
