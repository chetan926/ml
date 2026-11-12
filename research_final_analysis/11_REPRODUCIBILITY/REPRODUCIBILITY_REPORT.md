# Reproducibility report

- Scripts executed in order rfa01..rfa08 in this session; logs: `06_ABLATION_STUDY/ablation_run.log` and console captures.
- Integrity check: identical-seed shadow refits reproduce frozen phase-5 baseline hard predictions bit-for-bit on every OOF row (IMPLEMENTATION_INDEX.json: svm=True, rf=True).
- Leakage: premise/pair overlap checks PASS (DATASET_QUALITY_REPORT.json).
- Multi-seed stability verified in phase-6 artifacts (SD <= 0.010 MCC).
