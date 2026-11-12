# Phase 6 — Q1 Hardening

Standalone upgrade layer that fixes the six Q1 reviewer-attack items identified
after Phase 5, **without touching any frozen phase**. All artifacts land in
`artifacts/`; the definitive write-up is `artifacts/PHASE_6_REPORT.md` and the
revised manuscript is `paper/main_v2.tex` (phase-5 paper left untouched).

## The six items and where the evidence lives

| # | Item | Module / driver | Artifacts |
|---|------|-----------------|-----------|
| 1 | External validity | `prep_external_text.py` (Clough–Stevenson + MRPC fetch→freeze), `run_phase_6_execution.py` S5–S6, `external_ranking_metrics.py` | `PHASE_6_EXTERNAL_*_PREDICTIONS.parquet`, `PHASE_6_EXTERNAL_RANKING_METRICS.*`, `PHASE_6_CS_SEVERITY_RECALL.csv`, code L6 transfer in `PHASE_6_SUMMARY.json` |
| 2 | C4 human evidence | S7 walkthrough packet + `C4_HUMAN_REVIEW_PROTOCOL.md` | `PHASE_6_C4_REVIEW_PACKET.{csv,json}` |
| 3 | Conformal under-coverage (0.930 vs 0.95) | `h6src/conformal_variants.py`, S3, `repair_set_size_audit.py` | `PHASE_6_CONFORMAL_VARIANTS.csv`, `PHASE_6_CONFORMAL_SET_SIZE_AUDIT.json` |
| 4 | Neural cross-encoder-style verifier | S4 ablation (MLP vs RF/HGB/LR on identical interaction input) | `PHASE_6_VERIFIER_ABLATION.csv` |
| 5 | Multi-seed repeats | S2/S6 (seeds 42, 123, 2024, text + code LOGO) | `PHASE_6_MULTISEED_{TEXT,CODE}.csv` |
| 6 | Novelty/systems framing | `paper/main_v2.tex` | revised abstract, RQ5 repair narrative, RQ7, RQ8, updated threats |

## Protocol guarantees (carried forward from phases 4–5)

- **Evaluation-only external rule**: no external sample enters any fit,
  calibration, operating point, or conformal pool.
- Group-aware bootstrap / Wilson CIs everywhere; structurally-undefined metrics
  are reported as `NR`, never as zero.
- Cross-topic distractor negatives are reported **separately** and never pooled.
- Honest negative results are reported verbatim (C-S thresholded recall 0.00,
  MRPC thresholded collapse, MLP not the best verifier).

## Reproduce

```bash
./.venv/bin/python research_upgrade/phase_6_q1_hardening/prep_external_text.py
./.venv/bin/python research_upgrade/phase_6_q1_hardening/run_phase_6_execution.py   # ~11 min
./.venv/bin/python research_upgrade/phase_6_q1_hardening/external_ranking_metrics.py
./.venv/bin/python research_upgrade/phase_6_q1_hardening/repair_set_size_audit.py
./.venv/bin/python research_upgrade/phase_6_q1_hardening/generate_report.py
./.venv/bin/python -m pytest tests/test_conformal_variants.py -q
```

## External data provenance

- Clough–Stevenson corpus answers + `orig_task{a,b}` passages:
  `koustubhvijaykulkarni/Plagiarism-detection-on-Clough-Stevenson-corpus-dataset-using-unsupervised-learning-techniques-`
- Missing `orig_task{c,d,e}` passages (99% byte-identical for a/b):
  `arazosman/Code-Detective` → `Datasets/txt/orig_task*.txt`
- MRPC test split (1,725 pairs, full):
  `czx-li/DP2O` → `msr_paraphrase_test.txt`
