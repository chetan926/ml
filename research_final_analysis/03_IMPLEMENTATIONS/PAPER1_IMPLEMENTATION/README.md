# Paper 1 implementation — Setu et al. (2025)

Realization (no duplication, per §22C): frozen `TextBaselineSuite`
(`research_upgrade/phase_5_q1_upgrade/q1src/q1_baselines.py`) executes B-T2 dense semantic
MiniLM threshold + B-T1 TF-IDF sibling, thresholds tuned on validation only.
Evidence: `../../04_EXPERIMENTAL_RESULTS/PREDICTIONS/B-T1_text_predictions.csv`,
`B-T2_text_predictions.csv`, rows in `CLASSIFICATION_METRICS.csv`.
Dev note: full text absent; methodology per repository audits; published values NR.
