# Four research contributions — validation matrix

| Contribution | Protocol | Evidence | Result |
|---|---|---|---|
| C1 Modality-aware multi-view detection | Separate text & code evaluation | CLASSIFICATION_METRICS.csv | text acc 0.808 / F1 0.7273; code acc 0.8306 / F1 0.8287 (no pooled misleading score) |
| C2 Uncertainty-aware filtering | ECE/Brier + group conformal (phase-5) + repaired α* (phase-6) | UNCERTAINTY_METRICS.csv; phase-5 CONFORMAL_COVERAGE.csv; phase-6 CONFORMAL_VARIANTS.csv | group coverage 0.930 → repaired 1.000 (Wilson [0.996,1.000]); set-size cost disclosed |
| C3 Budget-constrained escalation | routing tables + budget sweep + cost ratios | ROUTING_METRICS.csv; BUDGET_ANALYSIS.csv; COMPUTATIONAL_* | 61.4% (text) / 27.4-34.7% (code) heavy-verifier calls avoided at ε-quality |
| C4 Cross-modality evaluation & audit dossiers | unified protocol + grey-zone review packet + external transfer probes | phase-6 C4 packet; external corpora artifacts | verified end-to-end; human-effect gains NOT claimed (documented) |

## Novelty statement (verified vs hypothesis)
- **Demonstrated:** joint accuracy-cost-uncertainty envelope with leak-free grouped protocol;
  conformal repair trade-off dial; evaluation-only external transfer characterization.
- **Hypothesis-class (unverified):** institution-scale human review time reduction (needs the
  IRB-style study in phase-6 C4 protocol); transfer with neural embeddings when MiniLM is
  available to the deployment site.
