# Phase 17 — Existing Research vs Proposed Framework

## Comparison matrix

Legend: **I** = implemented (with evidence) · **P** = partially implemented · **N** = not addressed ·
**NR** = not reported · **n/a** = not applicable.

| Research dimension | Paper 1 (Setu'25) | Paper 2 (Gandhi'24) | Paper 3 (Mehsen'24) | Proposed framework |
|---|---|---|---|---|
| Multi-view evidence | P (lexical+semantic via `max`) | P (7-D engineered vector) | P (token/syntactic vectors) | **I** — 12 text views / 13 code features, fused (`feature_pipeline.py`; ablation validates complementarity) |
| Candidate generation | N (all pairs scored) | N | P (n/a retrieval; ES-Plag-adjacent filtering is another paper) | **I** — full-pair coverage; recall-preserving screeners; measured 100% candidate recall vs ES-Plag 0.8627 |
| Uncertainty estimation | N | N | N | **I** — predictive entropy per pair (`uncertainty.py`), stored in prediction artifacts |
| Confidence calibration | N | P (uncalibrated `probability=True`) | N | **I** — Platt/isotonic/temperature + ECE/Brier/slope reports (`calibrators.py`, both phases) |
| Adaptive decisions | P (fixed dual band, E2) | N | N | **I** — budget-capped, recall-floored, ε-MCC cost-minimized OPs (`q1_stats_extras.py`) |
| Conformal prediction | N | N | N | **I** — group-wise split conformal, Wilson CIs (phase-5 correction) (`group_conformal.py`) |
| Budget-aware routing | P (workload reduction reported, no budget formalization) | N | N | **I** — β-constrained escalation, Pareto frontier, cost-min objective (`router.py`, `tune_routing_cost_minimizer`) |
| Computational efficiency | P (latency tabled) | NR (extraction cost unbudgeted) | P (latency tabled) | **I** — measured per-pair batched latencies + complexity ratios (`COST_ANALYSIS.csv`) |
| Explainability | N | N | N | **I** — operational dossiers, offsets/line indexing, no ground-truth leakage (`dossier_generator.py`) |
| Cross-modal aggregation | N (text only) | N (code only) | N (code only) | **I** — one framework, both modalities, shared dossier schema |
| Statistical validation | NR | NR | NR | **I** — McNemar + clustered bootstrap CIs + Wilson coverage CIs |
| Generalization evaluation | P (grouped corpus) | P (case-holdout) | P (case-holdout) | **I** — nested LOGO + grouped holdout + leakage quantification |

## Critical analysis

1. **Already existing capabilities:** engineered similarity features and ensemble discrimination;
   hybrid lexical-semantic fusion; informal cost reporting (Paper 1); ensemble robustness (Paper 3).
2. **Limitations remaining in the three papers:** no calibrated probabilities; no distribution-free
   uncertainty; no budget-formalized routing; no audit dossiers; no leak-free grouped evaluation with
   clustered statistics; no adversarial-augmentation defenses.
3. **What the framework introduces:** calibrated multi-view fusion (C1), group-conformal decision
   sets with binomial CIs (C2), ε-MCC cost-minimized budget routing with per-fold verifier selection
   (C3), dossier-based governance (C4) — plus the corrected protocols themselves.
4. **Overlap:** the proposed framework's *verifier anatomy* (RF/HGB, MLP) deliberately reuses the
   peer families (Paper 3's RF; GBDT lineage); overlap is acknowledged, not hidden.
5. **Meaningful methodological differences:** from *fixed single estimators* to a *calibrated
   cascade with guaranteed-marginal coverage accounting and budget-constrained objectives*.
6. **Experimentally demonstrated improvements:** text headline F1 0.7273 + 61.4% savings; code β=1
   F1 0.8481 with significant deltas vs 4/5 peers; iso-recall/iso-cost dominance; renaming/paraphrase
   robustness after augmentation.
7. **Theoretical/unvalidated improvements:** C4's faithfulness beyond ablation drop; institutional
   monetary-cost calibration reported as ms-latency ratios (wall-clock validated, currency-unit costs
   unvalidated); group-conditional (per-level) conformal coverage — listed as open question.
8. **Remaining weaknesses:** text = proxy corpus; code corpus small (720, 7 cases); code calibration
   under LOGO weak (ECE 0.181); group coverage slightly under nominal; span-level explainability
   absent.
