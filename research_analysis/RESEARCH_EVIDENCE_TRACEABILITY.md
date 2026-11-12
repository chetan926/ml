# Phase 23 — Research Evidence Traceability

## A. Literature claims (three repository papers)

| Claim | Paper | Evidence location (repo) | Note |
|---|---|---|---|
| Metadata (title/authors/venue/DOI) | all three | `literature/selected_papers.csv` rows M1/M2/M3 | verified bibliographic records |
| Setu methodology (max(TF-IDF,SBERT)) | P1 | `final_literature_verification.md` §2.1 | p. ref into PDF unavailable (PDF absent) |
| Setu reproduced metrics (0.7200/0.5697/0.8570/0.6844; AUC 0.8198; τ=0.5) | P1 | audit §2.1 | repository reproduction |
| Setu latency (4.163/2.875 ms; ~30% workload cut) | P1 | `literature/methodology_comparison.csv` | repository measurement |
| Gandhi methodology (7-D + RBF-SVM) | P2 | audit §2.2; `src/code_plagiarism.py` | code inspected |
| Gandhi reproduced metrics (0.7000/0.6258/0.9510/0.7549; τ=0.37) | P2 | audit §2.2 | — |
| Gandhi corrected-protocol results (F1 0.8363/MCC 0.6754) | P2 | `PHASE_5_HEADLINE_COMPARISON.csv` | nested LOGO |
| Mehsen methodology (RF 100×d6) | P3 | audit §2.3; `src/code_random_forest.py` | audit-corrected depth |
| Mehsen reproduced metrics (0.7333/0.6643/0.9118/0.7686) | P3 | audit §2.3 | — |
| L6 resilience 83.33% (adaptation record) | P3 | audit §2.3 | — |
| ES-Plag/RKRGST/Jaccard supporting roles | ref | `literature/selected_papers.csv` M4/B1/B2 | supporting baselines, not among "the three" |

## B. Experimental claims (this project)

| Claim | Dataset | Experiment ID | Source file | Value | Setting |
|---|---|---|---|---|---|
| Text proposed metrics (0.7893/0.7234/0.6218/0.6688/0.5188) | D1 | phase-4.1 frozen | `PHASE_4_1_PREDICTIONS_TEXT_TEST.csv` | see §15.1 | grouped test, n=3000 |
| Text proposed upgrade (0.8080/0.7273/0.5799; 61.37% saved) | D1 | phase-5 run5 | `PHASE_5_PREDICTIONS_TEXT_TEST.csv` + headline CSV | see §15.1 | grouped test, heavy S2 |
| Code proposed 4.1 (0.8486/0.8467/0.6996; 15.14% saved) | D2 | phase-4.1 frozen | `PHASE_4_1_PREDICTIONS_CODE_OOF.csv` | see §15.1 | nested LOGO OOF |
| Code proposed 5.0 (0.8287/0.6636; 27.36%) and β=1 (0.8481/0.6997) | D2 | phase-5 run5 | `PHASE_5_PREDICTIONS_CODE_OOF.csv` + headline CSV | see §15.1 | nested LOGO OOF |
| Statistical tests (McNemar + clustered bootstrap) | both | phase-5 | `PHASE_5_STATISTICAL_TESTS.json` | ΔF1 CIs per pair | cluster=case/premise |
| Conformal coverage + Wilson CI | D1 | phase-5 | `PHASE_5_CONFORMAL_COVERAGE.csv` | 0.930 [0.913,0.944] | group, α=0.05 |
| Calibration table | D1 | 4.1/5 | `PHASE_4_1_CALIBRATION_METRICS.csv` | ECE/Brier/slopes ×4 calibrators | test/prod in-fold |
| Cost ratios | both | 4.1/5 | `PHASE_4_1_COST_ANALYSIS.csv`, `PHASE_5_COST_ANALYSIS.csv` | 18.5×/20.2× batched; parameters ratios | measured host |
| Robustness (text attacks) | D1 | phase-5 | `PHASE_5_ROBUSTNESS_TEXT_ATTACKS.csv` | F1 per family | offline augmented variant |
| Robustness (renaming) | D2 | phase-5 | `PHASE_5_CODE_RENAMING_ATTACK.csv` | proposed β=1 0.8918 vs SVM 0.2778 | extra-benchmark attack |
| Obfuscation strata | D2 | phase-5 | `PHASE_5_CODE_OBFUSCATION_LEVELS.csv` | per L1–L6 pos-recall | LOGO OOF |
| Ablations | both | 4.1/5 | `PHASE_4_1_ABLATION_RESULTS.csv`, `PHASE_5_ABLATION.csv` | MCC ladders | frozen runs |
| Dataset purges/partitions | D1 | this analysis | `DATASET_1_PARTITIONS.csv` (script recompute) | 8 purged; partitions 4k/1.5k/1k/1k/3k | hash-deterministic |
| Dataset duplicate policies | D2 | this analysis | `DATASET_2_CLASS_DISTRIBUTION.csv` | 733→720/709 | Policy A/B |

## C. Dataset claims

Dataset provenance, label mapping, purging, and partition determinism trace to:
`experiments/results/dataset_provenance_audit.md`, `research_upgrade/src/data/{text_loader,
code_loader}.py`, `research_upgrade/src/config.py` constants, and re-computation in
`analysis_scripts/run_quantitative_analysis.py` (matching all asserted constants).

## D. Integrity statement

No DOI, author, venue, year, dataset name, or metric in this analysis was invented. Where evidence
was missing (paper PDFs, span-level labels, monetary costing), the corresponding cells say NR or the
limitation is written explicitly.
