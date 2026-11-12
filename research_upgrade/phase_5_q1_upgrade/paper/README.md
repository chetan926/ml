# Paper Package — Phase 5 Q1 Upgrade

`main.tex` is a self-contained IEEE two-column (IEEEtran, `conference`) manuscript.
No BibTeX pass is required (references are inlined via `thebibliography`).

## Build

```bash
# any TeX Live / MiKTeX / MacTeX installation with IEEEtran:
pdflatex main && pdflatex main
# or
latexmk -pdf main
```

Required inputs (all present): `main.tex`, `figures/figure_1_calibration.png`,
`figures/figure_2_risk_coverage.png`, `figures/figure_3_pareto_isocost.png`,
`figures/figure_4_ablation_mcc.png`, `figures/figure_5_robustness_attacks.png`.
(`figure_6_code_obfuscation_levels.png` is optional spare art for rebuttal/poster.)

If `IEEEtran.cls` is missing locally it is fetched automatically from CTAN by any
modern TeX distribution. Overleaf: just upload the folder — it works unchanged.

## Regenerating the numbers

Every table/figure value is produced deterministically (seed 42) by:

```bash
cd /home/user/ml
./.venv/bin/python research_upgrade/phase_5_q1_upgrade/run_phase_5_execution.py
```

Artifact → paper mapping:

| Paper element | Source artifact (`../artifacts/`) |
|---|---|
| Table II (headline) | `PHASE_5_HEADLINE_COMPARISON.csv` |
| Table III (stats)   | `PHASE_5_STATISTICAL_TESTS.json` |
| Table IV (iso-recall) | `PHASE_5_ISO_RECALL_COMPARISON_TEXT.csv` |
| Fig. 3 + RQ3 numbers | `PHASE_5_PARETO_FRONTIER.csv` |
| Table VI (text attacks) | `PHASE_5_ROBUSTNESS_TEXT_ATTACKS.csv` |
| Table VII (renaming) | `PHASE_5_CODE_RENAMING_ATTACK.csv` |
| L1–L6 obfuscation   | `PHASE_5_CODE_OBFUSCATION_LEVELS.csv` |
| Calibration ECE...  | `PHASE_5_RESEARCH_RESULTS.json` |
| Conformal + Wilson  | `PHASE_5_CONFORMAL_COVERAGE.csv` |
| Cost ratios         | `PHASE_5_COST_ANALYSIS.csv` |
| Ablation            | `PHASE_5_ABLATION.csv` |
| Dossier (C4) schema | `PHASE_5_PREDICTIONS_*.csv` |

Tests: `./.venv/bin/python -m unittest tests.test_phase_5_q1_upgrade -v` (17 tests).

## The six requested fixes → where they landed

1. **Earned "outperform" claim** — iso-recall table (IV), iso-cost figure (3) + cost-minimized
   OP (Eq. 1), β=1 cost-unconstrained rows; Section VI states the three-regime verdict honestly.
2. **Real code baselines as headline** — B-C3 (SVM'24), B-C4 (RF'24), B-C5 (ES-Plag), B-C2
   (RKRGST) in the headline table under nested LOGO; B-C1 Jaccard demoted to a footnote
   (degenerate near-all-positive behavior noted explicitly).
3. **Genuinely heavy Stage-2 verifier** — 175,873-param cross-style MLP (text) and
   per-fold selected HGB(400,d8)/RF(500,d12) verifiers (code); measured 18.5–20.2× batched
   latency and 628–14,656× architectural complexity over Stage-1 (Section VII-G).
4. **LLM-paraphrase robustness study** — deterministic surrogate attack suite (synonym light/heavy,
   word order, char noise, filler, combined) with leak-free adversarial augmentation; code
   extra-benchmark systematic identifier renaming. Limitation (surrogate attacks, LSA
   re-embedding) disclosed in §VIII.
5. **Binomial CI on conformal coverage** — Wilson 95% CIs at group and pair levels; the
   undercoverage (0.930 vs 0.95) is reported as a mixed result, not hidden.
6. **Realistic venues** — see checklist below.

## Venue checklist (tailoring, post-acceptance of this draft)

- **IEEE Trans. Learning Technologies** — already IEEEtran; add learning-analytics framing in
  intro, IRB/data-ethics statement.
- **Computers & Education: AI (Elsevier)** — switch to `elsarticle` (two-column), number →
  author-year citations, add CRediT roles + declaration of interest.
- **Assessment & Evaluation in Higher Education (T&F)** — APA7 prose style, expand
  Discussion/Limitations, de-emphasize equations.
- **IPM / ESWA (Elsevier)** — `elsarticle`; add graphical abstract and highlights file;
  ESWA expects a stronger "expert system" framing (emphasize dossiers/C4 as decision support).

## Known honesty checkpoints (do not 'fix' before review)

- ΔF1 vs B-C3 and B-T3 at headline operating points are **not significant**; the paper claims
  significance only where CIs exclude zero, and dominance at iso-recall / iso-cost / β=1.
- Group conformal coverage undercovers by 2 points (Wilson CI excludes 0.95).
- Text attack suite uses a deterministic LSA-surrogate offline variant where the MiniLM feature
  store is absent; latency claims are measured on the eval host (batched regime).
