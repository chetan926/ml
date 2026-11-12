# Phase 25 — Final Research Analysis Audit (Independent Verification)

**Scope of audit:** the 26 required deliverables + supporting CSVs/figures under
`research_analysis/`, against actual repository files. Verification steps below were executed as
commands or file comparisons; nothing was taken on faith from the analysis documents themselves.

## 1. Dataset verification — **PASS**

| Check | Method | Result |
|---|---|---|
| Both datasets correctly identified | `config.py` paths + provenance audit + raw files on disk | SNLI/Pereira parquet (9,842+9,824 rows read directly); IR-Plag 467 `.java` files counted on disk |
| Qualitative analyses complete | `DATASET_1/2_QUALITATIVE_ANALYSIS.md` sections map 1:1 to prompts | §3.1–3.4 and items 1–15 both present |
| Quantitative analyses complete | `*_QUANTITATIVE_ANALYSIS.md` + 4 machine CSVs per dataset | rows computed by `run_quantitative_analysis.py` from actual files |
| Statistical calculations verified | independent re-computation replicated loader logic exactly: **8 purged rows (6 crossing + 2 dupes), clean 19,658; Policy A=720, B=709, 11 duplicate groups/24 rows; partitions 4,000/1,500/1,000/1,000/3,000** — all equal to the frozen `config.py` constants | PASS |
| Class distributions verified | recomputed counts equal README/audit values (test 1,026/1,974; code 342/378 Policy A) | PASS |
| No dataset modifications | `git status`: only `research_analysis/` entries; datasets untouched | PASS |

## 2. Literature verification — **PASS (with documented limitation)**

- Exactly three designated papers identified from repository authority
  (`final_literature_verification.md` §2 "Three Recent Methodologies"; ambiguity over the 6-row
  `selected_papers.csv` resolved via the audit's recency-verdict table and category tags
  (`Recent Methodology` vs `Reference`/`Supporting`) — reported, not guessed.
- Individual qualitative + quantitative analyses exist for each paper (6 files).
- Qualitative + quantitative comparisons + machine CSV exist (3 files).
- Published results traceable: repository reproduction numbers labeled as such; paper-published-only
  values consistently marked NR; DOIs copied from verified repository metadata (no invented DOIs).
- **Limitation:** full-text PDFs absent from repository → page/section citations into the papers are
  replaced by repository-evidence citations (disclosed in Phase 8 §8.2 and every paper phase).
- No external/replacement papers introduced.

## 3. Proposed framework verification — **PASS (with declared partial items)**

- Four contributions analyzed against actual modules (`research_upgrade/src/*`, phase-5 `q1src/*`).
- Experimental results verified from stored per-pair predictions: recomputed confusion matrices and
  metric suites match the frozen headline values exactly (e.g., 4.1 text 0.7893/0.7234/0.6218/
  0.6688; code 0.8467 F1; phase-5 code β=1 TP 307/FN 35/FP 75/TN 303 reproducing F1 0.8481).
- Baseline comparisons completed for both protocols (frozen and upgrade), incl. degenerate-baseline
  measurement (Jaccard MCC 0.000 at frozen threshold; specificity 0.000).
- Ablations checked: contribution ladder exists both generations; two declared non-ablated cells
  (no-routing counterfactual, no-dossier counterfactual) marked *not experimentally validated*.
- Calibration/uncertainty claims checked: text ECE 0.0145–0.031 real; code LOGO ECE 0.181 disclosed;
  conformal group coverage 0.930 Wilson [0.9127, 0.9442] reported as shortfall.
- Computational cost claims checked: latencies cited from measured CSVs only (18.5×/20.2× batched;
  14,656×/628× complexity ratios); monetary costing left unvalidated.
- Explainability claims checked: dossier metrics quoted from frozen artifacts; span-level P/R/F
  correctly reported as structurally unavailable (no invented scores).

## 4. Research integrity verification — **PASS**

| Rule | Evidence |
|---|---|
| No fabricated metrics | every number traces to a repository file or a recomputation script listed in Phase 23 |
| No fabricated references | `.bib` uses repo-verified metadata + canonical foundational refs; suspicious entries annotated |
| No unsupported novelty | C1–C4 classified with an evidence level; C4 limited to "partially validated" |
| No misleading comparisons | comparability rules in Phase 13; ties vs B-C3/B-T3 disclosed; no cross-dataset model-transfer claim |
| No train/test leakage introduced | analysis is read-only; partition logic replicated, not re-run into datasets |
| No frozen artifacts modified | `git status` shows zero modifications outside `research_analysis/` |
| No unauthorized training/config changes | no training executed; venv creation is environment-only |

## 5. Deliverable completeness (Phase 24)

All 27 required files present (26 named + audit incl. this file), plus: 4 additional machine CSVs
(partitions, subgroups ×2, obfuscation, case distribution, comparison), `PROPOSED_SOLUTION_ROUTING_DETAIL.json`,
17 figures (`figures/fig01–fig17`), 3 analysis scripts, `repository_inventory.json`, `analysis_summary.json`.

## 6. Outstanding issues (carried as limitations, not failures)

1. Paper PDFs absent → NR-marked paper-side values; metadata-only verification.
2. Conformal group undercoverage (0.930 vs 0.95) — disclosed everywhere it appears.
3. Code calibration under LOGO (ECE 0.181) — disclosed; not corrected post-hoc.
4. C4 faithfulness partially validated (ablation drop only; no human study).
5. Text dataset is an SNLI proxy — ecological-validity boundary stated in Phases 3/7/20.
6. Single seed (42); no external institutional dataset.

## Final audit verdict

**`PASS_WITH_LIMITATIONS`**

Every mandatory requirement is satisfied or explicitly marked with a verifiable limitation. A
`FULL_PASS` is not issued because three mandatory-evidence classes are structurally unavailable in
the repository (full-text PDFs of the three papers; human span annotations; monetary unit costs),
which is itself verifiable and disclosed rather than masked.
