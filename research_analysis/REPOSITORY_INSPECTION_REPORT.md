# Phase 1 — Repository Inspection Report

**Analysis mode:** Independent, read-only research analysis. No project file was created, modified,
or deleted outside `research_analysis/` during this phase. Inventory generated programmatically by
`research_analysis/analysis_scripts/phase1_inventory.py` (full per-file list with 12-char SHA-256
fingerprints in `research_analysis/analysis_scripts/repository_inventory.json`).

**Repository:** `/home/user/ml` — *An Uncertainty-Aware, Budget-Constrained Multi-View Framework for
Cost-Efficient Academic Integrity Detection*. Branch under analysis: `arena/26b37a1c-ml` (post-merge
of PR #1, base commit `2796a31` of `main`).

---

## 1. Repository inventory (machine-generated)

| Axis | Result |
|---|---|
| Files tracked by inventory (excl. `.git`, venvs, caches) | **1,038 files, ≈ 26.1 MB** |
| Source code (`.py`) | 151 |
| Documentation (`.md/.txt/.pdf`) | 182 |
| Data CSV (`.csv`) | 102 |
| Structured (`.json/.toml/.bib`) | 69 |
| Figures (`.png`) | 43 |
| Archives (`.zip`) | 5 |
| Raw dataset files (datasets/, incl. 467 Java files) | 474 |
| Notebooks (`.ipynb`) | 5 |

## 2. Top-level structure

| Path | Role |
|---|---|
| `README.md` (17.4 KB) | Master project document: title, abstract, four contributions, datasets, methods, headline audited results, limitations |
| `GITHUB_PUBLICATION_AND_VERIFICATION_REPORT.md` | Publication & external verification record |
| `final_verified_results.csv`, `methodology_verification.csv` | Root-level verified metric snapshots |
| `reproducibility_checklist.md`, `research_execution_audit.md` | Process audits from earlier phases |
| `run_all.py` | Legacy phase driver (pre-upgrade) |
| `LICENSE` (Apache-2.0), `requirements.txt`, `pyproject.toml` | Packaging |
| `datasets/` | **Raw + processed data for both modalities** (Phase 2) |
| `literature/` | **`selected_papers.csv`, `methodology_comparison.csv`, `methodology_verification.md` — the paper selection evidence (Phase 8)** |
| `src/` | Original (phase ≤3) implementation: text/code pipelines, `bert_similarity.py`, `code_plagiarism.py` (SVM), `code_random_forest.py`, `typed_decision_pipeline.py`, `threshold_optimizer.py`, `cost_profiler.py`, `evaluation.py` |
| `src/code/` | ES-Plag implementation modules (tokenizer, indexer, rkr_gst, penalty, detector) |
| `notebooks/` | 5 exploratory notebooks (dataset analysis, text/code methods, typed pipeline, comparison) |
| `experiments/` | `results/` (27 CSV/MD/JSON result files incl. ES-Plag and Setu-hybrid pair-level results) and `figures/` (19 pre-rendered PNGs) |
| `reports/` | 11 documents incl. `final_research_report.pdf` (project's own report), qualitative/quantitative analyses, methodology and limitations documents |
| `research_audit/` | 8 audit documents (dataset validation, literature verification, metric audit, reproducibility, implementation, runtime, phase-1 gap analysis) |
| `scripts/` | repo inventory utilities, push script |
| `tests/` | 23 test files (framework + phase-4/4.1/5 regression tests) |
| `research_upgrade/` | **The upgraded research programme (phases 2–5):** `src/` (12 feature/calibration/routing/conformal/explainability modules), `artifacts/phase_3 + final_research`, `phase_4_corrections/`, `phase_4_1_corrections/` (audited frozen results + independent verification zip), `phase_5_q1_upgrade/` (latest Q1 upgrade: heavy verifiers, attacks, ISO comparisons, IEEE paper), `audits/`, `backups/`, `documentation/`, top-level design docs (`system_architecture.md`, `mathematical_formulations.md`, `four_contributions_formal_design.md`, `algorithm_pseudocode.md`, `novelty_and_literature_analysis.md`, `experimental_protocol.md`) |

## 3. Identified datasets (verified in Phase 2)

| # | Name | Raw location | Processed artifact |
|---|---|---|---|
| 1 | **SNLI/Pereira "MIT Plagiarism Detection" text-reuse proxy** | `datasets/raw/text_mit_plagiarism/{snli_val.parquet, snli_test.parquet}` (10,010 rows each → 9,842 / 9,824 retained) | `datasets/processed/text_plagiarism_benchmark.csv` (19,666 rows) |
| 2 | **IR-Plag Java source-code corpus** (Karnalim 2019) | `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` — 7 cases, **467 physical `.java` files** (verified count; `N_IR_PLAG_PHYSICAL_JAVA_FILES=467`) | `datasets/processed/code_plagiarism_benchmark.csv` (733 rows) |

## 4. Identified research papers (verified in Phase 8)

The three designated recent methodologies (per `research_audit/final_literature_verification.md`,
recency requirement "three methods published in 2024–2026"):

1. **Setu et al. (2025)** — *A comprehensive strategy for identifying plagiarism in academic submissions.* J. Umm Al-Qura Univ. Eng. Archit. (Springer). DOI 10.1007/s43995-025-00108-1.
2. **Gandhi, Gopalan & Prasad (2024)** — *A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings.* Frontiers in Computer Science 6:1393723. DOI 10.3389/fcomp.2024.1393723.
3. **Mehsen & Joshi (2024)** — *Detection of Source Code Plagiarism Utilizing an Approach Based on Machine Learning.* Int. J. Computing 23(1):78–84. DOI 10.47839/ijc.23.1.3438.

Supporting/reference literature also tracked: ES-Plag (Sulistiani & Karnalim 2019, DOI 10.1002/cae.22066),
Salton & Buckley (1988) / Sajid et al. (2025) TF-IDF, Faidhi & Robinson (1987) Jaccard.
**Verification status: bibliographic metadata present in repo; full-text PDFs of the three papers are
NOT stored in the repository** (see §7).

## 5. Relevant implementation modules (proposed framework)

| Layer | Files |
|---|---|
| Data/validation | `research_upgrade/src/data/{text_loader, code_loader, schema_validation, partition_validation}.py` |
| Features (C1) | `research_upgrade/src/features/*.py` — text lexical/character/semantic/structural; code token/AST/identifier/complexity; `feature_pipeline.py` |
| Calibration & UQ (C2) | `src/calibration/{calibrators, calibration_metrics, uncertainty, group_conformal}.py` |
| Routing & cost (C3) | `src/routing/{router, cost_model, decision_states, pareto, stage2_verifier}.py` |
| Explainability (C4) | `src/explainability/{dossier_generator, evidence_extractor, evidence_metrics}.py` |
| Candidate generation | `src/candidate_generation/*.py` (scorers, fusion, threshold selection, robustness, ablations) |
| Evaluation | `src/evaluation/{statistical_testing, feature_cost, feature_quality}.py`, `src/pipeline/integrated_framework.py` |
| Governance | `research_upgrade/src/governance.py` (SHA-256 frozen-baseline integrity verification) |
| Phases | `phase_4_corrections/`, `phase_4_1_corrections/` (+`run_phase_4_1_execution.py`), `phase_5_q1_upgrade/` (+`run_phase_5_execution.py`, own `q1src/` package with heavy verifiers, attacks, OP tuners) |

## 6. Available experimental results (existing, frozen — touched read-only)

| Source | Key contents |
|---|---|
| `research_upgrade/artifacts/final_research/` | FINAL_* CSVs/JSON: classification, calibration, cost, ablation, robustness, statistical analysis, SHA-256 manifest, execution report/paper drafts |
| `research_upgrade/phase_4_1_corrections/artifacts/` | Canonical audited results: per-pair predictions (text test n=3,000; code OOF n=720), baseline predictions, calibration per calibrator, ablations, pareto points, execution config & folds |
| `research_upgrade/phase_5_q1_upgrade/artifacts/` | Heavy-verifier upgrade: headline comparison (8 baselines), iso-recall / iso-cost tables, attack suites (text paraphrase; code identifier renaming + obfuscation levels), statistical tests (McNemar + clustered bootstrap), statistical Wilson-CI conformal coverage, cost analysis with batched latencies, figures ×6, execution log |
| `experiments/results/` | Legacy typed-decision pipeline results: ES-Plag pair-level results, Setu hybrid routing, recall-constrained comparison, dataset provenance audit, metric audit |
| Root | `final_verified_results.csv`, `methodology_verification.csv` |

## 7. Existing documentation, audits, frozen artifacts

- **Documentation set**: design docs under `research_upgrade/*.md` (six documents), `README.md`; narrative reports under `reports/` (11 files).
- **Audit set**: `research_audit/*.md` (8 files); `research_upgrade/audits/` incl. `phase_4_1_independent_verification.zip`, `phase_4_independent_verification.zip`, `final_research_source_audit.zip` + manifests.
- **Frozen artifacts**: SHA-256 manifests (`PHASE_4_1_SHA256_MANIFEST.json`, `FINAL_SHA256_MANIFEST.json`) and `governance.py::verify_baseline_integrity()`.
- **Missing / inaccessible information**:
  1. Full-text PDFs of the three designated papers are **not** in the repository (only verified bibliographic metadata, methodology summaries, and audited reproduction results in `literature/` + `research_audit/`). Page/section citations therefore reference the repository's own verified documentation, and the papers' *published* numbers are marked NR (not reported locally) unless captured by the repository docs.
  2. No human-annotated span-level ground truth exists for either dataset (declared in README §24 and `dossier_metrics`) — span-level precision/recall is structurally unavailable.
  3. The Kaggle "MIT Plagiarism Detection Dataset" hosting page is not archived locally (provenance is documented and re-derived from raw parquet files instead).
  4. `datasets/raw/ir_plag` tarball/zip provenance details beyond the extracted tree are not stored (dataset DOI not recorded in-repo).

## 8. Missing/inaccessible information statement

All gaps listed in §7 are propagated as **limitations** into the downstream phases instead of being
silently skipped, per the master execution rules.
