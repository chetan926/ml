# Phase 3 Experimental Evaluation & Test Execution Plan (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 16 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** FULLY SPECIFIED & READY FOR PHASE 3 EXECUTION

---

## 1. Experimental Conditions and Test Matrix

The Phase 3 experimental campaign evaluates eight structured experimental conditions designed to rigorously test all four research contributions:

| Exp ID | Experimental Condition | Modality | Dataset Split / Population | Primary Evaluation Metrics | Purpose & Hypothesis Tested |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **E1** | Classical Baseline Evaluation | Text & Code | $D_{\text{test}}$ / IR-Plag Held-Out | Precision, Recall, Macro-F1, Latency | Benchmarking BM25, TF-IDF, and Jaccard. |
| **E2** | Dense Bi-Encoder Baseline | Text | $D_{\text{test}}$ | Precision, Recall, Macro-F1, Latency | Benchmarking uncalibrated `all-MiniLM-L6-v2`. |
| **E3** | Uniform Cross-Encoder Baseline | Text & Code | $D_{\text{test}}$ / IR-Plag Held-Out | Precision, Recall, Macro-F1, Latency | Upper bound on detection accuracy; upper bound on cost. |
| **E4** | Multi-View Candidate Screening | Text & Code | $D_{\text{val}}, D_{\text{test}}$ | $\text{Recall}@K$, Pruning Ratio $\eta$, Zero-Overlap Recall | Tests $H_1$ (Recall $\ge 98\%$, empirical evaluation on the known 119 zero-overlap text positive pairs). |
| **E5** | Uncertainty Probability Calibration | Text | $D_{\text{test}}^{\text{esc}}$ | ECE ($M=10$), MCE, Brier, Clipped NLL | Tests $H_2$ (Calibration reliability on escalation slice). |
| **E6** | Group Split Conformal Prediction | Text | $D_{\text{test}}^{\text{esc}}$ | Empirical Coverage $\hat{C}$, Cardinality, Abstention | Tests $H_3$ (Marginal coverage $\ge 95\%$ under $\epsilon = 0.05$). |
| **E7** | Multi-Objective Pareto Routing | Text & Code | $D_{\text{val}}, D_{\text{test}}$ | Expected Latency, Macro-F1, FLOP Reduction | Tests $H_4$ ($\ge 80\%$ latency reduction vs uniform model). |
| **E8** | Full Multi-View Ablation Study | Text & Code | $D_{\text{test}}$ | Candidate Recall Delta ($\Delta R$), Latency Delta | Quantifies contribution of each individual view. |

---

## 2. Experimental Execution Protocol and Order of Runs

Phase 3 execution will follow an automated, scripted pipeline to guarantee that thresholds are frozen before calibration and that test data is evaluated strictly once:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 3 EXPERIMENTAL EXECUTION PIPELINE                         │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
    ┌────────────────────────────────────┴────────────────────────────────────┐
    ▼                                                                         ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ Run Step 1: Baseline Reproduction     │   │ Run Step 2: Multi-View Training       │
│ • Execute E1, E2, E3 on test splits   │   │ • Fit fusion model w* on D_train      │
│ • Record baseline latency and FLOPs   │   │ • Screen candidates on D_val          │
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Run Step 3: Pareto Frontier Search & Threshold Freezing                         │
│ • Execute dense grid search over D_val                                          │
│ • Select optimal thresholds (theta_low*, theta_high*, gamma_lower*, gamma_upper*) │
│ • PERMANENTLY FREEZE ALL THRESHOLDS                                             │
└─────────────────────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Run Step 4: Calibration on Escalated Calibration Splits                          │
│ • Route D_cal-prob through Stage 1; escalate ambiguous pairs                    │
│ • Fit Platt scaling calibrator (A*, B*) on escalated D_cal-prob (Text only)     │
│ • Route D_cal-conf through Stage 1; compute exact conformal quantile q_hat      │
└─────────────────────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Run Step 5: Final Untouched Test Evaluation                                     │
│ • Ingest untouched D_test (3,000 pairs)                                         │
│ • Route through Stage 1; evaluate Stage 2 on escalation slice                   │
│ • Compute ECE, MCE, Brier, clipped NLL, coverage, and end-to-end latency        │
│ • Run group-aware bootstrap (B = 1,000) and McNemar discordance tests           │
└─────────────────────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Run Step 6: Artifact Synthesis & Table Export                                   │
│ • Generate LaTeX tables, reliability diagrams, and Pareto curves                │
│ • Synthesize attribution dossiers for all flagged test pairs                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Test Harness and Automated Test Suites

A comprehensive `pytest` test harness will be implemented under `tests/`:

1. **`tests/test_governance.py` (Governance Test):**  
   Verifies that no files in `src/` have been altered and that all SHA-256 hashes match [repository_sha256_manifest.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/repository_sha256_manifest.csv) under status `CURRENT_SNAPSHOT_HASHED`.
2. **`tests/test_text_loader.py` (Text Loader Test):**  
   Verifies that `TextBenchmarkDataLoader` extracts exact partition counts (`train`: 4,000; `val`: 1,500; `cal_prob`: 1,000; `cal_conf`: 1,000; `test`: 3,000) with **zero premise overlap**.
3. **`tests/test_code_loader.py` (Code Loader Test):**  
   Verifies that `CodeBenchmarkDataLoader` correctly loads Policy A (720 pairs) and Policy B (709 pairs), and enforces Case-06 denominators (103 for Policy A, 101 for Policy B).
4. **`tests/test_feature_extractors.py` (Feature Extractors Test):**  
   Verifies that all multi-view features strictly lie in $[0, 1]$ and handle empty strings without crashing.
5. **`tests/test_conformal.py` (Conformal Test):**  
   Verifies that the exact order statistic $k = \lceil (n+1)(1-\epsilon) \rceil$ is calculated correctly and that empty prediction sets map to `ABSTAIN_UNRESOLVED`.
6. **`tests/test_calibration_metrics.py` (Metrics Test):**  
   Verifies ECE uniform binning ($M=10$), zero contribution for empty bins, and probability clipping for NLL ($[10^{-15}, 1 - 10^{-15}]$).

---

## 4. Expected Publication-Ready Deliverables

All experiment outputs will be saved automatically into structured machine-readable directories:
- `research_upgrade/results/tables/`:
  - `table1_candidate_screening_performance.tex`
  - `table2_calibration_and_uncertainty_metrics.tex`
  - `table3_conformal_coverage_efficiency.tex`
  - `table4_pareto_cost_latency_comparison.tex`
  - `table5_multiview_ablation_analysis.tex`
- `research_upgrade/results/figures/`:
  - `fig1_two_stage_architecture_diagram.pdf`
  - `fig2_reliability_diagrams_pre_post_calibration.pdf`
  - `fig3_pareto_frontier_latency_recall.pdf`
  - `fig4_conformal_set_size_distribution.pdf`
  - `fig5_attribution_dossier_samples.pdf`

---

## 5. First Phase 3 Implementation Task

Upon receiving user authorization to begin Phase 3, the initial implementation task will execute:

```powershell
# Task 1: Initialize research_upgrade/src/, verify governance, and build verified loaders
.venv\Scripts\python -m pytest tests/test_governance.py
.venv\Scripts\python research_upgrade/src/text_data_loader.py --verify-partitions
.venv\Scripts\python research_upgrade/src/code_data_loader.py --verify-denominators
```

