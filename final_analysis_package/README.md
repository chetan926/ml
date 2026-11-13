# Final Analysis Package — Typed Decision Pipeline for Paraphrase Detection

## Overview

This package contains the complete analysis for a research study comparing a **Proposed Typed Decision Pipeline** (integrating four original contributions C1–C4) against **three journal-published baseline methods** on **two original, non-synthetic datasets**.

**Branch:** `arena/e11d367e-ml` → merged into `main` (PR #12, commit `54c391cc`)

---

## Key Results (Out-of-Sample Test Sets)

| Dataset | Proposed MCC | Amirzhanov (2025) | Sajid (2025) | SinghGupta (2026) | Proposed > All Baselines? |
|---------|-------------|-------------------|--------------|-------------------|---------------------------|
| Dataset 1 (9,824 pairs) | **0.357970** | 0.247968 | 0.355919 | 0.213099 | ✅ Yes (6/6 MCC wins) |
| Dataset 2 (8,000 pairs) | **0.206178** | -0.004626 | 0.196047 | -0.015401 | ✅ Yes (6/6 MCC wins) |

---

## Package Structure

```
final_analysis_package/
├── datasets/                          # Original, non-synthetic datasets
│   ├── dataset1_text_plagiarism_benchmark.csv   # 19,666 rows (MSRP-style)
│   ├── dataset2_quora_train.tsv                 # 49,401 rows (Quora QP)
│   ├── dataset2_quora_test.tsv                  # 8,000 rows (held-out)
│   └── dataset2_quora_dev.tsv                   # 8,000 rows (validation)
├── source_code/
│   ├── baselines/                              # Three paper implementations + C1-C4
│   │   ├── implement_d1_all_three.py
│   │   ├── implement_d2_all_three.py
│   │   ├── implement_three_papers.py
│   │   └── four_new_contributions_c1c4.py
│   ├── proposed/                               # Proposed typed solution
│   │   └── proposed_typed_solution.py
│   └── generators/                             # Report generators
│       ├── generate_qualitative.py
│       ├── generate_linear_graph.py
│       ├── build_tables.py
│       └── build_qualitative.py
├── analysis/
│   ├── tables/                                 # 24 CSV result tables
│   │   ├── proposed_typed_solution_results.csv
│   │   ├── proposed_vs_baselines.csv
│   │   ├── ablation_study.csv
│   │   ├── four_contributions_results.csv
│   │   ├── qualitative_analysis.csv
│   │   └── ... (15 paper-vs-proposed tables)
│   ├── qualitative/                            # Qualitative analysis (180 rows)
│   ├── ablation/                               # Ablation study
│   └── comparison/                             # Proposed vs baselines
├── charts/
│   ├── metrics/                                # 7 metric charts (Accuracy, F1, MCC, ROC, etc.)
│   ├── summary/                                # Ablation + proposed-vs-baselines charts
│   └── linear_graphs/                          # C1-C4 linear graphs
└── docs/
    ├── dataset_analysis.md                     # Full dataset breakdown
    ├── novelty_and_contributions.md            # C1-C4 originality + effectiveness
    ├── complete_analysis_report.md             # Master report with all tables
    └── q1_suitability_analysis.md              # Q1 verdict: SUITABLE
```

---

## How to Reproduce

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run C1–C4 Contributions
```bash
python source_code/baselines/four_new_contributions_c1c4.py
```
Output: `analysis/tables/four_contributions_results.csv`, `charts/` (8 PNGs)

### Run Proposed Typed Solution
```bash
python source_code/proposed/proposed_typed_solution.py
```
Output: `analysis/tables/proposed_typed_solution_results.csv`, `analysis/tables/ablation_study.csv`, `analysis/tables/proposed_vs_baselines.csv`, `charts/metrics/`, `charts/summary/`

### Generate Qualitative Analysis
```bash
python source_code/generators/generate_qualitative.py
```
Output: `analysis/qualitative/qualitative_analysis.csv` (180 rows)

### Generate Linear Graphs for C1–C4
```bash
python source_code/generators/generate_linear_graph.py
```
Output: `charts/linear_graphs/q1_linear_graph_*.png`

---

## Deliverables Included

| Deliverable | Location |
|-------------|----------|
| Datasets (original, non-synthetic) | `datasets/` |
| Dataset analysis (train/val/test counts, splits) | `docs/dataset_analysis.md` |
| Three baseline paper implementations (source code) | `source_code/baselines/` |
| Proposed typed decision pipeline (source code) | `source_code/proposed/` |
| Quantitative results (all metrics, both datasets) | `analysis/tables/` |
| Proposed vs baselines comparison (6/6 MCC wins) | `analysis/tables/proposed_vs_baselines.csv` |
| Ablation study table | `analysis/tables/ablation_study.csv` |
| Qualitative analysis (12 methods × 15 dimensions) | `analysis/qualitative/qualitative_analysis.csv` |
| Pictorial Q1 graphs (7 metrics + 2 summary) | `charts/metrics/`, `charts/summary/` |
| Linear graph for C1–C4 contributions | `charts/linear_graphs/` |
| Contributions effectiveness analysis | `docs/novelty_and_contributions.md` |
| Q1 suitability analysis (verdict: SUITABLE) | `docs/q1_suitability_analysis.md` |
| Complete analysis report | `docs/complete_analysis_report.md` |

---

## Constraint Compliance

- ✅ Three baseline papers are journal-only
- ✅ Four contributions (C1–C4) are original (not from any paper)
- ✅ Proposed outperforms all baselines on MCC on both datasets
- ✅ Datasets are original/non-synthetic (verified by direct inspection)
- ✅ Absolute table, tabular analysis, pictorial analysis, qualitative analysis included
- ✅ Original values (not fake) — all computed from real data
- ✅ Q1 suitability analysis included (verdict: SUITABLE)
- ✅ Linear graph for four contributions included
- ✅ All metrics evaluated (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, MCC)
