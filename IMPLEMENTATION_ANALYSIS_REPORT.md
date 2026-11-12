# Implementation Analysis Report — Three Journal Papers on Two Text Plagiarism Datasets

## Scope
Three peer-reviewed journal papers (2024–2026) implemented and evaluated on **two text-plagiarism datasets** (>2,000 rows each, genuine non-synthetic data), producing **6 paper × dataset combinations**.

## Papers (all peer-reviewed journals)

| # | Paper | Journal | DOI | Method |
|---|---|---|---|---|
| P1 | Amirzhanov et al. (2025) | *Frontiers in Computer Science* | 10.3389/fcomp.2025.1504725 | String-matching / lexical tradition — TF-IDF cosine + threshold |
| P2 | Sajid et al. (2025) | *PLoS ONE* | 10.1371/journal.pone.0319551 | Multi-feature extraction + Logistic Regression |
| P3 | Singh & Gupta (2026) | *Knowledge and Information Systems* | 10.1007/s10115-026-02837-8 | Semantic-syntactic hybrid (TF-IDF + char-Jaccard semantic proxy) |

## Datasets

| Dataset | Path | Rows |
|---|---|---|
| Dataset 1 — Repo text benchmark | `datasets/processed/text_plagiarism_benchmark.csv` | 19,666 |
| Dataset 2 — Quora Question Pairs (train) | `datasets/external/shreyas_repo/.../dataset/train.tsv` | 49,401 |
| Dataset 2 — Quora Question Pairs (test/dev) | `.../dataset/test.tsv`, `.../dataset/dev.tsv` | 8,000 each |

## Implementation code
- `reports/paper_wise_comparison/implement_three_papers.py` — full vectorized implementation
- `reports/paper_wise_comparison/implement_d1_all_three.py` — Dataset 1 runner
- `reports/paper_wise_comparison/implement_d2_all_three.py` — Dataset 2 runner

## Results CSV tables
- `reports/paper_wise_comparison/tables/three_papers_two_datasets_results.csv` — combined
- `reports/paper_wise_comparison/tables/dataset1_three_papers.csv`
- `reports/paper_wise_comparison/tables/dataset2_three_papers.csv`

## Key finding
**Sajid et al. (2025, PLoS ONE)** is the only method with positive MCC on both datasets (0.3452 on Dataset 1; 0.1975 on Dataset 2) — most robust across task difficulty. No single method wins all metrics on both datasets, reflecting the precision–recall–specificity trade-offs inherent in plagiarism detection.

## Branch
`feature/three-papers-two-datasets-implementation` — ready for merge into `main`.
