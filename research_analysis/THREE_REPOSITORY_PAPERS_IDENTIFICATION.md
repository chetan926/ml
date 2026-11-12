# Phase 8 — Identification of the Three Repository Research Papers

## 8.1 Repository investigation (where the papers were searched)

Recursive search covered `literature/`, `research_audit/`, `reports/`, `research_upgrade/` (design
docs incl. `novelty_and_literature_analysis.md`), `docs`-equivalents, notebooks, and all bibliography
files. Findings:

- **No full-text PDFs of the three selected papers exist in the repository.** The only PDF is
  `reports/final_research_report.pdf` — the project's own report.
- The definitive selection record is `research_audit/final_literature_verification.md` §2
  ("Granular Verification of the **Three Recent Published Methodologies**", recency requirement
  "three methods published in 2024–2026"), corroborated by `literature/selected_papers.csv`
  (M1/M2/M3 with `category = Recent Methodology`) and `literature/methodology_verification.md`.
- Supporting (non-designated) reference works exist in the same files: ES-Plag 2019 (reference
  methodology), Salton & Buckley 1988 / Sajid et al. 2025 (TF-IDF baseline lineage), Faidhi &
  Robinson 1987 (Jaccard lineage).

**Ambiguity check.** `selected_papers.csv` lists six entries; the three-way designation is
disambiguated by the audit's explicit "Three Recent Methodologies" verdict table, so the selected set
is: **Setu (2025), Gandhi (2024), Mehsen (2024)**. This is a documented, non-guessed selection; the
ES-Plag/Salton/Faidhi entries are tagged `Reference Methodology`/`Traditional Baseline (Supporting)`
and are therefore excluded from the three-paper set.

## 8.2 Paper identification records

| # | Exact title | Authors | Year | Venue | DOI | Repository path (evidence) | Document verification status |
|---|---|---|---|---|---|---|---|
| 1 | A comprehensive strategy for identifying plagiarism in academic submissions | Deblina Mazumder Setu, Tania Islam, Md. Erfan, Samrat Kumar Dey, Md. Rashid Al Asif, Md. Golam Rabiul Alam | 2025 | Journal of Umm Al-Qura University for Engineering and Architecture (Springer Nature) | 10.1007/s43995-025-00108-1 | `literature/selected_papers.csv` row M1_SETU_2025; `research_audit/final_literature_verification.md` §2.1; `literature/methodology_verification.md` (M1/M2) | **Metadata verified; full-text PDF absent** — methodology and reproduced results recorded by repository audits; reproduction in `src/bert_similarity.py` |
| 2 | A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings | Nandini Gandhi, Kaushik Gopalan, Prajish Prasad | 2024 | Frontiers in Computer Science, vol. 6, art. 1393723 | 10.3389/fcomp.2024.1393723 | `literature/selected_papers.csv` row M2_GANDHI_2024; audit §2.2; implementation `src/code_plagiarism.py` | **Metadata verified; full-text PDF absent**; cross-language adaptation documented (Python→Java) |
| 3 | Detection of Source Code Plagiarism Utilizing an Approach Based on Machine Learning | Raddam Sami Mehsen, Hiren D. Joshi | 2024 | International Journal of Computing 23(1): 78–84 | 10.47839/ijc.23.1.3438 | `literature/selected_papers.csv` row M3_MEHSEN_2024; audit §2.3; implementation `src/code_random_forest.py` | **Metadata verified; full-text PDF absent**; audited parameter note (`max_depth=6`, not 10) |

## Consequence for Phases 9–13 (honesty rule)

Because full texts are absent, all paper-level claims use the repository's **verified metadata and
audited reproduction records** as evidence. Values from the papers' *own* published experiments are
marked **NR (Not Reported locally)** unless the repository audits captured them; repository
reproduction results are always labeled as such and never conflated with the papers' published
numbers. No replacement papers were searched for or introduced.
