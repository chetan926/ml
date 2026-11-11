# Reproducibility Checklist & Independent Verification Log

**Research Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Auditor:** Independent Research Evaluator  
**Audit Date:** November 2026  
**Operating Environment:** Windows x86_64, PowerShell 5.1 / 7, Python 3.11.18  
**Workspace Directory:** `d:\ml rearcch project ad`  

---

## 1. Environment & Dependency Audit

- [x] **Isolated Virtual Environment:** Dedicated virtualenv created at `.venv/` using Python 3.11.18.
- [x] **Core Scientific Libraries:**
  - NumPy: `2.4.6`
  - SciPy: `1.17.1`
  - Pandas: `3.1.0`
  - Scikit-learn: `1.4.2` (Verified signed binary distribution conforming to Windows Defender Application Control policies)
  - Matplotlib: `3.11.2`
  - Seaborn: `0.13.2`
  - PyArrow: `26.0.1`
- [x] **Deep Learning & NLP Libraries:**
  - PyTorch: `2.15.0+cpu`
  - Transformers: `5.22.1`
  - Sentence-Transformers: `6.2.0`
  - Pretrained Model Weights: `sentence-transformers/all-MiniLM-L6-v2` locally cached in offline mode (`HF_HUB_OFFLINE=1`).
- [x] **Requirements Specification:** `requirements.txt` accurately specifies all dependency constraints.

---

## 2. Dataset Integrity & Partitioning Audit

- [x] **Text Dataset (MIT / SNLI Benchmark):**
  - Raw files: `datasets/raw/text_mit_plagiarism/snli_val.parquet`, `snli_test.parquet`.
  - Processed file: `datasets/processed/text_plagiarism_benchmark.csv` (19,666 pairs).
  - Validation Split: 9,842 clean pairs.
  - Test Split: 9,824 clean pairs.
  - Label Mapping: Label 0 (entailment) $\rightarrow$ 1 (Plagiarized); Labels 1 (neutral) & 2 (contradiction) $\rightarrow$ 0 (Non-Plagiarized); Label -1 purged.
- [x] **Source Code Dataset (IR-Plag Java Corpus):**
  - Raw files: `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` (7 distinct tasks: `case-01` to `case-07`).
  - Processed file: `datasets/processed/code_plagiarism_benchmark.csv` (733 pairs).
  - Task-Disjoint Partitioning:
    - Training Set: `case-01`, `case-02`, `case-03`, `case-04` ($N = 416$).
    - Validation Set: `case-05` ($N = 107$).
    - Test Set: `case-06`, `case-07` ($N = 210$).
  - Obfuscation Breakdown: L1 (60), L2 (56), L3 (57), L4 (60), L5 (59), L6 (63).
- [x] **Data Leakage Guarantee:** Zero overlap exists between train, validation, and test splits across both modalities. All threshold tuning and feature transformations were fitted strictly on training/validation partitions.

---

## 3. Methodology & Paper Verification

- [x] **Method 1 (Lexical Baseline):** M. Sajid et al., *PLOS ONE*, 2025 (DOI: 10.1371/journal.pone.0319551). Sublinear TF-IDF + Cosine similarity.
- [x] **Method 2 (Contextual Semantic):** D. M. Setu et al., *Springer*, 2025 (DOI: 10.1007/s43995-025-00108-1). Sentence-BERT dense embeddings (`all-MiniLM-L6-v2`).
- [x] **Method 3 (Source Code Syntactic):** N. Gandhi et al., *Frontiers in Computer Science*, 2024 (DOI: 10.3389/fcomp.2024.1393723). Multi-dimensional feature extraction + SVM.
- [x] **Proposed Contribution:** Typed-Decision Pre-Filtering Pipeline combining modality routing with tri-band confidence escalation.

---

## 4. Execution & Verification Checklist

- [x] **Single-Command Execution:** `python run_all.py` executes without user interaction and completes in $\approx 95$ seconds.
- [x] **CSV Deliverables Generated:**
  - `experiments/results/text_experiments_summary.csv`
  - `experiments/results/code_experiments_summary.csv`
  - `experiments/results/dataset_statistics_summary.csv`
  - `experiments/results/code_obfuscation_level_recall.csv`
  - `experiments/results/threshold_sensitivity_data.csv` (Authentic validation sweep)
  - `experiments/results/qualitative_text_cases.csv`
  - `experiments/results/qualitative_code_cases.csv`
  - `final_verified_results.csv`
  - `methodology_verification.csv`
- [x] **Figure Artifacts (300 DPI Publication PNGs):**
  - `experiments/figures/01_accuracy_comparison.png`
  - `experiments/figures/02_precision_recall_f1_comparison.png`
  - `experiments/figures/03_confusion_matrices.png`
  - `experiments/figures/04_runtime_latency_comparison.png`
  - `experiments/figures/05_candidate_recall_vs_pruning_ratio.png`
  - `experiments/figures/06_expensive_detector_call_rate.png`
  - `experiments/figures/07_recall_vs_cost_tradeoff.png`
  - `experiments/figures/08_threshold_sensitivity_curves.png`
  - `experiments/figures/09_code_obfuscation_levels.png`
- [x] **Notebook Deliverables:** 5 fully documented Jupyter Notebooks in `notebooks/` (`01` through `05`).
- [x] **Academic Documentation:** 5 complete research reports in `reports/`.

---

## 5. Audit Verdict

| Evaluation Category | Audit Status | Auditor Notes |
| :--- | :---: | :--- |
| **Authenticity of Methodologies** | **PASSED** | 3 verified peer-reviewed publications from 2024 & 2025. |
| **Dataset Authenticity** | **PASSED** | Verified genuine SNLI and IR-Plag benchmarks; no fabricated samples. |
| **Split & Leakage Integrity** | **PASSED** | Strictly task-disjoint code split and independent text benchmark splits. |
| **Mathematical Precision** | **PASSED (FIXED)** | Candidate recall formula audited and corrected to evaluate pre-filter candidate retention. |
| **Visualization Veracity** | **PASSED (FIXED)** | Sensitivity curves re-plotted from genuine empirical validation sweep data. |
| **Reproducibility** | **PASSED** | Fully autonomous, offline-capable single reproduction command (`python run_all.py`). |

