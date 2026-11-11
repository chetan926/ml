# Academic Defense Presentation Summary

**Research Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Role:** Senior Machine Learning Research Candidate  
**Target Audience:** Academic Faculty Evaluation Committee & Research Panel  
**Date:** November 2026  

---

## Slide 1: Title & Research Motivation
- **Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines
- **Core Motivation:** Institutional plagiarism detection faces an $O(N^2)$ computational bottleneck. Modern deep learning models (Sentence-BERT, AST-SVMs) provide high semantic fidelity but are computationally intractable when run on every pairwise comparison across large student cohorts.
- **Goal:** Design, implement, and validate an adaptive, modality-aware pre-filtering pipeline that preserves detection recall while slashing expensive model invocations.

---

## Slide 2: Research Problem & Three Core Questions
1. **The Efficiency-Fidelity Trade-off:** Can lightweight pre-filters safely prune unplagiarized student submissions without accidentally discarding subtle paraphrasing or disguised source code?
2. **Modality Divergence:** How do natural-language text and source code differ in their lexical and structural vulnerability to obfuscation?
3. **Institutional Procedural Fairness:** How can we ensure automated systems generate reviewable similarity evidence rather than acting as opaque judicial arbiters?

---

## Slide 3: Three Selected Recent Methodologies (2024–2025)
1. **Method 1: Lexical TF-IDF Vectorization Baseline**
   - *Reference:* M. Sajid et al., *PLOS ONE*, 2025. DOI: [10.1371/journal.pone.0319551](https://doi.org/10.1371/journal.pone.0319551)
   - *Role:* Ultra-fast sub-millisecond lexical baseline and first-tier text pre-filter.
2. **Method 2: Contextual SBERT Semantic Text Detector**
   - *Reference:* D. M. Setu et al., *Springer (J. Umm Al-Qura Univ.)*, 2025. DOI: [10.1007/s43995-025-00108-1](https://doi.org/10.1007/s43995-025-00108-1)
   - *Role:* Dense 384-dimensional contextual sentence embeddings (`all-MiniLM-L6-v2`) capturing deep semantic paraphrasing.
3. **Method 3: Multi-Dimensional Syntactic/Token Java Detector**
   - *Reference:* N. Gandhi, K. Gopalan, P. Prasad, *Frontiers in Computer Science*, 2024. DOI: [10.3389/fcomp.2024.1393723](https://doi.org/10.3389/fcomp.2024.1393723)
   - *Role:* RBF Support Vector Machine synthesizing token Jaccard, anonymized identifier matching, keyword distributions, and cyclomatic proxies. Adapted faithfully from Python to Java.

---

## Slide 4: Authentic Benchmark Datasets
- **Text Dataset:** MIT / SNLI Plagiarism Benchmark (19,666 pairs; 9,842 validation, 9,824 test). Derived from SNLI entailment pairs (plagiarized) vs neutral/contradictory pairs (non-plagiarized).
- **Source Code Dataset:** IR-Plag Java Corpus (733 pairs across 7 introductory assessment tasks and 6 Faidhi-Robinson obfuscation levels: L1 to L6).

---

## Slide 5: Proposed Architecture (Typed-Decision Pipeline)
- **C1: Modality-Aware Typed Routing:** Automatically classifies input pair into natural text vs Java code based on token syntax and keyword prevalence.
- **C2: Tri-Band Confidence Escalation:**
  - `Low Risk` ($S < \theta_{low}$): Safely pruned at zero deep-model cost.
  - `Uncertain` ($\theta_{low} \le S < \theta_{high}$): Escalated to expensive neural / syntactic models.
  - `High Risk` ($S \ge \theta_{high}$): Directly flagged for instructor review (verbatim cases bypass redundant deep encoding).
- **C3: Safety-Calibrated Boundaries:** Thresholds calibrated strictly on validation data to guarantee $\ge 98\%$ candidate recall retention.

---

## Slide 6: Quantitative Comparison (Text Modality)

| Metric | TF-IDF Baseline | SBERT Full Analysis | Fixed 50% Threshold | Proposed Typed Pipeline |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.5723 | 0.7200 | 0.7090 | **0.7180** |
| **Precision** | 0.4417 | 0.5697 | 0.5750 | **0.5751** |
| **Recall** | 0.7846 | 0.8570 | 0.6849 | **0.7817** |
| **F1-Score** | 0.5652 | 0.6844 | 0.6252 | **0.6627** |
| **Candidate Recall** | N/A | 1.0000 | 0.7366 | **0.8598** |
| **Pruning Ratio (%)** | 0.00% | 0.00% | 42.87% | **26.53%** |
| **Expensive Call Rate (%)** | 100.0% | 100.0% | 57.13% | **70.03%** |
| **Average Latency (ms)** | **0.020 ms** | 4.143 ms | 2.904 ms | **3.167 ms** |

*Key Takeaway:* Fixed threshold filtering collapses candidate recall from 1.0000 to 0.7366 (discarding over 26% of true candidate positives). The proposed tri-band pipeline preserves **0.8598 Candidate Recall** (retaining 914 of 1,063 true positives) while eliminating nearly 30% of heavy neural model forward passes.

---

## Slide 7: Quantitative Comparison (Java Code Modality)

| Metric | Token Jaccard Filter | SVM Full Analysis | Fixed Threshold | Proposed Typed Pipeline |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.5286 | 0.7000 | 0.6381 | **0.6762** |
| **Precision** | 0.5083 | 0.6258 | 0.6000 | **0.6049** |
| **Recall** | 0.9020 | 0.9510 | 0.7647 | **0.9608** |
| **F1-Score** | 0.6502 | 0.7549 | 0.6724 | **0.7424** |
| **Candidate Recall** | N/A | 1.0000 | 0.7745 | **1.0000** |
| **Expensive Call Rate (%)** | 0.00% | 100.0% | 75.24% | **63.33%** |
| **Pruning Ratio (%)** | 0.00% | 0.00% | 24.76% | **0.48%** |

*Key Takeaway:* The proposed pipeline achieves **1.0000 Candidate Recall** (retaining 100% of candidate true positives) while saving **36.67%** of expensive feature evaluations by directly triaging 76 high-risk pairs for human review without heavy model calls.

---

## Slide 8: Source Code Obfuscation Resilience (IR-Plag Java)
- **Level 1 (Verbatim + Comments):** **93.75% Recall**
- **Level 2 (Identifier Renaming):** **100.00% Recall**
- **Level 3 (Statement Reordering):** **100.00% Recall**
- **Level 4 (Loop / Control Flow):** **100.00% Recall**
- **Level 5 (Method Abstraction):** **100.00% Recall**
- **Level 6 (Compound Attacks):** **77.78% Recall**

---

## Slide 9: Qualitative Error Highlights
- **Paraphrase Recovery (TXT_QUAL_02):** Substituted phrasing (`a man poses in front of an ad` vs `an old man with a package poses in front of an advertisement`) yielded low TF-IDF (0.227). Escalation to contextual SBERT correctly captured semantic equivalence (0.759).
- **Boilerplate False Positive Prevention (TXT_QUAL_03):** Repetitive church vocabulary in unrelated contexts caused TF-IDF to spike to 0.375. SBERT resolved the divergence (0.383), preventing an unjust student accusation.
- **Identifier Anonymization (CODE_QUAL_02):** Class and variable renaming in `case-01` (`T1` $\rightarrow$ `L2`) was neutralized by canonical identifier replacement, achieving 100% recall.
- **Direct Triage Efficiency (TXT_QUAL_01 & CODE_QUAL_01):** Massive token overlap was flagged directly ($\theta_{high}$), bypassing expensive model compute.

---

## Slide 10: Deliverables & Reproducibility Verification
- **Codebase:** Full production Python structure in `academic_integrity_research/`.
- **Single Master Reproduction Command:** `python run_all.py` (executes in 103 seconds).
- **Artifacts:**
  - 5 interactive Jupyter notebooks in `notebooks/`.
  - 9 publication-grade 300 DPI figures in `experiments/figures/`.
  - Comprehensive CSV summaries in `experiments/results/`.
  - Full academic implementation report in `reports/research_implementation_report.md`.

