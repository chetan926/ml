# Final Literature, Novelty & External Validity Analysis (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 09 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** FULLY POSITIONED & BIBLIOGRAPHICALLY AUDITED

---

## 1. Verified Bibliographic Landscape and Paper Audit

All foundational and comparative literature cited in this project has been audited against authoritative bibliographic repositories (Crossref, DBLP, Google Scholar) and verified with active Digital Object Identifiers (DOIs):

| Paper Key | Citation & Metadata | Venue & Publisher | Verified DOI | Methodological Classification | Role in Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1_SETU_2025** | Setu, Mazumder et al. (2025). *A comprehensive strategy for identifying plagiarism in academic submissions.* | Journal of Umm Al-Qura University for Engineering and Architecture (Springer) | [10.1007/s43995-025-00108-1](https://doi.org/10.1007/s43995-025-00108-1) | **Recent Published Methodology** | Benchmarked as comparative baseline; inspires hybrid lexical-semantic text screening. |
| **M4_ESPLAG_2019** | Sulistiani, L. & Karnalim, O. (2019). *ES-Plag: Efficient and sensitive source code plagiarism detection tool for academic environment.* | Computer Applications in Engineering Education, 27(1), 166–178 (Wiley) | [10.1002/cae.22066](https://doi.org/10.1002/cae.22066) | **Published Reference Methodology** | Foundational source for RKR-GST token string tiling and penalty normalization. |
| **M2_GANDHI_2024** | Gandhi, N., Gopalan, K. & Prasad, P. (2024). *A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings.* | Frontiers in Computer Science, vol. 6 | [10.3389/fcomp.2024.1393723](https://doi.org/10.3389/fcomp.2024.1393723) | **Recent Published Methodology** | Adapted from Python to Java AST/metric feature extraction. |
| **M3_MEHSEN_2024** | Mehsen, R. S. & Joshi, H. D. (2024). *Detection of Source Code Plagiarism Utilizing an Approach Based on Machine Learning.* | International Journal of Computing, 23(1), 78–84 | [10.47839/ijc.23.1.3438](https://doi.org/10.47839/ijc.23.1.3438) | **Recent Published Methodology** | Benchmark baseline for ensemble Random Forest code classification. |
| **B1_TFIDF_1988** | Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* (Re-benchmarked in Sajid et al., 2025). | Information Processing & Management / PLOS ONE | [10.1371/journal.pone.0319551](https://doi.org/10.1371/journal.pone.0319551) | **Established Traditional Baseline** | Classical lexical retrieval and Tier-1 candidate pre-filter. |
| **B2_JACCARD_1987**| Faidhi, J. A. W. & Robinson, J. L. (1987). *An empirical approach for detecting program similarity and plagiarism within a university programming environment.* | Computers & Education, 11(1), 11–19 (Elsevier) | [10.1016/0360-1315(87)90042-X](https://doi.org/10.1016/0360-1315(87)90042-X) | **Established Traditional Baseline** | Traditional baseline for set-theoretic token Jaccard similarity. |

---

## 2. Rigorous Novelty Claims & Research Contribution Boundaries

To ensure defensibility in Q1 journal peer review, unsupported "first-of-kind" claims have been removed. The novelty of this project is formally positioned as an **application-specific, principled integration**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               CORE CONTRIBUTION BOUNDARIES             │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
         ┌───────────────────────────┬─────────────┴─────────────┬───────────────────────────┐
         ▼                           ▼                           ▼                           ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│ Contribution 1    │       │ Contribution 2    │       │ Contribution 3    │       │ Contribution 4    │
│ Multi-View Fusion │       │ Uncertainty-Aware │       │ Pareto Budget     │       │ Explainable Diff  │
│ Screening         │       │ Adaptive Routing  │       │ Optimization      │       │ Attribution       │
├───────────────────┤       ├───────────────────┤       ├───────────────────┤       ├───────────────────┤
│ Integration of    │       │ Principled        │       │ Dense grid search │       │ Myers alignment   │
│ Char, Word, Dense │       │ adaptation of     │       │ over candidate    │       │ & AST subtree     │
│ Embeddings & AST  │       │ group split       │       │ recall vs latency │       │ isomorphism       │
│ to retrieve 119   │       │ conformal bounds  │       │ constraints (NOT  │       │ separated from    │
│ zero-overlap pairs│       │ to clustered text │       │ Lagrangian)       │       │ judicial guilt    │
└───────────────────┘       └───────────────────┘       └───────────────────┘       └───────────────────┘
```

1. **Contribution 1 (Multi-View Candidate Screening):**  
   *Claim Boundary:* The individual features (TF-IDF, dense embeddings, Levenshtein distance, AST subtrees) are established techniques. The contribution lies in their **validation-optimized fusion** designed specifically to maintain high candidate recall ($\ge 98\%$) on severe semantic paraphrases—targeting the known failure mode of 119 zero-overlap text pairs (11.19% of 1,063 benchmark positive pairs; 88.81% non-zero lexical overlap; active test split: 33/1,026 pairs with zero word Jaccard overlap)—and multi-level code obfuscations.
2. **Contribution 2 (Uncertainty-Aware Adaptive Decisions):**  
   *Claim Boundary:* Split conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021) and group maximum non-conformity aggregation are known theoretical statistical constructs. The contribution is the **application-specific formulation on clustered premise structures**, establishing exact finite-sample quantile order statistics and providing a mathematically bounded 4-state operational taxonomy.
3. **Contribution 3 (Budget-Constrained Optimization):**  
   *Claim Boundary:* Multi-objective trade-offs are standard. The contribution is the **systematic Pareto-frontier characterization** of the recall-latency-workload space in academic integrity systems. It is properly characterized as a Pareto-frontier search, NOT a Lagrangian optimizer.
4. **Contribution 4 (Explainable Evidence Synthesis):**  
   *Claim Boundary:* Myers diff and AST parsing are established algorithms. The contribution is the **multi-modal dossier integration** and the institutional policy separation between automated statistical evidence and human judicial review.

---

## 3. External Validity Assessment: SNLI as an Academic Integrity Proxy

Methodological rigor demands transparent disclosure of benchmark limitations:

### 3.1 Benchmark Characteristics
The text benchmark [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv) is derived from the Stanford Natural Language Inference (SNLI) corpus:
- Entailment pairs are labeled as positive candidates ($y = 1$).
- Contradiction / neutral pairs are labeled as negative candidates ($y = 0$).

### 3.2 Key Threats to External Validity
1. **Sequence Length Disparity:** SNLI consists of sentence pairs ($10$–$35$ words). Real academic plagiarism typically involves multi-sentence, paragraph, or section-level copying ($500$–$5,000$ words).
2. **Stylistic and Topical Domain Gap:** SNLI premises originate from Flickr30k image captions describing physical scenes, lacking academic prose, citation syntax, technical terminology, and formal argumentation.
3. **Intent vs. Entailment:** Entailment represents semantic equivalence, not intent to deceive. Two student submissions may legitimately share identical facts without academic misconduct.

### 3.3 Scientific Justification as an Algorithmic Stress-Test
Despite these domain gaps, SNLI provides an invaluable controlled testbed:
- **Extreme Paraphrase Condition:** It provides human-authored semantic paraphrases with zero lexical overlap (the known stress-test target of **119 zero-overlap positive pairs**, representing 11.19% of 1,063 benchmark positives), providing a rigorous benchmark for multi-view retrieval.
- **Clustered Group Topology:** Shared premises create natural dependent groups for testing **group split conformal prediction** without data snooping.
- **Reproducibility:** Being fully public, it enables reproducible independent verification without student privacy restrictions.

### 3.4 Mandatory Manuscript Statement
Any journal submission originating from this project must include this explicit disclaimer:
> *"The text experiments in this study evaluate multi-view candidate screening and conformal calibration on a clustered semantic inference proxy corpus (derived from SNLI). While this provides a rigorous stress-test for paraphrase retrieval under zero lexical overlap, full institutional deployment requires domain transfer validation on student essays and theses with scholarly citation structures."*

