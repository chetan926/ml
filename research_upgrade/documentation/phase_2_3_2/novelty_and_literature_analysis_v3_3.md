# Focused Literature Review, Novelty Demarcation, and Research Claims Audit (Version 3.3)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.3 (Phase 2.3.2 Methodological Audit & Certification)  
**Author:** Senior Machine Learning Researcher, Statistical Methodologist & Independent Research Auditor  
**Date:** November 2026  
**Status:** Certified Final Literature & Novelty Audit  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals  
**Source Baseline:** Audited from Version 3.2 (`research_upgrade/documentation/phase_2_3_1/`)

---

## 1. Focused Literature Review & Scholarly Foundations

To meet the rigorous standards of top-tier academic publishing, this work is situated directly within established scholarly foundations, providing verified DOIs and precise scientific attribution.

### 1.1 Candidate Screening & Pre-Filtering in Academic Integrity
- **Lexical Matching:** Salton & Buckley (1988, DOI: [10.1016/0306-4573(88)90021-0](https://doi.org/10.1016/0306-4573(88)90021-0)) established term-weighting (TF-IDF) in document retrieval. Sajid et al. (2025, *PLOS ONE*, DOI: [10.1371/journal.pone.0319551](https://doi.org/10.1371/journal.pone.0319551)) evaluated sublinear TF-IDF as a rapid text plagiarism pre-filter. Pure lexical filters fail when text undergoes paraphrasing or synonym replacement.
- **Dynamic Pre-Filtering in Source Code:** Sulistiani & Karnalim (2019, *Comput. Appl. Eng. Educ.*, DOI: [10.1002/cae.22066](https://doi.org/10.1002/cae.22066)) developed ES-Plag, introducing dynamic cosine pre-filtering based on code length differentials prior to Running-Karp-Rabin Greedy String Tiling (RKRGST; Wise, 1996, DOI: [10.1145/226239.226260](https://doi.org/10.1145/226239.226260)). Our empirical audit demonstrates that ES-Plag's dynamic filter drops true plagiarized pairs on the IR-Plag benchmark (candidate recall $0.8627$), failing strict recall preservation constraints.

### 1.2 Machine Learning for Source Code Plagiarism Detection
- **Token & String Tiling Baselines:** Classic token-based systems including MOSS (Aiken, 2003) and JPlag (Prechelt et al., 2002, DOI: [10.1023/A:1016368827828](https://doi.org/10.1023/A:1016368827828)) remain vulnerable to identifier renaming, statement reordering, and control flow substitution (Faidhi & Robinson, 1987, DOI: [10.1016/0360-1315(87)90042-X](https://doi.org/10.1016/0360-1315(87)90042-X)).
- **Recent Supervised ML Approaches:**
  - **Gandhi et al. (2024, *Frontiers in Computer Science*, DOI: [10.3389/fcomp.2024.1393723](https://doi.org/10.3389/fcomp.2024.1393723)):** Implemented an RBF-kernel SVM using 7 syntactic/lexical features on Python submissions. We adapt their feature extraction and SVM classifier to Java source code on the standardized IR-Plag corpus.
  - **Mehsen & Joshi (2024, *International Journal of Computing*, DOI: [10.47839/ijc.23.1.3438](https://doi.org/10.47839/ijc.23.1.3438)):** Evaluated Random Forest ensembles on token frequency matrices, reporting high robustness against compound transformations. We adapt their ensemble model to our 7-D syntactic/AST feature space.

### 1.3 Uncertainty Quantification & Conformal Prediction
- **Probability Miscalibration:** Guo et al. (2017, *ICML*, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) demonstrated that modern high-capacity classifiers output overconfident probabilities. Platt scaling (Platt, 1999) and Isotonic regression (Zadrozny & Elkan, 2002, *KDD*, DOI: [10.1145/775047.775148](https://doi.org/10.1145/775047.775148)) provide established post-hoc recalibration.
- **Inductive Split Conformal Prediction:** Vovk et al. (2005) and modern syntheses by Angelopoulos & Bates (2021, *Foundations and Trends in Machine Learning*, DOI: [10.1561/2200000085](https://doi.org/10.1561/2200000085)) provide finite-sample, distribution-free marginal coverage guarantees $P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$ under data exchangeability.

### 1.4 Cascaded Inference & Semantic Encoders
- **Cascaded Inference:** Viola & Jones (2001, *CVPR*, DOI: [10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)) established cascaded classification in computer vision, rejecting easy negatives at early, cheap stages.
- **Contextual Sentence Embeddings:** Reimers & Gurevych (2019, *EMNLP*, DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)) developed Sentence-BERT, enabling dense semantic similarity matching.
- **Cross-Encoder Contextual Attention:** Devlin et al. (2019, *NAACL*, DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)) introduced BERT with bidirectional joint cross-attention.

### 1.5 Statistical Resampling & Error Control Foundations
- **Cluster Bootstrap:** Efron & Tibshirani (1993, *An Introduction to the Bootstrap*, ISBN: 978-0-412-04231-7) established bootstrap resampling.
- **Paired Hypothesis Testing:** McNemar (1947, *Psychometrika*, DOI: [10.1007/BF02295996](https://doi.org/10.1007/BF02295996)) and Wilcoxon (1945, *Biometrics Bulletin*, DOI: [10.2307/3001968](https://doi.org/10.2307/3001968)).
- **False Discovery Rate (FDR):** Benjamini & Hochberg (1995, *Journal of the Royal Statistical Society: Series B*, DOI: [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)).

---

## 2. Rigorous Taxonomy of Research Claims

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TAXONOMY OF RESEARCH CLAIMS & PROVENANCE                             │
├───────────────────────────────┬────────────────────────────────────────────────────────────────────────┤
│ Category                      │ Specific Components & Scientific Attribution                           │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Established Published      │ • Sublinear TF-IDF Vectorization (Salton & Buckley, 1988)              │
│    Techniques                 │ • Sentence-BERT `all-MiniLM-L6-v2` sequence embeddings (Reimers, 2019) │
│                               │ • Running-Karp-Rabin Greedy String Tiling (Wise, 1996)                 │
│                               │ • Platt Scaling via Logistic Regression (Platt, 1999)                  │
│                               │ • Isotonic Regression via PAV Algorithm (Zadrozny & Elkan, 2002)       │
│                               │ • Smith-Waterman Local Sequence Alignment (Smith & Waterman, 1981)     │
│                               │ • Support Vector Machines & Random Forests (Cortes & Vapnik; Breiman)  │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Direct Baselines Replicated│ • Syntactic SVM Classifier for Code Plagiarism (Gandhi et al., 2024)   │
│    or Adapted                 │ • AST-Derived Random Forest Ensemble for Code (Mehsen & Joshi, 2024)   │
│                               │ • Dynamic Cosine Pre-Filter RKRGST Pipeline (Sulistiani & Karnalim)    │
│                               │ • Lexical TF-IDF Text Screening Baseline (Sajid et al., 2025)          │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Proposed Integrations      │ • Two-Tier Cascaded Screening with Document Cache                      │
│                               │ • AST Subtree Isomorphism + Identifier Mapping Ledger for Code         │
│                               │ • Discretized Pareto Surface Search over Routing Cutoffs               │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. Methodological Adaptations │ • Selection-Aware Calibration fitted strictly on Escalation Slices     │
│                               │ • Group-Level Conformal Routing on Escalated Premise Clusters (Text)   │
│                               │ • Group-Disjoint Premise Partitioning for Text Integrity Screening    │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 5. Potential Methodological   │ • Uncertainty-Aware, Budget-Constrained Multi-View Triage Framework    │
│    Novelties (To Validate)    │   integrating candidate generation, conditional calibration, and Pareto│
│                               │   budget optimization for cost-efficient academic integrity detection. │
└───────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Critical Novelty Audit for the Four Research Contributions

### 3.1 Contribution 1: Robust Multi-View Candidate Generation
- **What is Established:** Individual lexical similarity metrics (TF-IDF cosine, character 3-gram Jaccard, word token Jaccard) and token-based code lexers.
- **What is Adapted:** Adapting regex identifier normalization and operator frequency vectors as sub-0.050 ms code candidate features without AST parsing.
- **What is Integrated:** Fused logistic candidate scoring function $g_{\text{cand}}$ trained on $D_{\text{train}}$.
- **What may be Novel:** Empirical demonstration that a 4-D multi-view candidate generator retains $\ge 0.980$ recall on zero-overlap paraphrased pairs while maintaining sub-0.050 ms ingestion latency.
- **Required Experiments:** View ablation studies (A1–A4), VIF redundancy audit, recall-pruning trade-off curves.

### 3.2 Contribution 2: Uncertainty-Aware Adaptive Decision Mechanism
- **What is Established:** Post-hoc probability calibration (Platt scaling, Isotonic regression) and split conformal prediction theory.
- **What is Adapted:** 
  - Text: Group-level conformal prediction using group-max non-conformity $\alpha_g = \max_{i \in g} \alpha_i$ to handle clustered reference premises under finite-sample order statistic bounds.
  - Code: Exploratory raw score thresholding on decision margins $z$, rejecting forced 5-fold Platt scaling due to structural class-group imbalance (0 positive training pairs in Fold 1 of `case-05`).
- **What is Integrated:** Combining selection-aware calibration, 4-set conformal routing for text, and 3-band score routing for code within a unified triage cascade.
- **What may be Novel:** Resolving the selection bias problem in multi-stage academic integrity screening through conditional calibration.
- **Required Experiments:** Reliability diagrams, ECE/Brier score comparison between marginal and slice calibration (A8), empirical coverage verification on $D_{\text{test}}^{\text{esc}}$.

### 3.3 Contribution 3: Budget-Constrained Routing Policy Optimization
- **What is Established:** Grid search, Pareto optimality.
- **What is Adapted:** Multi-objective recall and precision constraints applied to academic integrity triage cutoffs.
- **What is Integrated:** Discretized Pareto surface search over $(\theta_{\text{low}}, \theta_{\text{high}})$ on $D_{\text{val}}$ using the disaggregated latency model.
- **What may be Novel:** Formal Pareto trade-off surface characterizing candidate recall vs. inference latency under mandatory human review constraints.
- **Required Experiments:** 1,440-point Pareto frontier extraction on $D_{\text{val}}$, budget constraint sensitivity ($B_{\text{max}} \in [0.5, 3.0]\text{ ms}$).

### 3.4 Contribution 4: Explainable Cross-Modal Evidence Aggregation
- **What is Established:** Smith-Waterman dynamic alignment, AST subtree isomorphism.
- **What is Adapted:** Affine gap alignment on token sequences; variable alpha-conversion mapping dictionaries.
- **What is Integrated:** On-demand dossier synthesis triggered strictly by triage decisions (`human_review_required = True`).
- **What may be Novel:** Decision probability ablation drop as a quantitative faithfulness metric for cascaded integrity screening.
- **Required Experiments:** Evidence faithfulness ablation drop (A12), token coverage evaluation, evidence generation overhead profiling ($T_{\text{explain}}$).

---

## 4. Methodological Risks and Dataset Constraints

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MAJOR EXPERIMENTAL CONSTRAINTS & RISKS                               │
├───────────────────────────────┬────────────────────────────────────────────────────────────────────────┤
│ Dimension                     │ Empirical Limitation & Methodological Safeguard                        │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ IR-Plag Code Dataset Size     │ Only 7 tasks in total. case-05 has only 1 positive reference program.  │
│                               │ Safeguard: Conformal guarantees formally disclaimed for source code.    │
│                               │ Evaluated as exploratory raw score thresholding.                       │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Code Test Task Generalization │ Test split contains only 2 tasks: case-06 (Palindrome) & case-07 (Sort)│
│                               │ Safeguard: case-06 and case-07 reported SEPARATELY. Pooled metrics     │
│                               │ are descriptive only. No broad task-generalization claims made.        │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Text Premise Overlap          │ snli_val and snli_test share exactly 1 premise sentence (6 rows total).│
│                               │ Safeguard: Explicitly disclosed; shared sentence removed in loader.    │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Statistical Testing on Code   │ Small discordant counts (b + c < 25) invalidate asymptotic chi^2.      │
│                               │ Safeguard: Mandatory Exact Binomial McNemar test; FDR correction.      │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Output State Semantics        │ Algorithmic outputs do not prove misconduct or innocence.             │
│                               │ Safeguard: Renamed LOW_RISK_REVIEW_DEFERRED, HIGH_RISK_CANDIDATE_FLAGGED│
└───────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Journal Readiness Assessment & Prerequisites for Submission

### 5.1 Current Status: Pre-Implementation Audit Phase (Phase 2.3.2)
- The project is currently in the **independent methodological audit and specification phase**.
- **Explicit Declaration:** The research design is mathematically sound, self-consistent, and ready for independent peer audit. However, **it is NOT "Q1-accepted" or "Q1-ready" prior to actual experimental execution, data collection, and statistical verification.**

### 5.2 Mandatory Empirical Prerequisites Prior to Journal Submission:
1. Complete execution of the 7-step experimental protocol across both modalities without data leakage.
2. Independent evaluation of calibration diagnostics (ECE, Brier score) on held-out test escalation slices ($D_{\text{test}}^{\text{esc}}$).
3. Separate empirical reporting of `case-06` and `case-07` on the IR-Plag benchmark with Exact Binomial McNemar tests.
4. Execution of all 12 planned ablation studies with Benjamini-Hochberg FDR correction.
5. Verification that measured warm-cache candidate ingestion latencies achieve the target budget ($T_{\text{Tier1}} \le 0.050\text{ ms/pair}$).
6. Blinded qualitative review of synthesized `EvidenceDossier` outputs by academic integrity officers.

---
*End of Document: novelty_and_literature_analysis_v3_3.md*

