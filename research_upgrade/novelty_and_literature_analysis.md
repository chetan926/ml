# Focused Literature Review, Novelty Demarcation, and Research Claims Audit
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Complete Literature & Novelty Audit  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Focused Literature Review

A rigorous submission to a Q1 journal requires establishing the exact scientific context of prior art and transparently delineating existing techniques, adaptations, and newly proposed mechanisms.

### 1.1 Candidate Generation & Pre-Filtering in Academic Integrity
- **Lexical Matching:** Early foundations by Salton & Buckley (1988) established term-frequency inverse-document-frequency (TF-IDF) as the gold standard for sparse text retrieval. In academic plagiarism, Sajid et al. (2025) benchmarked sublinear TF-IDF as a rapid candidate filter. However, lexical methods fail when students paraphrase or replace vocabulary.
- **Locality-Sensitive Hashing & MinHash:** Broder (1997) introduced MinHash for web document deduplication. While asymptotic complexity is sub-quadratic, MinHash requires shared $k$-shingles. When synonym replacement produces zero shared $k$-grams, MinHash suffers the identical zero-overlap failure mode as TF-IDF.
- **Dynamic Pre-Filtering in Source Code:** Sulistiani & Karnalim (2019) introduced ES-Plag, utilizing a dynamic cosine pre-filter based on submission length ratios to prune pairs prior to Running-Karp-Rabin Greedy String Tiling (RKRGST). However, our Phase 1 audit demonstrated that ES-Plag's dynamic filter drops 14 positive pairs (candidate recall $0.8627$), violating strict candidate preservation constraints.

### 1.2 Machine Learning for Source Code Plagiarism
- **Token & String Tiling Baselines:** Classic tools such as MOSS (Aiken, 2003) and JPlag (Prechelt et al., 2002) rely on winnowing and greedy string tiling. They are notoriously vulnerable to identifier renaming, statement reordering, and loop replacement (Faidhi & Robinson, 1987).
- **Recent Supervised ML Approaches:**
  - **Gandhi et al. (2024, *Frontiers in Computer Science*):** Proposed an RBF-kernel Support Vector Machine over 7 lexical and syntactic features on Python submissions, reporting resilience against variable renaming.
  - **Mehsen & Joshi (2024, *International Journal of Computing*):** Implemented Random Forest ensembles on token frequency vectors, demonstrating that multi-tree aggregation handles compound transformations more effectively than single decision boundaries.

### 1.3 Uncertainty Quantification & Conformal Prediction
- **Calibration in Machine Learning:** Guo et al. (2017) demonstrated that modern deep neural networks and high-capacity classifiers are severely miscalibrated, often outputting overconfident probabilities for incorrect predictions. Platt (1999) and Zadrozny & Elkan (2002) formulated Platt scaling (logistic calibration) and Isotonic regression as post-hoc remedies.
- **Selective Classification (Reject Option):** Geifman & El-Yaniv (2017, 2019) formalized selective classification with risk-coverage guarantees.
- **Split Conformal Prediction:** Vovk et al. (2005) and modern formulations by Angelopoulos & Bates (2021) established inductive split conformal prediction, providing finite-sample, distribution-free marginal coverage guarantees $P(Y \in \mathcal{C}(X)) \ge 1 - \epsilon$ under data exchangeability. To date, conformal prediction has rarely been formalized for budget-constrained academic integrity screening.

### 1.4 Cost-Sensitive Cascades & Early Exiting
- **Cascade Classifiers:** Viola & Jones (2001) pioneered cascaded classification in computer vision, rejecting negative instances at early, cheap stages. In NLP, early-exit transformers (e.g., DeeBERT, FastBERT) dynamically terminate inference. However, existing academic plagiarism pipelines lack a formal Lagrangian framework that optimizes thresholds directly against monetary/compute budgets while preserving high candidate recall.

---

## 2. Rigorous Demarcation of Research Contributions

To prevent scientific overclaiming, the table below partitions every component of the proposed framework into:
1. **Existing Published Techniques** (Unmodified foundations)
2. **Adaptations of Published Techniques** (Modified for our cross-modal setting)
3. **New Proposed Mechanisms** (Algorithmic / theoretical novelties)
4. **Experimental Contributions** (Empirical investigations and insights)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TAXONOMY OF RESEARCH CLAIMS & PROVENANCE                                     │
├───────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┤
│ Category                      │ Specific Components & Scientific Attribution                                   │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ 1. Existing Published         │ • Sublinear TF-IDF Vectorization (Salton & Buckley, 1988)                      │
│    Techniques                 │ • Sentence-BERT `all-MiniLM-L6-v2` contextual embeddings (Reimers et al., 2019)│
│                               │ • Running-Karp-Rabin Greedy String Tiling (Wise, 1996)                         │
│                               │ • Platt Scaling via Logistic Regression (Platt, 1999)                          │
│                               │ • Isotonic Regression via Pair-Adjacent Violators (Zadrozny & Elkan, 2002)     │
│                               │ • Split Conformal Prediction Quantile Calibration (Angelopoulos & Bates, 2021) │
│                               │ • Expected Calibration Error & Brier Score metrics (Guo et al., 2017)          │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ 2. Adaptations of Published   │ • Gandhi et al. (2024) SVM: Adapted from Python scripts to standardized Java   │
│    Techniques                 │   AST feature extraction over the IR-Plag benchmark.                           │
│                               │ • Mehsen & Joshi (2024) Random Forest: Adapted from private corpus TF-IDF      │
│                               │   matrices to our 7-D syntactic/AST feature space.                             │
│                               │ • Setu et al. (2025) Hybrid: Adapted from monolithic full-evaluation into      │
│                               │   a staged pipeline (Condition E2).                                            │
│                               │ • Sulistiani & Karnalim (2019) ES-Plag: Reimplemented faithfully in Python     │
│                               │   with 9 passing unit tests for Java IR-Plag benchmark evaluation.             │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ 3. New Proposed               │ • Multi-View Candidate Generator: Fusing orthogonal character n-grams, word    │
│    Mechanisms                 │   TF-IDF, and fast embedding proxies to eliminate the 10.57% zero-overlap     │
│                               │   text failure mode.                                                           │
│                               │ • Conformal-Bounded Tri-Band Router: Mapping distribution-free conformal       │
│                               │   prediction sets directly to institutional triage states (Pruned,             │
│                               │   Escalated, Flagged).                                                         │
│                               │ • Budget-Constrained Lagrangian Optimizer: Formalizing per-pair computational  │
│                               │   latency as an explicit optimization constraint on the Pareto frontier.      │
│                               │ • Cross-Modal Evidence Aggregator: A unified EvidenceDossier schema combining  │
│                               │   Smith-Waterman text alignment and AST subtree isomorphism.                   │
├───────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
│ 4. Experimental               │ • Empirical identification and resolution of the 119 zero-overlap text pairs   │
│    Contributions              │   in the SNLI benchmark.                                                       │
│                               │ • Disaggregated code latency audit (separating 2.050 ms AST extraction from    │
│                               │   0.013 ms SVM inference).                                                     │
│                               │ • Obfuscation-level breakdown across all 6 Faidhi-Robinson levels comparing    │
│                               │   SVM, Random Forest, RKRGST, and the Proposed Framework.                       │
│                               │ • Pareto-frontier analysis mapping candidate recall vs. computational budget.  │
│                               │ • Evidence faithfulness quantification via feature ablation probability drop.  │
└───────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Explicit Boundaries and Novelty Limitations

To ensure total academic transparency during peer review:
1. **Candidate Recall is an Optimization Target, Not a Guaranteed Theorem:**
   While conformal prediction provides marginal set coverage guarantees under exchangeability, candidate recall ($R_{\text{cand}} \ge 0.98$) is an empirical optimization target tuned on validation data. If test data exhibits extreme adversarial distribution shift, empirical recall may deviate from the validation target.
2. **Proxy Benchmark Limitation:**
   The text experiments utilize the SNLI/Pereira benchmark (sentence-level entailment proxy). While it rigorously benchmarks synonym substitution and lexical variation, findings must not be claimed as an absolute evaluation of multi-page academic essays.
3. **Introductory Code Scope:**
   The source code experiments utilize the authentic IR-Plag Java corpus (introductory assignments, 20–50 LOC). Generalizability to large multi-file software engineering repositories ($5,000+$ LOC) remains an open research question.
4. **Computational Speedup vs. Ingestion Overhead:**
   In code, bypassing heavy AST feature extraction provides massive speedup ($2.050\text{ ms} \to 0.050\text{ ms}$). However, the lightweight pre-filter itself must still inspect all $\frac{N(N-1)}{2}$ candidate pairs. In hyper-scale institutional cohorts ($N > 10,000$), sub-quadratic index clustering must precede pairwise screening.

---

## 4. Unresolved Research Questions for Peer Review Discussion

1. *How does probability calibration degrade under severe class imbalance?* When plagiarism prevalence drops from $35\%$ down to realistic production levels ($< 3\%$), do calibration curves require specialized cost-sensitive recalibration?
2. *Can conformal prediction guarantees hold conditionally across distinct obfuscation levels?* Can group-balanced conformal prediction provide independent guarantees for Level 6 compound attacks?
3. *Does evidence masking perfectly reflect model reasoning?* While feature ablation drop measures faithfulness, does token masking introduce out-of-distribution artifacts into neural language models?

