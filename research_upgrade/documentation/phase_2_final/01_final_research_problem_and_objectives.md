# Final Research Problem, Motivation & Objectives (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 01 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** FULLY SPECIFIED & CERTIFIED

---

## 1. Context and Problem Statement

Academic integrity detection in digital educational ecosystems faces an acute, unresolved tension between **computational scalability** and **detection reliability**:

1. **Quadratic Complexity Bottleneck:**  
   In institutional cohorts, comparing all submissions within an assignment cohort scales quadratically:
   $$\mathcal{O}(M^2) \quad \text{for } M \text{ submissions}$$
   For a typical large computer science or humanities course ($M = 1,000$), exhaustive pairwise inspection requires:
   $$\frac{1000 \times 999}{2} = 499,500 \text{ comparisons}$$
   For large multi-section courses ($M = 5,000$), this reaches $12,497,500$ pairwise comparisons.

2. **The Fragility of Traditional Heuristic Filters (The Zero-Overlap Failure Mode):**  
   Production systems (e.g., Turnitin, CopyLeaks, MOSS, JPlag) deploy single-view heuristic pre-filters (lexical word $n$-grams, token winnowing, or Karp-Rabin string matching). Two distinct zero-overlap populations are audited in this research:
   - **Historical Stress-Test Slice:** Historical baseline analysis identified **119 true positive candidate pairs** (11.19% of 1,063 benchmark positive pairs; 944 / 1,063 = 88.81% possessing non-zero word overlap) sharing zero lexical vocabulary ($\text{TF-IDF} = 0.0000$ and word Jaccard $= 0.0$) due to heavy synonym substitution.
   - **Active Test Partition Baseline:** In the active 3,000-pair group-partitioned test split, exactly **33 out of 1,026 positive pairs** (3.22%) exhibit zero word Jaccard overlap.
   A purely lexical filter set to prune candidates with zero lexical overlap misses 100% of these instances, capping candidate recall. Crucially, this research does **NOT claim that the 119 cases have already been recovered**, as Phase 3 implementation has not yet occurred. Instead, these 119 cases are established as a concrete **Phase 3 empirical evaluation stress-test target** for the multi-view candidate screening pipeline.

3. **The Intractability of Monolithic Deep Transformer Deployment:**  
   Deploying heavy cross-encoder neural models uniformly across all $M(M-1)/2$ pairs is computationally prohibitive. A bidirectional cross-encoder (`ms-marco-MiniLM-L-6-v2`) requires $\sim 4.046\text{ ms}$ per unbatched forward pass. Processing $500,000$ pairs sequentially would demand over **33.7 hours of high-utilization compute** per assignment submission deadline, incurring severe financial costs, carbon emissions, and turnaround delays.

4. **The High Cost of Uncalibrated Automated Decision-Making:**  
   Standard deep classifiers output uncalibrated decision margins or overconfident softmax probabilities. When academic misconduct charges carry disciplinary probation, suspension, or expulsion, false accusations cause severe reputational and legal harm, while undetected collusion undermines educational credentials. Institutional integrity review boards require **calibrated posterior probabilities**, **rigorous coverage guarantees**, and **interpretable attribution evidence**.

---

## 2. Research Objectives

This research project resolves these tensions through a cost-efficient, uncertainty-aware, two-stage detection architecture with four distinct technical objectives:

- **Objective 1 (Candidate Recall Preservation):**  
  Design a fast multi-view candidate screening mechanism that extracts complementary lexical, semantic, syntactic, and structural representations, targeting $\ge 98.0\%$ candidate recall while pruning $\ge 70.0\%$ of benign negative pairs, with specific empirical evaluation targets on the known 119 zero-overlap text pairs (11.19% of 1,063 benchmark positive pairs) and transformed code variants in Phase 3.

- **Objective 2 (Uncertainty-Aware Adaptive Escalation):**  
  Formulate an adaptive routing engine that partitions pairs into a 4-state decision taxonomy (`LOW_RISK_REVIEW_DEFERRED`, `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, `ABSTAIN_UNRESOLVED`). Guarantee finite-sample marginal coverage $\ge 1 - \epsilon$ over clustered premise groups in text via group-level split conformal prediction.

- **Objective 3 (Pareto-Optimal Budget Allocation):**  
  Formulate multi-objective optimization over candidate recall, inference latency, and human review burden, systematically characterizing the Pareto frontier and reducing average per-pair latency by $\ge 80.0\%$ relative to uniform heavy-model deployment.

- **Objective 4 (Faithful Explainable Attribution Evidence):**  
  Generate multi-modal attribution dossiers (token-level Myers diffs and isomorphic AST subtree matches) that strictly separate automated statistical indicators from university academic integrity board adjudication.

---

## 3. Scope Boundaries and Honest Methodological Constraints

To ensure uncompromising academic integrity, the following boundaries are formally established:

1. **Proxy Benchmark Limitation:**  
   The text benchmark is constructed from the Stanford Natural Language Inference (SNLI) corpus, where entailment serves as a proxy for semantic paraphrase. SNLI consists of sentence pairs ($10$–$35$ words) derived from image captions, lacking document-level discourse flow and academic citations. It is used as a controlled stress-test for paraphrase screening, not as an authentic student essay benchmark.

2. **Code Modality Single-Class Topology:**  
   In IR-Plag `case-05`, all 53 positive pairs share a single reference base program (`prog_original`). Group-disjoint cross-validation leaves training or test folds with zero positive instances. Consequently, code evaluation on `case-05` is restricted to **exploratory raw decision-margin thresholding** ($z \in \mathbb{R}$); probability calibration and conformal prediction apply strictly to the text modality.

3. **Separation of Evidence from Adjudication:**  
   The system never declares a student "guilty of plagiarism". It outputs an **Evidence Dossier** with statistical signals and diffs. The institutional integrity board maintains sole legal authority.

