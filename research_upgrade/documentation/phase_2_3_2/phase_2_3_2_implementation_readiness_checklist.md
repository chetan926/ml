# Phase 2.3.2 Implementation Readiness Checklist

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.2 (Final Statistical Validity and Cross-Document Certification)  
**Document Version:** Version 3.3  
**Status:** READY FOR PHASE 3 IMPLEMENTATION REVIEW (Documentation Complete; Phase 3 Implementation Pending Approval)  
**Date:** March 2026  

---

## 1. Five-Tier Evidence Taxonomy

In previous audit phases, readiness checklists prematurely marked unwritten algorithms as `PASS`, conflating mathematical design with empirical software completion. To enforce strict research and engineering rigor, Version 3.3 adopts a **Five-Tier Evidence Taxonomy**:

| Tier Level | Designation | Definition & Verification Criteria |
| :--- | :--- | :--- |
| **Tier 1** | `DESIGN_SPECIFIED` | The mathematical formulation, algorithmic pseudocode, data structures, and operational boundaries are fully specified in Version 3.3 documentation. No executable code has yet been authored in `research_upgrade/src/`. |
| **Tier 2** | `SOURCE_CODE_IMPLEMENTED` | Clean, modular Python 3.10+ source code has been authored in `research_upgrade/src/` adhering strictly to Version 3.3 specifications. |
| **Tier 3** | `UNIT_TESTED` | Automated unit tests (via `pytest`) pass completely on mock/synthetic data fixtures, verifying individual functions, error handling, edge cases, and numerical bounds. |
| **Tier 4** | `INTEGRATION_TESTED` | End-to-end multi-stage pipeline executes successfully on integrated benchmark sample batches, verifying tensor shapes, state transitions, and memory lifecycle. |
| **Tier 5** | `EMPIRICALLY_VERIFIED` | Full-scale experiments have been executed on the complete test partitions ($D_{\text{test}}$), metrics recorded, bootstrap confidence intervals computed, and statistical tests reported. |
| **Special** | `NOT VERIFIABLE` | Historical or retrospective properties that cannot be verified with mathematical certainty due to the absence of pre-existing cryptographic manifests or version-control tracking. |

> **Audit Policy:** In Phase 2.3.2, all new research upgrade components must be categorized as `DESIGN_SPECIFIED`. No component may be marked as `SOURCE_CODE_IMPLEMENTED` or higher until Phase 3 implementation is explicitly approved and executed.

---

## 2. Master 34-Requirement Implementation Readiness Checklist

The 34 formal requirements governing the research upgrade are audited below across the four contributions, system architecture, dataset pipeline, and experimental verification infrastructure.

### A. Contribution 1: Robust Multi-View Candidate Generation (Requirements 1–7)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R01** | Text Lexical View Extractor | Character & word n-gram TF-IDF similarity computation | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_3.md` Alg 1; Code target: `research_upgrade/src/views/text_lexical.py` |
| **R02** | Text Semantic View Extractor | MiniLM-L6-v2 dense cosine similarity computation | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_3.md` Alg 1; Code target: `research_upgrade/src/views/text_semantic.py` |
| **R03** | Text Candidate Fusion & Thresholding | Validation-tuned threshold $\tau_{\text{stage1}}$ achieving $R_{\text{target}} \ge 0.98$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §1.3; Threshold to be tuned on $D_{\text{val}}$ in Phase 3 |
| **R04** | Zero-Overlap Positive Text Recovery | Explicit evaluation on 119 zero-overlap positive candidate pairs | `TIER 1: DESIGN_SPECIFIED` | Ground truth audited; 119 pairs isolated; recovery script designed |
| **R05** | Code Lexical/Token View Extractor | Substring Jaccard and normalized Levenshtein distance on tokens | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §1.2; Code target: `research_upgrade/src/views/code_lexical.py` |
| **R06** | Code AST Syntactic View Extractor | Normalized tree edit distance (Zhang-Shasha) on normalized ASTs | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §1.2; Code target: `research_upgrade/src/views/code_ast.py` |
| **R07** | Code Structural & Metric View Extractor | Halstead metrics and Cyclomatic Complexity vector cosine distance | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §1.2; Code target: `research_upgrade/src/views/code_structural.py` |

---

### B. Contribution 2: Uncertainty-Aware Adaptive Decision Framework (Requirements 8–15)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R08** | Stage-2 Cross-Encoder Verification | Deep contextual re-ranking on escalated candidate pairs | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_3.md` §4.2; Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **R09** | Text Probability Calibration (Platt Scaling) | Logistic parameters $(A^*, B^*)$ fitted on $D_{\text{cal-prob}}^{\text{esc}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §2.1; Optimization via scikit-learn LogisticRegression |
| **R10** | Text Probability Calibration (Isotonic) | Non-parametric monotonic step calibration on $D_{\text{cal-prob}}^{\text{esc}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §2.2; Compared against Platt scaling |
| **R11** | Calibration Quality Audit Metrics | ECE, MCE, Brier score, and NLL evaluated strictly on $D_{\text{test}}^{\text{esc}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §2.3; Evaluated strictly on escalated test slice |
| **R12** | Group Conformal Non-Conformity Scoring | Worst-case premise cluster non-conformity $\alpha_g = \max_{i \in \mathcal{P}_g \cap \mathcal{E}} (1 - \hat{P}_{\text{cal}})$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §3.2; Grouping by reference premise string |
| **R13** | Finite-Sample Conformal Order Statistic | Quantile $\hat{q}_{1-\epsilon}$ with attainability check $n \ge \lceil (1-\epsilon)/\epsilon \rceil$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §3.3; Sets $\hat{q}_{1-\epsilon} = \infty$ if $n$ is insufficient |
| **R14** | Code Exploratory Raw Margin Engine | Decision bounds $(\gamma_{\text{lower}}, \gamma_{\text{upper}})$ on raw margin $z(\mathbf{x})$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §2.4; Conformal guarantees disclaimed for code |
| **R15** | Standardized 4-State Output Engine | Emission of `DEFERRED`, `FLAGGED`, `AMBIGUOUS`, and `ABSTAIN` | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_3.md` Types; 4 states for text, 3 states for code |

---

### C. Contribution 3: Budget-Constrained Multi-Stage Optimization (Requirements 16–21)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R16** | Stage-1 Candidate Pruning Efficiency | Screen $\ge 80\%$ unplagiarized pairs at Stage 1 under budget $B$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `four_contributions_formal_design_v3_3.md` §4; Verified during Phase 3 benchmark |
| **R17** | Global Computational Budget Formulation | Formal cost function $C_{\text{total}} = N \cdot c_{\text{stage1}} + |\mathcal{E}| \cdot c_{\text{stage2}} \le B$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §5.1; Unit costs benchmarked in Phase 3 |
| **R18** | Multi-Objective Pareto Frontier Profiling | Optimize (Candidate Recall vs. Computational Cost) on $D_{\text{val}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `experimental_protocol_v3_3.md` §3.2; Grid search across candidate thresholds $\tau$ |
| **R19** | Dynamic Batching & Memory Safety Engine | Bounded memory execution preventing OOM on large submission sets | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_3.md` §5.2; Batch size auto-scaling |
| **R20** | Wall-Clock Latency Profiling Engine | Sub-millisecond tracking per stage via `time.perf_counter_ns()` | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_3.md` §3; Standardized benchmark runner |
| **R21** | Empirical FLOPs / Energy Estimation | Standardized FLOP estimation based on transformer parameter counts | `TIER 1: DESIGN_SPECIFIED` | Spec: `experimental_protocol_v3_3.md` §5.2; Theoretical profiling model |

---

### D. Contribution 4: Faithful Attribution and Interpretability (Requirements 22–26)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R22** | Text Token Alignment Matrix Engine | Subsequence alignment mapping matching phrases between documents | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_3.md` §4.4; Myers / Smith-Waterman token aligner |
| **R23** | Code AST Subtree Diff Generator | Structural diff highlighting renamed variables and restructured loops | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_3.md` §4.4; AST node attribution extractor |
| **R24** | Attribution Saliency & Weight Heatmaps | Multi-view feature importance vector $\mathbf{w}$ emission | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_3.md` Types; View contribution weights |
| **R25** | Four-Level Adjudication Interface | Strict separation: Automated statistical signals vs Institutional audit | `TIER 1: DESIGN_SPECIFIED` | Spec: `four_contributions_formal_design_v3_3.md` §6; Report generation engine |
| **R26** | Audit Trail & Artifact Serialization | JSON / Markdown structured dossier generation per flagged pair | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_3.md` §4.4; Structured audit output |

---

### E. Experimental Infrastructure and Dataset Governance (Requirements 27–31)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R27** | Text Partition Ingestion & Hash Splitter | Zero-leakage SHA-256 group premise splitter on 19,666 rows | `TIER 1: DESIGN_SPECIFIED` | Spec: `dataset_accounting_and_partition_audit_v3_3.md`; Split logic verified via Python script |
| **R28** | Cross-Split Premise Purge Engine | Automatic filtering of 6 crossing rows for premise `"A dog catching a Frisbee."` | `TIER 1: DESIGN_SPECIFIED` | Exact row IDs isolated (`TXT_001181`–`83`, `TXT_016302`–`04`); Loader pre-filter |
| **R29** | Duplicate Pair Purge Engine | Automatic purging of 4 duplicate text rows and 24 duplicate code rows | `TIER 1: DESIGN_SPECIFIED` | Exact row IDs mapped; Ingestion deduplicator specified |
| **R30** | Exact Binomial McNemar Test Engine | Two-sided exact binomial significance calculation for paired discordance | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §6.1; SciPy exact binomial implementation |
| **R31** | Clustered & Stratified Bootstrap CI Engine | 1,000-resample cluster-level bootstrap confidence interval calculator | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_3.md` §6.2; Cluster resampling by premise $g$ |

---

### F. Legacy Baseline Preservation & Modality Boundaries (Requirements 32–34)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R32** | Legacy Baseline Source Code Isolation | Zero modification to 18 legacy Python files located in `src/` | `QUALIFIED PASS` (Current Manifest Verified) | 18 files hashed via SHA-256 (see Section 3); all new code directed to `research_upgrade/src/` |
| **R33** | Historical Baseline Retrospective Audit | Formal certification of legacy file immutability since project inception | `NOT VERIFIABLE` | Reclassified to `NOT VERIFIABLE — no trustworthy pre-change baseline available` |
| **R34** | Disaggregated Code Task Reporting | Independent evaluation and reporting of IR-Plag `case-06` and `case-07` | `TIER 1: DESIGN_SPECIFIED` | Spec: `experimental_protocol_v3_3.md` §5.3; Pooling prohibited for hypothesis tests |

---

## 3. Legacy Source Codebase Verification Table (Current SHA-256 Manifest)

As established in Phase 2.3.2 Issue 7, because Git version control was not initialized in this workspace (`Test-Path ".git"` is `False`) and no cryptographic manifest existed prior to Phase 2, retrospective immutability is formally classified as:  
**`NOT VERIFIABLE — no trustworthy pre-change baseline available`**.

However, to provide absolute verification going forward, SHA-256 cryptographic hashes were computed across all 28 legacy Python source files in `src/` (root and subpackages `src/code/`, `src/text/`) using PowerShell `Get-FileHash`. Any modification to these files during Phase 3 will trigger an immediate audit failure:

| File Relative Path | SHA-256 Cryptographic Checksum | Audit Status Going Forward |
| :--- | :--- | :--- |
| `src/bert_similarity.py` | `c1f23e3012ab02621da10f5c549fb2ff8a26061a850608c2b9dd5444ed0bca03` | **FROZEN / IMMUTABLE** |
| `src/code_plagiarism.py` | `f3847ff522513e94f6848841527a07a49097cb9fde4f09d520ebab06db60ae08` | **FROZEN / IMMUTABLE** |
| `src/code_preprocessing.py` | `d22a541f267c8e6109eb6ec1d5b75b758cbefd3ffd11db37c942ce773e99b31d` | **FROZEN / IMMUTABLE** |
| `src/code_random_forest.py` | `89f87edeb7df4e6054f88468a8adca4071c146605209e4d03db257927e60f8b8` | **FROZEN / IMMUTABLE** |
| `src/cost_profiler.py` | `f85da157177ea14ca0c4cef3b19845b2d77ddccb2d3ecc7df2ac8f6d27578126` | **FROZEN / IMMUTABLE** |
| `src/create_literature_docs.py` | `c4c27baa9dad511e254f51411731c7323e757a73f9f4a4055b173da6312463ec` | **FROZEN / IMMUTABLE** |
| `src/data_loader.py` | `42fb0a800b061f85e3c304f94dc0ee2251fa036c2509246dab2b173514bd1ea0` | **FROZEN / IMMUTABLE** |
| `src/evaluation.py` | `33e8dc49b18a5983d6f8c4bbb743317251028cf3bf41557a0cf1db0a5fc27e7e` | **FROZEN / IMMUTABLE** |
| `src/experiment_evaluator.py` | `c2388fef66f076c1d072a14d432771876c1f5cdbe146da31db4ec40a679838d9` | **FROZEN / IMMUTABLE** |
| `src/generate_notebooks.py` | `3921c75f30b177eacfec0b3107684ee9e9d2bffba713214476100c28148a96da` | **FROZEN / IMMUTABLE** |
| `src/inspect_data.py` | `a7d46b2cd40bc2f1b47ad11470184c4fa3c2cc73db3c4dd101dc1a3e1d684b60` | **FROZEN / IMMUTABLE** |
| `src/qualitative_analysis.py` | `461f9069d116a6734ee4f716e47663cd726783c3d5a267cb9887cea973a1a986` | **FROZEN / IMMUTABLE** |
| `src/text_preprocessing.py` | `4da62ade68255acb616384a22d160eea69dc21a403833b2ef1d10ce49441d24a` | **FROZEN / IMMUTABLE** |
| `src/text_tfidf.py` | `8b0229da7650f71db8524eb529aab459d0b32afedb28ec23763fb732291bee9c` | **FROZEN / IMMUTABLE** |
| `src/threshold_optimizer.py` | `1ae96084aea11053e745e9232a49f8b8276e609a8ef52948f6328d10e9f055a5` | **FROZEN / IMMUTABLE** |
| `src/typed_decision_pipeline.py` | `33d35f4f849b29e555a5d497ded9c08ec75dafc8db8d5e351921785a820c984b` | **FROZEN / IMMUTABLE** |
| `src/visualization.py` | `5bc9dfcb0576b9801aba1c435152ab170a21dd0756b9ed7a8304917db14aca2d` | **FROZEN / IMMUTABLE** |
| `src/code/esplag_detector.py` | `bed3875792e1af89c9523296813be7ff0bb8f018218604a06cfdc62cd32e6a6a` | **FROZEN / IMMUTABLE** |
| `src/code/esplag_tokenizer.py` | `c216dbda57fd971e42df14458fd78ce84dc9869026d8b7638581a0103bb79970` | **FROZEN / IMMUTABLE** |
| `src/code/penalty_similarity.py` | `d7105b33d927d2aa323d34b58c9d244e95b837d4ec790531e77f088d9acc69ff` | **FROZEN / IMMUTABLE** |
| `src/code/rkr_gst.py` | `58a1f797436d3a17f421f00cbe46fb5bc8c52504b0bfe3bffd18c80356822b41` | **FROZEN / IMMUTABLE** |
| `src/code/token_indexer.py` | `c6562b2fb73e12c80541d96cdc8f6f84123c770d8cc2e023dd7b067d9e1036df` | **FROZEN / IMMUTABLE** |
| `src/code/__init__.py` | `82fa8ef0a89ba4a0953d1bf671e4c2c1320fbcac8a6011b7a8c3df913598f82c` | **FROZEN / IMMUTABLE** |
| `src/text/hybrid_similarity.py` | `dce67503462320492ae39485cd7f325a92c82caa49c5b45a83cd8215ee1e1ce4` | **FROZEN / IMMUTABLE** |
| `src/text/lexical_detector.py` | `7e6de41cadc305d459c0026f43f5cf45b66536ca2bd689e47fd3c1dcb2aa0b6f` | **FROZEN / IMMUTABLE** |
| `src/text/semantic_detector.py` | `c20f3650dfd96b65f9a4e1cb98de2f50f77eb3ba3f3719efda8d30123b121453` | **FROZEN / IMMUTABLE** |
| `src/text/staged_hybrid_pipeline.py` | `4d719be882638ed5bab123d7a8554b2602e65dd8c19524f6e8c1f2b4a304b475` | **FROZEN / IMMUTABLE** |
| `src/text/__init__.py` | `12da859144ec1a8711b58f45e4793c28d6805a4b159b3891a35297a8a1210b95` | **FROZEN / IMMUTABLE** |

---

## 4. Phase 3 Implementation Prerequisite Verification

Before initiating any source code development in Phase 3, the following prerequisite checklist must be satisfied:

1. [x] **Documentation Completeness:** All 11 Version 3.3 deliverables authored, mutually verified, and checked into `research_upgrade/documentation/phase_2_3_2/`.
2. [x] **Mathematical Soundness:** Group conformal prediction, calibration lifecycles, and exact statistical testing rigorously formulated without unresolved edge cases.
3. [x] **Dataset Integrity:** Row counts, duplicate pairs, and shared-premise cross-contamination empirically audited and isolated.
4. [x] **Baseline Protection:** Legacy codebase frozen and checksummed; Phase 3 target directories defined exclusively in `research_upgrade/src/`.
5. [ ] **Human Review & Approval:** The research auditor and principal investigator must formally review this checklist and grant explicit authorization to proceed to Phase 3.

**Current Implementation Gate Status: BLOCKED PENDING USER AUTHORIZATION FOR PHASE 3.**
