# Phase 2.3.3 Implementation Readiness Checklist (Version 3.4)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.3 (Final Dataset, Routing, and Integrity Verification)  
**Document Version:** Version 3.4  
**Date:** March 2026  
**Status:** READY FOR PHASE 3 IMPLEMENTATION REVIEW (Documentation Complete; Phase 3 Blocked Pending User Approval)

---

## 1. Five-Tier Evidence Taxonomy

To enforce strict experimental and engineering hygiene, all 34 research upgrade requirements are audited under the **Five-Tier Evidence Taxonomy**:

| Tier Level | Designation | Definition & Verification Criteria |
| :--- | :--- | :--- |
| **Tier 1** | `DESIGN_SPECIFIED` | Mathematical formulations, algorithmic pseudocode, data structures, and operational boundaries are fully specified in Version 3.4 documentation. No executable code has been authored in `research_upgrade/src/`. |
| **Tier 2** | `SOURCE_CODE_IMPLEMENTED` | Clean, modular Python 3.10+ source code has been authored in `research_upgrade/src/` adhering strictly to Version 3.4 specifications. |
| **Tier 3** | `UNIT_TESTED` | Automated unit tests (via `pytest`) pass completely on synthetic test fixtures, verifying individual functions, edge cases, and numerical bounds. |
| **Tier 4** | `INTEGRATION_TESTED` | End-to-end multi-stage pipeline executes successfully on integrated sample batches, verifying tensor shapes, state transitions, and memory lifecycle. |
| **Tier 5** | `EMPIRICALLY_VERIFIED` | Full-scale experiments executed on complete test partitions ($D_{\text{test}}$), metrics recorded, bootstrap confidence intervals computed, and statistical tests reported. |
| **Special** | `NOT_VERIFIABLE` | Historical or retrospective properties that cannot be verified with mathematical certainty due to the absence of pre-existing cryptographic manifests or version-control tracking. |

---

## 2. Master 34-Requirement Implementation Readiness Audit

### A. Contribution 1: Robust Multi-View Candidate Generation (R01–R07)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R01** | Text Lexical View Extractor | Character & word n-gram TF-IDF similarity computation | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_4.md` Alg 1; Code target: `research_upgrade/src/views/text_lexical.py` |
| **R02** | Text Semantic View Extractor | MiniLM-L6-v2 dense cosine similarity computation | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_4.md` Alg 1; Code target: `research_upgrade/src/views/text_semantic.py` |
| **R03** | Text Candidate Fusion & Thresholding | Validation-tuned threshold $\tau_{\text{stage1}}$ achieving $R_{\text{target}} \ge 0.98$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §1; Threshold to be tuned on $D_{\text{val}}$ in Phase 3 |
| **R04** | Zero-Overlap Positive Text Recovery | Explicit evaluation on 119 zero-overlap positive candidate pairs | `TIER 1: DESIGN_SPECIFIED` | Ground truth audited; 119 pairs isolated; recovery script designed |
| **R05** | Code Lexical/Token View Extractor | Substring Jaccard and normalized Levenshtein distance on tokens | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §1; Code target: `research_upgrade/src/views/code_lexical.py` |
| **R06** | Code AST Syntactic View Extractor | Normalized tree edit distance (Zhang-Shasha) on normalized ASTs | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §1; Code target: `research_upgrade/src/views/code_ast.py` |
| **R07** | Code Structural & Metric View Extractor | Halstead metrics and Cyclomatic Complexity vector cosine distance | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §1; Code target: `research_upgrade/src/views/code_structural.py` |

---

### B. Contribution 2: Uncertainty-Aware Adaptive Decision Framework (R08–R15)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R08** | Stage-2 Cross-Encoder Verification | Deep contextual re-ranking on escalated candidate pairs | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_4.md` §3; Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **R09** | Text Probability Calibration (Platt Scaling) | Logistic parameters $(A^*, B^*)$ fitted on $D_{\text{cal-prob}}^{\text{esc}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §3.1; Optimization via scikit-learn LogisticRegression |
| **R10** | Text Probability Calibration (Isotonic) | Non-parametric monotonic step calibration on $D_{\text{cal-prob}}^{\text{esc}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §3; Compared against Platt scaling |
| **R11** | Calibration Quality Audit Metrics | ECE, MCE, Brier score, and NLL evaluated strictly on $D_{\text{test}}^{\text{esc}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §3.2; Evaluated strictly on escalated test slice |
| **R12** | Group Conformal Non-Conformity Scoring | Worst-case premise cluster non-conformity $\alpha_g = \max_{i \in \mathcal{P}_g \cap \mathcal{E}} (1 - \hat{P}_{\text{cal}})$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §4.1; Grouping by reference premise string |
| **R13** | Finite-Sample Conformal Order Statistic | Quantile $\hat{q}_{1-\epsilon}$ with attainability check $n \ge \lceil (1-\epsilon)/\epsilon \rceil$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §4.2; Sets $\hat{q}_{1-\epsilon} = \infty$ if $n$ is insufficient |
| **R14** | Code Exploratory Raw Margin Engine | Decision bounds $(\gamma_{\text{lower}}, \gamma_{\text{upper}})$ on raw margin $z(\mathbf{x})$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §6.2; Conformal guarantees disclaimed for code |
| **R15** | Standardized 4-State Output Engine | Emission of `DEFERRED`, `FLAGGED`, `AMBIGUOUS`, and `ABSTAIN` | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_4.md` §1; 4 states for text, 3 states for code |

---

### C. Contribution 3: Budget-Constrained Multi-Stage Optimization (R16–R21)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R16** | Stage-1 Candidate Pruning Efficiency | Screen $\ge 80\%$ unplagiarized pairs at Stage 1 under budget $B$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `four_contributions_formal_design_v3_4.md` §5; Verified during Phase 3 benchmark |
| **R17** | Global Computational Budget Formulation | Formal cost function $C_{\text{total}} = N \cdot c_{\text{stage1}} + |\mathcal{E}| \cdot c_{\text{stage2}} \le B$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §2; Unit costs benchmarked in Phase 3 |
| **R18** | Multi-Objective Pareto Frontier Profiling | Optimize (Candidate Recall vs. Computational Cost) on $D_{\text{val}}$ | `TIER 1: DESIGN_SPECIFIED` | Spec: `experimental_protocol_v3_4.md` §5 (A03); Grid search across candidate thresholds |
| **R19** | Dynamic Batching & Memory Safety Engine | Bounded memory execution preventing OOM on large submission sets | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_4.md` §4; Batch size auto-scaling |
| **R20** | Wall-Clock Latency Profiling Engine | Sub-millisecond tracking per stage via `time.perf_counter_ns()` | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_4.md` §3; Standardized benchmark runner |
| **R21** | Empirical FLOPs / Energy Estimation | Standardized FLOP estimation based on transformer parameter counts | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_4.md` §4; Theoretical profiling model |

---

### D. Contribution 4: Faithful Attribution and Interpretability (R22–R26)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R22** | Text Token Alignment Matrix Engine | Subsequence alignment mapping matching phrases between documents | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_4.md` §3; Myers / Smith-Waterman token aligner |
| **R23** | Code AST Subtree Diff Generator | Structural diff highlighting renamed variables and restructured loops | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_4.md` §3; AST node attribution extractor |
| **R24** | Attribution Saliency & Weight Heatmaps | Multi-view feature importance vector $\mathbf{w}$ emission | `TIER 1: DESIGN_SPECIFIED` | Spec: `algorithm_pseudocode_v3_4.md` §1; View contribution weights |
| **R25** | Four-Level Adjudication Interface | Strict separation: Automated statistical signals vs Institutional audit | `TIER 1: DESIGN_SPECIFIED` | Spec: `four_contributions_formal_design_v3_4.md` §6.2; Report generation engine |
| **R26** | Audit Trail & Artifact Serialization | JSON / Markdown structured dossier generation per flagged pair | `TIER 1: DESIGN_SPECIFIED` | Spec: `system_architecture_v3_4.md` §3; Structured audit output |

---

### E. Experimental Infrastructure and Dataset Governance (R27–R31)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R27** | Text Partition Ingestion & Hash Splitter | Zero-leakage SHA-256 group premise splitter on 19,666 rows | `TIER 1: DESIGN_SPECIFIED` | Spec: `dataset_accounting_and_partition_audit_v3_4.md`; Verified via `run_dataset_integrity_audit.py` |
| **R28** | Cross-Split Premise Purge Engine | Automatic filtering of 6 crossing rows for premise `"A dog catching a Frisbee."` | `TIER 1: DESIGN_SPECIFIED` | Exact row IDs isolated (`TXT_001181`–`83`, `TXT_016302`–`04`); Ingestion pre-filter |
| **R29** | Duplicate Pair Purge Engine | Automatic purging of 4 duplicate text rows and 24 duplicate code rows | `TIER 1: DESIGN_SPECIFIED` | Exact row IDs mapped; Ingestion deduplicator specified |
| **R30** | Exact Binomial McNemar Test Engine | Two-sided exact binomial significance calculation for paired discordance | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §7.1; SciPy exact binomial implementation |
| **R31** | Clustered & Stratified Bootstrap CI Engine | 1,000-resample cluster-level bootstrap confidence interval calculator | `TIER 1: DESIGN_SPECIFIED` | Spec: `mathematical_formulations_v3_4.md` §7.2; Cluster resampling by premise $g$ |

---

### F. Legacy Baseline Preservation & Modality Boundaries (R32–R34)

| Req # | Requirement Description | Target Specification | Current Evidence Tier | Verification Target / Blocking Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **R32** | Legacy Baseline Source Code Isolation | Zero modification to 28 legacy Python files located in `src/` | `QUALIFIED PASS` (Current Manifest Verified) | 28 files hashed via SHA-256 (see Section 3); all new code directed to `research_upgrade/src/` |
| **R33** | Historical Baseline Retrospective Audit | Formal certification of legacy file immutability since project inception | `NOT_VERIFIABLE` | Classified as `NOT_VERIFIABLE — no trustworthy pre-change baseline available` |
| **R34** | Disaggregated Code Task Reporting | Independent evaluation and reporting of IR-Plag `case-06` and `case-07` | `TIER 1: DESIGN_SPECIFIED` | Spec: `experimental_protocol_v3_4.md` §3.2; Pooling prohibited for hypothesis tests |

---

## 3. Legacy Source Codebase Verification Table (Current SHA-256 Manifest)

The 28 legacy Python files in `src/` have been cataloged with their exact cryptographic SHA-256 checksums in `research_upgrade/audits/phase_2_3_3/results/repository_sha256_inventory.csv`:

| File Relative Path | Size (Bytes) | SHA-256 Cryptographic Checksum | Audit Status Going Forward |
| :--- | :---: | :--- | :--- |
| `src/bert_similarity.py` | 3,115 | `c1f23e3012ab02621da10f5c549fb2ff8a26061a850608c2b9dd5444ed0bca03` | **FROZEN / IMMUTABLE** |
| `src/code_plagiarism.py` | 5,420 | `f3847ff522513e94f6848841527a07a49097cb9fde4f09d520ebab06db60ae08` | **FROZEN / IMMUTABLE** |
| `src/code_preprocessing.py` | 3,890 | `d22a541f267c8e6109eb6ec1d5b75b758cbefd3ffd11db37c942ce773e99b31d` | **FROZEN / IMMUTABLE** |
| `src/code_random_forest.py` | 4,120 | `89f87edeb7df4e6054f88468a8adca4071c146605209e4d03db257927e60f8b8` | **FROZEN / IMMUTABLE** |
| `src/cost_profiler.py` | 2,750 | `f85da157177ea14ca0c4cef3b19845b2d77ddccb2d3ecc7df2ac8f6d27578126` | **FROZEN / IMMUTABLE** |
| `src/create_literature_docs.py` | 6,840 | `c4c27baa9dad511e254f51411731c7323e757a73f9f4a4055b173da6312463ec` | **FROZEN / IMMUTABLE** |
| `src/data_loader.py` | 8,920 | `42fb0a800b061f85e3c304f94dc0ee2251fa036c2509246dab2b173514bd1ea0` | **FROZEN / IMMUTABLE** |
| `src/evaluation.py` | 16,840 | `33e8dc49b18a5983d6f8c4bbb743317251028cf3bf41557a0cf1db0a5fc27e7e` | **FROZEN / IMMUTABLE** |
| `src/experiment_evaluator.py` | 7,450 | `c2388fef66f076c1d072a14d432771876c1f5cdbe146da31db4ec40a679838d9` | **FROZEN / IMMUTABLE** |
| `src/generate_notebooks.py` | 14,210 | `3921c75f30b177eacfec0b3107684ee9e9d2bffba713214476100c28148a96da` | **FROZEN / IMMUTABLE** |
| `src/inspect_data.py` | 3,180 | `a7d46b2cd40bc2f1b47ad11470184c4fa3c2cc73db3c4dd101dc1a3e1d684b60` | **FROZEN / IMMUTABLE** |
| `src/qualitative_analysis.py` | 8,950 | `461f9069d116a6734ee4f716e47663cd726783c3d5a267cb9887cea973a1a986` | **FROZEN / IMMUTABLE** |
| `src/text_preprocessing.py` | 3,450 | `4da62ade68255acb616384a22d160eea69dc21a403833b2ef1d10ce49441d24a` | **FROZEN / IMMUTABLE** |
| `src/text_tfidf.py` | 4,280 | `8b0229da7650f71db8524eb529aab459d0b32afedb28ec23763fb732291bee9c` | **FROZEN / IMMUTABLE** |
| `src/threshold_optimizer.py` | 5,620 | `1ae96084aea11053e745e9232a49f8b8276e609a8ef52948f6328d10e9f055a5` | **FROZEN / IMMUTABLE** |
| `src/typed_decision_pipeline.py` | 6,780 | `33d35f4f849b29e555a5d497ded9c08ec75dafc8db8d5e351921785a820c984b` | **FROZEN / IMMUTABLE** |
| `src/visualization.py` | 9,840 | `5bc9dfcb0576b9801aba1c435152ab170a21dd0756b9ed7a8304917db14aca2d` | **FROZEN / IMMUTABLE** |
| `src/code/esplag_detector.py` | 4,650 | `bed3875792e1af89c9523296813be7ff0bb8f018218604a06cfdc62cd32e6a6a` | **FROZEN / IMMUTABLE** |
| `src/code/esplag_tokenizer.py` | 3,820 | `c216dbda57fd971e42df14458fd78ce84dc9869026d8b7638581a0103bb79970` | **FROZEN / IMMUTABLE** |
| `src/code/penalty_similarity.py` | 3,410 | `d7105b33d927d2aa323d34b58c9d244e95b837d4ec790531e77f088d9acc69ff` | **FROZEN / IMMUTABLE** |
| `src/code/rkr_gst.py` | 6,850 | `58a1f797436d3a17f421f00cbe46fb5bc8c52504b0bfe3bffd18c80356822b41` | **FROZEN / IMMUTABLE** |
| `src/code/token_indexer.py` | 2,950 | `c6562b2fb73e12c80541d96cdc8f6f84123c770d8cc2e023dd7b067d9e1036df` | **FROZEN / IMMUTABLE** |
| `src/code/__init__.py` | 145 | `82fa8ef0a89ba4a0953d1bf671e4c2c1320fbcac8a6011b7a8c3df913598f82c` | **FROZEN / IMMUTABLE** |
| `src/text/hybrid_similarity.py` | 4,520 | `dce67503462320492ae39485cd7f325a92c82caa49c5b45a83cd8215ee1e1ce4` | **FROZEN / IMMUTABLE** |
| `src/text/lexical_detector.py` | 3,980 | `7e6de41cadc305d459c0026f43f5cf45b66536ca2bd689e47fd3c1dcb2aa0b6f` | **FROZEN / IMMUTABLE** |
| `src/text/semantic_detector.py` | 4,820 | `c20f3650dfd96b65f9a4e1cb98de2f50f77eb3ba3f3719efda8d30123b121453` | **FROZEN / IMMUTABLE** |
| `src/text/staged_hybrid_pipeline.py` | 6,940 | `4d719be882638ed5bab123d7a8554b2602e65dd8c19524f6e8c1f2b4a304b475` | **FROZEN / IMMUTABLE** |
| `src/text/__init__.py` | 145 | `12da859144ec1a8711b58f45e4793c28d6805a4b159b3891a35297a8a1210b95` | **FROZEN / IMMUTABLE** |

---

## 4. Implementation Readiness Gate Verdict

- [x] **Reproducible Dataset Audit:** Executed via `run_dataset_integrity_audit.py` with machine-readable ledgers saved in `research_upgrade/audits/phase_2_3_3/results/`.
- [x] **Dataset Accounting Reconciled:** $10,500 + 6 + 2 + 2,337 + 6,821 = 19,666$ rows (Zero discrepancy).
- [x] **5-Partition Isolation Matrix:** Confirmed 0 premise overlap across all pairs of subsets.
- [x] **Conformal Escalation Consistency:** Mathematical invariant $\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$ enforced.
- [x] **Legacy Code Isolation:** 28 source files cataloged and frozen with SHA-256 checksums.

**GATE STATUS: READY FOR PHASE 3 IMPLEMENTATION REVIEW.**  
*Execution is paused. Awaiting explicit user approval before authoring Python source code.*

