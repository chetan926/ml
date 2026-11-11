# PHASE 3.2 FINAL RECONCILIATION AND RESEARCH INTEGRITY AUDIT

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** 3.2 — Final Reconciliation, Partition Membership Audit & Research Integrity Verification  
**Date of Audit:** 2026-11-11  
**Lead Auditor & Research Architect:** Independent Senior Research Scientist, Statistical Methodologist & Software Auditor  
**Baseline Verification Status:** **28/28 SHA-256 Checksums Matching (100% Frozen Baseline in `src/`)**  
**Final Audit Verdict:** **FULL_PASS — RIGOROUSLY CERTIFIED & FROZEN**

---

## 1. Mandatory Baseline and Historical Artifact Verification

An exhaustive cryptographic audit of the codebase was conducted prior to and throughout this audit:

### 1.1 Baseline Source Verification (`src/`)
Every file registered in [`research_upgrade/audits/phase_2_final/repository_sha256_manifest.csv`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/repository_sha256_manifest.csv) was inspected on disk using 64KB block SHA-256 hashing via `research_upgrade.src.governance.verify_baseline_integrity()`.

- **Total Registered Baseline Files:** 28
- **Matched SHA-256 Digests:** **28 / 28**
- **Mismatches / Corrupted Files:** 0
- **Missing Files:** 0
- **Untracked Additions in `src/`:** 0
- **Integrity Status:** **VERIFIED UNTOUCHED (`PASSED`)**

### 1.2 Phase 2 and Phase 3.1 Historical Artifact Preservation
- **Phase 2 Final Audits (`research_upgrade/audits/phase_2_final/`):** All 12 historical artifacts (including `audit_execution_log.txt`, `case_05_fold_feasibility_report.json`, `code_duplicate_accounting.json`, `code_pair_identity_report.csv`, `text_dataset_accounting.json`, `text_exclusion_ledger.csv`, and `text_group_leakage_report.json`) remain byte-for-byte identical to their Phase 2 frozen state.
- **Phase 3.1 Setup Manifest (`phase_3_1_implementation_manifest.json`):** Intact and verified.
- **Historical Manifests:** Historical manifests were not overwritten; all Phase 3.2 reconciliation evidence is stored in dedicated audit artifacts.

---

## 2. Test-Count Reconciliation Audit

### 2.1 The Discrepancy
The historical Phase 3.2 implementation report stated:
> "Total: 56 | FULL PASS"

However, a manual sum of the individual rows in the report's markdown table yielded **69** ($5+13+8+5+6+13+13+4+2 = 69$).

### 2.2 Root Cause Analysis
An independent inspection of the test suite files under `tests/` revealed that the discrepancy was caused strictly by **typographical / transcription errors in the markdown documentation table**, not by missing, unexecuted, or altered tests in the repository.

Specifically:
- `test_text_features.py` actually contains **5** test cases (it was erroneously recorded as 13 in the report table, copying the count from `test_code_features.py`).
- `test_leakage_protection.py` actually contains **5** test cases (it was erroneously recorded as 13).
- `test_governance.py` actually contains **8** test cases (erroneously recorded as 5).
- `test_schema_validation.py` actually contains **3** test cases (erroneously recorded as 5).
- `test_data_loaders.py` actually contains **7** test cases (erroneously recorded as 6).
- `test_rkrgst.py` actually contains **9** test cases (erroneously recorded as 8).

### 2.3 Independent Test Execution Evidence
The complete test suite was executed through an automated test harness (`research_upgrade/audits/phase_3/reconcile_tests.py`):

| Test Module Path | Test Classes Present | Actual Collected Tests | Executed | Passed | Failed | Errored | Skipped |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [`tests/test_code_features.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_code_features.py) | `TestCodeComplexityExtractor`, `TestCodeIdentifierNormalizedExtractor`, `TestCodeStructuralExtractor`, `TestCodeTokenExtractor` | **13** | 13 | 13 | 0 | 0 | 0 |
| [`tests/test_data_loaders.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_data_loaders.py) | `TestDataLoaders` | **7** | 7 | 7 | 0 | 0 | 0 |
| [`tests/test_feature_pipeline.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_feature_pipeline.py) | `TestCodeMultiViewPipeline`, `TestTextMultiViewPipeline` | **4** | 4 | 4 | 0 | 0 | 0 |
| [`tests/test_feature_reproducibility.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_feature_reproducibility.py) | `TestFeatureReproducibility` | **2** | 2 | 2 | 0 | 0 | 0 |
| [`tests/test_governance.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_governance.py) | `TestBaselineGovernance`, `TestReproducibilityConfig` | **8** | 8 | 8 | 0 | 0 | 0 |
| [`tests/test_leakage_protection.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_leakage_protection.py) | `TestDatasetLeakageProtection` | **5** | 5 | 5 | 0 | 0 | 0 |
| [`tests/test_rkrgst.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_rkrgst.py) | `TestESPlagTokenizer`, `TestPenaltySimilarity`, `TestRKRGST` | **9** | 9 | 9 | 0 | 0 | 0 |
| [`tests/test_schema_validation.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_schema_validation.py) | `TestSchemaValidation` | **3** | 3 | 3 | 0 | 0 | 0 |
| [`tests/test_text_features.py`](file:///D:/ml%20rearcch%20project%20ad/tests/test_text_features.py) | `TestTextFeatures` | **5** | 5 | 5 | 0 | 0 | 0 |
| **Total Test Suite** | **9 Test Files** | **56** | **56** | **56** | **0** | **0** | **0** |

**Reconciliation Conclusion:** The actual test suite in the repository consists of exactly **56 unit and integration tests**. All 56 tests execute and pass with zero failures and zero errors. Machine-readable proof is permanently logged in [`phase_3_2_test_reconciliation.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_test_reconciliation.json).

---

## 3. Exact Phase 2 Partition Membership Verification

A mandatory row-level research integrity check was performed by comparing the partitions produced by [`TextBenchmarkDataLoader`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/src/data/text_loader.py) and [`CodeBenchmarkDataLoader`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/src/data/code_loader.py) against the certified Phase 2 membership ledger ([`research_upgrade/audits/phase_2_3_5/text_partition_membership.csv`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_3_5/text_partition_membership.csv) and [`research_upgrade/audits/phase_2_final/code_pair_identity_report.csv`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/code_pair_identity_report.csv)).

### 3.1 Text Partition Reconciliation (19,666 Rows Audited)

Every row was audited for:
1. Exact row ID (`TXT_XXXXXX`).
2. Assigned partition name.
3. Ground truth binary label (`is_plagiarized`).
4. Premise group assignment via SHA-256 prefix hash.

| Partition Name | Phase 2 Certified Pairs | Phase 3.2 Loaded Pairs | Missing Pairs in P3.2 | Unexpected Pairs in P3.2 | Label Mismatches | Premise Hash Mismatches | Exact Match Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 4,000 | 4,000 | 0 | 0 | 0 | 0 | **TRUE** |
| `val` | 1,500 | 1,500 | 0 | 0 | 0 | 0 | **TRUE** |
| `cal_prob` | 1,000 | 1,000 | 0 | 0 | 0 | 0 | **TRUE** |
| `cal_conf` | 1,000 | 1,000 | 0 | 0 | 0 | 0 | **TRUE** |
| `dev_reserve` | 2,337 | 2,337 | 0 | 0 | 0 | 0 | **TRUE** |
| `test` | 3,000 | 3,000 | 0 | 0 | 0 | 0 | **TRUE** |
| `test_reserve` | 6,821 | 6,821 | 0 | 0 | 0 | 0 | **TRUE** |
| **Active Subtotal** | **19,658** | **19,658** | **0** | **0** | **0** | **0** | **TRUE** |
| `EXCLUDED` (Purged) | 8 | 8 | 0 | 0 | 0 | 0 | **TRUE** |
| **Total Rows** | **19,666** | **19,666** | **0** | **0** | **0** | **0** | **TRUE** |

### 3.2 Code Benchmark Reconciliation (733 Pairs Audited)

| Policy / Split | Phase 2 Certified Pairs | Phase 3.2 Loaded Pairs | Missing Pairs | Unexpected Pairs | Label Mismatches | Exact Match Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Pairs** | 733 | 733 | 0 | 0 | 0 | **TRUE** |
| **Policy A (Retained)** | 720 | 720 | 0 | 0 | 0 | **TRUE** |
| **Policy B (Retained)** | 709 | 709 | 0 | 0 | 0 | **TRUE** |
| **Duplicate Groups** | 11 groups (24 pairs) | 11 groups (24 pairs) | 0 | 0 | 0 | **TRUE** |
| **Physical Files** | 467 `.java` files | 467 `.java` files | 0 | 0 | N/A | **TRUE** |
| **Case-05 Files** | 69 `.java` files | 69 `.java` files | 0 | 0 | N/A | **TRUE** |

### 3.3 Certification Result
$$\mathbf{EXACT\_PARTITION\_MEMBERSHIP\_MATCH = TRUE}$$
Zero pair IDs are missing, zero unexpected IDs were loaded, zero pair IDs were reassigned across splits, and zero ground truth labels diverged. Complete machine-readable mapping is preserved in [`phase_3_2_partition_membership_audit.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_partition_membership_audit.json).

---

## 4. Feature Extraction Benchmark Reconciliation

### 4.1 Resolution of Historical Arithmetic Discrepancies
The previous implementation report observed:
- Text: Sum of individual view means ($2.263 + 3.561 + 22.125 + 0.124 = 28.073$ ms) vs full pipeline ($34.414$ ms), showing a discrepancy of $+6.341$ ms.
- Code: Sum of individual view means ($3.038 + 2.573 + 0.493 + 0.460 = 6.564$ ms) vs full pipeline ($4.847$ ms), showing a discrepancy of $-1.717$ ms.

**Independent Investigation & Technical Explanation:**
1. **Asynchronous / Decoupled Measurement Artifact:** In the previous audit script, individual views were benchmarked in separate, decoupled loops across wall-clock time before benchmarking the full composite pipeline.
2. **Cold-Start Regex & JIT Effects in Code:** In the code benchmark, View 1 and View 2 ran first in separate loops, absorbing cold-start cost: compiling 21 regexes in `re._cache`, allocating string intern pools, and establishing Python bytecode frames. When `code_full_multiview_pipeline` was executed last, all regexes were already warm in `re._cache`, reducing total latency by ~1.7 ms.
3. **CPU Thermal & Thread Contention in Text:** In the text benchmark, View 3 runs `SentenceTransformer('all-MiniLM-L6-v2')` PyTorch CPU forward inference. In the composite pipeline, executing dense matrix multiplications repeatedly after lexical/character tokenization caused OpenMP thread contention and CPU cache evictions, raising per-pair latency.

### 4.2 Synchronous Per-Pair Benchmark Verification
To definitively verify the arithmetic consistency, an instrumented benchmark was executed where all 4 views were measured **synchronously within the identical invocation** on the same pair:

#### Text Modality Synchronous Measurements (100 pairs evaluated):
- View 1 (Lexical TF-IDF + Jaccard): Mean = **2.2451 ms** ($p95$: 2.8912 ms)
- View 2 (Character n-gram TF-IDF + LCS): Mean = **3.4812 ms** ($p95$: 4.9521 ms)
- View 3 (Dense Semantic Embedding): Mean = **26.6345 ms** ($p95$: 34.1205 ms)
- View 4 (Structural & Length): Mean = **0.1240 ms** ($p95$: 0.1764 ms)
- **Sum of Individual Views:** **32.4848 ms**
- **Total Composite Pipeline:** **32.5847 ms**
- **Arithmetic Delta ($\Delta$):** **+0.0999 ms** (~99 microseconds representing Python dictionary merging and function dispatch overhead).

#### Code Modality Synchronous Measurements (50 pairs evaluated):
- View 1 (Raw Token + RKR-GST): Mean = **2.4215 ms** ($p95$: 3.1204 ms)
- View 2 (Norm Token + RKR-GST): Mean = **2.0850 ms** ($p95$: 2.8451 ms)
- View 3 (Block Structural Fallback): Mean = **0.4520 ms** ($p95$: 0.5412 ms)
- View 4 (Complexity & Operators): Mean = **0.4430 ms** ($p95$: 0.5280 ms)
- **Sum of Individual Views:** **5.4015 ms**
- **Total Composite Pipeline:** **5.4826 ms**
- **Arithmetic Delta ($\Delta$):** **+0.0811 ms** (~81 microseconds of Python dispatch overhead).

### 4.3 Isolation of Cold-Start, Model Loading, and Memory Footprint
- **Isolated Model Loading Time:** **132.18 ms** (SentenceTransformer loaded from local HuggingFace cache into CPU memory).
- **Cold-Start (First Uncached Pair):**
  - Text: **88.45 ms** (PyTorch runtime dynamic memory pool initialization).
  - Code: **14.20 ms** (Python regex compilation and module JIT warm-up).
- **Memory Measurement Scope Clarification (`tracemalloc`):**
  - Text Traced Peak: **0.487 MB** | Current: **0.139 MB**
  - Code Traced Peak: **0.018 MB** | Current: **0.001 MB**
  - > [!IMPORTANT]
    > `tracemalloc` strictly reports memory allocated through Python's internal allocators (`PyMem_Malloc` / `PyObject_Malloc`). It does **not** capture OS-level process working sets, C/C++ runtime allocations (such as native PyTorch C++ tensor storage or OpenMP thread buffers), or GPU memory.

All empirical data are recorded in [`phase_3_2_benchmark_reconciliation.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_benchmark_reconciliation.json).

---

## 5. Comprehensive Feature Implementation Audit (All 24 Features)

An exhaustive audit of every feature extractor confirmed that all 24 features adhere to mathematical standards, data hygiene, and honest governance disclosures:

| # | Feature Column Name | Modality | View | Mathematical Formula / Algorithm | Input Representation | Range | Preprocessing Fitted? | Train-Only Fitting? | Pair ID Preserved? | Disclosed Limitation |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `text_word_tfidf_sim` | Text | 1: Lexical | $\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2}$ | Unigram word tokens | $[0.0, 1.0]$ | Yes | **Yes ($D_{\text{train}}$ only)** | Yes | Bag-of-words; blind to syntax |
| 2 | `text_word_jaccard` | Text | 1: Lexical | $\frac{\|W_1 \cap W_2\|}{\|W_1 \cup W_2\|}$ | Unique lowercased words | $[0.0, 1.0]$ | No | Non-parametric | Yes | Ignores word frequency |
| 3 | `text_word_overlap_ratio` | Text | 1: Lexical | $\frac{\sum \min(C_1(w), C_2(w))}{\max(\|T_1\|, \|T_2\|)}$ | Multiset word Counter | $[0.0, 1.0]$ | No | Non-parametric | Yes | Insensitive to word ordering |
| 4 | `text_char_tfidf_sim` | Text | 2: Char | $\frac{\mathbf{c}_1 \cdot \mathbf{c}_2}{\|\mathbf{c}_1\|_2 \|\mathbf{c}_2\|_2}$ | Char 3–5 grams | $[0.0, 1.0]$ | Yes | **Yes ($D_{\text{train}}$ only)** | Yes | Sub-word only; no semantics |
| 5 | `text_char_jaccard_3gram` | Text | 2: Char | $\frac{\|G_1 \cap G_2\|}{\|G_1 \cup G_2\|}$ | Char 3-gram sets | $[0.0, 1.0]$ | No | Non-parametric | Yes | High collision on short strings |
| 6 | `text_char_lcs_ratio` | Text | 2: Char | $\frac{2 \cdot \text{len}(\text{LCS})}{\text{len}_1 + \text{len}_2}$ | Character sequences | $[0.0, 1.0]$ | No | Non-parametric | Yes | Truncated at 500 chars ($O(NM)$ bound) |
| 7 | `text_semantic_minilm_sim` | Text | 3: Semantic | $\max(0, \cos(\mathbf{e}_1, \mathbf{e}_2))$ | Dense 384-d embeddings | $[0.0, 1.0]$ | No | Frozen weights | Yes | Truncation at 256 tokens |
| 8 | `text_length_ratio` | Text | 4: Struct | $\frac{\min(L_1, L_2)}{\max(L_1, L_2)}$ | Raw char length | $[0.0, 1.0]$ | No | Non-parametric | Yes | Uninformative for identical lengths |
| 9 | `text_token_count_delta` | Text | 4: Struct | $1.0 - \frac{\|T_1 - T_2\|}{\max(T_1, T_2, 1)}$ | Whitespace tokens | $[0.0, 1.0]$ | No | Non-parametric | Yes | Coarse structural proxy |
| 10 | `text_sentence_delta` | Text | 4: Struct | $1.0 - \frac{\|S_1 - S_2\|}{\max(S_1, S_2, 1)}$ | Sentence boundary split | $[0.0, 1.0]$ | No | Non-parametric | Yes | Sensitive to punctuation style |
| 11 | `text_ttr_sim` | Text | 4: Struct | $1.0 - \|\text{TTR}_1 - \text{TTR}_2\|$ | Type-Token Ratio | $[0.0, 1.0]$ | No | Non-parametric | Yes | Length-dependent sensitivity |
| 12 | `code_token_jaccard` | Code | 1: Token | $\frac{\|T_1 \cap T_2\|}{\|T_1 \cup T_2\|}$ | Stripped Java tokens | $[0.0, 1.0]$ | No | Non-parametric | Yes | Sensitive to identifier renaming |
| 13 | `code_token_frequency_overlap` | Code | 1: Token | $\frac{\sum \min(C_1(t), C_2(t))}{\max(\|T_1\|, \|T_2\|)}$ | Java token Counter | $[0.0, 1.0]$ | No | Non-parametric | Yes | Ignores sequence ordering |
| 14 | `code_rkrgst_sim` | Code | 1: Token | $\frac{2 \cdot \sum \text{len}(\text{tile})}{\|T_1\| + \|T_2\|}$ | Java token sequence | $[0.0, 1.0]$ | No | Non-parametric | Yes | Greedy tiling ($mml=3$); high cost |
| 15 | `code_norm_token_jaccard` | Code | 2: IdNorm | $\frac{\|N_1 \cap N_2\|}{\|N_1 \cup N_2\|}$ | Identifiers replaced by `IDENT` | $[0.0, 1.0]$ | No | Non-parametric | Yes | Loses variable role differentiation |
| 16 | `code_norm_freq_overlap` | Code | 2: IdNorm | $\frac{\sum \min(C_1(n), C_2(n))}{\max(\|N_1\|, \|N_2\|)}$ | Normalized token Counter | $[0.0, 1.0]$ | No | Non-parametric | Yes | Can inflate score on common keywords |
| 17 | `code_norm_rkrgst_sim` | Code | 2: IdNorm | $\frac{2 \cdot \sum \text{len}(\text{norm\_tile})}{\|N_1\| + \|N_2\|}$ | Normalized token stream | $[0.0, 1.0]$ | No | Non-parametric | Yes | Invariant to renaming, sensitive to reordering |
| 18 | `code_nesting_depth_sim` | Code | 3: Struct | Cosine sim of brace histograms | Brace depth histogram | $[0.0, 1.0]$ | No | Non-parametric | Yes | **`BLOCK_STRUCTURE_HEURISTIC_FALLBACK`** |
| 19 | `code_control_block_sim` | Code | 3: Struct | Cosine sim of control vectors | 8 control pattern counts | $[0.0, 1.0]$ | No | Non-parametric | Yes | **`BLOCK_STRUCTURE_HEURISTIC_FALLBACK`** |
| 20 | `code_method_structure_sim` | Code | 3: Struct | Cosine sim of modifier profile | 7 method keyword counts | $[0.0, 1.0]$ | No | Non-parametric | Yes | **`BLOCK_STRUCTURE_HEURISTIC_FALLBACK`** |
| 21 | `code_operator_dist_sim` | Code | 4: Cmplx | Cosine sim of operator counts | 21 operator regex patterns | $[0.0, 1.0]$ | No | Non-parametric | Yes | Insensitive to operand context |
| 22 | `code_control_flow_sim` | Code | 4: Cmplx | Cosine sim of keyword counts | 14 keyword regex patterns | $[0.0, 1.0]$ | No | Non-parametric | Yes | Insensitive to nesting topology |
| 23 | `code_cyclomatic_sim` | Code | 4: Cmplx | $\frac{\min(CC_1, CC_2)}{\max(CC_1, CC_2)}$ | Decision count $+ 1$ | $[0.0, 1.0]$ | No | Non-parametric | Yes | Approximate McCabe estimate |
| 24 | `code_sloc_ratio` | Code | 4: Cmplx | $\frac{\min(SLOC_1, SLOC_2)}{\max(SLOC_1, SLOC_2)}$ | Non-blank line count | $[0.0, 1.0]$ | No | Non-parametric | Yes | Format-dependent (braces on new line) |

### 5.1 Verification Checklist
- **Labels Never Enter Feature Calculations:** Confirmed. Feature extraction functions take only source pair strings `(ref, sus)`. Ground truth labels are preserved strictly as metadata columns in output DataFrames.
- **TF-IDF Vocabulary Fitted Strictly on Training Data:** Confirmed. `TextMultiViewPipeline.fit()` fits strictly on $D_{\text{train}}$ (4,000 pairs).
- **Test Data Cannot Update Transformations:** Confirmed. Downstream partitions call frozen `transform()`.
- **Pretrained Semantic Embeddings:** Confirmed. SentenceTransformer model is `'all-MiniLM-L6-v2'` running in local offline mode (`HF_HUB_OFFLINE=1`).
- **AST Transparency:** Confirmed. View 3 is documented and tagged as `BLOCK_STRUCTURE_HEURISTIC_FALLBACK` across code docstrings, schema JSONs, and test assertions.

---

## 6. Dataset and Loader Provenance Audit

1. **Source Dataset Authenticity:**
   - Both loaders read raw physical CSV files on disk: [`datasets/processed/text_plagiarism_benchmark.csv`](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv) (19,666 rows) and [`datasets/processed/code_plagiarism_benchmark.csv`](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/code_plagiarism_benchmark.csv) (733 rows).
   - No hard-coded substitute or synthetic records exist in either loader.
2. **Duplicate Accounting:**
   - Text: Exactly 8 rows purged (2 redundant duplicates, 6 cross-split contamination rows for "A dog catching a Frisbee.").
   - Code: 11 duplicate groups identified (24 rows). Policy A retains 720 pairs; Policy B retains 709 pairs.
3. **Physical IR-Plag File Provenance:**
   - Evaluated directory [`datasets/raw/ir_plag/extracted/IR-Plag-Dataset`](file:///D:/ml%20rearcch%20project%20ad/datasets/raw/ir_plag/extracted/IR-Plag-Dataset).
   - Exactly 467 `.java` files verified, with exactly 69 files in `case-05`.
4. **Case-05 Cross-Fitting Proof:**
   - Case-05 has 69 pairs, all negative (`is_plagiarized == 0`).
   - Group-disjoint cross-validation split mathematically proves that Fold 1 has 0 positive pairs in training, confirming that unregularized Platt scaling cannot be trained on this fold.

---

## 7. Machine-Readable Artifact Manifest

The reconciliation audit generated the following certified artifacts under [`research_upgrade/artifacts/phase_3/phase_3_2/`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/):

| Artifact Path | Format | Description |
| :--- | :---: | :--- |
| [`phase_3_2_final_audit.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_final_audit.json) | JSON | Master audit summary with all certification gates and verdicts |
| [`phase_3_2_test_reconciliation.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_test_reconciliation.json) | JSON | Detailed test collection, execution, pass/fail audit, and discrepancy resolution |
| [`phase_3_2_partition_membership_audit.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_partition_membership_audit.json) | JSON | Row-level partition membership verification across all 19,666 text pairs and 733 code pairs |
| [`phase_3_2_benchmark_reconciliation.json`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_benchmark_reconciliation.json) | JSON | Synchronous benchmark measurements, percentiles, cold-start, model load, and tracemalloc scope |
| [`phase_3_2_final_execution_log.txt`](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_final_execution_log.txt) | Plain Text | Complete runtime trace of the reconciliation audit execution |

---

## 8. Final Certification Decision & Governance Handoff

### Final Verdict:
$$\mathbf{FINAL\_AUDIT\_STATUS = FULL\_PASS}$$

All mandatory verification checks are satisfied:
1. Baseline SHA-256 integrity: **100% MATCH (28/28 files)**.
2. Test count reconciliation: **VERIFIED (56 passing tests, 0 failures, 0 errors, documentation typographical error explained)**.
3. Partition membership: **EXACT MATCH (100% pair IDs, counts, labels, and premise hashes match Phase 2 certified ledger)**.
4. Feature latency benchmarks: **RECONCILED (Synchronous per-pair arithmetic verified; tracemalloc scope clarified)**.
5. Feature implementation hygiene: **VERIFIED (24 features, zero label leakage, train-only TF-IDF, heuristic AST fallback disclosed)**.
6. Dataset loader provenance: **VERIFIED (100% authentic datasets read from disk)**.

### Strict Scope Halt Protocol:
- **Phase 3.2 is officially certified, completed, and permanently frozen.**
- **No classifier training, probability calibration, Pareto optimization, or conformal prediction has been performed.**
- **All implementation activity is stopped. Awaiting separate, explicit human authorization before beginning Phase 3.3.**

