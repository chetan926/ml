# System Modular Architecture and Pipeline Specification (Version 3.4)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.3 (Final Dataset, Routing, and Integrity Verification)  
**Document Version:** Version 3.4  
**Date:** March 2026  
**Status:** FULL AUDIT CERTIFIED (Empirical Ground-Truth Verified)

---

## 1. System High-Level Architecture Overview

The system architecture addresses large-scale academic cohort plagiarism detection by decoupling cheap candidate screening from expensive deep-transformer re-ranking through an uncertainty-aware routing layer.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              END-TO-END SYSTEM PIPELINE ARCHITECTURE                                   │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ TIER 1: Ingestion & Multi-View Candidate Screening       │
                       │ • Text: Char/Word TF-IDF + MiniLM-L6-v2 + Ling          │
                       │ • Code: Token Jaccard + AST Isomorphism + Halstead/CC   │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Routing Logic on s_cand(x)    │
                                    └───────┬───────────────┬───────┘
                     ┌──────────────────────┘               └──────────────────────┐
                     ▼                                                             ▼
     [s_cand < theta_low]                                          [s_cand >= theta_high]
┌─────────────────────────────┐                               ┌─────────────────────────────┐
│  LOW_RISK_REVIEW_DEFERRED   │                               │ HIGH_RISK_CANDIDATE_FLAGGED │
│  (Pruned / Budget Conserved)│                               │ (Direct High-Risk Bypass)   │
└─────────────────────────────┘                               └──────────────┬──────────────┘
                     │                                                       │
                     │         [theta_low <= s_cand < theta_high]            │
                     └──────────────────────┬────────────────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ TIER 2: Deep Re-Ranking & Uncertainty Quantification    │
                       │ • Text: Cross-Encoder (MS-MARCO-MiniLM) + Platt + Conf  │
                       │ • Code: Cross-Vector Re-ranking + Raw Margin            │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ Standardized Output Decision  │
                                    └───────┬───────────────┬───────┘
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
        [C = {1} & P >= gamma_up]   [C = {0,1} or Margin]      [C = Empty (Text)]
┌─────────────────────────────┐┌─────────────────────────┐┌─────────────────────────┐
│ HIGH_RISK_CANDIDATE_FLAGGED ││ AMBIGUOUS_HUMAN_REVIEW  ││   ABSTAIN_UNRESOLVED    │
└──────────────┬──────────────┘└────────────┬────────────┘└────────────┬────────────┘
               │                            │                          │
               └────────────────────────────┼──────────────────────────┘
                                            ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ TIER 3: Explainable Evidence Dossier Synthesizer        │
                       │ • Text: Myers character span diffs & attention maps     │
                       │ • Code: AST subtree isomorphic diffs & alpha-ledger     │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │ TIER 4: Institutional Human Adjudication Interface      │
                       │ • Human Instructor / Disciplinary Committee Audit       │
                       │ • Final academic integrity findings (Non-delegable)     │
                       └─────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion and Invariant Dataset Path Architecture

### 2.1 Actual Repository Filesystem Paths
Audits of the physical disk established that no root `data/` directory exists. All data loaders and ingestion pipelines must reference the verified filesystem structure:
- **Processed Text Benchmark:** `datasets/processed/text_plagiarism_benchmark.csv` (19,666 raw rows).
- **Processed Code Benchmark:** `datasets/processed/code_plagiarism_benchmark.csv` (733 raw pairs).
- **Raw Natural Language Corpus:** `datasets/raw/text_mit_plagiarism/snli_val.parquet` (9,842 rows) and `snli_test.parquet` (9,824 rows).
- **Raw Java Source Code Corpus:** `datasets/raw/ir_plag/extracted/IR-Plag-Dataset/` (467 distinct `.java` files across `case-01` to `case-07`).

### 2.2 Data Ingestion Filter Pipeline
To eliminate data leakage and invalid duplicate pairs, data loaders in `research_upgrade/src/data/` apply strict pre-filtering:
1. **Premise Contamination Purge (Text):** 6 rows associated with the shared cross-split premise `"A dog catching a Frisbee."` are purged by premise ID:
   - `TXT_001181`, `TXT_001182`, `TXT_001183` in `val`.
   - `TXT_016302`, `TXT_016303`, `TXT_016304` in `test`.
2. **Exact Duplicate Purge (Text):** 2 duplicate rows are purged (keeping first occurrence):
   - `TXT_002505` (duplicate of `TXT_002504`).
   - `TXT_003942` (duplicate of `TXT_003940`).
3. **Exact Duplicate Purge (Code):** 24 redundant rows (12 duplicate pairs) across cases 01, 02, 03, and 06 are purged, yielding 709 unique pairs.

---

## 3. Four-Tier Processing Architecture Specification

### Tier 1: Candidate Generation & Multi-View Screening
- **Input:** Raw document pair $(D_A, D_B)$ from submission cohort.
- **Processing:** Multi-view extraction (TF-IDF, SBERT, AST, Metrics) with $O(M)$ document caching.
- **Routing:**
  - $s_{\text{cand}} < \theta_{\text{low}} \implies$ Emit `LOW_RISK_REVIEW_DEFERRED`.
  - $s_{\text{cand}} \ge \theta_{\text{high}} \implies$ Emit `HIGH_RISK_CANDIDATE_FLAGGED` (Direct Bypass).
  - $\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}} \implies$ Escalate to Tier 2.

### Tier 2: Deep Re-Ranking & Uncertainty Quantification
- **Input:** Escalated candidate pairs ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$).
- **Text Processing:**
  - Forward pass through `cross-encoder/ms-marco-MiniLM-L-6-v2` yielding logit $z$.
  - Calibrated posterior probability $\hat{P}_{\text{cal}} = \sigma(A^* z + B^*)$.
  - Conformal prediction set $\mathcal{C}(\mathbf{x}) = \{y : 1 - \hat{P}_{\text{cal}}(Y=y) \le \hat{q}_{1-\epsilon}\}$.
- **Code Processing:**
  - Deep cross-vector re-ranking producing raw decision margin $z_{\text{code}} \in \mathbb{R}$.
  - Conformal prediction sets formally disclaimed.

### Tier 3: Explainable Evidence Dossier Synthesizer
- **Trigger:** Invoked strictly when pair is routed to `HIGH_RISK_CANDIDATE_FLAGGED`, `AMBIGUOUS_HUMAN_REVIEW`, or `ABSTAIN_UNRESOLVED`.
- **Text Artifacts:** Myers diff character span alignments, token-level matching matrix, and multi-view weight attribution vectors.
- **Code Artifacts:** Normalized AST subtree diffs, statement permutation ledgers, and $\alpha$-conversion identifier renaming maps.

### Tier 4: Institutional Human Adjudication Interface
- **Role:** Independent evaluation by course instructors, academic integrity officers, or disciplinary committees.
- **Boundary:** Machine learning pipeline outputs empirical statistical evidence and confidence intervals; **human officers issue final institutional verdicts**.

---

## 4. Latency, Memory, and Computational Resource Profile

To prevent mixing disparate metrics, resource tracking is strictly categorized:

| Processing Stage | Modality | Measured Wall-Clock Latency | Peak Memory Consumption | Computational FLOPs | Provenance Classification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage-1 Screening** | Text | 0.045 ms / pair | 142 MB (Document Cache) | $\sim 1.2 \times 10^5$ FLOPs | Historical Measured Result |
| **Stage-1 Screening** | Code | 0.058 ms / pair | 186 MB (AST Forest Cache) | $\sim 3.8 \times 10^5$ FLOPs | Historical Measured Result |
| **Stage-1 Target** | Both | 0.050 ms / pair | $< 256$ MB | $< 5.0 \times 10^5$ FLOPs | System Design Target |
| **Stage-2 Transformer** | Text | 4.046 ms / pair | 512 MB (PyTorch Runtime) | $\sim 2.1 \times 10^9$ FLOPs | Historical Measured Result |
| **Stage-2 Parsing** | Code | 2.050 ms / pair | 320 MB (JVM Runtime) | $\sim 8.5 \times 10^6$ FLOPs | Historical Measured Result |
| **Evidence Dossier** | Both | 1.850 ms / pair | 64 MB (Diff Engine) | $\sim 4.2 \times 10^5$ FLOPs | Measured Benchmark |

> [!NOTE]
> Latency metrics represent CPU execution on dedicated x86-64 hardware with warm-cache vector representations. Memory allocations are non-additive to time durations.

---

## 5. Legacy Codebase Preservation & Isolation Boundaries

To ensure complete reproducibility and protect original research artifacts:
1. **Legacy Directory (`src/`):** Contains exactly 28 Python source files (17 in root, 6 in `src/code/`, 5 in `src/text/`). All 28 files remain completely frozen under their published SHA-256 hashes.
2. **Research Upgrade Directory (`research_upgrade/src/`):** All Phase 3 implementation code will be authored exclusively within this dedicated namespace. No production files in `src/` will ever be overwritten.
3. **Reproducible Audit Scripts (`research_upgrade/audits/`):** Houses standalone, read-only audit scripts (such as `run_dataset_integrity_audit.py`) and their machine-readable ledgers.

