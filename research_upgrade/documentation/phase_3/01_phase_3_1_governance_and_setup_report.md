# Phase 3.1 Implementation Report: Repository Setup, Governance Enforcement & Baseline Freezing

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 3.1 — Repository Preparation, Baseline Freezing & Governance  
**Date:** November 11, 2026  
**Auditor & Systems Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Execution Status:** ACCEPTANCE CRITERIA FULLY SATISFIED (Phase 3.2 Blocked Pending Approval)

---

## 1. Executive Summary

In accordance with explicit human authorization for **Phase 3.1 ONLY**, the repository infrastructure, governance enforcement mechanisms, reproducibility configurations, and automated anti-leakage test suites have been successfully established.

### Strict Governance Enforcements Verified:
1. **Baseline Immutability:** All 28 legacy Python source files in `src/` were audited byte-for-byte against the frozen [repository_sha256_manifest.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/repository_sha256_manifest.csv). Exactly 28/28 files match their SHA-256 hashes with zero discrepancies and zero untracked file injections.
2. **Phase 2 Baseline Preservation:** All Phase 2 documentation deliverables (Documents 01–16 in `research_upgrade/documentation/phase_2_final/`), dataset partitions, and 10 structured audit artifacts remain completely untouched.
3. **No Phase 3.2 Execution:** No Phase 3.2 modules (data loaders, feature extractors, cross-encoders, fusion models) were implemented. Phase 3 execution remains strictly halted at Phase 3.1.
4. **Automated Test Suite Success:** All 22 tests across `test_governance.py`, `test_leakage_protection.py`, and `test_rkrgst.py` executed cleanly with 0 failures and 0 errors in 1.586s.

---

## 2. Inventory of Newly Created Phase 3.1 Deliverables

All Phase 3.1 modules and test suites were created cleanly in new, dedicated namespaces without modifying the legacy `src/` codebase:

| Relative Path | Component Category | File Size (Bytes) | SHA-256 Fingerprint (Prefix) | Functional Purpose |
| :--- | :--- | :---: | :---: | :--- |
| `research_upgrade/src/__init__.py` | Package Root | 410 | `ac20dff1e82d` | Initializes `research_upgrade.src` package with version `3.1.0`. |
| `research_upgrade/src/config.py` | Configuration | 6,703 | `e004b99c738f` | Global paths, deterministic seed (`42`), partition sizes, routing states, and hyperparameters. |
| `research_upgrade/src/governance.py` | Governance | 5,698 | `1395ebb1db47` | Automated runtime hash auditing, tamper detection, and baseline protection assertions. |
| `research_upgrade/src/logging_utils.py` | Experiment Utilities | 4,061 | `0a56833e0cff` | Standardized logger, timestamped run directory generator, and JSON/manifest serializers. |
| `tests/test_governance.py` | Automated Test Suite | 4,702 | `95f7138b` | Unit tests asserting all 28 baseline hashes match, zero untracked files, and seed determinism. |
| `tests/test_leakage_protection.py` | Automated Test Suite | 7,990 | `462185a0` | Unit tests asserting 0 cross-split premise leaks, 8 purged rows, and Case-05 single-class proof. |
| `research_upgrade/audits/phase_3/run_phase_3_1_governance_audit.py` | Master Audit Script | 6,956 | `d437917e` | Master verification runner executing tests and generating machine-readable manifest & log. |
| `research_upgrade/artifacts/phase_3/phase_3_1_implementation_manifest.json` | Implementation Artifact | 2,750 | `b94c8e71` | Machine-readable manifest registering all created files, hashes, and acceptance verification. |
| `research_upgrade/artifacts/phase_3/phase_3_1_execution_log.txt` | Execution Log | 3,365 | `674a5802` | Clean execution trace capturing test results, hash matches, and audit output. |

---

## 3. Detailed Technical Tasks Addressed

### Task 1: Verification of Existing SHA-256 Baseline Manifest
The runtime verification engine in `research_upgrade/src/governance.py` loaded [repository_sha256_manifest.csv](file:///D:/ml%20rearcch%20project%20ad/research_upgrade/audits/phase_2_final/repository_sha256_manifest.csv) and computed the cryptographic SHA-256 digest for every file in `src/`.
- **Registered Files:** Exactly 28 files (17 in `src/`, 6 in `src/code/`, 5 in `src/text/`).
- **Matched Checksums:** 28 / 28 ($100.0\%$).
- **Mismatches / Missing:** 0.
- **Untracked / Injected Files:** 0.

### Task 2: Automated Baseline Integrity Tests (`tests/test_governance.py`)
Implemented test suite `TestBaselineGovernance` containing 8 rigorous tests:
1. `test_baseline_manifest_exists`: Verifies manifest presence.
2. `test_registered_file_count_is_exactly_28`: Verifies partition sum ($17 + 6 + 5 = 28$).
3. `test_all_28_files_exist_on_disk`: Verifies physical existence.
4. `test_all_28_sha256_hashes_match_manifest`: Cryptographic byte-level equality.
5. `test_no_untracked_files_in_src`: Detects unauthorized file injections into `src/`.
6. `test_governance_verification_function`: Asserts `verify_baseline_integrity()` returns `"PASSED"`.
7. `test_assert_baseline_intact_does_not_raise`: Asserts safety assertion executes cleanly.
8. `test_deterministic_seed_initialization`: Asserts Python and NumPy produce identical random sequences under `set_global_seed(42)`.

### Task 3: Reproducible Experiment Configuration (`research_upgrade/src/config.py`)
Established centralized, immutable constants:
- **Global Seed:** `RANDOM_SEED = 42`.
- **Text Modality Partitions:** Raw rows: 19,666; Purged: 8 (6 cross-split, 2 redundant duplicates); Clean: 19,658.
  - $D_{\text{train}}$: 4,000 pairs (1,349 premises)
  - $D_{\text{val}}$: 1,500 pairs (508 premises)
  - $D_{\text{cal-prob}}$: 1,000 pairs (336 premises)
  - $D_{\text{cal-conf}}$: 1,000 pairs (334 premises)
  - $D_{\text{dev-res}}$: 2,337 pairs (791 premises)
  - $D_{\text{test}}$: 3,000 pairs (1,015 premises)
  - $D_{\text{test-res}}$: 6,821 pairs (2,307 premises)
- **Code Modality Parameters:** Raw pairs: 733; 24 rows across 11 duplicate groups; Policy A retained: 720; Policy B retained: 709; Reversed duplicates: 0; Total IR-Plag Java files: 467.
- **Case-05 Constants:** 69 physical Java files; 107 pairs; 16 base program identities; all 53 positives share `prog_original`.
- **Conformal & Calibration Settings:** $\epsilon = 0.05$ (attainability $n \ge 19$); $M = 10$ uniform bins; NLL clipping $\delta = 10^{-15}$.

### Task 4: Dataset Partition and Anti-Leakage Protection Tests (`tests/test_leakage_protection.py`)
Implemented test suite `TestDatasetLeakageProtection` containing 5 rigorous tests:
1. `test_text_leakage_report_zero_cross_overlap`: Asserts that all off-diagonal entries in the $7 \times 7$ partition overlap matrix are strictly 0 (zero cross-split premise leakage).
2. `test_text_exclusion_ledger_exact_reconciliation`: Reconciles exactly 8 purged rows (6 cross-split premise contamination, 2 redundant exact duplicates).
3. `test_code_duplicate_accounting_reconciliation`: Asserts 733 raw pairs, 24 duplicate rows, 11 groups, Policy A (720 pairs), Policy B (709 pairs), and 0 reversed pairs.
4. `test_case_05_cross_fitting_proof_and_physical_files`: Asserts 69 physical Java files, 16 base identities, Fold 1 train infeasible ($N_{\text{pos}}^{\text{train}} = 0$), and Folds 2–5 test positives $= 0$ (Platt feasible on train, test sensitivity undefined, raw margin fallback enforced).
5. `test_zero_overlap_lexical_positives_accounting`: Validates separation between the 33 zero-word-overlap pairs in active test ($3.22\%$) and the 119 historical benchmark zero-lexical-overlap pairs ($11.19\%$).

### Task 5: Experiment Logging & Artifact Naming Conventions (`research_upgrade/src/logging_utils.py`)
- Standardized logging configuration routing timestamped records simultaneously to stdout and dedicated run log files.
- Experiment runs generate isolated directories under `research_upgrade/artifacts/phase_3/runs/{YYYYMMDD_HHMMSS}_{phase_tag}_{experiment_name}/`.
- Automated initialization of `run_config.json` and `run_execution.log` for future experimental runs.

### Task 6: Phase 3 Implementation Directory
Created clean directory structure:
- `research_upgrade/src/` (Package namespace)
- `research_upgrade/audits/phase_3/` (Phase 3 audit scripts)
- `research_upgrade/artifacts/phase_3/` (Phase 3 machine-readable artifacts)
- `research_upgrade/artifacts/phase_3/runs/` (Future experimental run directory)
- `research_upgrade/documentation/phase_3/` (Phase 3 reports and documentation)

---

## 4. Acceptance Criteria Verification Matrix

| Acceptance Criterion | Verification Method | Result / Evidence | Status |
| :--- | :--- | :--- | :---: |
| **1. All 28 Baseline Hashes Match** | `research_upgrade/src/governance.py` | 28/28 SHA-256 hashes match `repository_sha256_manifest.csv` byte-for-byte; 0 untracked files in `src/`. | **FULL PASS** |
| **2. Governance Tests Execute Successfully** | `tests/test_governance.py` | 8/8 unit tests passed cleanly in 1.586s. | **FULL PASS** |
| **3. Dataset Leakage Tests Execute Successfully** | `tests/test_leakage_protection.py` | 5/5 unit tests passed (0 leakage in $7\times 7$ matrix, 8 purged rows verified, Case-05 verified). | **FULL PASS** |
| **4. Reproducibility Configuration Documented** | `research_upgrade/src/config.py` | Seed 42, deterministic function, partition sizes, calibration hyperparameters fully defined. | **FULL PASS** |
| **5. Machine-Readable Manifest Created** | `phase_3_1_implementation_manifest.json` | All 7 newly created code/test/audit files registered with size and SHA-256 fingerprint. | **FULL PASS** |
| **6. Clean Execution Log Generated** | `phase_3_1_execution_log.txt` | Complete, timestamped execution log recording test run and verification passes with exit code 0. | **FULL PASS** |

---

## 5. Execution Command Trace

The master Phase 3.1 audit script was executed via `.venv\Scripts\python.exe research_upgrade/audits/phase_3/run_phase_3_1_governance_audit.py`:

```
================================================================================
PHASE 3.1: REPOSITORY SETUP & GOVERNANCE ENFORCEMENT AUDIT
Timestamp: 2026-11-11T02:10:57.466836
Project Root: D:\ml rearcch project ad
================================================================================
Execution started at: 2026-11-11 02:10:57

[TASK 1 & 2] Verifying Baseline SHA-256 Manifest and Integrity...
  Status: PASSED
  Total registered files: 28
  Matched files: 28
  Mismatched files: 0
  Missing files: 0
  Untracked files in src/: 0
  --> PASS: All 28 baseline files match registered SHA-256 checksums byte-for-byte.

[TASK 2 & 4] Executing Automated Test Suites under tests/...
Ran 22 tests in 1.586s
OK

  Total tests run: 22
  Failures: 0
  Errors: 0
  --> PASS: All 22 automated tests passed cleanly.

[TASK 5 & 6] Auditing Phase 3 Created Files & Generating Implementation Manifest...
  Tracked: research_upgrade/src/__init__.py (410 bytes, SHA-256: ac20dff1e82dee76...)
  Tracked: research_upgrade/src/config.py (6703 bytes, SHA-256: e004b99c738fdb91...)
  Tracked: research_upgrade/src/governance.py (5698 bytes, SHA-256: 1395ebb1db47f110...)
  Tracked: research_upgrade/src/logging_utils.py (4061 bytes, SHA-256: 0a56833e0cffb6fe...)
  Tracked: tests/test_governance.py (4702 bytes, SHA-256: 95f7138be43e4f68...)
  Tracked: tests/test_leakage_protection.py (7990 bytes, SHA-256: 462185a07bc93f89...)
  Tracked: research_upgrade/audits/phase_3/run_phase_3_1_governance_audit.py (6956 bytes, SHA-256: d437917ee5daaed2...)

Saved implementation manifest to: D:\ml rearcch project ad\research_upgrade\artifacts\phase_3\phase_3_1_implementation_manifest.json
Saved execution log to: D:\ml rearcch project ad\research_upgrade\artifacts\phase_3\phase_3_1_execution_log.txt

================================================================================
PHASE 3.1 ACCEPTANCE CRITERIA FULLY MET. READY FOR PHASE 3.2 AUTHORIZATION.
================================================================================
```

---

## 6. Implementation Gate Status & Next Steps

Phase 3.1 is **100% complete**. All acceptance criteria have been verified and recorded.

In accordance with strict project governance rules:
- **DO NOT proceed to Phase 3.2 automatically.**
- Phase 3.2 (Dataset Loaders & Multi-View Feature Extraction Pipelines) remains **STRICTLY BLOCKED** until explicit written human authorization is provided.
- The assistant has stopped and is waiting for your review and instruction.

