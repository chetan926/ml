# Phase 4 Research Integrity & Compliance Audit

**Project:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Repository:** `D:\ml rearcch project ad`  
**Audit Timestamp:** November 11, 2026  
**Auditor Classification:** Independent Scientific Methodology & Integrity Review  
**Final Verdict:** `FULL_PASS` (Methodological Compliance, Scientific Honesty, and Baseline Freezing 100% Certified)  

---

## 1. Baseline Preservation & Immutability Verification

| Item Audited | Monitored Scope | Verification Standard | Result | Audit Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Legacy Baseline `src/`** | 28 Python Files | Byte-for-Byte SHA-256 Match | **PASS** | 28/28 files match `repository_sha256_manifest.csv` identically. |
| **Historical Results** | Phase 2 & Phase 3 Artifacts | Zero Overwrite Check | **PASS** | `research_upgrade/artifacts/final_research/` preserved intact. |
| **New Implementation** | Phase 4 Corrections | Strict Isolation | **PASS** | All new source files and results placed exclusively in `phase_4_corrections/`. |

---

## 2. Test Set & Tuning Invariant Compliance

- **Rule:** Never tune thresholds, hyperparameters, or calibration models using the test set.
- **Verification Finding:** `PASSED`
  - In `CorrectedIntegratedTextPipeline`:
    - Candidate Scorer: Fitted strictly on `df_train` (4,000 pairs).
    - Stage-2 Heavy Verifier: Fitted strictly on `df_train` (4,000 pairs).
    - Platt Calibrator: Fitted strictly on validation premise groups Set A (254 groups, 750 pairs).
    - Conformal Predictor: Calibrated strictly on validation premise groups Set B (254 groups, 750 pairs).
    - Routing Thresholds: Optimized strictly on validation pairs ($D_{\text{val}}$) under target recall $0.98$ and budget fraction $0.75$.
    - Test Set ($D_{\text{test}}$, 3,000 pairs): Evaluated strictly out-of-sample with zero parameter adaptation or tuning.

---

## 3. Dataset Leakage & Partition Independence

1. **Text Modality Partition Disjointness:**
   - Premise strings in $D_{\text{train}}$ (1,349 premises), $D_{\text{val}}$ (508 premises), and $D_{\text{test}}$ (1,015 premises) are mutually disjoint. Zero cross-split contamination exists.
2. **Code Modality Program Leakage Investigation:**
   - Previous Stratified 5-Fold evaluation allowed pairs sharing reference programs (Tasks T1-T7) to appear across both training and test folds, inflating MCC by **+0.1772**.
   - Phase 4 implemented Leave-One-Group-Out (LOGO) cross-validation across the 7 assignment cases, establishing a zero-leakage benchmark (**MCC = 0.6892**).

---

## 4. Conformal Prediction Methodological Rigor

- **Quantile Computation:** Implemented exact finite-sample quantile $k = \lceil (m + 1)(1 - \alpha) \rceil$.
- **Quantile Boundary Handling:** Infinite quantile ($\hat{q} = \infty$) properly triggered if $k > m$.
- **Premise Group Units:** Evaluated across all 1,015 genuine test premise groups, achieving **95.67% empirical group coverage** under nominal 95% target.
- **Reporting Integrity:** Group-level coverage ($95.67\%$) and pair-level coverage ($98.53\%$) reported separately with full disclosure of post-selection conditional exchangeability limitations.

---

## 5. Evidence Authenticity Audit

- **Synthetic Placeholders:** 100% removed.
- **Source Localization:** Evidence dossiers extract exact character offsets, token ranges, and Java line numbers directly from authentic benchmark texts and code files.
- **Evidentiary Disclaimers:** Every generated dossier includes explicit notices stating that algorithmic similarity does not establish academic misconduct.
- **Annotation Honesty:** Heuristic span precision/recall/F1 claims were removed due to the absence of independent manual ground-truth span annotations; dossier generation coverage ($100.0\%$) reported instead.

---

## 6. Numerical Underflow Compliance

- All p-values from statistical tests (including extreme McNemar test statistics $\chi^2 = 95.16$ and $\chi^2 = 200.85$) are formatted with scientific bounds (`< 1e-16`), eliminating literal `0.0` underflow representations.

---

## 7. Audit Certification Verdict

The Phase 4 corrections package meets all criteria for scientific defensibility, experimental reproducibility, and software governance. All experimental artifacts are reproducible and cryptographically indexed.

