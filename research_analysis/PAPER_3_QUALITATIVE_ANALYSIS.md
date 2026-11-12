# Phase 12 — Paper 3 Qualitative Analysis

**Mehsen & Joshi (2024).** *Detection of Source Code Plagiarism Utilizing an Approach Based on
Machine Learning.* International Journal of Computing 23(1): 78–84. DOI 10.47839/ijc.23.1.3438.
Evidence: repository metadata + audit §2.3 (full-text PDF absent, Phase 8).

## Research background
- **Problem:** automating plagiarism detection when students deliberately disguise code.
- **Motivation/objectives:** exploit non-linear ensemble learning (Decision Trees, Random Forests,
  Hierarchical Clustering) over syntactic/token-frequency representations; quantify attack-level
  resilience.
- **Research questions:** which learner handles compound transformations best; how resilience varies
  across obfuscation levels.

## Methodology
- **Architecture:** Random Forest ensemble over structured token-frequency vectors + syntactic metrics.
- **Repository adaptation:** `src/code_random_forest.py`,
  `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)` — the audit
  records a parameter reconciliation (depth 10 claimed in earlier narrative vs depth 6 in audited code).
- **Evaluation:** adapted from the paper's private corpus to the IR-Plag Java benchmark; nested-LOGO
  OOF and L1–L6 per-level attack analysis under this project.

## Experimental investigation
- **Dataset:** original — private corpus; reproduction — IR-Plag (720 Policy-A pairs).
- **Baselines:** DT / hierarchical clustering within the paper; project compares vs Jaccard/RKRGST/SVM/ES-Plag.
- **Main findings (repository):** RF is the strongest *ensemble* peer on compound attacks; within the
  corrected protocol it trails the SVM comparator on balanced accuracy.

## Critical evaluation
- **Strengths:** ensemble robustness; per-level resilience reporting; decent inference cost
  (0.025 ms/pair).
- **Limitations:** bagging over a small hand-built feature space; no calibration/UQ; no cost-aware
  routing; no explainability layer; original evaluation corpus not public.
- **Uncertainty limitations:** none addressed.
- **Explainability limitations:** none addressed.
- **Unaddressed:** adversarial renaming (extra-benchmark) — project attack: RF'24 pos-recall 0.3216
  under systematic renaming (vs proposed β=1 at 0.8918).
- **Relevance to framework:** B-C4 comparator; intellectual parent of the phase-5 heavy-verifier
  choice (RF(500,d12) selected in 5/7 folds by inner cross-validation) — the framework operationalizes
  this lineage inside a calibrated, budget-aware, leak-free protocol rather than as a standalone
  classifier.
