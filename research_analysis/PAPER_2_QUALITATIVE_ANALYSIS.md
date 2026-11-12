# Phase 11 — Paper 2 Qualitative Analysis

**Gandhi, Gopalan & Prasad (2024).** *A Support Vector Machine based approach for plagiarism
detection in Python code submissions in undergraduate settings.* Frontiers in Computer Science 6,
art. 1393723. DOI 10.3389/fcomp.2024.1393723. Evidence: repository metadata + audit §2.2
(full-text PDF absent, Phase 8).

## Research background
- **Problem:** still-relevant operational failure — token-similarity tools (MOSS/JPlag-type) degrade
  under obfuscation: identifier renaming, comment manipulation, structural rearrangement.
- **Motivation/objectives:** replace single-signal tiling with a supervised multi-dimensional
  representation of code similarity for undergraduate assignments.
- **Research questions:** which engineered syntactic/token features discriminate plagiarized pairs
  under disguise; whether an RBF-SVM outperforms standard token baselines.

## Methodology
- **Approach:** 7-D feature vector — `/` standardized `[token-Jaccard, cosine, Levenshtein,
  keyword-distribution overlap, operator overlap, token-length, cyclomatic-complexity differential]`
  — via comment stripping and anonymized-identifier matching; RBF-kernel SVM classifier.
- **Repository adaptation:** Python `ast` analysis translated to **Java** over IR-Plag:
  `src/code_plagiarism.py`; `SVC(kernel='rbf', C=1.0, probability=True, random_state=42)`, operating
  threshold τ=0.3700 tuned on validation case-05; nested-LOGO evaluation under this project's
  corrected protocol.
- **Evaluation protocol (reproduction):** case-06/07 held-out adaptation tests; nested LOGO OOF on
  the full Policy-A benchmark in the upgrade.

## Experimental investigation (as recorded)
- **Datasets:** original paper — Python undergraduate submissions (private); this project — IR-Plag
  Java (720 Policy-A pairs).
- **Baselines:** token Jaccard-style comparators.
- **Main findings (repository):** multi-dimensional SVM beats token baselines; high resilience to
  renaming/comment manipulation within benchmark distributions.

## Critical evaluation
- **Strengths:** obfuscation-aware engineering; the strongest single-peer comparator in this
  project's corrected benchmark (LOGO F1 0.8363 / MCC 0.6754).
- **Methodological limitations:** fixed operating threshold; single-view learner over small hand-built
  space; no nested group-disjoint evaluation in the paper's own protocol as reproduced.
- **Computational limitations:** negligible inference cost (0.013 ms) but *extraction* dominates
  (2.050 ms AST parsing/pair — separated in the novelty audit); no budget formalization.
- **Generalization limitations:** original validation confined to Python coursework; Java adaptation
  adds validity boundary questions.
- **Uncertainty-handling limitations:** `probability=True` outputs uncalibrated pseudo-probabilities;
  no calibration evidence; no conformal coverage.
- **Explainability limitations:** none — SVM scores without casewise evidence traces.
- **Unaddressed challenges:** systematic extra-benchmark identifier renaming still collapses it in
  this project (pos-recall 0.2778) — its anonymized-identifier feature does not survive whole-sale
  renaming attacks.

## Research relevance
- **Learnings:** engineered syntactic features are a strong baseline; pure renaming-resilience claims
  do not transfer to extra-benchmark adversarial conditions.
- **Relevant to framework:** B-C3 headline comparator; the "hard peer" the proposed framework must
  at least match (it does ε-parity, plus wins on cost/robustness).
- **Unresolved:** calibrated decisions, budget economics, audit dossiers, adversarial augmentation.
- **Difference of proposed solution:** adapts the multi-view spirit but routes between a cheap
  screener and a per-fold-selected heavy verifier with calibration, conformal sets, budget
  constraints, and adversarial-augmented training — none of which Paper 2 addresses.
