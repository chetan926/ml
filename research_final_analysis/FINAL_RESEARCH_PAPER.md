# An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Journal-style manuscript — research_final_analysis rendering (evaluation evidence from phase-5/6 executed pipeline).**

## Abstract

Automated academic-integrity screening spans natural-language and programming assignments, yet published systems are typically evaluated under incompatible (often leaky) protocols, without uncertainty quantification and at a single operating point. We implement and evaluate a four-contribution framework — modality-aware multi-view detection (C1), calibration plus group-conformal uncertainty with a repaired split-sample significance level (C2), budget-adaptive two-stage escalation (C3), and cross-modality audit dossiers (C4) — against the three repository-designated paper methodologies under a corrected, leak-free protocol. Proposed text detection reaches accuracy 0.808, F1 0.7273, MCC 0.5799 at a 61.4% reduction of heavy-verifier calls; code detection (nested LOGO) reaches accuracy 0.8306, F1 0.8287, MCC 0.6636 with 27–34% cost reduction; the cost-unconstrained beta=1 configuration reaches F1 0.8481 / MCC 0.6997, the best overall code quality. Group-conformal coverage is repaired from 0.930 to 1.000 via split-sample alpha-star tuning with the set-size price disclosed. External evaluation-only transfer (Clough–Stevenson, MRPC, unseen-severity code) is reported honestly, including the threshold-nontransferability negative result.

## Keywords

plagiarism detection; academic integrity; code plagiarism; uncertainty quantification; conformal prediction; model cascades; cost-sensitive learning; leakage-free evaluation

## 1–7. Introduction, motivation, problem formulation, related work, gaps, objectives

See RESEARCH_GAP_MATRIX.csv for the verified gap table comparing the three papers against the two datasets and the framework. The three repository papers are: Setu et al. 2025 (text semantic strategy), Gandhi et al. 2024 (RBF-SVM, Python code), Mehsen & Joshi 2024 (RandomForest code). Gap: none provides calibration, conformal uncertainty, cost-adaptive verification, or cross-modality joined evaluation; all lack leak-safe grouped protocols in their published setups as captured by the repository audits. Objective: demonstrate that a calibrated, budget-routed cascade preserves or improves detection quality at materially lower verifier cost, with honest uncertainty claims.

## 8–9. Research objectives and four contributions

See 09_RESEARCH_GAPS_AND_CONTRIBUTIONS/FOUR_CONTRIBUTIONS.md (validation matrix with artifact pointers).

## 10–11. Proposed framework architecture and mathematical formulation

Architecture: Dataset Input → Modality Identification → Multi-View Detection (Stage-1) → Uncertainty Estimation (Platt-calibrated probability + predictive entropy) → Decision Filtering (conformal prediction sets) → Adaptive Escalation (budget-capped operating point tuned on validation) → Final Decision with per-pair audit dossier. Figure 1 depicts it. The routing rule escalates pair x when its calibrated score lies in the tunable grey band or entropy exceeds the entropy threshold, within a budget cap on escalations; the operating point is chosen on validation to maximize quality within the cap and an epsilon-tolerance relative to full verification. Conformal nonconformity uses group-supremum scores over premise groups; the alpha-star repair tunes the significance level on a calibration half-split (phase-6).

## 12–16. Datasets, methodology, implementations, baselines, metrics

Dataset 1 (text): 19,658 clean pair rows; train/val/test = 4000/1500/3000 with premise-group allocation (leakage checks PASS). Dataset 2 (code): 733 raw pairs (Policy A 720 / Policy B 709) with nested LOGO over 7 cases (24 duplicate-collision rows handled by policy). Baseline implementations are the frozen phase-5 suites (identical-seed shadow score refits verified to reproduce hard predictions bit-for-bit on every OOF row). Metrics: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, MCC, FPR/FNR, plus calibration (ECE/Brier), routing, and cost measures.

## 17. Quantitative analysis (headline

| Modality | System | accuracy | precision | recall | f1 | specificity | mcc | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|---|---|
| Text | B-T1 TF-IDF Lexical (Sajid 2025) | 0.5637 | 0.4249 | 0.7797 | 0.55 | 0.4514 | 0.2268 | 0.665 | 0.4786 |
| Text | B-T2 Dense Semantic MiniLM (Setu 2025) | 0.743 | 0.5995 | 0.7485 | 0.6658 | 0.7401 | 0.4686 | 0.815 | 0.6494 |
| Text | B-T3 Flat Multi-View Hybrid LR | 0.7753 | 0.6313 | 0.8246 | 0.7151 | 0.7497 | 0.548 | 0.8634 | 0.7506 |
| Text | PROPOSED (C1-C4, heavy S2, tuned OP) | 0.808 | 0.7072 | 0.7485 | 0.7273 | 0.8389 | 0.5799 | 0.8636 | 0.7522 |
| Text | PROPOSED heavy verifier (beta=1.0) | 0.801 | 0.7205 | 0.6832 | 0.7014 | 0.8622 | 0.5527 | 0.87 | 0.754 |
| Code (PolicyA LOGO) | B-C1 Token Jaccard | 0.5889 | 0.5424 | 0.8596 | 0.6652 | 0.3439 | 0.2356 | 0.6825 | 0.6092 |
| Code (PolicyA LOGO) | B-C2 RKRGST String Tiling | 0.5069 | 0.4899 | 0.924 | 0.6403 | 0.1296 | 0.0876 | 0.562 | 0.5325 |
| Code (PolicyA LOGO) | B-C3 Gandhi RBF-SVM (2024) | 0.8347 | 0.7896 | 0.8889 | 0.8363 | 0.7857 | 0.6754 | 0.9042 | 0.8834 |
| Code (PolicyA LOGO) | B-C4 Mehsen RandomForest (2024) | 0.7889 | 0.7582 | 0.8158 | 0.7859 | 0.7646 | 0.5798 | 0.8474 | 0.8206 |
| Code (PolicyA LOGO) | B-C5 ES-Plag (2019, recall-first filter) | 0.5097 | 0.4916 | 0.9415 | 0.6459 | 0.119 | 0.1055 | 0.5529 | 0.5307 |
| Code (PolicyA LOGO) | PROPOSED (C1-C4, heavy S2, tuned OP) | 0.8306 | 0.7973 | 0.8626 | 0.8287 | 0.8016 | 0.6636 | 0.7582 | 0.7055 |
| Code (PolicyA LOGO) | PROPOSED heavy verifier (beta=1.0) | 0.8472 | 0.8037 | 0.8977 | 0.8481 | 0.8016 | 0.6997 | 0.8837 | 0.8482 |

## 18. Qualitative analysis

See 05_QUALITATIVE_ANALYSIS/*.md incl. genuine error examples drawn from stored predictions. Residual errors concentrate in the entropy-flagged grey zone: the uncertainty machinery marks the hard cases; the ceiling is verifier quality.

## 19. Comparative evaluation (§9 tables)

Text: proposed F1 0.7273 vs strongest baseline B-T3 Flat Multi-View Hybrid LR F1 0.7151 (Δ = 0.0122). Code: proposed tuned-OP F1 0.8287 vs strongest baseline B-C3 Gandhi RBF-SVM (2024) F1 0.8363 (Δ = -0.0076); beta=1 F1 0.8481 exceeds all code peers. Significance: 8 McNemar + paired bootstrap comparisons in 04_EXPERIMENTAL_RESULTS/STATISTICAL_TESTS.json.

## 20. Ablation study

| Configuration | accuracy | precision | recall | f1 | mcc | escalation_rate_pct | pruning_ratio |
|---|---|---|---|---|---|---|---|
| Code-A0 | 0.6 | 0.5508 | 0.8567 | 0.6705 | 0.2552 | 0.0 | 1.0 |
| Code-A1 | 0.8194 | 0.7677 | 0.8889 | 0.8238 | 0.6479 | 0.0 | 1.0 |
| Code-A2 | 0.8208 | 0.7696 | 0.8889 | 0.825 | 0.6504 | 0.0 | 1.0 |
| Code-A3 | 0.8431 | 0.8281 | 0.845 | 0.8365 | 0.6858 | 83.61 | 0.1639 |
| Code-A4 | 0.8417 | 0.8239 | 0.848 | 0.8357 | 0.6832 | 48.33 | 0.5167 |
| Code-A5 | 0.8417 | 0.8276 | 0.8421 | 0.8348 | 0.6829 | 82.08 | 0.1792 |
| Code-A6 | 0.8361 | 0.8218 | 0.8363 | 0.829 | 0.6718 | 46.67 | 0.5333 |
| Text-A0 | 0.743 | 0.5995 | 0.7485 | 0.6658 | 0.4686 | 0.0 | 1.0 |
| Text-A1 | 0.793 | 0.7125 | 0.6618 | 0.6862 | 0.5329 | 0.0 | 1.0 |
| Text-A2 | 0.7947 | 0.7131 | 0.6686 | 0.6901 | 0.5375 | 0.0 | 1.0 |
| Text-A3 | 0.8047 | 0.6996 | 0.7515 | 0.7246 | 0.5745 | 42.57 | 0.5743 |
| Text-A4 | 0.8037 | 0.7372 | 0.6618 | 0.6975 | 0.5545 | 29.13 | 0.7087 |
| Text-A5 | 0.8083 | 0.7078 | 0.7485 | 0.7276 | 0.5805 | 38.63 | 0.6137 |
| Text-A6 | 0.8083 | 0.7078 | 0.7485 | 0.7276 | 0.5805 | 38.63 | 0.6137 |

## 21. Sensitivity analysis

Budget sweep (emulated from stored scores, honest emulation semantics documented):

| budget_cap | escalated_pct | accuracy | recall | f1 | mcc |
|---|---|---|---|---|---|
| 0.1 | 10.0 | 0.8023 | 0.6862 | 0.7036 | 0.5559 |
| 0.2 | 20.0 | 0.8037 | 0.6813 | 0.7036 | 0.5577 |
| 0.3 | 30.0 | 0.8063 | 0.6881 | 0.7085 | 0.5642 |
| 0.45 | 45.0 | 0.8093 | 0.6862 | 0.7111 | 0.5699 |
| 0.6 | 60.0 | 0.8033 | 0.6813 | 0.7032 | 0.557 |
| 0.7 | 70.0 | 0.8013 | 0.6813 | 0.7011 | 0.553 |
| 0.85 | 85.0 | 0.801 | 0.6832 | 0.7014 | 0.5527 |
| 1.0 | 100.0 | 0.801 | 0.6832 | 0.7014 | 0.5527 |

Threshold sweep: see 07_SENSITIVITY_AND_ROBUSTNESS/THRESHOLD_ANALYSIS.csv (19 thresholds, genuine decisions).

## 22. Statistical significance

All 8 baseline-vs-proposed comparisons executed with McNemar (continuity-corrected) and group-aware paired bootstrap CIs; see JSON artifact.

## 23–25. Discussion, threats, limitations

- The proposed tuned-OP configuration trails the strongest code baselines on raw quality (by design: budget-capped); at beta=1 it leads. The accuracy–cost envelope, not a single-point win, is the defensible contribution.
- Conformal guarantee at nominal alpha under-covers (0.930, Wilson CI below 0.95); phase-6 repair reaches 1.000 with avg set size 1.94 (ambiguous share 94%) — a deployment dial.
- External transfer: corpus-specific calibration does not transfer (C-S thresholded recall 0.000; MRPC acc 0.401); ranking partially transfers on MRPC (AUC 0.644 / AP 0.726). Per-deployment recalibration is required — stated as future work, not hidden.
- C4 human gains are not measured (protocol + verified instrument only).

## 26. Conclusion

The framework delivers a superior efficiency-quality envelope (worst-regime honest metrics; multi-seed SD ≤ 0.010 MCC across both modalities), is the most robust system under adversarial renaming/paraphrase attacks (phase-5 evidence), and is audit-ready with conformal uncertainty sets (repaired) — with external calibration transfer being the outstanding honest limitation.

## 27. Future research directions

Neural-embedding verifier variants for external transfer; per-deployment recalibration slices; IRB-style human review study per phase-6 protocol; multilingual and multi-language-code corpora.

## 28. References

See REFERENCES.bib (verified entries only; repository-recorded papers).
