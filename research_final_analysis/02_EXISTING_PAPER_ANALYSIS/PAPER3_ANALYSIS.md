# PAPER3 — Detection of Source Code Plagiarism Utilizing an Approach Based on Machine Learning

- **Authors:** Mehsen, Joshi
- **Year/venue:** 2024, International Journal of Computing 23(1):78-84
- **DOI:** 10.47839/ijc.23.1.3438
- **Modality:** Code (applied to Code dataset (Dataset 2))
- **Methodology:** RandomForest classifier over code similarity features (audited parameter note: max_depth as per audit not 10).
- **Local implementation:** B-C4 (RF n=100, LOGO, CODE_RF_FEATURES) — corrected leak-free protocol

## Published results (from paper) vs independently reproduced
| metric | published (paper) | reproduced (this repo, corrected protocol) |
|---|---|---|
| accuracy | NR (full text absent) | 0.7889 |
| precision | NR (full text absent) | 0.7582 |
| recall | NR (full text absent) | 0.8158 |
| f1 | NR (full text absent) | 0.7859 |

**Reproduced extras (same protocol):** MCC 0.5798; ROC-AUC 0.8474; PR-AUC 0.8206; specificity 0.7646.

## Advantages
- Nonlinear feature interactions
- Feature-importance interpretability

## Limitations
- No calibration
- No group-aware eval
- No cost/uncertainty model

## Research gap addressed by proposed framework
- No uncertainty quantification or conformal sets → C2.
- No budget-adaptive heavy verification → C3.
- No cross-modality unified evaluation → C4.
- No leakage-safe grouping protocol in published setup → corrected protocol.

**Deviation note:** implementation diverges only where the repository audits record a deviation (e.g., Python→Java adaptation for Paper 2; RF depth note for Paper 3).