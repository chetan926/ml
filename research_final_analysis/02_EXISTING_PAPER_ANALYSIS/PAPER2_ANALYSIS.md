# PAPER2 — A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings

- **Authors:** Gandhi, Gopalan, Prasad
- **Year/venue:** 2024, Frontiers in Computer Science 6:1393723
- **DOI:** 10.3389/fcomp.2024.1393723
- **Modality:** Code (applied to Code dataset (Dataset 2))
- **Methodology:** RBF-SVM over token/structural similarity features (Python); cross-language adaptation to Java documented.
- **Local implementation:** B-C3 (SVC rbf, LOGO, CODE_SVM_FEATURES) — corrected leak-free protocol

## Published results (from paper) vs independently reproduced
| metric | published (paper) | reproduced (this repo, corrected protocol) |
|---|---|---|
| accuracy | NR (full text absent) | 0.8347 |
| precision | NR (full text absent) | 0.7896 |
| recall | NR (full text absent) | 0.8889 |
| f1 | NR (full text absent) | 0.8363 |

**Reproduced extras (same protocol):** MCC 0.6754; ROC-AUC 0.9042; PR-AUC 0.8834; specificity 0.7857.

## Advantages
- Strong margin classifier on structural features
- Robust on held-out cases

## Limitations
- Hard decision only (no calibrated probability)
- No routing/cost model
- Cross-language adaption required for Java corpus

## Research gap addressed by proposed framework
- No uncertainty quantification or conformal sets → C2.
- No budget-adaptive heavy verification → C3.
- No cross-modality unified evaluation → C4.
- No leakage-safe grouping protocol in published setup → corrected protocol.

**Deviation note:** implementation diverges only where the repository audits record a deviation (e.g., Python→Java adaptation for Paper 2; RF depth note for Paper 3).