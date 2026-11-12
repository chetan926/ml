# PAPER1 — A comprehensive strategy for identifying plagiarism in academic submissions

- **Authors:** Setu, Islam, Erfan, Dey, Al Asif, Alam
- **Year/venue:** 2025, Journal of Umm Al-Qura University for Engineering and Architecture (Springer Nature)
- **DOI:** 10.1007/s43995-025-00108-1
- **Modality:** Text (applied to Text dataset (Dataset 1))
- **Methodology:** Dense semantic similarity over embeddings (MiniLM-class sentence embeddings) with threshold decision; lexical pre-screening lineage from TF-IDF.
- **Local implementation:** B-T2 (text_semantic_minilm_sim threshold, val-tuned) + B-T1 lexical sibling — corrected leak-free protocol

## Published results (from paper) vs independently reproduced
| metric | published (paper) | reproduced (this repo, corrected protocol) |
|---|---|---|
| accuracy | NR (full text absent) | 0.743 |
| precision | NR (full text absent) | 0.5995 |
| recall | NR (full text absent) | 0.7485 |
| f1 | NR (full text absent) | 0.6658 |

**Reproduced extras (same protocol):** MCC 0.4686; ROC-AUC 0.815; PR-AUC 0.6494; specificity 0.7401.

## Advantages
- Simple, fast, deployable threshold rule
- Semantic view catches paraphrase reuse

## Limitations
- No calibration/uncertainty
- No leakage-safe protocol in-paper as captured
- Single threshold trades precision/recall poorly
- No cost accounting

## Research gap addressed by proposed framework
- No uncertainty quantification or conformal sets → C2.
- No budget-adaptive heavy verification → C3.
- No cross-modality unified evaluation → C4.
- No leakage-safe grouping protocol in published setup → corrected protocol.

**Deviation note:** implementation diverges only where the repository audits record a deviation (e.g., Python→Java adaptation for Paper 2; RF depth note for Paper 3).