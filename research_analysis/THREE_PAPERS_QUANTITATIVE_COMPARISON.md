# Phase 13 — Quantitative Comparison of the Three Repository Papers

Companion machine table: `THREE_PAPERS_COMPARISON.csv`.

## Reproduction results (repository-audited implementations, this project's protocols)

| Metric | Paper 1 (Setu'25) | Paper 2 (Gandhi'24) | Paper 3 (Mehsen'24) |
|---|---|---|---|
| Accuracy | 0.7200 (text test) | 0.7000 (case-06/07) | 0.7333 (case-06/07) |
| Precision | 0.5697 | 0.6258 | 0.6643 |
| Recall | 0.8570 | 0.9510 | 0.9118 |
| F1 | 0.6844 | 0.7549 | 0.7686 |
| ROC-AUC | 0.8198 | NR | NR |
| Corrected-protocol F1 / MCC | — (0.666 F1 grouped headline B-T2) | **0.8363 / 0.6754** | 0.7859 / 0.5798 |
| Inference latency | 4.143 ms/pair (E1: 4.163; E2: 2.875) | 0.013 ms/pair (+2.050 extraction) | 0.025 ms/pair |
| Training time | NR (pretrained encoder) | NR | NR |
| Memory | NR | NR | NR |
| Formal complexity | transformer fwd ×2/pair | SVM-RBF over 7-D | 100 trees, depth ≤6 |
| Systematic renaming pos-recall | n/a (text) | 0.2778 | 0.3216 |

## Comparability assessment (required cautions)

1. **Dataset compatibility:** Paper 1's numbers are on the *text* benchmark; Papers 2–3 on the *code*
   benchmark — **no cross-column ranking is defensible**.
2. **Protocol compatibility:** the case-06/07 adaptation numbers are not nested; the nested-LOGO row
   is the only fully comparable cross-peer code comparison; headings are labeled accordingly.
3. **Split compatibility:** text threshold tuning used validation data (F1-optimal); identical for
   all text comparators, so the text comparison is internally fair.
4. **Metric definitions:** consistent (binary plagiarism decision) across all rows.
5. **Class distribution:** text ~34% positives; code ~47.5% — absolute values are not commensurable
   across modalities.
6. **Experimental conditions:** single seed 42; all latencies measured on the same host inside this
   project (batched amortized regime in phase 5).
7. **Lineage separation:** "published" = NR throughout (PDFs absent); all numbers are
   **independently reproduced** results, explicitly separated from any paper-published claims.

**Bottom line.** Within the corrected protocol the paper-era methods form a strong but
classification-only field: best code peer (B-C3 SVM) reaches F1 0.8363/MCC 0.6754; best text peer
(B-T2 MiniLM) F1 0.666/MCC 0.469 at headline, 0.684/0.500 at matched recall — both fields lack UQ,
budget routing, and audit dossiers, which the proposed framework adds while matching or exceeding
their discrimination quality (Phase 14–17 evidence).
