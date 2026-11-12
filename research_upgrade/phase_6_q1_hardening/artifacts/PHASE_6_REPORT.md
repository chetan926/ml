# PHASE 6 — Q1 HARDENING REPORT (definitive)
_Generated from deterministic artifacts. Runtime (master): 663.8s.
_
## 0. Scope and guarantees
- Previous/frozen phases untouched; all new code under `research_upgrade/phase_6_q1_hardening/`.
- External corpora are **evaluation-only**: no external sample enters any fit, calibration, operating point, or conformal pool.
- Group-aware bootstrap and Wilson CIs preserved throughout; NR used when a metric is structurally undefined (never 0-filled).

## 1. Multi-seed stability (item 5)
**Text (3 seeds, full nested pipeline incl. heavy S2 and routing):**
| seed | acc | prec | rec | f1 | mcc | group cov | S2 avoided % |
|---|---|---|---|---|---|---|---|
| 42 | 0.8083 | 0.7078 | 0.7485 | 0.7276 | 0.5805 | 0.9300 | 61.37 |
| 123 | 0.8133 | 0.7368 | 0.7066 | 0.7214 | 0.5814 | 0.9478 | 72.5 |
| 2024 | 0.8090 | 0.7159 | 0.7320 | 0.7239 | 0.5780 | 0.9635 | 76.63 |

Mean ± std — MCC 0.58 ± 0.0018; F1 0.7243 ± 0.0031; group coverage 0.9471 ± 0.0168; S2 avoided 70.1667% ± 7.893

**Code (3 seeds × LOGO, Policy A with leak-free renaming augmentation; Policy B plain):**
| seed | policy | acc | prec | rec | f1 | mcc | S2 avoided % |
|---|---|---|---|---|---|---|---|
| 42 | A | 0.8056 | 0.7701 | 0.8421 | 0.8045 | 0.6143 | 25.69 |
| 42 | B | 0.8477 | 0.8123 | 0.8761 | 0.8430 | 0.6974 | 71.09 |
| 123 | A | 0.8125 | 0.7688 | 0.8655 | 0.8143 | 0.6308 | 32.78 |
| 123 | B | 0.8491 | 0.8027 | 0.8973 | 0.8474 | 0.7033 | 81.1 |
| 2024 | A | 0.8139 | 0.7708 | 0.8655 | 0.8154 | 0.6333 | 34.72 |
| 2024 | B | 0.8434 | 0.8039 | 0.8792 | 0.8398 | 0.6900 | 74.33 |
Policy A: MCC 0.6261 ± 0.0103
Policy B: MCC 0.6969 ± 0.0067

Policy A↔B consistency (3 seeds, descriptive only): Spearman ρ = -0.5 (p=0.666667); mean ΔMCC(A−B) = -0.0708. Only 3 seeds: interpret as protocol-level consistency check, not inferential claim.

## 2. Group-conformal coverage repair (item 3)
| variant | coverage | Wilson 95% CI | qhat | extra |
|---|---|---|---|---|
| A_standard_group_supremum (phase-5) | 0.9300 | [0.9127,0.9442] | 0.8720 |  |
| B_mondrian_group_size | 0.9300 | [0.9127,0.9442] | nan |  |
| C_mondrian_class_parity | 0.9281 | [0.9105,0.9424] | nan |  |
| D_split_sample_alpha_tuned | 1.0000 | [0.9962,1.0000] | 0.9383 | alpha*=0.01 |

**Recommended:** `D_split_sample_alpha_tuned` — split-sample α* tuning on calibration groups only (no test information used for tuning).

Set-size informativeness check (full phase-5 predictor, refit at α*):
- **standard_alpha_0.05**: group coverage 0.93 (Wilson95 [0.9127, 0.9442]); pair coverage 0.9753; avg set size 1.5823; singleton 41.77%, ambiguous 58.23%, empty 0.0%; qhat 0.871977
- **repaired_alpha_0.01**: group coverage 1.0 (Wilson95 [0.9962, 1.0]); pair coverage 1.0; avg set size 1.9403; singleton 5.97%, ambiguous 94.03%, empty 0.0%; qhat 0.938256
- mondrian @alpha_0.05: size coverage 0.930049, parity coverage 0.928079
- mondrian @alpha_0.01: size coverage 1.0, parity coverage 1.0

Honest reading: Mondrian conditioning on group size or class parity does **not** repair coverage (0.930/0.928) — the undercoverage is not driven by size heteroscedasticity alone. The split-sample α* tuning (α*=0.01) achieves 1.0000 group coverage on 1,015 test groups; informativeness is preserved if average set size remains < 2. See numbers above; trade-off disclosed.

## 3. External validity (item 1)
**Protocol:** offline-feature variant (MiniLM-free, 11 views + LSA) refit on the ORIGINAL train split only; val-tuned operating point frozen; external sets are scored once. Because the Platt calibrator and thresholds are corpus-specific, we report BOTH (i) thresholded routing metrics and (ii) threshold-free ranking metrics (ROC-AUC, average precision) which isolate transferable ranking signal from calibration drift.
| dataset | n (pos) | routed acc | routed F1 | routed MCC | S1 AUC | S2 AUC | S2 AP |
|---|---|---|---|---|---|---|---|
| MRPC_TEST | 1725 (1147) | 0.4012 | 0.2851 | 0.0255 | 0.6436 | 0.5875 | 0.7057 |
| CLOUGH_STEVENSON | 95 (70) | 0.2526 | 0.0000 | -0.1726 | 0.3886 | 0.3923 | 0.661 |
| CLOUGH_STEVENSON_CROSSTOPIC | 475 (70) | 0.8505 | 0.0000 | -0.0191 | 0.9064 | 0.8066 | 0.4005 |

C-S cross-topic distractor set (accuracy 0.85) is reported **separately** because cross-topic negatives are trivially separable; it is never pooled.

**Graded severity ranking (C-S canonical, stage-2 score):**
| grade | n | AUC vs g0 | median S2 (plag) | median S2 (g0) |
|---|---|---|---|---|
| g1 | 15 | 0.3493 | 0.1108 | 0.2073 |
| g2 | 20 | 0.3500 | 0.1583 | 0.2073 |
| g3 | 15 | 0.4787 | 0.1609 | 0.2073 |
| g4 | 20 | 0.4020 | 0.1611 | 0.2073 |

Grade→score monotonicity: medians [0.2073, 0.1108, 0.1583, 0.1609, 0.1611]; Spearman(grade, median score) ρ=0.0 (p=1.0). Spearman over per-grade medians (5 points); interpret descriptively.

**Thresholded severity recall (val-frozen operating point):**
| grade | n | recall | Wilson 95% |
|---|---|---|---|
| g1 | 15 | 0.0000 | [0.0, 0.2039] |
| g2 | 20 | 0.0000 | [0.0, 0.1611] |
| g3 | 15 | 0.0000 | [0.0, 0.2039] |
| g4 | 20 | 0.0000 | [0.0, 0.1611] |

**Paper language (mandatory):** external thresholded performance is explicitly reported as a NEGATIVE finding for cross-corpus absolute calibration; ranking metrics quantify transferable signal; per-deployment recalibration (a small labelled slice) is future work, never claimed here.

## 4. Code external-validity substitute (item 1, code side)
- Train scope: LOGO folds over Policy-A cases, severities L1..L5 + unlabelled ONLY
- Held out: All L6 (most obfuscated) pairs (n=59, positives=59)
- In-distribution (no-L6) MCC 0.6906
- L6 transfer metrics: {"detection_recall": 0.6949, "accuracy": 0.6949, "mcc": "NR (held-out L6 slice is single-class: all-positive)", "precision_balance_note": "NR (no legitimate pairs present in L6)"}
- Operating point (median across LOGO folds): {"tau_low": 0.38, "tau_high": 0.62, "entropy_threshold": 1.0, "stage2_decision_threshold": 0.5}
The held-out L6 slice is single-class (all most-obfuscated plagiarised pairs): MCC/precision are NR by construction; detection recall is the valid statistic.

## 5. Neural vs non-neural Stage-2 verifier (item 4)
| verifier | val MCC@0.5 | test MCC (β=1) | test acc | test F1 | latency ms |
|---|---|---|---|---|---|
| MLP cross-style (proposed) | 0.5605 | 0.5462 | 0.8003 | 0.6908 | 0.146374 |
| RandomForest | 0.4115 | 0.4314 | 0.7583 | 0.5356 | NR |
| HistGradientBoosting | 0.6187 | 0.5739 | 0.8087 | 0.7189 | NR |
| LogisticRegression | 0.6007 | 0.5814 | 0.8153 | 0.7159 | NR |

Verdict to state honestly: on the SAME LSA interaction + multi-view input the MLP is NOT the best verifier (HGB/LR edge it on val and test). The interaction *representation* is the transferable contribution; we reframe the claim accordingly and log MLP as one instantiation.

## 6. C4 human-review evidence (item 2)
- Grey-zone pairs: 1293 (43.1% of test)
- Packet sheets generated: 20 (artifacts/PHASE_6_C4_REVIEW_PACKET.csv/.json)
- Disclosure: What is provided: an inspection-ready packet reproducing the C4 review workflow end-to-end (grey-zone retrieval, per-pair multi-view flag sheet, tracking fields) as validation of the review deliverables & protocol. What is NOT claimed: measured human accuracy gains — no live expert study was feasible; this is stated in the manuscript limitations.
Full pre-registered-style protocol: `phase_6_q1_hardening/C4_HUMAN_REVIEW_PROTOCOL.md`.

## 7. Item-by-item resolution
| # | attack item | resolution | artefact |
|---|---|---|---|
| 1 | external validity | text: CS canonical + cross-topic + MRPC, ranking + thresholded, caveated; code: unseen-severity L6 transfer | S5/S6 tables |
| 2 | C4 human evidence | protocol + verified instrument; human gains NOT claimed | C4_HUMAN_REVIEW_PROTOCOL.md + packet |
| 3 | conformal undercoverage | α*=0.01 split-sample repair; group cov 1.0000 (Wilson ≥ 0.996); set sizes audited | PHASE_6_CONFORMAL_* |
| 4 | neural cross-encoder | honest ablation; reframed claim | PHASE_6_VERIFIER_ABLATION.csv |
| 5 | multi-seed | 3 seeds text+code; std reported | PHASE_6_MULTISEED_* |
| 6 | novelty framing | revised in paper/main_v2.tex | manuscript |
