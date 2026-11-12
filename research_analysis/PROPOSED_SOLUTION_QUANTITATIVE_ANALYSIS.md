# Phase 15 — Proposed Solution Quantitative Analysis
*Source: existing prediction/result artifacts only (no retraining). Machine outputs:
`PROPOSED_SOLUTION_METRICS.csv`, `PROPOSED_SOLUTION_CALIBRATION.csv`,
`PROPOSED_SOLUTION_ROUTING.csv` (+`_DETAIL.json`), `PROPOSED_SOLUTION_ABLATION.csv`.*

## 15.1 Classification performance (from stored per-pair predictions)

**Text, held-out test (n=3,000; 1,026 pos / 1,974 neg):**

| Generation | Acc | Prec | Rec | Spec | F1 | macroF1 | wtdF1 | ROC-AUC | PR-AUC | FPR | FNR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.1 frozen | 0.7893 | 0.7234 | 0.6218 | 0.8764 | 0.6688 | 0.7572 | 0.7851 | 0.8634 | 0.7506 | 0.1236 | 0.3782 | 0.5188 |
| 5.0 upgrade | 0.8080 | 0.7072 | 0.7485 | 0.8389 | 0.7273 | 0.7896 | 0.8092 | 0.8636 | 0.7522 | 0.1611 | 0.2515 | 0.5799 |

Confusion: 4.1 [TN 1730, FP 244; FN 388, TP 638]; 5.0 [TN 1656, FP 318; FN 258, TP 768].

**Code, Policy A nested-LOGO OOF (n=720; 342 pos / 378 neg):**

| Generation | Acc | Prec | Rec | Spec | F1 | ROC-AUC | PR-AUC | FPR | FNR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.1 frozen | 0.8486 | 0.8157 | 0.8801 | 0.8201 | 0.8467 | 0.8322 | 0.7492 | 0.1799 | 0.1199 | 0.6996 |
| 5.0 upgrade (cost-min OP) | 0.8306 | 0.7973 | 0.8626 | 0.8016 | 0.8287 | 0.7582 | 0.7055 | 0.1984 | 0.1374 | 0.6636 |
| 5.0 upgrade β=1 (heavy verifier) | 0.8472 | 0.8037 | 0.8977 | 0.8016 | 0.8481 | — | — | 0.1984 | 0.1023 | 0.6997 |

Confusion: 4.1 [TN 310, FP 68; FN 41, TP 301]; 5.0-OP [TN 303, FP 75; FN 47, TP 295].

## 15.2 Uncertainty and calibration

| Setting | ECE | adaptive ECE | MCE | Brier | log-loss | slope | intercept |
|---|---|---|---|---|---|---|---|
| Text (4.1, uncalibrated) | 0.0145 | 0.0144 | 0.0314 | 0.1413 | 0.4331 | 1.074 | 0.015 |
| Text (4.1, Platt) | 0.0315 | 0.0321 | 0.0701 | 0.1423 | 0.4397 | 1.010 | −0.093 |
| Text (4.1, isotonic) | 0.0224 | 0.0202 | 0.1189 | 0.1426 | 0.4405 | 0.869 | −0.131 |
| Text (4.1, temperature) | 0.0156 | 0.0116 | 0.0473 | 0.1412 | 0.4327 | 0.969 | 0.014 |
| Text (5.0, in-prod isotonic) | 0.0310 | 0.0310 | 0.0685 | 0.1420 | — | 1.017 | −0.087 |
| Code (4.1, LOGO OOF) | **0.1812** | 0.1846 | **0.9130** | 0.1698 | 0.5525 | 1.051 | −0.009 |

Selective prediction: risk–coverage behavior documented in phase-5 figure; abstention = escalation
state; code calibration weakness under LOGO is a declared limitation (fewer inner folds → isotonic
variance). NLL = log-loss column above (where stored).

**Conformal (group-wise, α=0.05).** Phase-4.1: empirical group coverage **93.69%**, pair coverage
97.80%, mean set size 1.608, singletons 39.2%, m=254 calib groups → n=1,015 test groups.
Phase-5 corrected with **Wilson binomial CI: coverage 0.9300 [0.9127, 0.9442]** (excludes 0.95 →
honest undercoverage), pair coverage 0.9753 [0.9691, 0.9803]. Validity note: finite-sample marginal
guarantee holds under premise-group exchangeability; group-heteroscedasticity explains the gap,
disclosed in both generations.

## 15.3 Routing and computational efficiency

| Setting | Stage-1 ms/pair | Stage-2 ms/pair (single) | Escalated | Avoided % | Peak mem |
|---|---|---|---|---|---|
| Text (4.1) | 0.000195 (offline-precomputed ETL) | 0.0689 | 1,583/3,000 | 47.23% | 0.385 MB |
| Code (4.1, LOGO) | 0.000371 | 0.0675 | 611/720 | 15.14% | 0.279 MB |
| Text (5.0, batched amortized) | 0.000299/pair | 0.005535/pair (18.5×) | 1,159/3,000 | 61.37% | — |
| Code (5.0, batched amortized) | 0.000671/pair | 0.01354/pair (20.2×) | 523/720 | 27.36% | — |

Routing distributions (decision states): 4.1 text {pruned 1,174, escalated 1,583, flagged 243};
4.1 code {82/611/27}; 5.0 text {1,199/1,159/642}; 5.0 code {145/523/52}. Retrieval/routing
overhead per pair recorded: 0.000129 ms (text), 3.8e-05 ms (code) — included in cost accounting.
Single-pair Stage-2 latency (phase 5): 0.226 ms text MLP; 1.245 ms code HGB (fold-0 selected).

## 15.4 Explainability

Measured indicators (existing): dossier sampling 50/1,826 eligible (2.74% representative audit
sample), 100% generation success, mean 0.204 ms/dossier, p95 0.427 ms; evidence faithfulness via
feature-ablation probability drop (novelty audit); span-level P/R/F **structurally omitted** (no
human span ground truth). No invented explanation-quality scores.

## 15.5 Statistical reliability

- **Grouped bootstrap CIs (4.1, 1,000 premise-cluster resamples):** text accuracy 0.7896
  [0.7765, 0.8028]; F1 0.6690 [0.6461, 0.6917]; MCC 0.5195 [0.4896, 0.5507].
- **Paired Δ vs semantic baseline:** Δacc +0.0906 [+0.076, +0.107], Δprecision +0.1868 [+0.164,
  +0.212], Δrecall −0.2429 (trade-off), ΔMCC +0.0638 (p<1e-16) (McNemar).
- **Phase-5 tests:** McNemar + clustered bootstrap on both modalities; code β=1 significant wins
  vs B-C1/B-C2/B-C4/B-C5 (ΔF1 +0.181/+0.207/+0.063/+0.202), parity with B-C3 (ΔF1 +0.012
  [−0.035, +0.068]); text significant vs B-T1/B-T2 (+0.177/+0.062), tie vs B-T3.
- **Effect sizes:** ΔMCC +0.46 vs B-C1 (β=1), magnitude reported directly from bootstrap deltas.
- **Sensitivity:** operating-point robustness via 20-budget Pareto sweep; verifier choice sensitivity
  recorded (RF 5/7 folds vs HGB 2/7).
- **Group-aware methods:** all bootstrap resampling is cluster-stratified (premise/case), per the
  audit notices embedded in `STATISTICAL_TESTS.json`.
