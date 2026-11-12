# Phase 16 — Baseline and Ablation Analysis
*Sources (all existing artifacts): `PHASE_4_1_ABLATION_RESULTS.csv`,
`PHASE_4_1_BASELINE_PREDICTIONS_*`, `PHASE_5_HEADLINE/ISO_RECALL/PARETO/STATISTICAL_*`,
`experiments/results/recall_constrained_multi_method_comparison.csv`.*

## Baseline comparison

**Text (held-out test, n=3,000):**

| System | Acc | Prec | Rec | F1 | MCC | Verdict vs proposed (4.1 frozen / 5.0) |
|---|---|---|---|---|---|---|
| Proposed (4.1) | 0.7893 | 0.7234 | 0.6218 | 0.6688 | 0.5188 | — |
| TF-IDF baseline | 0.6143 | 0.4572 | 0.6823 | 0.5475 | 0.2480 | proposed dominates every summary metric |
| MiniLM baseline | 0.6987 | 0.5369 | 0.8645 | 0.6624 | 0.4549 | proposed higher Acc/Prec/F1/MCC; baseline recall-optimal posture |
| Flat hybrid LR (B-T3, phase 5) | 0.7753 | 0.6313 | 0.8246 | 0.7151 | 0.5480 | proposed 5.0 OP: F1 +0.012 (CI [−0.004,+0.030], tie) |
| Proposed (5.0) | 0.8080 | 0.7072 | 0.7485 | 0.7273 | 0.5799 | best headline text row |

At **iso-recall (target 0.93)** the proposed framework exceeds *all* text baselines on F1 and MCC
(0.719/0.558 vs 0.684/0.500 next-best) — see `PHASE_5_ISO_RECALL_COMPARISON_TEXT.csv`.

**Code (Policy A nested-LOGO, n=720):**

| System | F1 | MCC | sig. vs proposed |
|---|---|---|---|
| Proposed 4.1 (routed) | 0.8467 | 0.6996 | — |
| Proposed 5.0 β=1 | **0.8481** | **0.6997** | — |
| B-C1 token Jaccard | 0.6652 | 0.2356 | proposed sig. (+0.161/+0.181) |
| B-C2 RKRGST | 0.6403 | 0.0876 | sig. (+0.187/+0.207) |
| B-C3 SVM'24 | 0.8363 | 0.6754 | **tie** (−0.008/+0.012, CIs span 0) |
| B-C4 RF'24 | 0.7859 | 0.5798 | sig. (+0.043/+0.063) |
| B-C5 ES-Plag | 0.6459 | 0.1055 | sig. (+0.182/+0.202) |

Uncertainty quality: only the proposed framework reports calibration suites and conformal coverage
(no baseline computes calibrated probabilities natively). Computational efficiency: proposed avoids
47–61%/15–27% of expensive calls; baselines pay flat cost. Routing effectiveness: measured frontier
dominance (Phase-5 Pareto CSV). Calibration: text ECE 0.014–0.031 vs uncalibrated baselines.
Explainability: dossiers exist only in the proposed framework. Robustness: under renaming, proposed
β=1 pos-recall 0.8918 vs baselines 0.17–0.80; text attacks: proposed top F1 in 6/7 families.

## Ablation studies (existing results only)

**Text ablation (`PHASE_4_1_ABLATION_RESULTS.csv`):**

| Configuration | Acc | F1 | MCC | Reading |
|---|---|---|---|---|
| Word lexical only | 0.7243 | 0.4633 | 0.3370 | C1 view ablated |
| Char n-gram only | 0.7027 | 0.3780 | 0.2695 | C1 view ablated |
| Dense semantic only | 0.7460 | 0.5942 | 0.4155 | C1 view ablated |
| LSA semantic only (phase 5) | 0.6583 | 0.1465 | 0.0844 | C1 view ablated |
| Structural only | 0.6583 | 0.0154 | 0.0286 | C1 view ablated |
| **C1 multi-view fusion (12 views)** | 0.7930 | 0.6862 | 0.5329 | fusion payoff |
| **C1 + C2 calibration** | 0.7947 | 0.6901 | 0.5375 | calibration marginal on MCC; needed for routing/UQ semantics |
| **C1+C2+C3+C4 full framework** | 0.8080 | 0.7273 | 0.5799 | routing+heavy-S2 ⇒ best overall |

**Code ablation (existing):** C1 fusion reference (Phase-4.1 LOGO, per ablation row) MCC 0.6996 ≈
full framework β=1 (0.6997) — confirms the routed OP gap (+/−) is a cost-vs-quality trade, not a
regression; full framework 5.0 cost-min OP: 0.6636 at 27% savings.

**Ablations NOT experimentally validated (declared, not invented):**
- *Without C3 (routing removed entirely at fixed budget ≠ flat baseline)* is represented implicitly
  by the flat-fusion reference rows; a held-budget "no-routing but calibrated heavy verifier at
  intermediate β" counterfactual was not run.
- *Without C4 (dossiers removed)* has no quantitative row — C4 is a governance/audit artifact whose
  removal does not alter metrics; its value is qualitative (traceability). Marked as not
  experimentally validated, per the master rules.
