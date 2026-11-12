# Phase 20 — Cross-Dataset Generalization Analysis

## What the experiments actually test

- **Dataset 1 (text):** premise-group-disjoint holdout (train/val/cal/test disjoint by premise) —
  tests *within-dataset generalization across premises*.
- **Dataset 2 (code):** nested Leave-One-Case-Out (LOGO) over 7 assignment cases — tests
  *within-dataset generalization across programming tasks*; Policy B (709 pairs) is a stricter
  duplicate-policy variant (per README code results: Policy B MCC 0.7222 in 4.1; phase-5 Policy B
  MCC 0.6974), holding out at the same case granularity.

This is **same-task-family transfer** inside each dataset, not transfer of one trained model between
the text corpus and the code corpus. No experiment trains on one dataset and tests on the other —
and none is claimed.

## Performance differences (framework level)

| Dimension | Text | Code |
|---|---|---|
| Frozen F1 / MCC | 0.6688 / 0.5188 | 0.8467 / 0.6996 |
| Upgrade F1 / MCC | 0.7273 / 0.5799 | 0.8287 / 0.6636 (β=1: 0.8481 / 0.6997) |
| Subgroup spread | length-tercile F1 0.563–0.717 | case F1 0.494–0.982 |
| Calibration | strong (ECE ≤0.031) | weak under LOGO (ECE 0.181) |
| Routing savings | 47–61% | 15–27% |
| Conformal | group coverage 0.930–0.937 at 0.95 nominal | n/a (group granularity = 7 cases too sparse) |

## Interpretation

1. **Within-dataset performance:** demonstrated for both (grouped holdout; nested LOGO).
2. **Cross-dataset generalization:** **not tested** as model transfer; what is demonstrated is
   **cross-modal framework transfer** — the same architecture and protocol machinery instantiate
   successfully in both domains.
3. **Domain robustness:** stronger on code (procedural ground truth, higher separability, MCC 0.70);
   text is harder (proxy semantics, MCC 0.58) — hypotheses, not conclusion claims.
4. **Calibration/routing generalization:** calibration transfers well in text but poorly under LOGO
   in code (fewer inner folds → isotonic variance); routing transfers with different savings rates —
   a meaningful, measured heterogeneity in cost economics.
5. **Limitations:** single corpus per modality; 7 code cases; proxy labels in text; no external
   (institutional) dataset. Generalization to unseen institutions/languages remains future work.

**Verdict:** reliable *within-dataset* performance in both modalities + *cross-modal framework
transfer*, with explicit refusal to claim model-level cross-dataset generalization.
