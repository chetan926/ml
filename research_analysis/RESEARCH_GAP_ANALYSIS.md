# Phase 18 — Research Gap Identification
*Derived strictly from the three repository papers + the two datasets; no manufactured gaps.*

### Gap 1 — Uncertainty-blind binary decisions
- **Paper 1:** static threshold decisions; no UQ (metadata: metrics = Acc/P/Rec/F1/ROC-AUC only).
- **Paper 2:** `probability=True` outputs without calibration evidence (audit §2.2).
- **Paper 3:** RF votes without calibration treatment (audit §2.3).
- **Dataset support:** text test has 1,015 premise groups — exactly the unit that grouped coverage
  quantification requires; code cases (7) make task-level uncertainty meaningful.
- **Why it matters:** wrongful-accusation regimes need bounded-risk outputs; a bare label cannot be
  escalated/schedulable.
- **Framework response:** C2 calibration + group-conformal sets + Wilson CIs.
- **Validation evidence:** text ECE 0.0145–0.031; group coverage measured (0.930, CI given);
  selective abstention wired into routing states.

### Gap 2 — Static, budget-unaware inference
- **Paper 1:** E2 staging reduces workload ≈30% but has no formal budget knob; no per-pair cost
  ledger.
- **Papers 2/3:** exhaustive scoring only.
- **Dataset support:** text corpus scale (n=8.5k core, 19.7k pool) makes quadratic/full sweeps the
  operative constraint; code's AST extraction (2.050 ms/pair) magnifies it.
- **Why it matters:** institutional screening cost governs feasibility.
- **Framework response:** C3 routing with ε-MCC cost-minimized OP + measured latencies (18.5–20.2×
  heavy-vs-light ratios).
- **Validation evidence:** 47–61% (text) / 15–27% (code) heavy-call savings at recorded quality;
  iso-cost frontier dominance.

### Gap 3 — Single-view evidence fragility
- **Paper 1:** `max(TF-IDF, SBERT)` is brittle on zero-overlap positives.
- **Papers 2/3:** small hand-built metric vectors; renaming collapses SVM/RF reproductions
  (pos-recall 0.28/0.32 under systematic renaming).
- **Dataset support:** text zero-overlap failure mode measured (119 pairs, 10.57%); code obfuscation
  levels L1–L6 give a severity axis that single views span poorly.
- **Why it matters:** evasion is the operational failure mode.
- **Framework response:** C1 complementary views + leak-free adversarial augmentation.
- **Validation evidence:** fusion ablation MCC 0.533 vs best single view 0.416; attacked recall
  0.89 (β=1) / 0.75 (OP) vs peer collapses; text attack table dominance in 6/7 families.

### Gap 4 — Opaque scoring, no adjudication trail
- **Papers 1–3:** none emit audit trails; all score-only systems.
- **Dataset support:** IR-Plag's procedural labels enable exact offset/line tracing; text premise
  groups enable casewise dossier aggregation.
- **Why it matters:** disciplinary decisions require appealable records.
- **Framework response:** C4 dossiers (offsets, line indices, routing state, conformal set, verifier
  identity; ground-truth withheld).
- **Validation evidence:** dossier metrics (100% generation success; 0.204 ms mean); span-level
  scoring transparently declared unavailable.

### Gap 5 — Leaky evaluation protocols in the subfield
- **Papers 2/3 (as reproduced):** case-holdout, no nested group-disjoint CV in the original designs
  (as recorded repo-side).
- **Dataset support:** task-level and premise-level grouping quantifiable; the project measured the
  inflation: stratified vs LOGO ΔMCC +0.122 (4.1) / +0.153 (5.0) on code.
- **Why it matters:** inflated literature claims misdirect deployment decisions.
- **Framework response:** corrected protocols are part of the contribution; baselines re-run inside
  them.
- **Validation evidence:** controlled leakage experiments recorded in both generations.

### Gap 6 — Uncorrected uncertainty communication (coverage without CIs)
- Evidence: phase-4.1 reported coverage 93.69% bare; phase 5 added Wilson binomial intervals —
  reviewer-grade uncertainty-on-uncertainty that papers 1–3 never touch.
- Validation: `PHASE_5_CONFORMAL_COVERAGE.csv`.
