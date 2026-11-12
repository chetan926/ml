# Phase 21 — Final Research Synthesis

## Connected picture

**Dataset 1 (SNLI/Pereira text-reuse proxy)** provides the volume + premise-group structure that
makes calibration, group conformal prediction, and cost-routing economics *measurable*. **Dataset 2
(IR-Plag Java)** provides authentic, procedurally labeled, obfuscation-graded code pairs that make
*resilience and leak-free LOGO* measurable. **Paper 1 (Setu'25)** supplies the text lineage
(hybrid lexical-semantic) plus a primitive cost narrative; **Papers 2–3 (Gandhi'24, Mehsen'24)**
supply the strongest peer code learners. **The proposed framework** integrates all of it into one
auditable machine: multi-view evidence (C1) → calibrated uncertainty (C2) → budgeted escalation (C3)
→ casewise dossiers (C4).

## Central research questions — answered

- **Why is academic integrity detection important?** False accusations and missed misconduct both
  have institutional and student-welfare consequences; detection systems are decision infrastructure,
  not demos.
- **What limits the three papers?** No UQ/calibration discipline, no budget-formalized routing, no
  audit dossiers, no leak-free grouped evaluation (Phases 9–13).
- **Why two datasets?** Each dataset stresses complementary axes: paraphrase semantics at scale with
  grouped structure (text), and authenticated obfuscation with task-level clustering (code). The
  framework's transfer claim is *cross-modal instantiation*, evidenced by both.
- **Why multi-view evidence?** Single views measurably cap the task (best text view MCC 0.42; Jaccard
  degenerate at MCC 0); fusion lifts to 0.53+ and survives renaming/paraphrase attacks that collapse
  single views.
- **Why uncertainty estimation?** Adjudication requires bounded, honest risk reporting: calibrated
  probabilities + conformal prediction sets + Wilson CIs on coverage.
- **Why budget awareness?** Heavy verification is 18–20× costlier per pair; flat application is
  operationally infeasible; routing saves 27–61% at ε-parity.
- **Why explainable aggregation?** Disciplinary processes are appealable; dossiers translate model
  state into reviewable records without leaking ground truth.
- **Measurable benefits demonstrated:** F1 0.7273 text / 0.8481 code (β=1) − best headline rows in
  both modalities; significant deltas vs 4/5 code peers; iso-recall and iso-cost dominance;
  robustness wins under renaming/paraphrase; 47–61%/15–27% compute savings; leakage inflation
  quantified and eliminated.
- **What remains unresolved?** Group conformal undercoverage (0.93 vs 0.95); code calibration under
  LOGO; span-level explainability; proxy-corpus ecological validity; small code corpus; no real-world
  monetary costing; single-seed determinism.
- **Strongest defensible contribution:** the **calibrated, budget-adaptive, leak-free
  evaluation-and-routing methodology** (C2+C3 designed and *validated as a pair*) — the statistically
  honest version of an "outperform" claim plus its cost accounting — rather than any single accuracy
  number.

## What a Q1 manuscript may defensibly claim

1. A cross-modal framework unifying text+code integrity screening with calibrated UQ and dossiers.
2. Leak-free protocols with measured leakage inflation (+0.12…+0.15 MCC correction artifact).
3. Dominance at matched recall and matched cost; best-in-field code quality at β=1 with significant
   wins vs 4/5 peers; explicit ties disclosed against the strongest single peer.
4. Parsimonious robustness defense via leak-free adversarial augmentation, quantified on both
   modalities.

What it must **not** claim: unconditional guarantees; span-level ground truth; cross-dataset model
transfer; blanket superiority over SVM'24 at a fixed threshold.
