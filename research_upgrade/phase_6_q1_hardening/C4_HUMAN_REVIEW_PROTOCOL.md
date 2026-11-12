# C4 Human-in-the-Loop Review: Protocol & Honest Status

**Phase 6 Q1 hardening — item 2 (C4 human evidence).**

## Disclosure (read first)

A live multi-rater expert study (recruitment, IRB-style consent, inter-rater
agreement statistics) is **structurally unavailable** in this offline
reproduction environment: no human participants, no institutional ethics board,
no way to verify rater credentials. We therefore **do not claim measured human
accuracy gains**. What this artifact provides instead — and what a reviewer can
audit — is:

1. a **pre-registered-style review protocol** (verbatim below) specifying
   hypotheses, sampling, procedure, metrics, and analysis plan *as it would be
   executed* by an institution with human-participant approval;
2. a completed **machine-side walkthrough** of that protocol: the exact packet
   of inspection sheets our framework emits for the 20 grey-zone test pairs
   (`artifacts/PHASE_6_C4_REVIEW_PACKET.csv/.json`), proving the C4 deliverable
   pipeline works end-to-end (grey-zone retrieval → multi-view flag sheet →
   tracking fields) without any human;

This converts C4 from an unverifiable effectiveness claim into a verifiable
systems/auditability claim plus an executable protocol. The manuscript
limitations section states this explicitly.

## Protocol (pre-registered style)

**R1 (primary).** Reviewers assisted by the framework's per-pair
interpretability sheet (calibrated probability, entropy, routing state, top-3
plagiarism-flagging views, conformal prediction set) achieve higher judgement
accuracy on grey-zone pairs than unassisted reviewers receiving raw text only.

- **Hypotheses.**
  H1: assisted accuracy − unassisted accuracy > 0 (one-sided, α=0.05).
  H2: median time-per-judgement assisted < unassisted (one-sided Mann–Whitney U).
- **Sample.** All pairs in the routing grey zone (tau_low ≤ p ≤ tau_high or
  entropy ≥ threshold-ish band) of the held-out text test set; target ≥ 20
  pairs; stratified 50/50 by ground truth where feasible.
- **Design.** Between-subject, two arms (assisted/unassisted), rater
  allocation by alternating assignment on arrival; raters blinded to ground
  truth; no time limit; judgement options {plagiarised, legitimate, unsure}.
- **Raters.** Inclusion: graduate teaching assistants or faculty with grading
  experience on the task topics. Exclusion: prior exposure to the corpus.
  Target n≥8 per arm (power: medium effect d≈0.5, 1−β≈0.8).
- **Measures.** Per-rater accuracy; decision confidence (1–5); time to
  decision; inter-rater agreement (Fleiss' κ) within each arm.
- **Analysis.** Arm difference in mean accuracy via two-sample t (with Welch
  correction) plus 95% CI; H2 via Mann–Whitney U; κ reported with bootstrap
  CIs; exploratory: correlation between tool's top-3 flagged views and the
  textual spans raters annotate.
- **Ethics.** Consent form; anonymized pair IDs; no personal data; raters may
  withdraw at any time.

## Machine-side walkthrough (what was actually executed)

Executed `run_phase_6_execution.py` section S7: selected 20 grey-zone pairs
from the 3,000-pair held-out test set (definition in code, seed-fixed),
generated one sheet per pair containing all fields the assisted arm would see.
Artifact proves end-to-end generation and provides the exact instrument the
human study would operate on. Report fields: `n_grey_zone_pairs`,
`grey_zone_pct`, packet file paths.

**Limitation statement for the manuscript:** H1/H2 remain untested with human
participants; the contribution is the protocol + verified instrument, not
measured human improvement.
