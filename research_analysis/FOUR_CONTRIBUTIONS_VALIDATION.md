# Phase 19 — Validation of the Four Research Contributions

### C1 — Robust Multi-View Candidate Generation
- **Novelty claim:** complementary multi-view fusion as the evidence layer of an auditable,
  cost-aware detection framework (fusion itself is established art; the *combination-in-context* is
  the claim).
- **Already established:** TF-IDF/embedding views (Paper 1), 7-D engineered code features (Papers
  2–3), tiling (RKRGST).
- **Methodological distinction:** fusion tuned for *routed* decisions; plus leak-free adversarial
  augmentation fixing measured evasion failure modes.
- **Experimental evidence:** ablation (fusion MCC 0.533 vs ≤0.416 any single text view); candidate
  recall 100% vs ES-Plag filter 0.8627; renaming resilience 0.89 (β=1) vs 0.28 (B-C3).
- **Limitations:** view set hand-chosen; LSA surrogate where MiniLM stores absent; no learned
  cross-modal encoder.
- **Classification:** **Experimentally validated** (within the two benchmarks).

### C2 — Uncertainty-Aware Adaptive Decisions
- **Mechanisms implemented:** predictive entropy, Platt/isotonic/temperature calibration, group-wise
  conformal sets, three-state adaptive routing.
- **Calibration evidence:** text ECE 0.0145–0.031, slopes ≈1.0; code honest weakness (LOGO ECE 0.181)
  disclosed.
- **Measurable decision improvements:** routed OP adds +0.045 MCC over fusion-only on text; selective
  abstention cuts heavy calls 47–61%.
- **Justified guarantees:** finite-sample *marginal* group coverage under exchangeability;
  conditional/guaranteed candidate recall is explicitly **not** claimed (novelty audit boundary 1).
- **Limitations:** group undercoverage 0.93 vs 0.95 (CI given); small calibration corpus (m=254).
- **Classification:** **Experimentally validated** (with a documented coverage shortfall).

### C3 — Budget-Constrained Routing Optimization
- **Strategy implemented:** entropy+dual-threshold tri-band router, budget cap β, ε-MCC
  cost-minimized OP (phase 5), per-fold verifier selection.
- **Savings demonstrated:** 61.4% text / 27.4% code verifier calls avoided; S2/S1 amortized ratios
  18.5×/20.2×.
- **Trade-offs:** MCC gap vs β=1 recorded — text 0.580 vs 0.553 at β=1 (routed OP higher due to
  precision-first gate), code 0.664 vs 0.700.
- **Vs fixed inference:** iso-cost frontier dominates all alternative routing signals ≤45% budget;
  parity by construction at β=1.
- **Limitations:** β=0.1-level claims rely on grid approximations; monetary unit-cost mapping
  unvalidated (latency ratios only).
- **Classification:** **Experimentally validated.**

### C4 — Explainable Cross-Modal Evidence Aggregation
- **Evidence integrated:** multi-view features, calibrated scores, routing state, conformal set,
  verifier identity, offsets/line indices.
- **Aggregation:** unified EvidenceDossier schema across modalities.
- **Explanations generated:** operational dossiers (50-doc audit sample; 100% generation success).
- **Faithfulness evaluation:** feature-ablation probability-drop exists; token-masking OOD caveat
  acknowledged; span-level P/R/F structurally omitted.
- **Measurable benefits:** explanation latency 0.204 ms; governance constraint (no ground-truth in
  reviewer dossiers) implemented.
- **Limitations:** no human-factors study; faithfulness quantification partial.
- **Classification:** **Methodologically implemented but partially validated.**

| Contribution | Verdict |
|---|---|
| C1 | Experimentally validated |
| C2 | Experimentally validated (coverage shortfall disclosed) |
| C3 | Experimentally validated |
| C4 | Methodologically implemented; partially validated |
