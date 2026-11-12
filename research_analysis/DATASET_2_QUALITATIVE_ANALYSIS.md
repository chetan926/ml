# Phase 5 — Dataset 2 Qualitative Analysis
*(IR-Plag Java corpus → `code_plagiarism_benchmark.csv`)*

1. **Dataset origin.** Authentic corpus by Oscar Karnalim (2019), comprising introductory
   programming submissions with systematically graded transformations; "fully authentic" per the
   repository provenance audit.
2. **Research purpose.** Source-code plagiarism detection under pedagogically realistic obfuscation —
   exactly the deployment scenario of undergraduate assignment screening.
3. **Data structure.** 467 physical `.java` files organized as 7 cases (`case-01`…`case-07`), each
   with `original` solutions, `non-plagiarized` student submissions (numbered 01–15), and
   `plagiarized` directories grouped into obfuscation levels L1–L6.
4. **Feature characteristics.** Raw Java source strings; pairs carry `case`, `ref_type`,
   `sus_type`, and `obfuscation_level`. All numeric features (token/AST/complexity) are derived by
   `src/features/code_*.py` (token histograms, AST shape, identifier anonymization, Halstead-style
   complexity).
5. **Label definitions.** `is_plagiarized` = 1 when the suspicious file descends from the original
   via a documented transformation chain; obfuscation_level ∈ {L1,L2,L3,L4,L5,L6} grades severity
   (L1 verbatim/light edits → L6 compound multi-transform disguise).
6. **Annotation quality.** Transformation-procedural labels are unambiguous by construction
   (positives are *generated*), a qualitatively stronger ground truth than crowd labels.
7. **Missing information.** 378 structural NaNs in `obfuscation_level` — *by design*, negatives are
   not obfuscation instances. No accidental missingness (verified: all other columns complete).
8. **Duplicate records.** **24 rows in 11 collision groups** of identical (ref,sus) pairs (verified);
   the project freezes two transparent policies: Policy A (drop subsequent → 720 rows) and Policy B
   (drop entire groups → 709 rows).
9. **Data consistency.** Case sizes 94–108 pairs; positive share 42.6%–50.0% per case — a stable,
   balanced design.
10. **Class ambiguity.** Minimal: label provenance is procedural. Residual ambiguity lies in
    *independent-work similarity* (negatives sharing library/boilerplate patterns).
11. **Dataset complexity.** High for a 7-case corpus: identifier-renaming (L2), structural reordering
    (L3–L4), compound transformations (L6); obfuscation strata make it possible to measure resilience
    *graded by severity*.
12. **Potential leakage.** The dominant risk is *task/premise leakage*: programs solving the same
    assignment share algorithmic skeletons, so stratified pair-level folds leak. The project
    quantified this directly: stratified 5-fold inflates MCC by **+0.1221** (phase-4.1) over nested
    LOGO; the upgrade re-measured an inflation of **+0.1534**. Case-05 is a documented singularity
    (all 53 positives derive from one base program → cross-case training needed).
13. **Generalization challenges.** 7 cases only → bounded cross-case variance; Java-only; small
    corpus (720 pairs) → wide CIs; L6 positives are few (63 pairs).
14. **Suitability for academic integrity detection.** **High** — this is the modality where the
    framework's claims (renaming resilience, heavy-verifier economics, leak-free LOGO) carry direct
    ecological meaning.
15. **Suitability for the proposed framework.** Strong: graded difficulty enables ablation by attack
    level; the 7-case structure is exactly the grouping unit LOGO/grouped bootstrap require; the
    measured heavy-verifier economics (Stage-2 = 20× Stage-1) are realistic here.

**Strengths.** Authentic, procedurally labeled, graded obfuscation, balanced classes, leakage precisely
measurable. **Weaknesses.** Small; one language; 7 cases; duplicate collisions need policy handling;
no span-level ground truth. **Limitations.** Cross-case variance and CIs are necessarily wide;
conclusions transfer cautiously to large-cohort MOOC-scale corpora.
