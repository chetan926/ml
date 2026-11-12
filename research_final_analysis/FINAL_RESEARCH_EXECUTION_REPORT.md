# FINAL RESEARCH EXECUTION REPORT

## §20 answers (evidence-linked)
1. Datasets integrated: YES (leakage checks PASS; zero premise/pair overlap).
2. Three papers identified: YES (Setu 2025, Gandhi 2024, Mehsen 2024 — verified metadata).
3. Three methodologies implemented: YES (frozen phase-5 suites; shadow refits bit-exact).
4. All evaluated: YES (12 systems, full metric sets, ROC/PR curves).
5. Proposed framework implemented: YES (phase-5 pipeline; predictions stored).
6. Four contributions validated: YES (see FOUR_CONTRIBUTIONS.md matrix; C4 without human-effect claim).
7. Highest accuracy — text: proposed 0.808; code: B-C4 RF 0.7889 vs proposed beta=1 0.8472.
8. Highest precision — computed per system in CLASSIFICATION_METRICS.csv.
9. Highest recall — B-C5 ES-Plag (recall-first) as expected; proposed competitive.
10. Highest F1 — text: proposed 0.7273; code: beta=1 0.8481.
11. Lowest cost — proposed (adaptive routing; 61.4% text / 27-34% code avoided calls).
12. Proposed outperforms all three methods? TEXT: yes on F1/accuracy with large cost savings (McNemar significant vs all). CODE (tuned OP): trades -0.0076 F1 vs strongest SVM for 27-34% cost savings; beta=1 exceeds all on F1=0.8481.
13. Issues identified: conformal under-coverage (repaired in phase-6); external calibration non-transfer (documented).
14. Improvements implemented: phase-5 (leak-safe protocol, augmented robustness) + phase-6 (six hardening items).
15. Measurable benefits: yes (statistics in STATISTICAL_TESTS.json; multi-seed SD <= 0.010).
16. Ablation establishes: multi-view fusion is the dominant quality term; calibration adds small MCC; routing+calibration achieve the quality-cost envelope; budget cap binds only when intended.
17. Qualitative findings: errors concentrate in entropy-flagged grey zone; uncertainty marks hard cases.
18. Remaining limitations: see 12_FINAL_AUDIT/LIMITATIONS.md.
19. Reproducible: YES (guide + seeds + freeze; bit-exact integrity check).
20. Q1 manuscript support: see verdict.

## Verdicts
- **Implementation verdict:** COMPLETE.
- **Experimental verdict:** EFFICIENT-TRADE-OFF (text wins outright with cost savings; code tuned-OP trades 0.8 F1 points for 27-34% savings; beta=1 leads code quality).
- **Research manuscript verdict:** REQUIRES MAJOR REVISION.

Justification: every internally-valid claim is experimentally grounded; the external-transfer
(calibration drift) result and absence of a live human study are material gaps for Q1
deployment claims — both are stated and scoped precisely rather than hidden.
