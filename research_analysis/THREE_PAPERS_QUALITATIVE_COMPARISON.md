# Phase 13 — Qualitative Comparison of the Three Repository Papers

| Dimension | Paper 1 — Setu et al. 2025 | Paper 2 — Gandhi et al. 2024 | Paper 3 — Mehsen & Joshi 2024 |
|---|---|---|---|
| Research objective | Hybrid lexical+semantic text plagiarism detection with workload awareness | Obfuscation-resilient code detection via multi-feature SVM | Ensemble ML code detection under disguise |
| Problem addressed | Paraphrase defeats lexical matching; dense scoring is costly | Renaming/comment edits defeat tiling | Compound disguises defeat linear separators |
| Methodology | `max(TF-IDF, SBERT)`; E1 full vs E2 staged variants | 7-D syntactic/token features → RBF-SVM | RF ensemble over token-frequency/syntactic vectors |
| Algorithms | TF-IDF cosine; Siamese MiniLM mean-pooling | SVC RBF (C=1.0) | RF (100×d6), DT, hierarchical clustering |
| Architecture | 384-d dense embeddings + sparse lexical view | hand-built 7-D metric vector | ensemble of trees over engineered vectors |
| Dataset (repo) | SNLI/Pereira sentence-level proxy | IR-Plag Java (Python→Java adaptation) | IR-Plag Java (private→IR-Plag adaptation) |
| Feature extraction | TF-IDF + contextual embeddings | comment stripping; anonymized identifiers; complexity | structured token frequencies; syntactic metrics |
| Uncertainty handling | None | None (uncalibrated Platt-style outputs) | None |
| Explainability | Similarity scores only | None | None |
| Computational efficiency | Explicit but informal (expensive-call rate; E2 ≈30% workload cut) | Classifier cheap; extraction cost unbudgeted | 0.025 ms/pair; no budget model |
| Evaluation strategy | Threshold tuned on validation; repository reproduces on grouped benchmark | case-holdout; nested-LOGO in repo reproduction | case-holdout; nested-LOGO in repo reproduction |
| Strengths | Covers verbatim+paraphrase; cost awareness exists | Strongest deep comparator; obfuscation-aware features | Best compound-attack resilience among peers (L6) |
| Limitations | No calibration/UQ; no budget formalization; no dossiers | No UQ; no budget model; renaming attack still collapses it | No UQ; no budget model; small private-corpus origin |

**Cross-cutting.** All three papers share the same three structural omissions: (i) no calibrated
probability discipline or distribution-free uncertainty; (ii) no formal, budget-first cost model
(Paper 1's workload cut is the nearest approach but stays heuristic); (iii) no casewise audit
explainability. None evaluates under premise/task-grouped protocols with clustered statistics —
the methodological correction this project installs.
