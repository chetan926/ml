# Phase 3 — Dataset 1 Qualitative Analysis
*(SNLI/Pereira "MIT Plagiarism Detection" text-reuse benchmark → `text_plagiarism_benchmark.csv`)*

## 3.1 Dataset characteristics

- **Research purpose (as used here):** sentence-level similarity detection benchmark standing in for
  short-text academic plagiarism/paraphrase reuse. The project's provenance audit
  (`experiments/results/dataset_provenance_audit.md`) explicitly reclassified it from "MIT academic
  plagiarism" to an **SNLI-derived semantic-equivalence proxy** — an honest labeling the analysis
  preserves.
- **Data origin:** SNLI (Bowman et al., EMNLP 2015): premises are Flickr30k image captions;
  hypotheses were written by crowd-workers under entail/neutral/contradict instructions. Kaggle
  re-export by Ruvel Pereira.
- **Domain characteristics:** short everyday-scene descriptions; vocabulary is concrete, low-register,
  non-academic. References (premises) are longer (≈13.9 words) than suspicious texts (≈7.5 words).
- **Data structure:** pair table (`reference_text`, `suspicious_text`, 3-way `label`, binary
  `is_plagiarized`, source `split`).
- **Feature types:** no precomputed columns beyond raw strings; all modeling features (TF-IDF, char
  n-grams, sentence embeddings, stylometry) are derived downstream.
- **Label definitions:** positive = entailment (semantic equivalence ≈ paraphrase reuse);
  negative = neutral ∨ contradiction. Conceptually reasonable as a *similarity* proxy; **not**
  a faithful operationalization of academic plagiarism (no authorship/intent dimension).
- **Annotation methodology:** SNLI 5-way annotator consensus; only consensus pairs retained
  (label `-1` dropped by the Kaggle export — none present in-repo).
- **Collection methodology:** image-caption elicitation; hence premise-group structure (each premise
  caption spawns multiple hypotheses) — the basis for premise-level exchangeability in the project's
  group-conformal evaluation.

## 3.2 Data quality

- **Missing information:** none (0 null cells; verified).
- **Duplicate records:** 2 ordered redundant duplicate rows, purged (Policy-A style) in the clean
  pool; 2 full duplicated rows over the 5-column identity.
- **Inconsistent records:** none observed; schema validator enforces types.
- **Noisy observations:** SNLI hypotheses are telegraphic; some entailment pairs share few tokens
  (hard positives), some neutral pairs share many (hard negatives).
- **Ambiguous labels:** the entailment↔plagiarism mapping is the dataset's main *conceptual* caveat
  (Lexical overlap ≠ misconduct); label noise as annotation error is limited by consensus filtering.
- **Potential annotation errors:** inherent SNLI crowd noise remains; consensus filtering removes the
  worst cases.
- **Potential data leakage:** **the primary risk** — 1 premise appears in both the raw `val` and
  `test` pools (6 rows), purged by the loader (verified: 6 crossing rows + 2 dupes = 8 purged rows).
  Without purging, premise-level leakage would inflate any pair-level evaluation. The deterministic
  allocator guarantees premise-disjoint train/val/cal/test partitions (verified with 1,349/508/336/334/
  1,015 unique premises per partition, all disjoint).

## 3.3 Research suitability

- **Academic-integrity relevance:** moderate — tests lexical/semantic discrimination of paraphrase-like
  reuse, but not misconduct intent, citation norms, or document-level attitudes.
- **Plagiarism-case representation:** positives concentrate exactly the paraphrase regime that defeats
  TF-IDF — appropriate to stress Paper-1-style semantic methods.
- **Classification support:** well-formed binary task (34.2% positives on test); enough volume
  (19.7k pool) for statistically meaningful CIs.
- **Multi-view support:** strong — orthogonal lexical/character/semantic/structural views are
  meaningful on sentence pairs and empirically non-redundant (ablation: fusion ≫ any single view).
- **Uncertainty estimation support:** natural premise-group structure enables group-wise conformal
  prediction (254 calibration groups → 1,015 test groups); class balance stable across partitions
  (33.6–34.9%).
- **Cost-aware routing support:** ideal — a calibrated screener resolves most pairs, ambiguity mass
  concentrates in a minority, enabling measured 47–61% heavy-verifier savings.
- **Generalization challenges:** single genre (image captions → sentences); very short suspicious
  texts; conclusions must not be extrapolated to full-length essays.

## 3.4 Critical assessment

**Strengths.** (i) Large, clean, consensus-labeled; (ii) explicit group structure enabling
distribution-free UQ and leak-free partitioning; (iii) balanced positives (~1/3) yielding meaningful
precision/recall trade-off analyses; (iv) cheap to iterate on.

**Weaknesses.** (i) It is an **SNLI proxy, not a real misconduct corpus** — ecological validity for
academic integrity is limited; (ii) texts are very short (median hypothesis 7 words), compressing
feature signal; (iii) premise reuse concentrates topical similarity; (iv) no span-level annotations.

**Research opportunities.** (i) A clean testbed for the *calibrated cascade* thesis (C2+C3);
(ii) premise-level exchangeability is exactly what group conformal needs; (iii) well-suited for
measuring workload-reduction economics.

**Evaluation challenges.** Pair-level random splits would leak premises — this is the dataset's
principal trap, and the project's grouped protocol exists precisely because of it.

**Limitations affecting the proposed framework.** External validity toward full-length essay
plagiarism is bounded; results should be framed as *sentence-level text-reuse detection*. Group-count
granularity (254 calibration groups) leaves the conformal finite-sample margin thin (observed 2-point
undercoverage at nominal 95%).
