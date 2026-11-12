# Phase 9 — Paper 1 Qualitative Analysis

**Paper 1:** Setu, D. M., Islam, T., Erfan, M., Dey, S. K., Al Asif, M. R., & Alam, M. G. R. (2025).
*A comprehensive strategy for identifying plagiarism in academic submissions.* Journal of Umm
Al-Qura University for Engineering and Architecture (Springer Nature). DOI 10.1007/s43995-025-00108-1.
Evidence base: repository-verified metadata (`literature/selected_papers.csv` M1; audit §2.1) —
full-text PDF not stored in repo (Phase 8 §8.2).

## Research background
- **Problem:** detecting both verbatim copying and paraphrased reuse in academic submissions at scale.
- **Motivation:** lexical matchers fail on paraphrase; running dense transformer scoring on all pairs
  is computationally expensive (the paper itself reports an *expensive-call rate / workload-reduction*
  dimension — the repository's core motivation for staged routing).
- **Objectives/questions:** combine lexical and semantic evidence so that direct copies and semantic
  paraphrase shifts are both captured, with measured workload impact.

## Methodology (as recorded by the project audits)
- **Approach:** hybrid lexical-semantic similarity: sublinear TF-IDF cosine + Sentence-BERT
  (`all-MiniLM-L6-v2`, 384-d) dense embeddings, mean-pooled, L2-normalized; combined decision score
  `max(TF-IDF, SBERT)` categorizing direct-copies vs paraphrases.
- **Feature extraction:** word n-gram TF-IDF on one view; contextual sentence embeddings on the other.
- **Evaluation protocol (repository reproduction):** validation-tuned F1-optimal threshold
  τ = 0.5000 (val F1 = 0.6690) on the SNLI/Pereira proxy, evaluated out-of-sample on the same
  grouped benchmark as the proposed framework.
- **Two repository conditions:** E1 = paper-faithful full hybrid (both scorers on every pair,
  4.163 ms/pair) and E2 = staged dual-threshold routing (2.875 ms/pair, ≈30% workload reduction) —
  an explicit cost analysis of the paper's design choices.

## Experimental investigation (repository reproduction context)
- **Dataset:** adapted to the sentence-level SNLI/Pereira corpus (the paper's original context was
  full-length academic documents/thesis chapters — marked as an adaptation boundary).
- **Baselines:** sublinear TF-IDF lexical baseline (Salton/Sajid lineage).
- **Metrics:** Accuracy/Precision/Recall/F1/ROC-AUC plus expensive-call rate and workload reduction.
- **Main findings (repository-audited):** contextual embeddings succeed where lexical methods fail
  (paraphrase substitutions), at high per-pair compute; the hybrid max score covers both regimes.

## Critical evaluation
- **Strengths:** conceptually simple, competitive sentence-level discrimination; acknowledges the
  compute axis explicitly (workload reduction is a first-class metric).
- **Methodological limitations:** single-threshold decisions; no calibrated probabilities; no
  uncertainty quantification; `max()` fusion is brittle for correlated score failures.
- **Computational limitations:** E1 pays the transformer cost on every pair; E2's fixed-band routing
  has no budget formalization and no per-pair cost accounting tied to a target budget.
- **Generalization limitations:** validated (in-repo) only on sentence-level proxy data.
- **Uncertainty-handling limitations:** none present.
- **Explainability limitations:** numeric similarity scores only; no casewise evidence records.
- **Unaddressed challenges:** false-accusation control under regulated error budgets; grouped
  evaluation (premise leakage); distribution-free coverage.

## Research relevance
- **Learnings:** hybrid lexical+semantic fusion materially helps; cost of dense scoring must be
  managed — the direct ancestor of the proposed C3 staged routing.
- **Relevant to framework:** Paper 1 supplies B-T2 (dense semantic) and the lineage of the hybrid
  comparator in the iso-recall/iso-cost studies; E2 is the closest "ancestor" of tri-band routing.
- **Unresolved by Paper 1:** calibrated, budget-formalized, uncertainty-aware, auditable decisions.
- **Difference of proposed solution:** replaces static thresholding/`max` fusion with calibrated
  fusion, group-conformal prediction sets, formal budget routing (Eq. OP minimization), and
  dossier-level auditability — with pairwise statistics under premise-disjoint protocols.
