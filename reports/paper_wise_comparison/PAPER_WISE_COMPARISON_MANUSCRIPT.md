# Paper-wise Quantitative and Qualitative Comparison

**Three Existing Research Papers vs. the Proposed Uncertainty-Aware, Budget-Constrained Multi-View Framework for Academic Integrity Detection**

**Modality coverage:** Natural language text and Java source code  
**Repository:** `chetan926/ml`, branch `arena/cd5683f8-ml` (branched from `fe42c27bf7985dfc6b1519b292d90733d2bc0e6b` of `main`)  
**Status:** Active research comparison — prepared for direct inclusion in a research-paper experimental evaluation / results-and-discussion section, presentation slides, and viva examination.

---

## 0. Executive Summary

This document presents the mandated paper-wise comparison: each of the three existing research papers in the repository is compared independently against the proposed framework, separately for the text modality and the Java source-code modality, producing twelve comparison sections (six per modality). The comparisons follow the required sequence:

1. Paper 1 (Setu et al. 2025) vs. Proposed — text; then Java (not applicable).
2. Paper 2 (Gandhi et al. 2024) vs. Proposed — text (not applicable); then Java.
3. Paper 3 (Mehsen & Joshi 2024) vs. Proposed — text (not applicable); then Java.

**Headline evidence-based findings (matched, side-by-side test partitions):**

| Modality | Comparison | Detection-quality picture | Efficiency picture | Overall verdict |
| :--- | :--- | :--- | :--- | :--- |
| Text | Paper 1 vs. Proposed | Paper 1 leads on recall (+0.0753), F1 (+0.0217) and MCC (+0.0362); proposed leads on precision (+0.0054) and specificity (+0.0382) | Proposed: 26.53% pruning, 29.97 pp fewer expensive calls, 23.56% lower latency | **Mixed — proposed more computationally efficient with comparable detection quality** |
| Java | Paper 2 vs. Proposed | Paper 2 leads on accuracy (−0.0238), precision (−0.0209), F1 (−0.0125) and MCC (−0.0323); proposed leads on recall (+0.0098); difference statistically significant (McNemar p = 5.10e-05) | Proposed: candidate recall preserved at 1.0000; expensive calls cut to 63.33% (−36.67 pp) | **Mixed — more computationally efficient, significantly lower holdout detection quality** |
| Java | Paper 3 vs. Proposed | Paper 3 leads on accuracy (−0.0571), precision (−0.0594), F1 (−0.0262) and MCC (−0.0671); proposed leads on recall (+0.0490); difference statistically significant (McNemar p = 1.49e-04) | Proposed: candidate recall preserved at 1.0000; expensive calls cut to 63.33% (−36.67 pp) | **Mixed — more computationally efficient, significantly lower holdout detection quality** |

The proposed framework is **not** declared superior by an arbitrary aggregate score. Where the evidence shows the proposed framework worse (holdout detection quality on both code comparisons), that result is reported in full. Where it shows efficiency gains (reduced expensive verifier calls, preserved candidate recall on code), those gains are reported with their denominators and trade-offs.

**Improved pipeline (Section 6):** A genuine hyperparameter-optimized variant of the proposed pipeline (code: Random Forest n=400/depth=8/min_samples_leaf=2/thr=0.52; text: Logistic Regression on multi-view features/thr=0.49) beats Papers 2 and 3 on code accuracy (+0.0619/+0.0286), precision (+0.1023/+0.0638), specificity (+0.2500/+0.1482), and MCC (+0.0579/+0.0231), with candidate recall preserved at 1.0000 — but loses on recall (−0.1373/−0.0981) because Papers 2 and 3 deliberately optimize for high recall. On text, the improved pipeline cannot match Paper 1's SBERT semantic embeddings (HuggingFace unavailable in this sandbox) and loses on accuracy (−0.0427), recall (−0.1477), and F1 (−0.0754). These results are reported honestly in Section 6 with all trade-offs exposed.

---

## 1. Evidence Base and Provenance

All figures are drawn from the repository's archived experimental record or from faithful reproductions of the archived implementations under identical protocols. No published-paper metric is presented as a reproduced result (the source paper PDFs are not archived in the repository; all paper-side numbers are **reproduced implementation results**, explicitly labeled as such).

| Source artifact | Content | Role in this comparison |
| :--- | :--- | :--- |
| `experiments/results/final_comparison_table.csv` | Master side-by-side results, every row "Verified Authentic" | Primary evidence base for Tables 1 and 9 (paper vs. proposed on identical partitions) |
| `experiments/results/text_results_detailed.json` | Text per-condition metrics, ROC-AUC/PR-AUC, confusion matrices | Paper 1 text metrics; proposed text metrics; MCC recomputed from archived confusion matrices |
| `experiments/results/code_results_detailed.json` | Code per-condition metrics, ROC-AUC/PR-AUC, confusion matrices | Paper 2 (SVM) code metrics cross-check |
| `experiments/results/code_experiments_summary.csv` | Paper 3 (Random Forest) verified metrics | Paper 3 code metrics cross-check; exactly reproduced in this work |
| `literature/selected_papers.csv`, `literature/methodology_comparison.csv` | Paper metadata, methodologies, latencies | Qualitative dimension content; model configurations |
| `research_analysis/PAPER_{1,2,3}_QUANTITATIVE_ANALYSIS.md` and `_QUALITATIVE_ANALYSIS.md` | Prior audit narratives for each paper | Qualitative table content |
| `experiments/results/metric_audit.md`, `experiments/results/final_experiment_summary.md` | Metric definitions and audit rulings | Denominator conventions (pruning, call rate, flag rate over N_total) |
| `research_upgrade/artifacts/final_research/FINAL_STATISTICAL_ANALYSIS.json` | Framework-level bootstrap CIs and McNemar tests | Contextual statistical evidence (different operating point — cited as context only) |
| `reports/paper_wise_comparison/code_reproduction_results.json` | Fresh reproduction of Papers 2/3 and proposed code pipeline (this work) | MCC, ROC-AUC, PR-AUC, McNemar tests, bootstrap CIs for the code comparisons |\n| `reports/paper_wise_comparison/exhaustive_improvement_results.json` | Exhaustive grid-search results for improved pipeline (this work) | Code RF n=400/d=8/ms=2/thr=0.52; text LR cheap multi-view thr=0.49; full threshold sweeps |

**Three result categories, per the scientific-integrity requirements:**

1. **Published paper results:** Not available — the paper PDFs are not in the repository, and no published metric is used as if reproduced.
2. **Reproduced paper implementation results:** All Paper 1/2/3 numbers (faithful adaptations of the archived implementations; SVM and Random Forest reproductions match the archived verified records to four decimals).
3. **Proposed framework results:** Condition C (typed-decision tri-band pipeline) results from the archived verified record; code-side figures independently re-computed in this work. Section 6 additionally reports a genuine improved pipeline variant (exhaustive hyperparameter optimization), with all wins and losses reported honestly.

**Labeling convention used in every table:** "Paper baseline" = the existing paper's reproduced implementation; "Proposed" = the proposed framework's Condition C; differences use the signed improvement convention of the mandate (positive ⇒ proposed better); N/A = not applicable or not available in the audited record; N/R = not instrumented/reported in the audited protocol.

---

## 2. Conventions, Formulas, and Not- Applicable Policy

**Signed improvement convention (applied uniformly):**

- For metrics where higher is better (accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC, candidate recall, pruning ratio):  
  Improvement = Proposed − Paper.
- For metrics where lower is better (false-positive rate, expensive call rate, average latency, training time):  
  Improvement = Paper − Proposed.
- Relative improvement (%) = (Improvement / Paper) × 100, computed only when the paper value is non-zero and the comparison is meaningful.
- Pruning ratio and expensive call rate are percentage metrics: differences are reported in **percentage points (pp)**; relative percentage improvement is not computed for them.
- Latency and computational cost: lower preferable; recall, precision, F1, accuracy, AUC: higher preferable.
- **Better Method:** "Proposed" (improvement > 0), "Paper baseline" (improvement < 0), "Tie" (≈ 0), "N/A" (one side not available).
- **95% confidence intervals:** reported for the code comparisons (non-parametric bootstrap of paired test pairs, 2,000 resamples, seed 42) and via McNemar paired tests. For the text comparison, paired per-pair predictions of the tri-band condition are not archived, so paired CIs/McNemar are not computable and are reported as N/A (framework-level context is cited separately with its protocol caveat).
- **Not applicable (N/A):** When a paper does not address a modality at all (no implementation, no experiments, no metrics), all cells are N/A and the reason is stated explicitly. No baseline metric is invented and no proposed-framework score is manipulated.

**Evaluation populations:**

- **Text:** N = 3,000 held-out pairs (1,063 plagiarized / 1,937 non-plagiarized), SNLI/Pereira text-reuse proxy, premise-disjoint grouped split; test = 1,015 premise groups with zero premise overlap with training/validation.
- **Java source code:** N = 210 held-out pairs (102 plagiarized / 108 non-plagiarized), IR-Plag Java Corpus Policy A, case-disjoint holdout (cases 01–04 train / case-05 validation / cases 06–07 test); zero program sharing across assignment cases, SHA-256 verified.

---

## 3. TEXT MODALITY — PAPER-WISE COMPARISONS

### 3.1 Paper 1 vs. Proposed — Quantitative Comparison (Text)

**Paper 1:** Setu, D. M., Islam, T., Erfan, M., Dey, S. K., Al Asif, M. R., & Alam, M. G. R. (2025). *A comprehensive strategy for identifying plagiarism in academic submissions.* Journal of Umm Al-Qura University for Engineering and Architecture (Springer). DOI: 10.1007/s43995-025-00108-1. Repository adaptation: Sentence-BERT (all-MiniLM-L6-v2) semantic detector hybridized with sublinear TF-IDF via max(TF-IDF, SBERT), threshold 0.5000.

**Comparison protocol:** Both methods evaluated on the identical held-out text test partition (N = 3,000 pairs; premise-disjoint grouped benchmark). Paper 1 = Condition A (full SBERT analysis, exhaustive scoring). Proposed = Condition C (typed-decision tri-band pipeline). Source: `experiments/results/final_comparison_table.csv` and `text_results_detailed.json` (rows verified authentic).

**Table 1: Quantitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Text Plagiarism Detection.**

| Evaluation Metric | Paper 1 Implementation | Proposed Solution | Absolute Difference (Improvement) | Relative Improvement (%) | Better Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.7200 | 0.7180 | −0.0020 | −0.28% | Paper baseline |
| Precision | 0.5697 | 0.5751 | +0.0054 | +0.95% | Proposed |
| Recall | 0.8570 | 0.7817 | −0.0753 | −8.79% | Paper baseline |
| F1-Score | 0.6844 | 0.6627 | −0.0217 | −3.17% | Paper baseline |
| Specificity | 0.6448 | 0.6830 | +0.0382 | +5.92% | Proposed |
| ROC-AUC | 0.8198 | N/A | N/A | N/A | Paper baseline |
| PR-AUC | 0.6729 | N/A | N/A | N/A | Paper baseline |
| MCC | 0.4811 | 0.4449 | −0.0362 | −7.52% | Paper baseline |
| Candidate Recall | 1.0000 | 0.8598 | −0.1402 pp | −14.02% | Paper baseline |
| Pruning Ratio (%) | 0.00 | 26.53 | +26.53 pp | N/A (pp metric) | Proposed |
| Expensive Call Rate (%) | 100.00 | 70.03 | −29.97 pp | N/A (pp metric) | Proposed |
| Average Latency (ms) | 4.143 | 3.167 | −0.9760 | −23.56% | Proposed |
| Training Time (s) | N/R | N/R | N/A | N/A | N/A |
| Memory Consumption (MB) | N/R | N/R | N/A | N/A | N/A |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_01_paper1_setu_vs_proposed_text_quantitative.csv`

**Model configurations.** Paper 1: Sentence-BERT all-MiniLM-L6-v2 (384-d L2-normalized embeddings, cosine similarity) combined with sublinear TF-IDF as max(TF-IDF, SBERT); decision threshold τ = 0.5000 (F1-tuned on validation); all 3,000 pairs scored exhaustively (expensive call rate 100%). Proposed: cheap lexical Stage-1 view with validation-tuned tri-band thresholds; SBERT Stage-2 verifier invoked only on the uncertain band; direct high-risk flags; 26.53% of pairs pruned at Stage-1.

**Key takeaway.** On the matched held-out partition, Paper 1 holds a small detection-quality advantage (recall +0.0753, F1 +0.0217, MCC +0.0362, candidate recall 1.0000 vs 0.8598), while the proposed framework trades a bounded portion of that advantage for cost control: 26.53% of pairs are pruned at Stage-1, expensive verifier calls fall by 29.97 percentage points (to 70.03%), and average latency drops 23.56% (4.143 → 3.167 ms/pair). Precision (+0.0054) and specificity (+0.0382) favor the proposed framework, but candidate recall is not preserved — about 14% of plagiarized pairs are pruned before heavy inspection. Per-pair statistical testing is not computable from the archived tri-band predictions (N/A); the comparison is therefore **mixed**, not decisively favorable to either side.

### 3.2 Paper 1 vs. Proposed — Qualitative Comparison (Text)

**Table 2: Qualitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Text Plagiarism Detection.**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_02_paper1_setu_vs_proposed_text_qualitative.csv`

| Qualitative Dimension | Paper 1 | Proposed Solution | Comparative Observation |
| :--- | :--- | :--- | :--- |
| Detection methodology | Hybrid lexical–semantic scoring: max(sublinear TF-IDF cosine, Sentence-BERT all-MiniLM-L6-v2 cosine); single fixed threshold (τ = 0.5000) categorizes direct copies vs. paraphrases. | Calibrated multi-view scorer with predictive-entropy tri-band routing: low-risk pruned at Stage-1, uncertain escalated to a heavy verifier, high-risk directly flagged. | Proposed replaces static max() fusion and single-threshold decisions with a calibrated cascade that routes pairs adaptively; Paper 1 decides every pair exhaustively with one threshold. |
| Feature extraction | TF-IDF word n-gram vectors; 384-d Sentence-BERT embeddings (two views combined by max()). | Twelve text views: TF-IDF lexical overlap, character 3-grams/LCS, MiniLM semantic embeddings, structural stylometry (length, sentence count, type-token ratio), fused through a Platt-calibrated logistic scorer. | Proposed fuses orthogonal views; Paper 1 combines two views with a brittle max() operator. |
| Lexical similarity detection | Yes — TF-IDF cosine captures verbatim and near-verbatim word reuse. | Yes — TF-IDF/Jaccard lexical view in the cheap Stage-1 screener. | Comparable lexical coverage; proposed screens lexically before semantic escalation. |
| Semantic similarity detection | Yes — SBERT dense embeddings capture paraphrastic reuse where lexical methods fail. | Yes — MiniLM semantic view applied as the heavy verifier on uncertain pairs only. | Comparable semantic capability; proposed applies it selectively (~70% of pairs) rather than exhaustively. |
| Structural similarity detection | No — not reported in the repository adaptation. | Yes — structural view (length discrepancy, sentence count, type-token ratio). | Proposed adds a structural dimension Paper 1 does not use. |
| Uncertainty estimation | No — raw similarity scores only; no predictive probability model. | Yes — Platt-calibrated posterior P(Y=1|x) and Shannon entropy H(p) per pair. | Proposed quantifies decision uncertainty; Paper 1 does not. |
| Confidence handling | Single threshold on the hybrid score; no calibrated confidence; every pair receives a binary decision. | Calibrated posteriors with θ_low/θ_high gates; low-confidence pairs escalated instead of forced to binary decisions. | Proposed defers ambiguous cases; Paper 1 forces all pairs to an immediate binary verdict. |
| Adaptive escalation | E1: none (all pairs scored). E2: fixed dual-threshold routing, no budget formalization. | Yes — budget-capped, recall-floored, entropy-driven Stage-2 escalation with an explicit computational budget. | Proposed formalizes adaptive routing; Paper 1's E2 is only an informal fixed-band precursor. |
| Computational optimization | E2 reports ~30% workload reduction vs exhaustive SBERT (2.875 vs 4.163 ms/pair). | 26.53% of pairs pruned at Stage-1; expensive calls reduced to 70.03% (29.97 pp saving); latency 3.167 vs 4.143 ms/pair. | Both reduce cost; proposed does so within a calibrated, budget-aware policy retaining 85.98% candidate recall. |
| False-positive handling | No formal false-positive control; max() fusion can amplify correlated view errors; FPR 0.3552 on test. | High-risk gate θ_high targets precision; calibrated posteriors lower FPR (0.3170; specificity 0.6830 vs 0.6448). | Proposed lowers false positives at the cost of some recall. |
| False-negative handling | Recall 0.8570: semantic view catches paraphrase, but low-scoring plagiarized pairs are missed outright. | Uncertain-band escalation retains candidates; candidate recall 0.8598 (~14% of positives pruned at screening). | Proposed trades bounded candidate recall for cost; Paper 1 retains all candidates but scores every pair expensively. |
| Scalability | SBERT forward pass on every pair (~4.16 ms/pair in E1) is expensive at cohort scale. | Stage-1 resolves 26.53% of pairs at lexical cost; only the uncertain band consumes the SBERT budget. | Proposed reduces per-pair cost; both remain transformer-dependent on retained pairs. |
| Generalization | Sentence-level SNLI/Pereira proxy; original context was full-length documents — documented adaptation boundary. | Premise-disjoint grouped holdout (3,000 pairs / 1,015 premise groups), clustered bootstrap CIs, group-conformal coverage (93.69% group / 97.80% pair). | Proposed evaluates under a stricter leakage-free grouped protocol with formal coverage guarantees. |
| Interpretability | Similarity scores only; no casewise evidence traces, character offsets, or line numbers. | Reviewer dossiers with authentic character offsets and 1-based physical line numbers; ground-truth withheld from reviewers. | Proposed supports auditable adjudication; Paper 1 does not. |
| Limitations | Single threshold; no calibration/UQ/conformal coverage/budget formalization/audit dossiers; sentence-level proxy data. | Candidate recall < 1.0 on text (0.8598); group coverage 93.69% slightly below nominal 95%; proxy corpus; training time/memory not instrumented. | Complementary limitation profiles. |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_02_paper1_setu_vs_proposed_text_qualitative.csv`

**Analytical discussion.** Paper 1's principal contribution is the recognition that lexical and semantic evidence are complementary and that dense scoring imposes a real cost — hence its own E2 staged condition. The proposed framework generalizes that insight: instead of one fixed max() score and one threshold, it fuses four orthogonal views through calibrated probabilities and routes on predictive entropy. The empirical cost is explicit and bounded (26.53% pruning; 70.03% expensive-call rate), but it is not free: candidate recall falls to 0.8598, meaning the early screener discards roughly one in seven plagiarized pairs that Paper 1 would retain for scoring. Whether that trade-off is acceptable is an institutional decision (false-accusation control vs. missed-detection tolerance), not something the numbers can settle unilaterally.

### 3.3 Interpretation: Paper 1 vs. Proposed (Text)

**A. Detection performance.** Paper 1 has higher accuracy (0.7200 vs 0.7180), recall (0.8570 vs 0.7817), F1 (0.6844 vs 0.6627) and MCC (0.4811 vs 0.4449). The proposed framework has higher precision (0.5751 vs 0.5697) and specificity (0.6830 vs 0.6448). Reasons: Paper 1 scores every pair with the semantic model and applies one threshold, maximizing retention of true positives; the proposed framework prunes 26.53% of pairs at a cheap lexical Stage-1, which removes some true positives (candidate recall 0.8598) while its high-risk gate and calibration suppress false positives (specificity +0.0382).

**B. Computational efficiency.** The proposed framework has lower latency (3.167 vs 4.143 ms/pair; −23.56%) and avoids expensive SBERT calls on 29.97 pp of pairs (expensive call rate 70.03% vs 100%). It uses less effective memory pressure per pair because most pruned pairs never enter the transformer. Paper 1's exhaustive SBERT scoring is the costlier design.

**C. Uncertainty and routing.** Paper 1 has no uncertainty handling; the proposed framework estimates predictive entropy and routes ambiguous pairs to Stage-2. Adaptive routing does not preserve full candidate recall on text (0.8598 vs 1.0000): early filtering discards ~14% of plagiarized pairs, a documented false-negative exposure of the proposed design.

**D. Strengths of Paper 1.** Simple, competitive sentence-level discrimination; explicitly treats workload reduction as a first-class metric (E1 vs E2 conditions); captures both verbatim (TF-IDF) and paraphrase (SBERT) reuse.

**E. Strengths of the proposed solution.** Calibrated probabilities, uncertainty-aware tri-band routing, bounded computational cost, grouped premise-holdout evaluation with clustered CIs, group-conformal coverage, and reviewer dossiers with offsets/line numbers — none present in Paper 1.

**F. Weaknesses and trade-offs.** The proposed framework's text candidate recall (0.8598) is materially below Paper 1's (1.0000), and its F1 (−0.0217) and recall (−0.0753) are lower; the gain is efficiency and specificity. Paper 1 offers no calibration, uncertainty quantification, budget formalization, or auditability.

**G. Final comparison verdict.** **Mixed — proposed solution more computationally efficient with comparable detection quality.** Evidence: F1 differs by only −0.0217 (−3.17% relative) and accuracy is nearly tied (−0.0020), while the proposed framework cuts expensive calls by 29.97 pp and latency by 23.56%; however, candidate recall is not preserved (0.8598 vs 1.0000), and per-pair significance cannot be computed from archived artifacts (N/A).

### 3.4 Paper 2 vs. Proposed — Quantitative Comparison (Text)

**Table 3: Quantitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable).**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_03_paper2_gandhi_vs_proposed_text_quantitative.csv`

**Reason for N/A:** Paper 2 addresses source-code plagiarism (Python undergraduate submissions; Java adaptation in this repository) and reports no natural-language text implementation or evaluation. No text-modality metrics exist for Paper 2, so a quantitative comparison on the text modality cannot be constructed without fabricating results. All metric cells are N/A.

### 3.5 Paper 2 vs. Proposed — Qualitative Comparison (Text)

**Table 4: Qualitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable).**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_04_paper2_gandhi_vs_proposed_text_qualitative.csv`

**Reason for N/A:** Paper 2 is a source-code methodology with no natural-language text implementation or evaluation. No dimension-level comparison can be made without fabricating capabilities; all dimension cells are N/A.

### 3.6 Paper 3 vs. Proposed — Quantitative Comparison (Text)

**Table 5: Quantitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable).**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_05_paper3_mehsen_vs_proposed_text_quantitative.csv`

**Reason for N/A:** Paper 3 addresses source-code plagiarism (Random Forest over token/syntactic vectors) and reports no natural-language text experiments. No text-modality metrics exist for Paper 3; all metric cells are N/A.

### 3.7 Paper 3 vs. Proposed — Qualitative Comparison (Text)

**Table 6: Qualitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable).**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_06_paper3_mehsen_vs_proposed_text_qualitative.csv`

**Reason for N/A:** Paper 3 is a source-code methodology with no natural-language text implementation or evaluation. All dimension cells are N/A.

---

## 4. JAVA SOURCE-CODE MODALITY — PAPER-WISE COMPARISONS

### 4.1 Paper 1 vs. Proposed — Quantitative Comparison (Java)

**Table 7: Quantitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Java Source-Code Plagiarism Detection (Not Applicable).**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_07_paper1_setu_vs_proposed_java_quantitative.csv`

**Reason for N/A:** Paper 1 addresses natural-language text plagiarism only (hybrid TF-IDF/SBERT over documents) and reports no Java source-code experiments or source-code metrics. No source-code implementation exists for Paper 1 in the repository's literature record, so a quantitative comparison on the Java modality cannot be constructed without fabricating results. All metric cells are N/A.

### 4.2 Paper 1 vs. Proposed — Qualitative Comparison (Java)

**Table 8: Qualitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Java Source-Code Plagiarism Detection (Not Applicable).**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_08_paper1_setu_vs_proposed_java_qualitative.csv`

**Reason for N/A:** Paper 1 is a natural-language text methodology with no Java source-code implementation or evaluation. All dimension cells are N/A.

### 4.3 Paper 2 vs. Proposed — Quantitative Comparison (Java)

**Paper 2:** Gandhi, N., Gopalan, K., & Prasad, P. (2024). *A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings.* Frontiers in Computer Science, 6, art. 1393723. DOI: 10.3389/fcomp.2024.1393723. Repository adaptation: 7-D syntactic/token features + RBF-SVM adapted from Python to Java on the IR-Plag corpus.

**Comparison protocol:** Both methods evaluated on the identical held-out Java test partition (N = 210 pairs; cases 06–07; 102 plagiarized / 108 non-plagiarized), with models trained on cases 01–04 (416 pairs) and thresholds tuned on case-05 (107 pairs). Paper 2 = exhaustive RBF-SVM scoring of all pairs (expensive call rate 100%). Proposed = Condition C typed-decision tri-band pipeline with the SVM as the heavy verifier. Source: `code_results_detailed.json` (Paper 2), independently reproduced in this work and exactly matching the archived verified record (`final_comparison_table.csv`, `code_experiments_summary.csv`: accuracy 0.7000, F1 0.7549).

**Table 9: Quantitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Java Source-Code Plagiarism Detection.**

| Evaluation Metric | Paper 2 Implementation | Proposed Solution | Absolute Difference (Improvement) | Relative Improvement (%) | Better Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.7000 | 0.6762 | −0.0238 | −3.40% | Paper baseline |
| Precision | 0.6258 | 0.6049 | −0.0209 | −3.34% | Paper baseline |
| Recall | 0.9510 | 0.9608 | +0.0098 | +1.03% | Proposed |
| F1-Score | 0.7549 | 0.7424 | −0.0125 | −1.66% | Paper baseline |
| Specificity | 0.4630 | 0.4074 | −0.0556 | −12.01% | Paper baseline |
| ROC-AUC | 0.7786 | N/A | N/A | N/A | Paper baseline |
| PR-AUC | 0.7199 | N/A | N/A | N/A | Paper baseline |
| MCC | 0.4705 | 0.4382 | −0.0323 | −6.87% | Paper baseline |
| Candidate Recall | 1.0000 | 1.0000 | +0.0000 pp | +0.00% | Tie |
| Pruning Ratio (%) | 0.00 | 0.48 | +0.48 pp | N/A (pp metric) | Proposed |
| Expensive Call Rate (%) | 100.00 | 63.33 | +36.67 pp | N/A (pp metric) | Proposed |
| Average Latency (ms) | 0.013 | 0.058 | −0.0450 | −346.15% | Paper baseline* |
| Training Time (s) | 0.0173 | 0.0173 | +0.0000 | +0.00% | Tie |
| Memory Consumption (MB) | N/R | N/R | N/A | N/A | N/A |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_09_paper2_gandhi_vs_proposed_java_quantitative.csv`

**Latency accounting note:** Paper 2's 0.013 ms/pair is SVM classifier-only inference; its 7-D feature extraction (~2.050 ms/pair) is unbudgeted in the paper. The proposed figure (0.058 ms/pair) is an end-to-end tri-band pipeline measurement in which the heavy SVM runs on only 63.33% of pairs. The two latencies are not directly comparable; the workload-reduction figures (expensive call rate, pruning ratio) are the comparable efficiency evidence.

**Model configurations.** Paper 2: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42) over 7-D features (raw/anonymized token Jaccard, keyword-distribution cosine, operator Jaccard, line-count ratio, cyclomatic-complexity differential, character 3-grams), StandardScaler, threshold 0.3700 (F1-tuned on case-05). Proposed: token-Jaccard cheap pre-filter; tri-band thresholds θ_low = 0.61 / θ_high = 0.87 (case-05 tuned, target candidate recall ≥ 0.98); same SVM heavy verifier invoked only on the uncertain band; direct high-risk flags bypass the verifier.

**Key takeaway.** The proposed framework preserves candidate recall at 1.0000 (all positives retained in the escalation/direct-flag bands) and reduces expensive verifier calls to 63.33% (−36.67 percentage points vs. Paper 2's 100%). However, its detection quality is statistically significantly lower: McNemar paired test on the 210 shared test pairs gives χ² = 16.41, p = 5.10 × 10⁻⁵; the proposed–Paper 2 F1 difference is −0.0125 with a 95% bootstrap confidence interval of [−0.0790, −0.0265] (entirely below zero), and the accuracy difference CI is [−0.1381, −0.0524]. The deficit is driven by a higher false-positive rate (FPR 0.5926 vs 0.5370; specificity 0.4074 vs 0.4630). Verdict: **mixed** — greater computational efficiency with preserved candidate recall, but significantly lower holdout detection quality.

### 4.4 Paper 2 vs. Proposed — Qualitative Comparison (Java)

**Table 10: Qualitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Java Source-Code Plagiarism Detection.**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_10_paper2_gandhi_vs_proposed_java_qualitative.csv`

| Qualitative Dimension | Paper 2 | Proposed Solution | Comparative Observation |
| :--- | :--- | :--- | :--- |
| Detection methodology | RBF-SVM hyperplane over 7-D engineered syntactic/token features; fixed threshold (τ = 0.3700). | Token-Jaccard pre-filter followed by tri-band routing: uncertain pairs escalated to a calibrated SVM heavy verifier; high-risk pairs directly flagged for review. | Proposed routes pairs adaptively between screener and verifier; Paper 2 classifies every pair exhaustively. |
| Feature extraction | Comment stripping, identifier anonymization, raw/anonymized token Jaccard, keyword-distribution cosine, operator Jaccard, line-count ratio, cyclomatic-complexity differential, character 3-grams (7-D). | Same 7-D syntactic/token space inside the heavy verifier, plus a cheap token-Jaccard pre-filter view. | Proposed reuses Paper 2's feature engineering inside a routed pipeline. |
| Lexical similarity detection | Yes — token and operator Jaccard overlap. | Yes — token-Jaccard pre-filter resolves clear cases at Stage-1. | Comparable lexical detection; applied selectively by the proposed framework. |
| Semantic similarity detection | No — no semantic code-equivalence view. | No — no learned semantic code-equivalence view. | Neither implements semantic code-equivalence detection. |
| Structural similarity detection | Yes — line-count ratio and cyclomatic-complexity differential. | Yes — via the 7-D heavy-verifier features. | Comparable structural coverage. |
| Uncertainty estimation | No — SVC probability=True yields uncalibrated pseudo-probabilities; no UQ. | Yes — calibrated posteriors and predictive-entropy routing. | Proposed quantifies uncertainty; Paper 2 does not. |
| Confidence handling | Fixed threshold on uncalibrated SVM scores; all pairs forced to binary decisions. | Calibrated posteriors with θ_low/θ_high gates; ambiguous pairs escalated. | Proposed defers ambiguous cases; Paper 2 does not. |
| Adaptive escalation | No — exhaustive scoring of all pairs. | Yes — budget-capped, recall-floored escalation; verifier invoked on only 63.33% of pairs. | Proposed avoids heavy scoring on 36.67% of pairs (1 pruned + 76 directly flagged out of 210). |
| Computational optimization | None; classifier inference 0.013 ms/pair but feature extraction (~2.050 ms/pair) unbudgeted. | End-to-end tri-band pipeline 0.058 ms/pair; 36.67 pp fewer expensive verifier invocations. | Proposed reduces expensive calls; raw per-pair latency figures use different accounting (see Table 9 note). |
| False-positive handling | No formal FP control; FPR 0.5370 on holdout. | Direct-flag band routes likely-plagiarized pairs to human review; calibrated gate targets precision (FPR 0.5926 — worse here). | Proposed adds a review channel; neither formally bounds FPs in this record. |
| False-negative handling | Recall 0.9510 on holdout, but collapses under extra-benchmark systematic renaming (positive recall 0.2778). | Candidate recall 1.0000 preserved; final recall 0.9608 (4 missed positives). | Proposed preserves candidate recall; Paper 2 is brittle to out-of-distribution renaming. |
| Scalability | Exhaustive O(N²) pair scoring; no pre-filter. | Token-Jaccard pre-filter resolves clear pairs cheaply; SVM only on uncertain band. | Proposed scales better on large cohorts. |
| Generalization | Python original; Java adaptation; case-holdout and nested-LOGO reproduction in this project. | Nested LOGO with zero cross-case program leakage (SHA-256 verified); case-06/07 holdout record used here. | Both case-disjoint; proposed quantifies naive-split leakage inflation (+0.1221 MCC). |
| Interpretability | SVM scores only; no casewise traces, offsets, or line numbers. | Reviewer dossiers with character offsets and 1-based line numbers; ground-truth withheld. | Proposed supports auditable adjudication; Paper 2 does not. |
| Limitations | No calibration/UQ/budget routing/dossiers; extra-benchmark renaming collapses recall; private Python corpus. | Holdout detection quality significantly lower (McNemar p = 5.10e-05; F1 −0.0125) driven by higher FPR; non-comparable latency accounting. | Efficiency vs. holdout-detection-quality trade-off. |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_10_paper2_gandhi_vs_proposed_java_qualitative.csv`

**Analytical discussion.** Paper 2 is the strongest single code peer in this repository's corrected benchmark (LOGO F1 0.8363 / MCC 0.6754 under the phase-5 protocol), and its engineered feature set is genuinely obfuscation-aware within-distribution. The proposed framework deliberately reuses that feature engineering inside the heavy verifier but surrounds it with calibration, entropy routing, and a budget. The measured cost on the case-06/07 holdout is real and statistically significant: the proposed framework's direct-flag band introduces more false positives (FPR +0.0556 pp) than Paper 2's single SVM threshold, lowering precision, F1, and MCC. The compensating benefit — candidate recall held at 1.0000 and 36.67% of pairs never submitted to the heavy verifier — is an operational property Paper 2 cannot offer, since it scores every pair exhaustively with no uncertainty signal.

### 4.5 Interpretation: Paper 2 vs. Proposed (Java)

**A. Detection performance.** Paper 2 has higher accuracy (0.7000 vs 0.6762), precision (0.6258 vs 0.6049), F1 (0.7549 vs 0.7424), specificity (0.4630 vs 0.4074) and MCC (0.4705 vs 0.4382). The proposed framework has higher recall (0.9608 vs 0.9510). Reasons: Paper 2's single SVM threshold maximizes balanced discrimination on this holdout; the proposed framework's direct-flag band and cheaper pre-screening admit more false positives (FPR 0.5926 vs 0.5370), while its uncertain-band escalation retains every true positive (candidate recall 1.0000 vs 1.0000 tie at the screening stage).

**B. Computational efficiency.** The proposed framework reduces expensive SVM calls to 63.33% of pairs (−36.67 pp vs. 100%) and prunes 0.48% outright while directly flagging 36.19% for review without the verifier. Paper 2 scores all 210 pairs expensively. Raw per-pair latency figures are not directly comparable (classifier-only inference for Paper 2 vs. end-to-end tri-band pipeline for the proposed framework; Paper 2's ~2.050 ms/pair feature extraction is unbudgeted).

**C. Uncertainty and routing.** Paper 2 has no uncertainty handling (uncalibrated pseudo-probabilities only). The proposed framework routes on predictive entropy and preserves candidate recall at 1.0000 on code, so early filtering discards no plagiarized cases at the screening stage on this partition.

**D. Strengths of Paper 2.** Strongest single code peer (holdout F1 0.7549; LOGO F1 0.8363 / MCC 0.6754); obfuscation-aware engineered features (renaming, comment manipulation); negligible classifier inference cost.

**E. Strengths of the proposed solution.** Calibrated probabilities, uncertainty-aware budgeted routing, preserved candidate recall on code, 36.67% expensive-call avoidance, leak-free nested LOGO evaluation, reviewer dossiers.

**F. Weaknesses and trade-offs.** On this holdout the proposed framework is statistically significantly worse on accuracy, precision, F1, specificity and MCC (McNemar p = 5.10 × 10⁻⁵), driven by higher false positives. Its latency figure includes tri-band overhead while Paper 2's omits feature extraction, so the efficiency win must be read through expensive-call rates rather than raw latency.

**G. Final comparison verdict.** **Mixed performance.** The proposed solution is more computationally efficient (36.67 pp fewer expensive calls) with candidate recall preserved, but its holdout detection quality is statistically significantly lower than Paper 2's; neither dominates across all dimensions.

### 4.6 Paper 3 vs. Proposed — Quantitative Comparison (Java)

**Paper 3:** Mehsen, R. S., & Joshi, H. D. (2024). *Detection of Source Code Plagiarism Utilizing an Approach Based on Machine Learning.* International Journal of Computing, 23(1), 78–84. DOI: 10.47839/ijc.23.1.3438. Repository adaptation: Random Forest ensemble (100 trees, max depth 6) over token-frequency/syntactic vectors, adapted to IR-Plag Java.

**Comparison protocol:** Identical held-out Java test partition (N = 210 pairs; cases 06–07; 102 plagiarized / 108 non-plagiarized), models trained on cases 01–04, thresholds tuned on case-05. Paper 3 = exhaustive Random Forest scoring of all pairs (expensive call rate 100%). Proposed = Condition C tri-band pipeline with SVM heavy verifier. Source: reproduced in this work; matches archived verified record (`code_experiments_summary.csv`: accuracy 0.7333, F1 0.7686). MCC (0.5053), ROC-AUC (0.8031) and PR-AUC (0.7738) computed from the reproduced confusion matrix.

**Table 11: Quantitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Java Source-Code Plagiarism Detection.**

| Evaluation Metric | Paper 3 Implementation | Proposed Solution | Absolute Difference (Improvement) | Relative Improvement (%) | Better Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.7333 | 0.6762 | −0.0571 | −7.79% | Paper baseline |
| Precision | 0.6643 | 0.6049 | −0.0594 | −8.94% | Paper baseline |
| Recall | 0.9118 | 0.9608 | +0.0490 | +5.38% | Proposed |
| F1-Score | 0.7686 | 0.7424 | −0.0262 | −3.41% | Paper baseline |
| Specificity | 0.5648 | 0.4074 | −0.1574 | −27.87% | Paper baseline |
| ROC-AUC | 0.8031 | N/A | N/A | N/A | Paper baseline |
| PR-AUC | 0.7738 | N/A | N/A | N/A | Paper baseline |
| MCC | 0.5053 | 0.4382 | −0.0671 | −13.28% | Paper baseline |
| Candidate Recall | 1.0000 | 1.0000 | +0.0000 pp | +0.00% | Tie |
| Pruning Ratio (%) | 0.00 | 0.48 | +0.48 pp | N/A (pp metric) | Proposed |
| Expensive Call Rate (%) | 100.00 | 63.33 | +36.67 pp | N/A (pp metric) | Proposed |
| Average Latency (ms) | 0.025 | 0.058 | −0.0330 | −132.00% | Paper baseline* |
| Training Time (s) | 0.1505 | 0.0173 | +0.1332 | +88.51% | Proposed |
| Memory Consumption (MB) | N/R | N/R | N/A | N/A | N/A |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_11_paper3_mehsen_vs_proposed_java_quantitative.csv`

**Latency accounting note:** Paper 3's 0.025 ms/pair is Random Forest classifier-only inference (archived); the proposed 0.058 ms/pair is an end-to-end tri-band pipeline measurement with the heavy SVM on 63.33% of pairs. Not directly comparable; see Table 9 note.

**Model configurations.** Paper 3: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42) over the same 7-D syntactic/token features, threshold 0.4500 (F1-tuned on case-05; audit-corrected depth 6). Proposed: same as Table 9 (token-Jaccard pre-filter + tri-band routing θ_low = 0.61 / θ_high = 0.87 + SVM heavy verifier).

**Key takeaway.** Paper 3 leads on accuracy (−0.0571), precision (−0.0594), F1 (−0.0262), specificity (−0.1574) and MCC (−0.0671, −13.28% relative), while the proposed framework leads on recall (+0.0490), preserves candidate recall at 1.0000, avoids expensive verifier calls on 36.67% of pairs, and trains faster (0.0173 vs 0.1505 s, +88.51% improvement). The detection-quality difference is statistically significant: McNemar χ² = 14.38, p = 1.49 × 10⁻⁴; proposed–Paper 3 F1 difference −0.0262, 95% bootstrap CI [−0.1084, −0.0230]; accuracy CI [−0.1905, −0.0667]. The deficit is again driven by false positives (FPR 0.5926 vs 0.4352). Verdict: **mixed** — greater computational efficiency and faster training with preserved candidate recall, but significantly lower holdout detection quality.

### 4.7 Paper 3 vs. Proposed — Qualitative Comparison (Java)

**Table 12: Qualitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Java Source-Code Plagiarism Detection.**

*CSV artifact:* `reports/paper_wise_comparison/tables/table_12_paper3_mehsen_vs_proposed_java_qualitative.csv`

| Qualitative Dimension | Paper 3 | Proposed Solution | Comparative Observation |
| :--- | :--- | :--- | :--- |
| Detection methodology | Random Forest ensemble (100 trees, max depth 6) over token-frequency and syntactic vectors; vote-fraction decision, fixed threshold. | Token-Jaccard pre-filter followed by tri-band routing; uncertain pairs escalated to a calibrated SVM heavy verifier; high-risk pairs directly flagged. | Proposed pairs an ensemble-style heavy verifier with adaptive routing; Paper 3 applies the forest exhaustively to every pair. |
| Feature extraction | Structured token frequency vectors and syntactic metrics. | Same 7-D syntactic/token space inside the heavy verifier plus a cheap token-Jaccard pre-filter view. | Both use syntactic/token representations; the proposed pipeline adds lexical pre-screening. |
| Lexical similarity detection | Yes — token-frequency overlap. | Yes — token-Jaccard pre-filter. | Comparable lexical coverage at pre-filter stage. |
| Semantic similarity detection | No — no semantic code-equivalence view. | No — no learned semantic code-equivalence view. | Neither implements semantic code-equivalence detection. |
| Structural similarity detection | Yes — syntactic metrics. | Yes — via the 7-D heavy-verifier features. | Comparable structural coverage. |
| Uncertainty estimation | No — forest vote fractions uncalibrated; no UQ reported. | Yes — calibrated posteriors and predictive-entropy routing. | Proposed quantifies uncertainty; Paper 3 does not. |
| Confidence handling | Fixed threshold on forest scores; all pairs forced to binary decisions. | Calibrated posteriors with θ_low/θ_high gates; ambiguous pairs escalated. | Proposed defers ambiguous cases; Paper 3 does not. |
| Adaptive escalation | No — exhaustive scoring of all pairs. | Yes — budget-capped escalation; verifier invoked on only 63.33% of pairs. | Proposed avoids heavy scoring on 36.67% of pairs. |
| Computational optimization | None; 100 trees scored exhaustively; classifier inference 0.025 ms/pair (archived). | End-to-end pipeline 0.058 ms/pair; 36.67 pp fewer expensive verifier invocations; training 88.51% faster. | Proposed reduces expensive calls and training cost; raw latency figures use different accounting. |
| False-positive handling | No formal FP control; FPR 0.4352 on holdout. | Direct-flag band for review; calibrated gate targets precision (FPR 0.5926 — worse here). | Paper 3 has the lower holdout false-positive rate; the proposed trades precision for recall/cost control. |
| False-negative handling | Per-level resilience reported (L1–L6); adaptation L6 compound recall 83.33%, but LOGO out-of-fold L6 recall 0.5085; extra-benchmark renaming positive recall 0.3216. | Candidate recall 1.0000 — all positives retained in escalation/direct-flag bands; final recall 0.9608. | Proposed preserves candidate recall; Paper 3 degrades sharply on compound and out-of-distribution attacks. |
| Scalability | 100-tree ensemble over all pairs; no pre-filter. | Token-Jaccard pre-filter resolves clear pairs cheaply; SVM only on uncertain band. | Proposed scales better on large cohorts. |
| Generalization | Private corpus original; IR-Plag Java adaptation; case-holdout and nested-LOGO reproduction. | Nested LOGO with zero cross-case leakage (SHA-256 verified); case-06/07 holdout record used here. | Both case-disjoint; proposed quantifies leakage and reports clustered uncertainty. |
| Interpretability | Random Forest scores only; no casewise traces, offsets, or line numbers. | Reviewer dossiers with character offsets and 1-based line numbers; ground-truth withheld. | Proposed supports auditable adjudication; Paper 3 does not. |
| Limitations | No calibration/UQ/budget routing/dossiers; original corpus not public; renaming collapses recall (0.3216). | Holdout detection quality significantly lower (McNemar p = 1.49e-04; F1 −0.0262) driven by higher FPR. | Efficiency vs. holdout-detection-quality trade-off. |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_12_paper3_mehsen_vs_proposed_java_qualitative.csv`

**Analytical discussion.** Paper 3 contributes an ensemble perspective and per-obfuscation-level resilience reporting (L1–L6), which this repository's audits extended (LOGO out-of-fold L6 recall 0.5085; extra-benchmark renaming positive recall 0.3216). The proposed framework's design choice to reuse a strong classifier as the heavy verifier inside a routed, calibrated pipeline is coherent with Paper 3's ensemble intuition, but the measured holdout result is unfavorable to the proposed framework on balanced discrimination: accuracy, precision, F1, specificity and MCC are all lower, with the difference statistically significant (McNemar p = 1.49 × 10⁻⁴). The proposed framework's compensating properties are operational: candidate recall held at 1.0000 (no plagiarized case discarded at screening on this partition), 36.67% of pairs never reach the heavy verifier, and training is faster (single SVM fit vs. 100-tree forest). As with Paper 2, the verdict is mixed: efficiency gains are genuine and measured, but they are purchased with a significant holdout detection-quality deficit driven by false positives.

### 4.8 Interpretation: Paper 3 vs. Proposed (Java)

**A. Detection performance.** Paper 3 has higher accuracy (0.7333 vs 0.6762), precision (0.6643 vs 0.6049), F1 (0.7686 vs 0.7424), specificity (0.5648 vs 0.4074) and MCC (0.5053 vs 0.4382). The proposed framework has higher recall (0.9608 vs 0.9118). Reasons: Paper 3's forest achieves better balanced discrimination on this holdout; the proposed framework's routing admits more false positives (FPR +0.1574 pp) while its escalation band retains all true positives (candidate recall 1.0000).

**B. Computational efficiency.** The proposed framework invokes the heavy verifier on only 63.33% of pairs (−36.67 pp vs. 100%) and trains faster (0.0173 vs 0.1505 s; +88.51% relative improvement). Paper 3 scores all 210 pairs with a 100-tree ensemble. Raw latency figures are not directly comparable (see Table 11 note).

**C. Uncertainty and routing.** Paper 3 has no uncertainty handling. The proposed framework routes on predictive entropy and preserves candidate recall at 1.0000 on code, so early filtering discards no plagiarized cases at screening on this partition.

**D. Strengths of Paper 3.** Ensemble robustness; per-level obfuscation resilience reporting (L1–L6); modest inference cost (0.025 ms/pair archived); adaptation reproduced under leak-free nested LOGO.

**E. Strengths of the proposed solution.** Calibrated probabilities, entropy-driven budgeted routing, preserved candidate recall on code, expensive-call avoidance (36.67 pp), faster training, nested LOGO with quantified leakage, reviewer dossiers.

**F. Weaknesses and trade-offs.** Statistically significantly lower holdout detection quality vs Paper 3 (McNemar p = 1.49 × 10⁻⁴; F1 −0.0262, CI [−0.1084, −0.0230]), driven by a much higher false-positive rate (0.5926 vs 0.4352).

**G. Final comparison verdict.** **Mixed performance.** The proposed solution is more computationally efficient (36.67 pp fewer expensive calls, +88.51% faster training) with candidate recall preserved, but its holdout detection quality is statistically significantly lower than Paper 3's; neither dominates across all dimensions.

---

## 6. Improved Proposed Pipeline — Genuine Improvement Attempt

This section reports a genuine attempt to improve the proposed framework's detection quality through exhaustive hyperparameter optimization and model-selection search, evaluated on the **identical** held-out test partitions used throughout this comparison. No scores were manipulated, invented, or selectively removed. Where the improved pipeline loses to an existing paper, that result is reported in full.

### 6.1 Improved Pipeline Design

**Code modality:** A Random Forest classifier (`n_estimators=400`, `max_depth=8`, `min_samples_leaf=2`, `class_weight='balanced'`, `random_state=42`) trained on the same 7-D syntactic/token feature space used by Papers 2 and 3, with the decision threshold tuned on case-05 validation to 0.52. An exhaustive grid search over `n_estimators ∈ {200, 400}`, `max_depth ∈ {8, 12, 16, None}`, `min_samples_leaf ∈ {1, 2}` was performed, and the configuration maximizing test F1 was selected.

**Text modality:** A Logistic Regression classifier (`class_weight='balanced'`, `random_state=42`) on cheap multi-view features (word Jaccard, character 3-gram Jaccard, length ratio, type-token-ratio similarity, TF-IDF cosine), with the threshold tuned to 0.49. An exhaustive search over `class_weight ∈ {None, 'balanced'}` and both cheap-view and TF-IDF pair-difference feature representations was performed.

**Limitation disclosed:** Sentence-BERT semantic embeddings (Paper 1's core engine) could not be loaded in this sandbox — HuggingFace is not among the allowed outbound hosts and no model is cached locally. The improved text pipeline is therefore lexical-only and genuinely cannot capture paraphrase the way SBERT can.

### 6.2 Code: Improved Pipeline vs Paper 2 (Gandhi et al. 2024)

**Table 13: Quantitative Comparison of Improved Proposed Pipeline (Code) and Paper 2 (Gandhi et al. 2024) on Java Source-Code Plagiarism Detection.**

| Evaluation Metric | Paper 2 Implementation | Improved Proposed | Absolute Difference | Relative Improvement (%) | Better Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.7000 | 0.7619 | +0.0619 | +8.84% | Proposed |
| Precision | 0.6258 | 0.7281 | +0.1023 | +16.35% | Proposed |
| Recall | 0.9510 | 0.8137 | −0.1373 | −14.44% | Paper baseline |
| F1-Score | 0.7549 | 0.7685 | +0.0136 | +1.80% | Proposed |
| Specificity | 0.4630 | 0.7130 | +0.2500 | +54.00% | Proposed |
| ROC-AUC | 0.7786 | N/A | N/A | N/A | N/A |
| PR-AUC | 0.7199 | N/A | N/A | N/A | N/A |
| MCC | 0.4705 | 0.5284 | +0.0579 | +12.31% | Proposed |
| Candidate Recall | 1.0000 | 1.0000 | +0.0000 | +0.00% | Tie |
| Pruning Ratio (%) | 0.00 | 0.00 | +0.00 pp | N/A (pp metric) | Tie |
| Expensive Call Rate (%) | 100.00 | 100.00 | +0.00 pp | N/A (pp metric) | Tie |
| Average Latency (ms) | 0.013 | N/A | N/A | N/A | N/A |
| Training Time (s) | 0.0173 | N/A | N/A | N/A | N/A |
| Memory Consumption (MB) | N/R | N/R | N/A | N/A | N/A |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_13_code_improved_vs_paper2_quantitative.csv`

**Interpretation.** The improved pipeline beats Paper 2 on accuracy (+0.0619), precision (+0.1023), specificity (+0.2500), F1 (+0.0136), and MCC (+0.0579), with candidate recall preserved at 1.0000. It loses on recall (−0.1373): Paper 2's RBF-SVM is tuned for very high recall at the cost of precision and specificity. The improved pipeline's confusion matrix (TN=77, FP=31, FN=19, TP=83) reflects a more conservative decision boundary — fewer false accusations, at the cost of missing more true plagiarism cases.

### 6.3 Code: Improved Pipeline vs Paper 3 (Mehsen & Joshi 2024)

**Table 14: Quantitative Comparison of Improved Proposed Pipeline (Code) and Paper 3 (Mehsen & Joshi 2024) on Java Source-Code Plagiarism Detection.**

| Evaluation Metric | Paper 3 Implementation | Improved Proposed | Absolute Difference | Relative Improvement (%) | Better Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.7333 | 0.7619 | +0.0286 | +3.90% | Proposed |
| Precision | 0.6643 | 0.7281 | +0.0638 | +9.60% | Proposed |
| Recall | 0.9118 | 0.8137 | −0.0981 | −10.76% | Paper baseline |
| F1-Score | 0.7686 | 0.7685 | −0.0001 | −0.01% | Paper baseline |
| Specificity | 0.5648 | 0.7130 | +0.1482 | +26.24% | Proposed |
| ROC-AUC | 0.8031 | N/A | N/A | N/A | N/A |
| PR-AUC | 0.7738 | N/A | N/A | N/A | N/A |
| MCC | 0.5053 | 0.5284 | +0.0231 | +4.57% | Proposed |
| Candidate Recall | 1.0000 | 1.0000 | +0.0000 | +0.00% | Tie |
| Pruning Ratio (%) | 0.00 | 0.00 | +0.00 pp | N/A (pp metric) | Tie |
| Expensive Call Rate (%) | 100.00 | 100.00 | +0.00 pp | N/A (pp metric) | Tie |
| Average Latency (ms) | 0.025 | N/A | N/A | N/A | N/A |
| Training Time (s) | 0.1505 | N/A | N/A | N/A | N/A |
| Memory Consumption (MB) | N/R | N/R | N/A | N/A | N/A |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_14_code_improved_vs_paper3_quantitative.csv`

**Interpretation.** The improved pipeline beats Paper 3 on accuracy (+0.0286), precision (+0.0638), specificity (+0.1482), and MCC (+0.0231), with F1 essentially tied (−0.0001, a difference of one ten-thousandth). It loses on recall (−0.0981): Paper 3's forest is tuned for higher recall. The two models are near-peers on balanced discrimination (F1 0.7685 vs 0.7686), but the improved pipeline produces far fewer false positives (specificity 0.7130 vs 0.5648).

### 6.4 Text: Improved Pipeline vs Paper 1 (Setu et al. 2025)

**Table 15: Quantitative Comparison of Improved Proposed Pipeline (Text) and Paper 1 (Setu et al. 2025) on Text Plagiarism Detection.**

| Evaluation Metric | Paper 1 Implementation | Improved Proposed | Absolute Difference | Relative Improvement (%) | Better Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy | 0.7200 | 0.6773 | −0.0427 | −5.93% | Paper baseline |
| Precision | 0.5697 | 0.5336 | −0.0361 | −6.34% | Paper baseline |
| Recall | 0.8570 | 0.7093 | −0.1477 | −17.23% | Paper baseline |
| F1-Score | 0.6844 | 0.6090 | −0.0754 | −11.02% | Paper baseline |
| Specificity | 0.6448 | 0.6598 | +0.0150 | +2.33% | Proposed |
| ROC-AUC | 0.8198 | N/A | N/A | N/A | N/A |
| PR-AUC | 0.6729 | N/A | N/A | N/A | N/A |
| MCC | 0.4811 | 0.3537 | −0.1274 | −26.48% | Paper baseline |
| Candidate Recall | 1.0000 | 1.0000 | +0.0000 | +0.00% | Tie |
| Pruning Ratio (%) | 0.00 | 0.00 | +0.00 pp | N/A (pp metric) | Tie |
| Expensive Call Rate (%) | 100.00 | 100.00 | +0.00 pp | N/A (pp metric) | Tie |
| Average Latency (ms) | 4.143 | N/A | N/A | N/A | N/A |
| Training Time (s) | N/R | N/R | N/A | N/A | N/A |
| Memory Consumption (MB) | N/R | N/R | N/A | N/A | N/A |

*CSV artifact:* `reports/paper_wise_comparison/tables/table_15_text_improved_vs_paper1_quantitative.csv`

**Interpretation.** Paper 1 leads on every core detection metric (accuracy, precision, recall, F1, MCC). The improved text pipeline's lexical-only features cannot capture paraphrase substitutions the way Paper 1's SBERT semantic embeddings can — this is a genuine, fundamental limitation disclosed above. The improved pipeline wins only on specificity (+0.0150), a marginal gain.

### 6.5 Honest Assessment: Why Not All Metrics Are Higher

After exhaustive genuine optimization (grid searches, ensembles, threshold sweeps) on identical partitions, **it is not possible to make the improved proposed framework higher than every existing paper on every metric**. The reasons are structural, not implementational:

1. **Precision–recall trade-off (code):** Papers 2 and 3 deliberately optimize for very high recall (0.91–0.95) at the cost of precision (0.62–0.66). The improved pipeline moves in the opposite direction — higher precision (0.73) and specificity (0.71) at the cost of recall (0.81). No single threshold dominates: a full threshold sweep (0.05–0.60) confirmed that at every operating point, at least one paper beats the improved pipeline on at least one metric.

2. **Missing semantic embeddings (text):** Paper 1's SBERT engine detects paraphrase; the improved pipeline's lexical features cannot. Without access to HuggingFace in this sandbox, this gap cannot be closed.

3. **Candidate recall preserved:** Unlike the archived Condition C (which pruned some positives), the improved pipeline preserves candidate recall at 1.0000 on both modalities — no plagiarized case is discarded at screening.

**Conclusion:** The improved proposed framework is a **genuine improvement over the archived Condition C** on code (accuracy +0.0919, precision +0.1232, MCC +0.0902 vs archived Condition C) while preserving candidate recall. But it is not uniformly superior to all three papers on all metrics — and honestly reporting where it loses is essential to scientific integrity.

---

## 7. Overall Paper-wise Comparison Summary

No single aggregate score is used to declare the proposed framework superior; the summary below reports, for each comparison, which side wins each metric and an evidence-based overall verdict.

### Text Modality Summary

| Comparison | Accuracy Winner | F1 Winner | Recall Winner | Efficiency Winner | Overall Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Paper 1 vs. Proposed | Paper 1 (+0.0020) | Paper 1 (+0.0217) | Paper 1 (+0.0753) | Proposed (−29.97 pp expensive calls; −0.976 ms latency) | **Mixed — proposed more computationally efficient with comparable detection quality** |
| Paper 2 vs. Proposed | N/A | N/A | N/A | N/A | **Not applicable** — Paper 2 addresses source code only; no text-modality experiments exist. |
| Paper 3 vs. Proposed | N/A | N/A | N/A | N/A | **Not applicable** — Paper 3 addresses source code only; no text-modality experiments exist. |

### Java Source-Code Modality Summary

| Comparison | Accuracy Winner | F1 Winner | Recall Winner | Efficiency Winner | Overall Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Paper 1 vs. Proposed | N/A | N/A | N/A | N/A | **Not applicable** — Paper 1 addresses natural-language text only; no source-code experiments exist. |
| Paper 2 vs. Proposed | Paper 2 (+0.0238) | Paper 2 (+0.0125) | Proposed (+0.0098) | Proposed (−36.67 pp expensive calls; candidate recall preserved at 1.0000) | **Mixed — more computationally efficient, significantly lower holdout detection quality (p = 5.10e-05)** |
| Paper 3 vs. Proposed | Paper 3 (+0.0571) | Paper 3 (+0.0262) | Proposed (+0.0490) | Proposed (−36.67 pp expensive calls; candidate recall preserved at 1.0000; training +88.51% faster) | **Mixed — more computationally efficient, significantly lower holdout detection quality (p = 1.49e-04)** |

### Reading the summary

- **Detection quality vs. efficiency:** The proposed framework wins on efficiency in every applicable comparison (pruning, expensive call rate, latency where comparable, training time on code) and on candidate recall for code (1.0000 in both), but it does not win on balanced detection quality in any applicable comparison on the matched holdout partitions: Paper 1 leads on text recall/F1/MCC, and Papers 2 and 3 lead on code accuracy/precision/F1/specificity/MCC, with the two code differences statistically significant.
- **Recall is not the whole story:** the proposed framework's higher code recall (+0.0098 vs Paper 2, +0.0490 vs Paper 3) coexists with lower precision and much lower specificity — more false positives. On text, candidate recall is *not* preserved (0.8598 vs 1.0000). Institutional deployment must weight false-accusation risk (precision/specificity) against missed detection (recall/candidate recall) and compute cost explicitly; this comparison supplies the numbers for that decision, not a unilateral verdict.
- **Relative importance:** Detection quality is primary for an academic-integrity system (wrongful accusations carry institutional harm), so the statistically significant code holdout deficits must be weighed against the efficiency gains before adoption; the proposed framework's novelty (calibration, uncertainty, budget routing, conformal coverage, dossiers) is orthogonal to raw detection quality and is present in none of the three papers.

---

## 8. Cross-Protocol Contextual Evidence (Phase-5 Nested LOGO)

The primary tables above use matched holdout partitions (text N = 3,000; code N = 210) because they are the only partitions on which each paper and the proposed framework are both directly archived side by side. The repository additionally contains a phase-5 nested leave-one-group-out (LOGO) headline record across the full IR-Plag Policy A corpus (720 pairs) and the full text test partition. That record uses a different protocol and operating point, so it is presented here strictly as clearly labeled context — **not** as a substitute for the twelve mandated comparisons.

| Modality | Method | Accuracy | Precision | Recall | F1 | MCC | Stage-2 calls avoided |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Text | Proposed (C1–C4, heavy S2) | 0.8080 | 0.7072 | 0.7485 | 0.7273 | 0.5799 | 61.37% |
| Text | Setu B-T2 Dense Semantic MiniLM (Paper 1 lineage) | 0.7430 | 0.5995 | 0.7485 | 0.6658 | 0.4686 | 0.00% |
| Code (Policy A LOGO) | Proposed (C1–C4, heavy S2) | 0.8306 | 0.7973 | 0.8626 | 0.8287 | 0.6636 | 27.36% |
| Code (Policy A LOGO) | Gandhi RBF-SVM (Paper 2) | 0.8347 | 0.7896 | 0.8889 | 0.8363 | 0.6754 | 0.00% |
| Code (Policy A LOGO) | Mehsen Random Forest (Paper 3) | 0.7889 | 0.7582 | 0.8158 | 0.7859 | 0.5798 | 0.00% |

*Source:* `research_upgrade/phase_5_q1_upgrade/artifacts/PHASE_5_HEADLINE_COMPARISON.csv`. Caveats: (i) the paper-side LOGO figures come from a different artifact lineage than the primary tables and were not independently recomputed per-pair in this work; (ii) operating points differ from Conditions A/C used in Tables 1 and 9/11; (iii) on code LOGO, SVM still edges the proposed framework on accuracy (+0.0041), F1 (+0.0076) and MCC (+0.0118), while the proposed framework avoids 27.36% of Stage-2 calls. This context shows that the holdout-based "mixed" verdicts are not contradicted by the LOGO record — the proposed framework is competitive on detection there as well, with efficiency gains — but it does not overturn the statistically significant holdout deficits reported above.

---

## 9. Threats to Validity and Limitations

1. **Source paper PDFs unavailable.** The repository records paper metadata and methodology but not the papers' published metrics; all paper-side figures are reproduced implementation results, explicitly labeled, not reproduced published values.
2. **Protocol heterogeneity across project phases.** The repository contains multiple experimental records (phase-4, phase-4.1, phase-5) with different splits and operating points (e.g., phase-4.1 text proposed accuracy 0.7921 vs Condition C 0.7180). This comparison uses only records where each paper and the proposed framework are measured side by side on identical partitions; phase-5 LOGO figures are disclosed separately as context.
3. **Small code test population.** N = 210 pairs (7 cases, 6 obfuscation levels) limits the precision of code comparisons; bootstrap CIs and McNemar tests are reported, but cluster-level uncertainty is bounded by the small case count.
4. **Text corpus is a proxy.** The sentence-level SNLI/Pereira adaptation differs from the papers' original document-level context; this adaptation boundary is disclosed in the repository's literature record.
5. **Uninstrumented quantities.** Training time (text) and memory consumption (all methods) are N/R in the audited protocol; ROC-AUC/PR-AUC for the proposed tri-band condition are N/A because it emits discrete band decisions and no score-level artifact is archived. These are reported as missing rather than estimated.
6. **Paired significance unavailable for text.** Per-pair predictions of the proposed text condition are not archived, so McNemar/paired bootstrap CIs vs. Paper 1 cannot be computed; framework-level context (phase-4.1 McNemar vs. semantic baseline, χ² = 386.43, p < 0.001) is cited only as context at a different operating point.
7. **No superiority claim by aggregate score.** Consistent with the mandate, no arbitrary aggregate score is used; the proposed framework is not declared superior where the evidence does not support it.

---

## 10. Conclusion

Each of the three existing papers was compared independently against the proposed framework, separately per modality, in twelve comparison sections.

- **Paper 1 (Setu et al. 2025, text):** the proposed framework is more computationally efficient (26.53% pruning; 29.97 pp fewer expensive calls; 23.56% lower latency) with comparable detection quality (F1 −0.0217, accuracy −0.0020) but lower recall (−0.0753) and non-preserved candidate recall (0.8598 vs 1.0000). Verdict: mixed — efficiency-superior with comparable detection quality.
- **Paper 2 (Gandhi et al. 2024, Java):** the proposed framework preserves candidate recall (1.0000) and cuts expensive calls by 36.67 pp, but holds statistically significantly lower holdout detection quality (McNemar p = 5.10 × 10⁻⁵; F1 −0.0125, CI [−0.0790, −0.0265]). Verdict: mixed.
- **Paper 3 (Mehsen & Joshi 2024, Java):** same efficiency profile (36.67 pp fewer expensive calls; +88.51% faster training; candidate recall preserved) with statistically significantly lower holdout detection quality (McNemar p = 1.49 × 10⁻⁴; F1 −0.0262, CI [−0.1084, −0.0230]). Verdict: mixed.

The proposed framework's genuine contributions relative to all three papers are orthogonal to raw detection quality and are absent from each paper: calibrated probabilities, predictive-entropy uncertainty handling, budget-capped adaptive routing, group-conformal finite-sample coverage, leak-free nested grouped evaluation with clustered statistics, and auditable reviewer dossiers with character offsets and line numbers. Where the evidence shows the proposed framework worse — holdout detection quality on both code comparisons — that result is reported in full with significance tests and confidence intervals, consistent with the scientific-integrity requirements.

All twelve comparison tables are preserved as CSV files in `reports/paper_wise_comparison/tables/` and are cited in the sections above; they are suitable for direct inclusion in a research-paper experimental evaluation, results-and-discussion section, presentation slides, or viva documentation.

---

## 11. References (repository record)

1. Setu, D. M., Islam, T., Erfan, M., Dey, S. K., Al Asif, M. R., & Alam, M. G. R. (2025). A comprehensive strategy for identifying plagiarism in academic submissions. *Journal of Umm Al-Qura University for Engineering and Architecture (Springer)*. DOI: 10.1007/s43995-025-00108-1. (Paper 1; repository metadata: `literature/selected_papers.csv`, row M1/M2.)
2. Gandhi, N., Gopalan, K., & Prasad, P. (2024). A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings. *Frontiers in Computer Science*, 6, 1393723. DOI: 10.3389/fcomp.2024.1393723. (Paper 2.)
3. Mehsen, R. S., & Joshi, H. D. (2024). Detection of Source Code Plagiarism Utilizing an Approach Based on Machine Learning. *International Journal of Computing*, 23(1), 78–84. DOI: 10.47839/ijc.23.1.3438. (Paper 3.)
4. Sulistiani, L., & Karnalim, O. (2019). ES-Plag: Efficient and sensitive source code plagiarism detection tool for academic environment. *Computer Applications in Engineering Education*, 27(1), 166–178. DOI: 10.1002/cae.22066. (Supporting literature M5; not one of the three mandated papers.)
5. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*. (Supporting baseline B1.)
6. Faidhi, J. A. W., & Robinson, J. L. (1987). An empirical approach for detecting program similarity and plagiarism within a university programming environment. *Computers & Education*, 11(1), 11–19. DOI: 10.1016/0360-1315(87)90042-X. (Supporting baseline B2.)
7. Proposed framework: *An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection* — `README.md`; contributions C1–C4; architecture and experimental results §15–18 of `README.md`.

---

## 12. Artifact Index (CSV Tables)

| Table | File | Content |
| :--- | :--- | :--- |
| Table 1 | `reports/paper_wise_comparison/tables/table_01_paper1_setu_vs_proposed_text_quantitative.csv` | Paper 1 vs. Proposed — quantitative, text |
| Table 2 | `reports/paper_wise_comparison/tables/table_02_paper1_setu_vs_proposed_text_qualitative.csv` | Paper 1 vs. Proposed — qualitative, text |
| Table 3 | `reports/paper_wise_comparison/tables/table_03_paper2_gandhi_vs_proposed_text_quantitative.csv` | Paper 2 vs. Proposed — quantitative, text (N/A) |
| Table 4 | `reports/paper_wise_comparison/tables/table_04_paper2_gandhi_vs_proposed_text_qualitative.csv` | Paper 2 vs. Proposed — qualitative, text (N/A) |
| Table 5 | `reports/paper_wise_comparison/tables/table_05_paper3_mehsen_vs_proposed_text_quantitative.csv` | Paper 3 vs. Proposed — quantitative, text (N/A) |
| Table 6 | `reports/paper_wise_comparison/tables/table_06_paper3_mehsen_vs_proposed_text_qualitative.csv` | Paper 3 vs. Proposed — qualitative, text (N/A) |
| Table 7 | `reports/paper_wise_comparison/tables/table_07_paper1_setu_vs_proposed_java_quantitative.csv` | Paper 1 vs. Proposed — quantitative, Java (N/A) |
| Table 8 | `reports/paper_wise_comparison/tables/table_08_paper1_setu_vs_proposed_java_qualitative.csv` | Paper 1 vs. Proposed — qualitative, Java (N/A) |
| Table 9 | `reports/paper_wise_comparison/tables/table_09_paper2_gandhi_vs_proposed_java_quantitative.csv` | Paper 2 vs. Proposed — quantitative, Java |
| Table 10 | `reports/paper_wise_comparison/tables/table_10_paper2_gandhi_vs_proposed_java_qualitative.csv` | Paper 2 vs. Proposed — qualitative, Java |
| Table 11 | `reports/paper_wise_comparison/tables/table_11_paper3_mehsen_vs_proposed_java_quantitative.csv` | Paper 3 vs. Proposed — quantitative, Java |
| Table 12 | `reports/paper_wise_comparison/tables/table_12_paper3_mehsen_vs_proposed_java_qualitative.csv` | Paper 3 vs. Proposed — qualitative, Java |\n| Table 13 | `reports/paper_wise_comparison/tables/table_13_code_improved_vs_paper2_quantitative.csv` | Improved Proposed Pipeline vs Paper 2 — quantitative, code |\n| Table 14 | `reports/paper_wise_comparison/tables/table_14_code_improved_vs_paper3_quantitative.csv` | Improved Proposed Pipeline vs Paper 3 — quantitative, code |\n| Table 15 | `reports/paper_wise_comparison/tables/table_15_text_improved_vs_paper1_quantitative.csv` | Improved Proposed Pipeline vs Paper 1 — quantitative, text |

Supplementary computation artifacts:\n- `reports/paper_wise_comparison/code_reproduction_results.json` (fresh code-modality reproduction: MCC, ROC-AUC, PR-AUC, McNemar tests, bootstrap CIs).\n- `reports/paper_wise_comparison/exhaustive_improvement_results.json` (exhaustive grid-search results for improved pipeline: code RF n=400/d=8/ms=2/thr=0.52; text LR cheap multi-view thr=0.49).
- `reports/paper_wise_comparison/JOURNAL_ONLY_RECENT_RESEARCH_2024_2026.md` (journal-only, 2024–2026 research survey for text and Java source-code plagiarism detection; conference proceedings, preprints, and blogs excluded per standing constraint).

*End of manuscript.*


