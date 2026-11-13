"""Generate qualitative comparison CSVs (Tables 2, 4, 6, 8, 10, 12) and N/A quantitative CSVs."""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "Detection methodology",
    "Feature extraction",
    "Lexical similarity detection",
    "Semantic similarity detection",
    "Structural similarity detection",
    "Uncertainty estimation",
    "Confidence handling",
    "Adaptive escalation",
    "Computational optimization",
    "False-positive handling",
    "False-negative handling",
    "Scalability",
    "Generalization",
    "Interpretability",
    "Limitations",
]


def write_qual_csv(path, title, caption, modality_note, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TABLE", title])
        w.writerow(["CAPTION", caption])
        w.writerow(["MODALITY_NOTE", modality_note])
        w.writerow([])
        w.writerow(["Qualitative Dimension", "Paper Implementation", "Proposed Solution", "Comparative Observation"])
        for r in rows:
            w.writerow(r)
    print("wrote", path)


def write_na_quant_csv(path, title, caption, reason):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TABLE", title])
        w.writerow(["CAPTION", caption])
        w.writerow(["STATUS", "NOT APPLICABLE"])
        w.writerow(["REASON", reason])
        w.writerow([])
        w.writerow(["Evaluation Metric", "Paper Implementation", "Proposed Solution",
                    "Absolute Difference (Improvement)", "Relative Improvement (%)",
                    "Better Method"])
        for m in [
            "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
            "ROC-AUC", "PR-AUC", "MCC", "Candidate Recall",
            "Pruning Ratio (%)", "Expensive Call Rate (%)",
            "Average Latency (ms)", "Training Time (s)", "Memory Consumption (MB)",
        ]:
            w.writerow([m, "N/A", "N/A", "N/A", "N/A", "N/A"])
    print("wrote", path)


# ----------------------------------------------------------------------------
# TABLE 2 — TEXT: Paper 1 (Setu) vs Proposed — Qualitative
# ----------------------------------------------------------------------------
t2_rows = [
    ["Detection methodology",
     "Hybrid lexical–semantic scoring: max(sublinear TF-IDF cosine, Sentence-BERT all-MiniLM-L6-v2 cosine); a single fixed threshold (tau=0.5000) categorizes pairs as direct copies vs. paraphrases.",
     "Calibrated multi-view scorer with predictive-entropy tri-band routing: low-risk pairs pruned at Stage-1, uncertain pairs escalated to a heavy verifier, high-risk pairs directly flagged for review.",
     "Proposed replaces static max() fusion and single-threshold decisions with a calibrated cascade that routes pairs adaptively; Paper 1 decides every pair exhaustively with one threshold."],
    ["Feature extraction",
     "TF-IDF word n-gram vectors and 384-d Sentence-BERT dense embeddings (two views combined by max()).",
     "Twelve text views: TF-IDF lexical overlap, character 3-grams/LCS, MiniLM semantic embeddings, and structural stylometry (length, sentence count, type-token ratio), fused through a Platt-calibrated logistic scorer.",
     "Proposed fuses orthogonal views; Paper 1 combines two views with a brittle max() operator."],
    ["Lexical similarity detection",
     "Yes — TF-IDF cosine captures verbatim and near-verbatim word reuse.",
     "Yes — TF-IDF/Jaccard lexical view in the cheap Stage-1 screener.",
     "Comparable lexical coverage; proposed screens lexically before any semantic escalation."],
    ["Semantic similarity detection",
     "Yes — SBERT dense embeddings capture paraphrastic reuse where lexical methods fail.",
     "Yes — MiniLM semantic view applied as the heavy verifier on uncertain pairs only.",
     "Comparable semantic capability; proposed applies it selectively (~70% of pairs) rather than exhaustively."],
    ["Structural similarity detection",
     "No — not reported in the repository adaptation of the paper.",
     "Yes — structural view (length discrepancy, sentence count, type-token ratio).",
     "Proposed adds a structural dimension Paper 1 does not use."],
    ["Uncertainty estimation",
     "No — raw similarity scores only; no predictive probability model.",
     "Yes — Platt-calibrated posterior P(Y=1|x) and Shannon entropy H(p) computed per pair.",
     "Proposed quantifies decision uncertainty; Paper 1 does not."],
    ["Confidence handling",
     "Single threshold on the hybrid score; no calibrated confidence; every pair receives a binary decision.",
     "Calibrated posteriors with theta_low/theta_high gates; low-confidence pairs are escalated instead of forced to a binary decision.",
     "Proposed defers ambiguous cases; Paper 1 forces all pairs to an immediate binary verdict."],
    ["Adaptive escalation",
     "Condition E1: none (all pairs scored). Condition E2: fixed dual-threshold routing with no budget formalization.",
     "Yes — budget-capped, recall-floored, entropy-driven Stage-2 escalation with an explicit computational budget.",
     "Proposed formalizes adaptive routing; Paper 1's E2 is only an informal fixed-band precursor."],
    ["Computational optimization",
     "E2 reports ~30% workload reduction vs exhaustive SBERT (2.875 vs 4.163 ms/pair).",
     "26.53% of pairs pruned at Stage-1; expensive calls reduced to 70.03% of pairs (29.97 percentage-point saving); latency 3.167 vs 4.143 ms/pair.",
     "Both reduce cost; proposed does so within a calibrated, budget-aware policy while retaining 85.98% candidate recall."],
    ["False-positive handling",
     "No formal false-positive control; max() fusion can amplify correlated view errors; FPR 0.3552 on the test partition.",
     "High-risk gate theta_high targets precision; calibrated posteriors lower the false-positive rate (FPR 0.3170; specificity 0.6830 vs 0.6448).",
     "Proposed lowers false positives at the cost of some recall."],
    ["False-negative handling",
     "Recall 0.8570: the semantic view catches paraphrase, but low-scoring plagiarized pairs are missed outright once scored below the threshold.",
     "Uncertain-band escalation retains candidates for heavy verification; candidate recall 0.8598 (about 14% of positives pruned at screening).",
     "Proposed trades a bounded amount of candidate recall for cost efficiency; Paper 1 retains all candidates but scores every pair expensively."],
    ["Scalability",
     "SBERT forward pass on every pair (~4.16 ms/pair in E1) is expensive at cohort scale.",
     "Stage-1 resolves 26.53% of pairs at lexical cost; only the uncertain band consumes the SBERT budget.",
     "Proposed reduces per-pair cost; both remain transformer-dependent on retained pairs."],
    ["Generalization",
     "Evaluated on a sentence-level SNLI/Pereira proxy; the paper's original context was full-length academic documents — an adaptation boundary documented in the repository.",
     "Premise-disjoint grouped holdout (3,000 test pairs / 1,015 premise groups), clustered bootstrap confidence intervals, and group-conformal coverage (93.69% group / 97.80% pair).",
     "Proposed evaluates under a stricter leakage-free grouped protocol with formal coverage guarantees."],
    ["Interpretability",
     "Similarity scores only; no casewise evidence traces, character offsets, or line numbers.",
     "Reviewer dossiers with authentic character offsets and 1-based physical line numbers; ground-truth labels withheld from reviewers.",
     "Proposed supports auditable adjudication; Paper 1 does not."],
    ["Limitations",
     "Single threshold; no calibrated probabilities, uncertainty quantification, conformal coverage, budget formalization, or audit dossiers; sentence-level proxy data.",
     "Candidate recall below 1.0 on text (0.8598); group coverage 93.69% slightly below the nominal 95%; text corpus is a proxy; training time and memory not instrumented.",
     "Complementary limitation profiles: Paper 1 lacks uncertainty and audit machinery; the proposed framework accepts a bounded candidate-recall trade-off for cost control."],
]
write_qual_csv(OUT / "table_02_paper1_setu_vs_proposed_text_qualitative.csv",
    "Table 2: Qualitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Text Plagiarism Detection",
    "Capability-by-capability comparison grounded in the implemented methodologies; claims restricted to features demonstrated in the repository record.",
    "Natural language text. Paper 1 addresses natural-language text plagiarism only.",
    t2_rows)

# ----------------------------------------------------------------------------
# TABLE 10 — JAVA: Paper 2 (Gandhi) vs Proposed — Qualitative
# ----------------------------------------------------------------------------
t10_rows = [
    ["Detection methodology",
     "RBF-SVM hyperplane over 7-D engineered syntactic/token features; fixed decision threshold (tau=0.3700).",
     "Token-Jaccard pre-filter followed by tri-band routing: uncertain pairs escalated to a calibrated SVM heavy verifier; high-risk pairs directly flagged for review.",
     "Proposed routes pairs adaptively between a cheap screener and a heavy verifier; Paper 2 classifies every pair exhaustively with the SVM."],
    ["Feature extraction",
     "Comment stripping, identifier anonymization, raw and anonymized token Jaccard, keyword-distribution cosine, operator Jaccard, line-count ratio, cyclomatic-complexity differential, character 3-grams (7-D feature vector).",
     "The same 7-D syntactic/token feature space (reused inside the heavy verifier) plus a cheap token-Jaccard pre-filter view.",
     "Proposed operationalizes Paper 2's feature engineering inside a routed pipeline rather than as a standalone scorer."],
    ["Lexical similarity detection",
     "Yes — token and operator Jaccard overlap.",
     "Yes — token-Jaccard pre-filter resolves clear cases at Stage-1.",
     "Comparable lexical detection; proposed applies it selectively before any heavy scoring."],
    ["Semantic similarity detection",
     "No — no semantic or model-based code-equivalence view.",
     "No — no learned semantic code-equivalence view; the heavy verifier operates on syntactic/token features.",
     "Neither methodology implements semantic code-equivalence detection."],
    ["Structural similarity detection",
     "Yes — line-count ratio and cyclomatic-complexity differential features.",
     "Yes — via the 7-D syntactic features used by the heavy verifier.",
     "Comparable structural coverage."],
    ["Uncertainty estimation",
     "No — SVC(probability=True) yields uncalibrated pseudo-probabilities; no uncertainty quantification.",
     "Yes — calibrated posteriors and predictive-entropy routing decide which pairs need verification.",
     "Proposed quantifies uncertainty; Paper 2 does not."],
    ["Confidence handling",
     "Fixed threshold on uncalibrated SVM scores; all pairs forced to binary decisions.",
     "Calibrated posteriors with theta_low/theta_high gates; ambiguous pairs are escalated to the verifier.",
     "Proposed defers ambiguous cases; Paper 2 does not."],
    ["Adaptive escalation",
     "No — exhaustive scoring of all pairs.",
     "Yes — budget-capped, recall-floored escalation; expensive verifier invoked on only 63.33% of pairs (1 pruned, 76 directly flagged out of 210).",
     "Proposed avoids heavy scoring on 36.67% of pairs."],
    ["Computational optimization",
     "No optimization — all pairs scored by SVM; classifier inference 0.013 ms/pair but 7-D feature extraction (~2.050 ms/pair) unbudgeted in the paper.",
     "One pair pruned (0.48%), 76 directly flagged (36.19%) bypass the SVM; end-to-end tri-band pipeline 0.058 ms/pair (archived).",
     "Proposed reduces expensive verifier invocations by 36.67 percentage points."],
    ["False-positive handling",
     "No formal false-positive control; FPR 0.5370 on the case-06/07 holdout.",
     "Direct-flag band routes likely-plagiarized pairs to human review rather than automatic accusation; calibrated gate targets precision.",
     "Proposed adds a review channel; neither methodology formally bounds false positives in this record."],
    ["False-negative handling",
     "Recall 0.9510 on holdout, but collapses under systematic extra-benchmark identifier renaming (positive recall 0.2778).",
     "Candidate recall 1.0000 — all positives retained in the escalation/direct-flag bands; final F1 lower (0.7424 vs 0.7549) because of more false positives.",
     "Proposed preserves candidate recall; Paper 2 is brittle to out-of-distribution renaming attacks."],
    ["Scalability",
     "Exhaustive O(N^2) pair scoring; no candidate pre-filter.",
     "Token-Jaccard pre-filter resolves clear pairs cheaply; the SVM runs only on the uncertain band.",
     "Proposed scales better on large submission cohorts."],
    ["Generalization",
     "Originally validated on Python coursework; adapted to Java (IR-Plag) in this project; case-holdout and nested-LOGO reproduction reported.",
     "Nested leave-one-group-out evaluation with zero program leakage across assignment cases (SHA-256 verified); case-06/07 holdout record used here.",
     "Both use case-disjoint protocols; the proposed framework additionally quantifies naive-split leakage inflation (+0.1221 MCC)."],
    ["Interpretability",
     "SVM scores only; no casewise evidence traces, offsets, or line numbers.",
     "Reviewer dossiers with authentic character offsets and 1-based physical line numbers; ground-truth withheld from reviewers.",
     "Proposed supports auditable adjudication; Paper 2 does not."],
    ["Limitations",
     "No calibration, uncertainty handling, budget routing, or audit dossiers; extra-benchmark renaming collapses recall; original corpus is private Python coursework.",
     "Detection quality statistically significantly lower on the holdout (McNemar p=5.10e-05; F1 −0.0125) driven by a higher false-positive rate; latency accounting is non-comparable (classifier-only vs end-to-end tri-band).",
     "Trade-off: the proposed framework is more computationally efficient with preserved candidate recall, but its holdout detection quality is significantly lower than Paper 2's."],
]
write_qual_csv(OUT / "table_10_paper2_gandhi_vs_proposed_java_qualitative.csv",
    "Table 10: Qualitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Java Source-Code Plagiarism Detection",
    "Capability-by-capability comparison grounded in the implemented methodologies; claims restricted to features demonstrated in the repository record.",
    "Java source code. Paper 2 addresses source-code plagiarism (Python original; Java adaptation in this repository).",
    t10_rows)

# ----------------------------------------------------------------------------
# TABLE 12 — JAVA: Paper 3 (Mehsen) vs Proposed — Qualitative
# ----------------------------------------------------------------------------
t12_rows = [
    ["Detection methodology",
     "Random Forest ensemble (100 trees, max depth 6) over structured token-frequency and syntactic feature vectors; vote-fraction decision with a fixed threshold.",
     "Token-Jaccard pre-filter followed by tri-band routing: uncertain pairs escalated to a calibrated SVM heavy verifier; high-risk pairs directly flagged for review.",
     "Proposed pairs an ensemble-style heavy verifier with a cheap pre-filter and adaptive routing; Paper 3 applies the forest exhaustively to every pair."],
    ["Feature extraction",
     "Structured token frequency vectors and syntactic metrics.",
     "Seven-D syntactic/token feature vector inside the heavy verifier plus a cheap token-Jaccard pre-filter view.",
     "Both use syntactic/token representations; the proposed pipeline adds a lexical pre-screening view."],
    ["Lexical similarity detection",
     "Yes — token-frequency vectors capture lexical overlap.",
     "Yes — token-Jaccard pre-filter.",
     "Comparable lexical coverage at the pre-filter stage."],
    ["Semantic similarity detection",
     "No — no semantic code-equivalence view.",
     "No — no learned semantic code-equivalence view.",
     "Neither implements semantic code-equivalence detection."],
    ["Structural similarity detection",
     "Yes — syntactic metrics in the feature vectors.",
     "Yes — via the 7-D heavy-verifier features.",
     "Comparable structural coverage."],
    ["Uncertainty estimation",
     "No — forest vote fractions are not calibrated and no uncertainty quantification is reported.",
     "Yes — calibrated posteriors and predictive-entropy routing.",
     "Proposed quantifies uncertainty; Paper 3 does not."],
    ["Confidence handling",
     "Fixed threshold on forest scores; all pairs forced to binary decisions.",
     "Calibrated posteriors with theta_low/theta_high gates; ambiguous pairs escalated.",
     "Proposed defers ambiguous cases; Paper 3 does not."],
    ["Adaptive escalation",
     "No — exhaustive scoring of all pairs.",
     "Yes — budget-capped escalation; expensive verifier invoked on only 63.33% of pairs.",
     "Proposed avoids heavy scoring on 36.67% of pairs."],
    ["Computational optimization",
     "No optimization — all pairs scored by the 100-tree forest; classifier inference 0.025 ms/pair (archived).",
     "One pair pruned (0.48%), 76 directly flagged (36.19%) bypass the verifier; end-to-end pipeline 0.058 ms/pair (archived).",
     "Proposed reduces expensive verifier invocations by 36.67 percentage points."],
    ["False-positive handling",
     "No formal false-positive control; FPR 0.4352 on holdout.",
     "Direct-flag band for human review; calibrated gate targets precision (FPR 0.5926 on holdout — worse than Paper 3).",
     "Paper 3 has the lower false-positive rate; the proposed framework trades precision for recall/cost control."],
    ["False-negative handling",
     "Per-level resilience reported (L1–L6); adaptation record: L6 compound recall 83.33%, but LOGO out-of-fold L6 recall 0.5085; extra-benchmark renaming positive recall 0.3216.",
     "Candidate recall 1.0000 — all positives retained in escalation/direct-flag bands.",
     "Proposed preserves candidate recall; Paper 3 degrades sharply on compound and out-of-distribution attacks."],
    ["Scalability",
     "100-tree ensemble scored exhaustively over all pairs; no pre-filter.",
     "Token-Jaccard pre-filter resolves clear pairs cheaply; SVM runs only on the uncertain band.",
     "Proposed scales better on large cohorts."],
    ["Generalization",
     "Originally evaluated on a private corpus; adapted to IR-Plag Java; case-holdout and nested-LOGO reproduction in this project.",
     "Nested leave-one-group-out with zero cross-case program leakage; case-06/07 holdout record used here.",
     "Both use case-disjoint protocols; proposed quantifies leakage and reports clustered uncertainty."],
    ["Interpretability",
     "Random Forest ensemble scores only; no casewise evidence traces, offsets, or line numbers.",
     "Reviewer dossiers with authentic character offsets and 1-based physical line numbers; ground-truth withheld from reviewers.",
     "Proposed supports auditable adjudication; Paper 3 does not."],
    ["Limitations",
     "No calibration, uncertainty handling, budget routing, or audit dossiers; original corpus not public; extra-benchmark renaming collapses recall (0.3216).",
     "Detection quality statistically significantly lower on the holdout (McNemar p=1.49e-04; F1 −0.0262) driven by a higher false-positive rate.",
     "Trade-off: the proposed framework is more computationally efficient with preserved candidate recall, but its holdout detection quality is significantly lower than Paper 3's."],
]
write_qual_csv(OUT / "table_12_paper3_mehsen_vs_proposed_java_qualitative.csv",
    "Table 12: Qualitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Java Source-Code Plagiarism Detection",
    "Capability-by-capability comparison grounded in the implemented methodologies; claims restricted to features demonstrated in the repository record.",
    "Java source code. Paper 3 addresses source-code plagiarism (private corpus original; Java adaptation in this repository).",
    t12_rows)

# ----------------------------------------------------------------------------
# N/A quantitative tables: Tables 3, 5 (text modality: Papers 2 & 3 not applicable)
# and Tables 7, 8 (java modality: Paper 1 not applicable) -- quant + qual
# ----------------------------------------------------------------------------
na_quant_text_p2 = ("Not applicable: Paper 2 (Gandhi et al. 2024) addresses source-code plagiarism "
                    "(Python undergraduate submissions; Java adaptation in this repository) and reports no "
                    "natural-language text experiments. No text-modality metrics exist for this paper, so a "
                    "quantitative comparison on the text modality cannot be constructed without fabricating results.")
write_na_quant_csv(OUT / "table_03_paper2_gandhi_vs_proposed_text_quantitative.csv",
    "Table 3: Quantitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable)",
    "Paper 2 is a source-code methodology; it has no text-modality implementation or evaluation. All metric cells are N/A.",
    na_quant_text_p2)

na_quant_text_p3 = ("Not applicable: Paper 3 (Mehsen & Joshi 2024) addresses source-code plagiarism and reports no "
                    "natural-language text experiments. No text-modality metrics exist for this paper.")
write_na_quant_csv(OUT / "table_05_paper3_mehsen_vs_proposed_text_quantitative.csv",
    "Table 5: Quantitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable)",
    "Paper 3 is a source-code methodology; it has no text-modality implementation or evaluation. All metric cells are N/A.",
    na_quant_text_p3)

na_quant_java_p1 = ("Not applicable: Paper 1 (Setu et al. 2025) addresses natural-language text plagiarism "
                    "(hybrid TF-IDF/SBERT) and reports no Java source-code experiments. No source-code metrics exist "
                    "for this paper, so a quantitative comparison on the Java modality cannot be constructed without "
                    "fabricating results.")
write_na_quant_csv(OUT / "table_07_paper1_setu_vs_proposed_java_quantitative.csv",
    "Table 7: Quantitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Java Source-Code Plagiarism Detection (Not Applicable)",
    "Paper 1 is a text modality methodology; it has no Java source-code implementation or evaluation. All metric cells are N/A.",
    na_quant_java_p1)

def na_qual_rows():
    return [[d, "N/A", "N/A",
             "Not applicable: this paper does not address the compared modality, so no dimension-level comparison can be made without fabricating capabilities."]
            for d in DIMENSIONS]

write_qual_csv(OUT / "table_04_paper2_gandhi_vs_proposed_text_qualitative.csv",
    "Table 4: Qualitative Comparison of Paper 2 (Gandhi et al. 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable)",
    "Paper 2 is a source-code methodology with no natural-language text implementation or evaluation.",
    "Natural language text. Not applicable: Paper 2 addresses source code only.",
    na_qual_rows())

write_qual_csv(OUT / "table_06_paper3_mehsen_vs_proposed_text_qualitative.csv",
    "Table 6: Qualitative Comparison of Paper 3 (Mehsen & Joshi 2024) and Proposed Framework on Text Plagiarism Detection (Not Applicable)",
    "Paper 3 is a source-code methodology with no natural-language text implementation or evaluation.",
    "Natural language text. Not applicable: Paper 3 addresses source code only.",
    na_qual_rows())

write_qual_csv(OUT / "table_08_paper1_setu_vs_proposed_java_qualitative.csv",
    "Table 8: Qualitative Comparison of Paper 1 (Setu et al. 2025) and Proposed Framework on Java Source-Code Plagiarism Detection (Not Applicable)",
    "Paper 1 is a natural-language text methodology with no Java source-code implementation or evaluation.",
    "Java source code. Not applicable: Paper 1 addresses text only.",
    na_qual_rows())

print("\nDone.")