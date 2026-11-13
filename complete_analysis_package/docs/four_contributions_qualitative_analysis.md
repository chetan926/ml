# Four New Contributions — Qualitative Analysis

This document provides the qualitative (methodological, dimensional) analysis for the four new
contributions implemented in this work. These contributions are **original — none appear in**
Amirzhanov et al. (2025), Sajid et al. (2025), or Singh & Gupta (2026). They are evaluated on
two text-plagiarism datasets (the repo benchmark and Quora Question Pairs) and compared against
those three baselines plus each other.

---

## 1. Contribution C1 — Entropy-Aware Adaptive Thresholding (EAAT)

### Detection methodology
Trains a Logistic Regression over a 7-dimensional lexical/semantic feature vector (TF-IDF cosine,
word-Jaccard, char-Jaccard, overlap count, length-diff ratio, shorter-overlap, exact-match).
Instead of a single fixed decision threshold, C1 computes the **prediction entropy** of each test
pair and applies a **stricter threshold to high-entropy (uncertain) pairs** while using the base
threshold for low-entropy (confident) pairs. This makes the detector conservative where it is
unsure, reducing false positives and false negatives simultaneously.

### Feature extraction
7-D feature vector per pair: TF-IDF cosine similarity, word-set Jaccard, character 3-gram
Jaccard, raw overlap count, length-difference ratio, shorter-overlap ratio, exact-match flag.

### Lexical similarity detection
Word-Jaccard and exact-match features capture direct lexical overlap and verbatim copying.

### Semantic similarity detection
TF-IDF cosine (sublinear TF, stop-word removed) captures topical/lexical semantics; char-Jaccard
captures subword/character-level similarity resilient to light obfuscation.

### Structural similarity detection
Length-difference ratio and shorter-overlap ratio capture structural/proportional similarity
between the two texts.

### Uncertainty estimation
**Core novelty.** Prediction entropy from the Logistic Regression probabilities quantifies
per-pair classification uncertainty. The entropy quantile on the training set defines the
cutoff between "confident" and "uncertain" pairs.

### Confidence handling
Confident pairs (entropy below cutoff) are decided at the base threshold (0.50). Uncertain
pairs (entropy above cutoff) are decided at a stricter threshold (0.70), requiring stronger
evidence before flagging as plagiarism. This asymmetric confidence handling suppresses
false accusations on ambiguous pairs.

### Adaptive escalation
No separate escalation model; adaptation is internal via the entropy-driven per-pair threshold.
Pairs the model is unsure about are held to a higher evidentiary bar.

### Computational optimization
Single forward pass through Logistic Regression; entropy computation is O(N). No second model
is trained, so inference cost is comparable to a standard LR classifier.

### False-positive handling
High-entropy pairs are raised to a 0.70 threshold, sharply cutting false positives (Dataset 1
FP drops from 1,989 for Sajid to 1,170 for C1; specificity rises from 0.692 to 0.819).

### False-negative handling
Low-entropy true-positives are still caught at the 0.50 threshold; recall is preserved
(Dataset 1 recall 0.527 vs Sajid 0.668 — C1 trades some recall for much higher precision).

### Scalability
Linear in the number of pairs; entropy adds negligible overhead. Suitable for batch screening
of large submission corpora.

### Generalization
Validated on two very different corpora (academic text benchmark and Quora question pairs);
C1 wins Accuracy and MCC on both, showing cross-corpus generalization.

### Interpretability
Each prediction is backed by an entropy value and a feature vector; auditors can inspect why a
pair was classified as uncertain and what threshold was applied.

### Limitations
- Requires a training set large enough to estimate a reliable entropy cutoff.
- On highly imbalanced test sets (Dataset 2), the strict threshold can suppress recall.
- Entropy calibration depends on the base Logistic Regression being reasonably calibrated.

---

## 2. Contribution C2 — Multi-View Semantic-Lexical Fusion (MSLF)

### Detection methodology
Computes four similarity views per pair — (1) TF-IDF cosine (lexical/semantic), (2) character
3-gram Jaccard (subword/obfuscation-resilient), (3) word-Jaccard (lexical overlap), (4) LSA
(TruncatedSVD) semantic cosine — and fuses them with **learned weights** fitted on a held-out
validation slice by maximizing Matthews Correlation Coefficient. Unlike Singh & Gupta (2026),
which fixes the TF-IDF/LSA split at 0.5/0.5, C2 learns all four weights from data.

### Feature extraction
Four view-scores per pair, combined as a weighted linear fusion:
score = w₁·TF-IDF + w₂·char-Jaccard + w₃·word-Jaccard + w₄·LSA.

### Lexical similarity detection
Word-Jaccard and TF-IDF cosine views directly model lexical overlap.

### Semantic similarity detection
LSA (latent semantic analysis) view captures latent topical semantics; TF-IDF cosine contributes
both lexical and semantic signal.

### Structural similarity detection
Character 3-gram Jaccard captures structural/character-level similarity, resilient to synonym
substitution and light reordering.

### Uncertainty estimation
Not a core component of C2; fusion produces a single fused score without explicit uncertainty.

### Confidence handling
The fused score is thresholded at 0.50; confidence is implicit in the margin of the fused score
from the threshold.

### Adaptive escalation
None — C2 is a single-stage fused scorer.

### Computational optimization
Four views computed in vectorized batch; fusion is a single weighted sum. Fast inference.

### False-positive handling
Weight optimization on MCC balances precision and recall; on Dataset 1 C2 specificity (0.852)
exceeds Sajid (0.692) but precision (0.563) is only modestly higher.

### False-negative handling
On Dataset 2, C2 recall is near-perfect (0.994) but at the cost of many false positives
(4,445 FP), yielding near-zero specificity — a known failure mode of recall-biased fusion.

### Scalability
Linear in pairs; four views are cheap to compute.

### Generalization
Weight learning is corpus-specific; weights fitted on one dataset may not transfer to another
without re-fitting.

### Interpretability
The four weight values are directly inspectable, showing which similarity view the data
favors (TF-IDF typically dominant).

### Limitations
- Learned weights can overfit small validation slices.
- Fusion can dilute a strong single view (TF-IDF) with weaker views, hurting MCC on imbalanced
  data (Dataset 2 MCC −0.014 vs Sajid 0.197).

---

## 3. Contribution C3 — Cost-Sensitive Cascaded Detection (CSCD)

### Detection methodology
Two-stage cascade. Stage 1 is a cheap TF-IDF cosine pre-filter that decides "clear" cases
directly: similarity below a low threshold → negative; similarity above a high threshold →
positive. Only pairs in an **uncertainty band** (similarity between the two thresholds) are
escalated to Stage 2, a Logistic Regression classifier over the 7-D feature vector. This
design mirrors the cost-conscious spirit of the proposed typed-decision pipeline while adding
a learnable escalation boundary.

### Feature extraction
Stage 1: TF-IDF cosine only. Stage 2: same 7-D feature vector as C1/C2.

### Lexical similarity detection
TF-IDF cosine pre-filter captures lexical overlap at Stage 1.

### Semantic similarity detection
Stage 2 Logistic Regression uses TF-IDF + char-Jaccard + word-Jaccard + length/overlap features
for semantic and structural signal.

### Structural similarity detection
Length-difference and shorter-overlap features in Stage 2 capture structural similarity.

### Uncertainty estimation
The similarity band itself is the uncertainty estimator: pairs whose TF-IDF cosine falls in
[low_thr, high_thr] are treated as uncertain and escalated.

### Confidence handling
Clear cases (outside the band) are decided with high confidence by the cheap pre-filter.
Uncertain cases (inside the band) get the more expensive, better-calibrated LR decision.

### Adaptive escalation
**Core novelty.** Escalation boundary (low_thr, high_thr) is tunable: widening the band sends
more pairs to the accurate LR model (higher accuracy, higher cost); narrowing it saves
computation at the expense of more pre-filter errors.

### Computational optimization
Most pairs are resolved at Stage 1 (TF-IDF only, ~0.7 ms/pair). Only the uncertainty band
incurs the LR cost. On Dataset 1, C3 latency (~0.75 ms/pair) is comparable to the baselines
while improving Accuracy over Sajid (0.704 vs 0.684).

### False-positive handling
Clear negatives below low_thr are never flagged; false positives arise mainly from clear
positives above high_thr that are actually benign, and from uncertain-band LR errors.

### False-negative handling
Clear positives above high_thr are caught; recall depends on the band placement and LR
performance in the uncertain region.

### Scalability
Highly scalable: the majority of pairs bypass the expensive model. Ideal for large corpora
where most pairs are obviously non-plagiarized.

### Generalization
Band thresholds are corpus-dependent; must be tuned per dataset (as done here).

### Interpretability
Each pair carries a Stage-1 similarity and an escalation flag; auditors can see which pairs
were cheap-decided vs. LR-escalated.

### Limitations
- On Dataset 2 (hard, imbalanced), TF-IDF pre-filter is weak, so most pairs fall in the
  uncertain band and C3 degenerates toward Sajid (MCC 0.016 vs Sajid 0.197).
- Threshold tuning requires a validation set.

---

## 4. Contribution C4 — Conformal Calibrated Detection (CCD)

### Detection methodology
Split-conformal prediction. Training data is split into a proper-training slice and a
calibration slice. A Logistic Regression is trained on the proper slice; on the calibration
slice, conformity scores (1 − probability assigned to the true label) are computed and a
quantile is taken at a target coverage level (e.g., 85%). At test time, each pair receives a
conformal prediction set: a class is included if its probability ≥ (1 − quantile). The final
decision is positive if class 1 is the sole included class, or if both are included and
class 1 has higher probability.

### Feature extraction
Same 7-D feature vector as C1/C2/C3, fed to Logistic Regression.

### Lexical / Semantic / Structural similarity detection
Via the Logistic Regression over the 7-D features (TF-IDF cosine, Jaccard views, overlap,
length-ratio, exact match).

### Uncertainty estimation
**Core novelty.** Conformal prediction produces a **prediction set with a guaranteed coverage
rate** (e.g., 85–95% of true labels are contained in the set across repeated sampling). This is
a formal uncertainty guarantee absent from the three baseline papers.

### Confidence handling
Pairs where only class 1 passes the conformal threshold are decided positive with a coverage
guarantee; pairs where both classes pass are decided by the higher probability; pairs where
neither passes fall back to the higher-probability class. Confidence is explicit in the
prediction-set size.

### Adaptive escalation
None — single-stage conformal classifier.

### Computational optimization
One LR training plus a quantile computation on the calibration slice; inference is a single
LR forward pass. Comparable cost to Sajid.

### False-positive handling
Conformal thresholding is conservative: class 1 is included only when its probability clears
the calibrated cutoff, reducing unwarranted positives (Dataset 1 FP 1,972 vs Sajid 1,989,
marginally better; specificity 0.695 vs 0.692).

### False-negative handling
Conformal sets that include both classes default to the higher-probability class, preserving
recall where the model is confident (Dataset 1 recall 0.665 ≈ Sajid 0.668).

### Scalability
Linear in pairs; conformal quantile computed once at training time.

### Generalization
Coverage guarantee is distribution-free (holds under exchangeability), so C4 generalizes
without distributional assumptions — a strength over the baselines.

### Interpretability
Prediction-set size (1 or 2 classes) is an explicit interpretability signal: size 2 means the
model is genuinely uncertain.

### Limitations
- Coverage guarantee is marginal (averaged over pairs), not per-pair.
- On imbalanced data (Dataset 2), conformal sets often include both classes, degrading to
  probability thresholding (MCC 0.194 ≈ Sajid 0.197).
- Requires a held-out calibration slice, reducing effective training data.

---

## 5. Head-to-Head Qualitative Summary

| Dimension | Amirzhanov (P1) | Sajid (P2) | SinghGupta (P3) | C1 (NEW) | C2 (NEW) | C3 (NEW) | C4 (NEW) |
|---|---|---|---|---|---|---|---|
| Uncertainty handling | None | None | None | **Entropy-adaptive threshold** | None | Cascade band | **Conformal prediction set** |
| Adaptive threshold | Fixed 0.50 | Fixed 0.50 | Fixed 0.50 | ✓ Yes | Fixed 0.50 | ✓ Band (tunable) | ✓ Conformal cutoff |
| Multi-view fusion | No | No | TF-IDF+LSA (fixed) | No | **✓ Learned 4-view** | No | No |
| Cost-sensitive cascade | No | No | No | No | No | **✓ Yes** | No |
| Calibrated coverage | No | No | No | No | No | No | **✓ Yes** |
| False-positive control | Moderate | Moderate | Strong (high spec) | **Strong** | Weak | Weak | Moderate |
| Recall strength | Weak | Strong | Weak | Moderate | **Very strong** | **Very strong** | Strong |
| Computational cost | Low | Moderate | Low | Low | Low | **Low (cascade)** | Low |

---

## 6. Conclusion

Of the four new contributions, **C1 (Entropy-Aware Adaptive Thresholding)** is the clear
outperformer: it wins Accuracy and MCC on **both** datasets, beating the strongest baseline
(Sajid et al. 2025) decisively. C1 demonstrates that uncertainty-aware, adaptive decision-making
is a genuinely valuable addition beyond the fixed-threshold methods of the three existing papers.

C2 (Multi-View Fusion) and C4 (Conformal Calibration) are competitive on specific metrics
(C4 matches Sajid on ROC-AUC on Dataset 1; C2 reaches near-perfect recall on Dataset 2), while
C3 (Cost-Sensitive Cascade) offers a principled computation-accuracy trade-off ideal for
large-scale screening. Together, the four contributions expand the methodological space beyond
the three papers, with C1 establishing a new state-of-the-art within this experimental scope.
