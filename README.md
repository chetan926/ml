# An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Audit Status](https://img.shields.io/badge/Audit-Passed_Phase_4.1-brightgreen.svg)]()
[![Verification](https://img.shields.io/badge/Verification-Independent_Archive_Verified-success.svg)]()

---

## 1. Research Title
**An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection**

---

## 2. Abstract & Research Motivation
Institutional plagiarism and academic integrity detection platforms face severe operational bottlenecks when scanning massive submission pools. Modern deep verification models (such as cross-encoders, AST-based graph alignment models, and heavy language models) offer high discriminative power but demand prohibitive computational resources $\mathcal{O}(N^2)$ during cohort-wide screening. Conversely, surface lexical heuristics suffer from high false-positive rates, disproportionately penalizing students and overburdening adjudicatory committees.

This framework introduces a staged, uncertainty-aware decision pipeline that combines lightweight multi-view screening with calibrated uncertainty and budget-constrained routing. Decisive pairs (blatantly innocent or blatantly plagiarized) are resolved immediately at Stage-1 at sub-millisecond latencies, while genuinely ambiguous cases are selectively escalated to heavy Stage-2 verification models within an explicit computational budget.

---

## 3. Problem Statement
Applying computationally heavy verification architectures across all possible submission pairs is financially and computationally unfeasible in large classes and cross-institutional repositories. The core problem is to design an automated screening and verification architecture that minimizes expensive verifier invocations while maintaining competitive detection accuracy and strictly bounding false-accusation rates under rigorous group-disjoint evaluation.

---

## 4. Research Objectives
1. **Maximize Workload Reduction:** Prune clear non-plagiarized and direct plagiarism cases early to avoid unnecessary heavy verifier calls.
2. **Control False-Accusation Rates:** Ensure high precision ($>0.70$) to prevent cognitive bias and wrongful disciplinary escalations.
3. **Rigorous Uncertainty Calibration:** Provide well-calibrated class-posterior probabilities and exact finite-sample coverage guarantees at the assignment premise-group level.
4. **Leakage-Free Source Code Evaluation:** Enforce strict Leave-One-Group-Out (LOGO) cross-validation across distinct programming tasks, eliminating cross-fold program identity leakage.
5. **Auditable Explainability:** Supply authentic character offsets and physical line numbers in adjudicatory dossiers without revealing ground-truth labels to reviewers.

---

## 5. Four Principal Research Contributions

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUR RESEARCH CONTRIBUTIONS                              │
├─────────────────────────┬───────────────────────────┬──────────────────────────────────┤
│ C1: Multi-View Feature   │ C2: Uncertainty & Group   │ C3: Budget-Adaptive Routing      │
│     Fusion Mechanism    │     Conformal Prediction  │     & Cost Optimization          │
│ • Lexical + Char +      │ • Platt & Temperature     │ • Epistemic Entropy Routing      │
│   Semantic + Structure  │   Scaling Calibration     │ • Stage-2 Escalation Bounding    │
│ • AST + Token for Code  │ • Exact Finite-Sample $k$ │ • Hardware Profiling & Latency   │
├─────────────────────────┴───────────────────────────┴──────────────────────────────────┤
│ C4: Explainable Reviewer Dossiers & Governance Verification                             │
│ • Authentic Character Offsets & 1-Based Physical Line Indexing                          │
│ • Unbiased Operational Reviewer Schemas (No Ground-Truth Leakage)                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **C1 — Multi-View Complementary Feature Fusion:** Integrates word-level lexical overlap (TF-IDF, Jaccard), character substrings (3-grams, LCS), dense contextual semantic similarities (MiniLM), and structural stylometry.
2. **C2 — Predictive Uncertainty & Exact Finite-Sample Group-Conformal Prediction:** Combines post-hoc probability calibration with group-conformal prediction at the premise-group level, handling finite-sample quantiles with formal fallback guarantees.
3. **C3 — Budget-Adaptive Routing & Cost Optimization:** Routes candidate pairs through an entropy-driven triage policy, deciding decisive candidates at Stage-1 and invoking heavy Stage-2 models only for high-entropy instances.
4. **C4 — Adjudicatory Explainability:** Generates standardized operational reviewer dossiers that map authentic character slices and physical line numbers, omitting ground-truth labels to ensure unbiased human adjudication.

---

## 6. Proposed Architecture

```
[Candidate Pair (Text or Code)]
               │
               ▼
┌──────────────────────────────────────────────┐
│ Stage-1: Lightweight Multi-View Feature Ext. │
│ (Lexical, Character, Semantic, Structural)   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage-1 Candidate Scorer & Platt Calibrator  │──> Calibrated Probability p = P(Y=1|x)
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Uncertainty Evaluation & Predictive Entropy  │──> H(p) = -p log2(p) - (1-p) log2(1-p)
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
Low Uncertainty: H(p) < H_thresh   High Uncertainty: H(p) >= H_thresh
(Decisive at Stage-1)              (Ambiguous Pair)
       │                               │
       ├──────────────┐                ▼
       ▼              ▼          ┌──────────────────────────────────────────┐
p < tau_low     p > tau_high     │ Stage-2: Heavy Verification Model        │
[PRUNED (0)]    [FLAGGED (1)]    └────────────────────┬─────────────────────┘
                                                      │
                                                      ▼
                                         [Final Prediction + Reviewer Dossier]
```

---

## 7. Dataset Descriptions & Benchmark Sources
- **Text Modality (SNLI Integrity Corpus):** 8,500 total pairs partitioned strictly by premise text groups into:
  - Training: 4,000 pairs
  - Validation: 1,500 pairs across 254 distinct premise groups
  - Test: 3,000 pairs across 1,015 distinct premise groups (zero premise overlap with training/validation)
- **Code Modality (IR-Plag Java Corpus):** Benchmark of authentic student Java submissions across 7 distinct programming assignments (Cases 01–07):
  - Policy A: 720 pairs
  - Policy B: 709 pairs
  - Cryptographic Overlap Check: Verified via SHA-256 program content hashing; 0 reference programs and 0 suspicious programs are shared across assignment cases.

---

## 8. Data Preprocessing Methodology
- **Text Preprocessing:** Whitespace normalization, sentence splitting, token extraction, and character n-gram indexing with exact preservation of original raw text slices for offset recovery.
- **Code Preprocessing:** Java AST parsing (`javalang`), comment and whitespace stripping for canonical representation, token sequence extraction, and 1-based physical line index mapping preserving original source layout.

---

## 9. Machine Learning Algorithms & Models Used
- **Stage-1 Screening Scorer:** L2-regularized Logistic Regression ($C=1.0$, L-BFGS solver).
- **Probability Calibrators:** Platt Scaling (Sigmoidal Logistic), Isotonic Regression, Temperature Scaling.
- **Stage-2 Heavy Verifiers:** Ensemble Random Forest (100 estimators, max depth 8) and gradient-boosted decision trees.

---

## 10. Multi-View Framework Explanation
A single similarity metric (e.g. lexical overlap) fails against paraphrased text or renamed variables. The multi-view framework extracts orthogonal representations:
- **Lexical View:** Captures direct word reuse and bag-of-words overlap.
- **Character View:** Detects partial token matches, stem variations, and character subsequences.
- **Semantic View:** Encodes sentence-level contextual embeddings invariant to lexical substitutions.
- **Structural View:** Quantifies length discrepancies, sentence counts, and type-token ratios.

---

## 11. Uncertainty Estimation Methodology
Predictive uncertainty is quantified via the Shannon entropy of the calibrated class-posterior:
$$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$
Pairs with $H(p) \ge H_{\text{thresh}}$ indicate high ambiguity where the Stage-1 model is uncertain between classes, triggering escalation to Stage-2.

---

## 12. Calibration Methodology
Raw classifier outputs are mapped to true posterior probabilities using independent validation sets:
- **Platt Scaling:** Fits parameters $A, B$ such that $P(Y=1 \mid s) = \frac{1}{1 + \exp(As + B)}$.
- **In Code LOGO:** Implements nested inner cross-validation: for each outer held-out case, Platt scaling is fit strictly on inner out-of-fold predictions from the remaining 6 cases, ensuring zero leakage.

---

## 13. Conformal Prediction Methodology
Provides distribution-free, finite-sample predictive guarantees at the premise-group unit level:
- Nonconformity score: Group supremum $S_g = \max_{i \in g} (1 - \hat{P}(Y = y_i \mid x_i))$.
- Exact finite-sample rank: $k = \lceil (m+1)(1-\alpha) \rceil$.
- If $k > m$, the quantile defaults conservatively to $\hat{q} = \infty$.
- Evaluated on 1,015 held-out test premise groups ($m=254$ calibration groups, $\alpha=0.05$):
  - **Empirical Group Coverage:** **93.69%**
  - **Empirical Pair Coverage:** **97.80%**
  - **Average Prediction Set Size:** **1.6080**

---

## 14. Budget-Constrained Routing Methodology
Operating thresholds ($\tau_{\text{low}}, \tau_{\text{high}}, H_{\text{thresh}}$) are tuned on the validation set given a target computational budget $\beta \in [0, 1]$. Pairs falling into decisive low-entropy regions are classified immediately; remaining pairs consume the Stage-2 evaluation budget.

---

## 15. Cost-Reduction Mechanism
- **Text Modality:** 1,417 out of 3,000 pairs decided at Stage-1 $\rightarrow$ **47.23% reduction** in heavy verifier calls.
- **Code Modality (Policy A):** 109 out of 720 pairs decided at Stage-1 $\rightarrow$ **15.14% reduction** in heavy verifier calls under zero program leakage.

---

## 16. Experimental Methodology
- Independent split isolation: Train, Validation, and Test sets never share premise groups.
- Nested Leave-One-Group-Out (LOGO) cross-validation for code across 7 assignment cases.
- Clustered bootstrap inference (1,000 resamples at the premise group level) for robust confidence interval estimation.

---

## 17. Baseline Comparisons
Compared against:
1. **Dense Semantic Baseline (MiniLM):** Cosine similarity thresholding on dense contextual embeddings.
2. **TF-IDF Lexical Baseline:** Word-level TF-IDF similarity thresholding.
3. **Token Jaccard Baseline (Code):** Standard token-level Jaccard overlap.

---

## 18. Actual Experimental Results

### A. Text Modality (3,000 Held-Out Pairs, 1,015 Premise Groups)

| Metric | Proposed Framework | 95% Clustered Bootstrap CI | Dense Semantic Baseline | TF-IDF Baseline | Paired $\Delta$ vs. Semantic | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **0.7893** | [0.7747, 0.8037] | 0.6987 | 0.6143 | **+0.0907** | $p < 10^{-16}$ |
| **Precision** | **0.7234** | [0.6946, 0.7516] | 0.5369 | 0.4572 | **+0.1865** | $p < 10^{-16}$ |
| **Recall** | **0.6218** | [0.5904, 0.6534] | 0.8645 | 0.6823 | **-0.2427** | Trade-off |
| **F1-Score** | **0.6688** | [0.6433, 0.6931] | 0.6624 | 0.5475 | **+0.0064** | $p = 0.591$ |
| **MCC** | **0.5188** | [0.4851, 0.5519] | 0.4549 | 0.2480 | **+0.0638** | $p < 10^{-16}$ |
| **Stage-2 Avoided** | **47.23%** | — | 0.00% | 0.00% | **+47.23%** | Measured Savings |

### B. Code Modality (IR-Plag Java Corpus, 720 Policy A Pairs)

| Evaluation Setting | Accuracy | Precision | Recall | F1-Score | MCC | Program Leakage | Stage-2 Avoided |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Policy A (Nested LOGO)** | **0.8486** | **0.8157** | **0.8801** | **0.8467** | **0.6996** | **ZERO_LEAKAGE** | **15.14%** (109 / 720) |
| **Proposed Policy B (Nested LOGO)** | **0.8604** | **0.8277** | **0.8852** | **0.8555** | **0.7222** | **ZERO_LEAKAGE** | **15.37%** (109 / 709) |
| **Stratified 5-Fold (Naive Split)** | 0.9111 | 0.9137 | 0.8977 | 0.9056 | 0.8217 | Task/Program Overlap | 29.72% (214 / 720) |
| **Token Jaccard Baseline** | 0.4750 | 0.4750 | 1.0000 | 0.6441 | 0.0000 | Baseline | 0.00% (0 / 720) |

*Controlled Leakage Impact:* Identical estimators and thresholds demonstrate that naive cross-validation inflates MCC by **$+0.1221$** due to program leakage.

---

## 19. Evaluation Metrics
Evaluates: Accuracy, Precision, Recall, F1, F0.5, F2, Specificity, NPV, FPR, FNR, FDR, FOR, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Cohen's Kappa, Youden's J, Geometric Mean, Threat Score, ROC-AUC, PR-AUC, ECE, Brier Score, and empirical Stage-2 workload avoidance ratios.

---

## 20. Installation Instructions

```bash
git clone https://github.com/chetan926/ml.git
cd ml
pip install -r requirements.txt
```

---

## 21. Execution Instructions

To execute the complete research pipeline:
```bash
python research_upgrade/phase_4_1_corrections/run_phase_4_1_execution.py
```

---

## 22. Testing Instructions

To run automated unit and regression tests:
```bash
python -m unittest tests/test_phase_4_1_corrections.py
```

---

## 23. Reproducibility Instructions
- Frozen baseline SHA-256 integrity verification: `research_upgrade/src/governance.py::verify_baseline_integrity()`.
- Random seed: Set globally to `42`.
- All outputs are stored in `research_upgrade/phase_4_1_corrections/artifacts/`.
- Independent verification archive: `research_upgrade/audits/phase_4_1_independent_verification.zip`.

---

## 24. Limitations
1. **IR-Plag Case Count:** Contains 7 assignment cases; cross-case variance is bounded by the small cluster count.
2. **Case-05 Singularity:** All 53 plagiarized pairs in Case-05 share a single base program (`prog_original`). Isolating it leaves zero positive pairs in training, necessitating cross-case training.
3. **Absence of Human Span Ground Truth:** Span-level precision/recall/F1 are omitted as no human-annotated character-level span dataset exists for this benchmark.

---

## 25. Future Research Directions
- Scaling conformal risk control to large generative AI detection benchmarks.
- Incorporating neural AST representation models (e.g. Tree-LSTM, CodeBERT) as Stage-2 verifiers.
- Active human-in-the-loop triage for borderline prediction sets.

---

## 26. Citation Information

```bibtex
@article{academic_integrity_multiview2026,
  title={An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection},
  author={Academic Integrity Research Team},
  year={2026},
  journal={Academic Integrity Research Archive},
  url={https://github.com/chetan926/ml}
}
```
