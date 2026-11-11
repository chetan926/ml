# Experimental Protocol and Evaluation Framework (Version 3.2)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.2 (Phase 2.3.1 Independent Audit Revision)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Finalized Experimental Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## 1. Group-Aware Data Partitioning Protocols & Leakage Audit

To guarantee scientific validity and prevent data contamination, all data splits enforce strict group-level boundaries.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FULL BENCHMARK DATASET                                            │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
         ┌───────────────────┬──────────────────────┴───────────────┬───────────────────┐
         ▼                   ▼                                      ▼                   ▼
┌─────────────────┐ ┌─────────────────┐                    ┌─────────────────┐ ┌─────────────────┐
│  TRAIN SPLIT    │ │ VALIDATION      │                    │ CALIBRATION     │ │ TEST SPLIT      │
│  (D_train)      │ │ SPLIT (D_val)   │                    │ SPLIT (D_cal)   │ │ (D_test)        │
├─────────────────┤ ├─────────────────┤                    ├─────────────────┤ ├─────────────────┤
│ • Model Fitting │ │ • Threshold     │                    │ • Calibrator    │ │ • Held-out test │
│ • Vocab & Index │ │   Optimization  │                    │   Fitting       │ │   tasks/groups  │
│ • Group Isolated│ │ • Group Isolated│                    │ • Conformal     │ │ • Evaluated ONCE│
│                 │ │ • Frozen Policy │                    │   Quantile      │ │   at the end    │
└─────────────────┘ └─────────────────┘                    └─────────────────┘ └─────────────────┘
```

---

### 1.1 Text Modality Group-Aware Partitioning (SNLI / Pereira Benchmark)

#### Forensic Provenance & Grouping Audit:
In the SNLI text corpus, multiple hypothesis sentences are paired with the exact same reference premise sentence (an average of 3 hypotheses per premise). A naive random split scatters pairs sharing the identical premise across train, validation, and test partitions, creating **premise-level data leakage**.

#### Full Dataset Accounting & Reconciliation (Issue 7):
A forensic audit of `datasets/processed/text_plagiarism_benchmark.csv` and the raw parquet files reveals:
- Total available pairs in corpus: **19,666 pairs**.
  - `snli_val.parquet`: 9,842 pairs across 3,304 unique premises.
  - `snli_test.parquet`: 9,824 pairs across 3,261 unique premises.
- **Cross-File Intersection:** Exactly **one premise sentence** appears in both `snli_val.parquet` and `snli_test.parquet`, resulting in $99.98\%$ premise isolation. Our data loader explicitly excludes this single shared premise sentence during partitioning.
- **Active Experimental Partitions ($10,500$ pairs total):**
  1. Base Training Split ($D_{\text{train}}^{\text{text}}$): $N = 4,000$ pairs ($1,340$ unique premises). Used for vocabulary extraction and candidate model fitting.
  2. Threshold Validation Split ($D_{\text{val}}^{\text{text}}$): $N = 1,500$ pairs ($505$ unique premises, 100% disjoint). Used exclusively for Pareto surface search.
  3. Probability Calibration Split ($D_{\text{cal-prob}}^{\text{text}}$): $N = 1,000$ pairs ($335$ unique premises, 100% disjoint). Used for fitting Platt scaling / Isotonic regression on conditional escalated pairs ($D_{\text{cal-prob}}^{\text{esc}}$).
  4. Conformal Calibration Split ($D_{\text{cal-conf}}^{\text{text}}$): $N = 1,000$ pairs ($335$ unique premises, 100% disjoint). Used for estimating empirical quantile $\hat{q}_{1-\epsilon}$ on conditional escalated pairs ($D_{\text{cal-conf}}^{\text{esc}}$).
  5. Held-Out Test Split ($D_{\text{test}}^{\text{text}}$): $N = 3,000$ pairs ($1,010$ unique premises, derived from `snli_test.parquet` with the shared premise removed). Untouched throughout development; evaluated exactly once.
- **Reserve Unselected Partitions ($9,166$ pairs total):**
  - Development Reserve: $9,842 - 7,500 = 2,342$ unselected development pairs.
  - Test Reserve: $9,824 - 3,000 = 6,824$ unselected test pairs.
  - Total Reserve: $2,342 + 6,824 = 9,166$ pairs held in reserve for sample size sensitivity analysis.

*Recalculation of Empirical Statistics:*
All class proportions, duplicate statistics, and zero-overlap statistics must be re-computed dynamically on the newly partitioned test split. The historical $119$-pair ($10.57\%$) zero-overlap figure from the legacy random split is **not** assumed; it will be formally recalculated and reported on $D_{\text{test}}^{\text{text}}$.

---

### 1.2 Source Code Modality: Task-Disjoint Partitioning & Statistical Limitations

#### Forensic Task & File Structure Audit (Issue 7):
The authentic IR-Plag Java corpus contains 467 source files yielding **733 pairs** across 7 tasks:
- `case-01`: 56 files, 94 pairs (54 non-plag, 40 plag).
- `case-02`: 70 files, 108 pairs (54 non-plag, 54 plag).
- `case-03`: 68 files, 106 pairs (54 non-plag, 52 plag).
- `case-04`: 70 files, 108 pairs (54 non-plag, 54 plag).
- `case-05`: 69 files, 107 pairs (54 non-plag, 53 plag). Contains 1 original file, 15 non-plagiarized files, and 53 plagiarized variants.
- `case-06` (Palindrome): 67 files, 105 pairs (54 non-plag, 51 plag).
- `case-07` (Sorting): 67 files, 105 pairs (54 non-plag, 51 plag).

#### Methodological Constraints & Mandates:
1. **The 15-Group Constraint in `case-05`:** With only 15 reference programs, `case-05` is statistically insufficient to support a valid finite-sample conformal quantile ($N_{\text{ref}} < \lceil (1-\epsilon)/\epsilon \rceil = 19$). Therefore:
   - Conformal prediction is **formally disclaimed for source code**.
   - `case-05` is evaluated via group-aware 5-fold cross-fitting (3 reference programs per fold) for threshold validation and exploratory calibrated routing.
2. **Mandatory Disaggregated Test Reporting (Issue 6):**
   - **`case-06` (Palindrome, $N = 105$) and `case-07` (Sorting, $N = 105$) must be reported SEPARATELY.**
   - Pooled metrics over `case-06` + `case-07` ($N = 210$) may be reported strictly as descriptive summary statistics.
   - **Generalization Disclaimer:** Broad claims about generalizing to unseen programming assignments are strictly prohibited; two tasks cannot establish cross-task generalization.
   - Pair-level McNemar tests on $N = 210$ cannot be cited as proof of task-level generalization.

---

## 2. Irreversible Seven-Step Experimental Lifecycle (Issue 1)

The experimental protocol enforces a strict, irreversible sequence to prevent data leakage and selection bias:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE SEVEN-STEP SEQUENTIAL EXECUTION LIFECYCLE                             │
├──────┬───────────────────────────────────┬───────────────────────────┬─────────────────────────────────┤
│ Step │ Operation                         │ Target Partition          │ Invariant & Safeguard           │
├──────┼───────────────────────────────────┼───────────────────────────┼─────────────────────────────────┤
│ 1    │ Train Candidate & Heavy Models    │ D_train                   │ Group-isolated fitting          │
│ 2    │ Optimize Routing Thresholds       │ D_val                     │ Pareto surface grid search      │
│ 3    │ FREEZE Routing Thresholds         │ (theta_low, theta_high)   │ Thresholds locked               │
│ 4    │ Filter & Fit Probability Calibrat │ D_cal-prob^esc            │ Fitted on conditional slice     │
│ 5    │ Fit Conformal Quantile (Text Only)│ D_cal-conf^esc            │ q_hat estimated via exact order │
│ 6    │ FREEZE Complete Pipeline          │ All components            │ Pipeline fully locked           │
│ 7    │ Single-Pass Final Evaluation      │ D_test                    │ Untouched test split (ONCE)     │
└──────┴───────────────────────────────────┴───────────────────────────┴─────────────────────────────────┘
```

*Crucial Protocol Invariant:* Any modification to routing thresholds $(\theta_{\text{low}}, \theta_{\text{high}})$ invalidates $D_{\text{cal-prob}}^{\text{esc}}$ and $D_{\text{cal-conf}}^{\text{esc}}$, requiring a complete re-run of Steps 4 through 6 before test evaluation.

---

## 3. Comprehensive Evaluation Metrics Suite

### 3.1 Candidate Screening Metrics (Contribution 1)
- **Candidate Recall ($R_{\text{cand}}$):** Fraction of ground-truth positive pairs retained after Tier-1 screening:
  $$R_{\text{cand}} = \frac{\sum_{i \in D_{\text{test}}} y_i \cdot \mathbb{I}(s_{\text{cand}}(\mathbf{x}_i) \ge \theta_{\text{low}})}{\sum_{i \in D_{\text{test}}} y_i}$$
- **Pruning Rate ($P_{\text{prune}}$):** Fraction assigned `LOW_RISK_REVIEW_DEFERRED` at Tier 1 ($s_{\text{cand}} < \theta_{\text{low}}$).
- **Direct Candidate Flag Rate ($P_{\text{flag}}$):** Fraction assigned `HIGH_RISK_CANDIDATE_FLAGGED` at Tier 1 ($s_{\text{cand}} \ge \theta_{\text{high}}$).
- **Escalation Rate ($P_{\text{esc}}$):** Fraction escalated to Tier 2 ($\theta_{\text{low}} \le s_{\text{cand}} < \theta_{\text{high}}$).

### 3.2 Classification & Detection Metrics (Overall Pipeline)
- **Accuracy, Precision, Recall, Specificity, Macro F1:** Evaluated on final decisions.
- **Selective Accuracy:** Accuracy evaluated across pairs meeting specific uncertainty criteria.
- **Direct Flag Precision:** Precision of pairs flagged directly at Tier 1 ($\ge P_{\text{target}}$).

### 3.3 Uncertainty Quantification & Calibration Diagnostics (Contribution 2, Issue 1)
- **Expected Calibration Error (ECE) & Maximum Calibration Error (MCE):** Evaluated with 10 bins on independent evaluation data ($D_{\text{test}}^{\text{esc}}$).
- **Brier Score & Negative Log-Likelihood (NLL).**
- **Text Conformal Metrics:** Empirical marginal coverage rate ($1 - \epsilon$), Set size distribution ($|\mathcal{C}| \in \{0, 1, 2\}$), and Singleton Efficiency (fraction of singleton sets).
- **Code Uncertainty Metrics:** Selective accuracy and error rates across predictive entropy tiers.

### 3.4 Computational Cost & Latency Metrics (Contribution 3, Issue 5)
- **8 Explicit Latency Stages (Milliseconds):** Ingestion ($T_{\text{prep}}$), Text Vectorization ($T_{\text{vec-text}}$), Code Tokenization ($T_{\text{tok-code}}$), AST Parsing ($T_{\text{ast}}$), Cache Lookup ($T_{\text{cache}}$), Candidate Scorer ($T_{\text{Tier1}}$), Heavy Inference ($T_{\text{infer-heavy}}$), Evidence Synthesis ($T_{\text{explain}}$), End-to-End Latency ($T_{\text{e2e}}$).
- **Resource Overhead (Megabytes):** Process Resident Set Size ($M_{\text{RSS}}$ via `psutil`), Cache Footprint ($M_{\text{cache}}$), Model Memory ($M_{\text{model}}$).
- **Cold-Start vs. Warm-Cache Latency:** Reported separately.

### 3.5 Explainability Quality Metrics (Contribution 4)
- **Evidence Faithfulness (Probability Drop):** $\Delta \hat{P}_{\text{faith}} = \hat{P}(Y=1 \mid \mathbf{x}) - \hat{P}(Y=1 \mid \mathbf{x} \setminus \mathcal{E}(\mathbf{x}))$.
- **Evidence Coverage:** Fraction of suspicious submission tokens covered by evidence spans.
- **Alignment Precision:** Fraction of detected identifier substitutions matching verified transformations.

---

## 4. Statistical Hypothesis Testing & Significance Protocol (Issue 6)

### 4.1 Paired Classifier Significance Testing
To rigorously compare the proposed multi-stage cascade against baselines:
- **Discordant Cell Count ($b + c$):** Let $b$ denote pairs where Proposed Model is correct and Baseline is wrong; let $c$ denote pairs where Proposed Model is wrong and Baseline is correct.
- **Asymptotic McNemar Test ($b + c \ge 25$):**
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1$$
- **Exact Binomial McNemar Test ($b + c < 25$):** Mandatory for small subsets (`case-06` and `case-07`):
  $$p = 2 \cdot \sum_{k=0}^{\min(b, c)} \binom{b + c}{k} \left(\frac{1}{2}\right)^{b + c}$$

### 4.2 Group-Aware Cluster Bootstrap Confidence Intervals
- All 95% confidence intervals are computed via **1,000-resample cluster bootstrap**, resampling unique group clusters (`reference_text` premises for text; assignment tasks for code) with replacement.
- Pair-level bootstrap is strictly prohibited as it underestimates variance under group correlation.

### 4.3 Multiple Testing Correction
- False Discovery Rate (FDR) is controlled across all baseline comparisons and ablation configurations using the **Benjamini-Hochberg procedure** at $\alpha = 0.05$.

---

## 5. Baseline Comparison Framework

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 STANDARDIZED BASELINE BENCHMARKS                                       │
├──────────┬─────────────────────────────┬────────────────────────────────┬──────────────────────────────┤
│ Modality │ Baseline Identifier         │ Scientific Source Reference    │ Representation & Model       │
├──────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ Text     │ Lexical TF-IDF Pre-Filter   │ Sajid et al. (2025, PLOS ONE)  │ Sublinear TF-IDF Cosine      │
│ Text     │ Dense SBERT Bi-Encoder      │ Reimers & Gurevych (2019)      │ `all-MiniLM-L6-v2` Cosine    │
│ Text     │ Cross-Encoder Transformer   │ Devlin et al. (2019)           │ Pairwise Joint Attention     │
├──────────┼─────────────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ Code     │ ES-Plag String Tiling       │ Sulistiani & Karnalim (2019)   │ Dynamic Length Pre-Filter    │
│ Code     │ Syntactic RBF-SVM           │ Gandhi et al. (2024, Front CS) │ 7 Syntactic Features + SVM   │
│ Code     │ AST Random Forest Ensemble  │ Mehsen & Joshi (2024, IJC)     │ 7 AST Features + RF Ensemble │
└──────────┴─────────────────────────────┴────────────────────────────────┴──────────────────────────────┘
```

---

## 6. Comprehensive Ablation Study Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     TWELVE PLANNED ABLATION STUDIES                                    │
├────┬─────────────────────────────┬────────────────────────────────┬────────────────────────────────────┤
│ #  │ Ablation Experiment         │ Condition Evaluated            │ Target Hypothesis Tested           │
├────┼─────────────────────────────┼────────────────────────────────┼────────────────────────────────────┤
│ A1 │ Single-View TF-IDF Only     │ Remove views v2, v3, v4 (Text) │ Drops candidate recall on 0-overlap│
│ A2 │ Character 3-Grams Only      │ Remove views v1, v3, v4 (Text) │ Evaluates morphological recall     │
│ A3 │ Raw Code Tokens Only        │ Remove views u2, u3, u4 (Code) │ Fails under identifier renaming    │
│ A4 │ Normalized Code Tokens Only │ Remove views u1, u3, u4 (Code) │ Isolates alpha-renaming robustness │
│ A5 │ No Probability Calibration  │ Raw margin z -> Sigmoid        │ Measures ECE degradation           │
│ A6 │ Platt vs. Isotonic          │ Parametric vs. Non-parametric  │ Compares calibration fit on slices │
│ A7 │ Fixed Arbitrary Thresholds  │ (0.20, 0.80) without Pareto    │ Measures budget violation / recall │
│ A8 │ Marginal vs. Slice Calib    │ Calibrate on D_cal vs D_cal^esc│ Proves selection bias shift        │
│ A9 │ Conformal vs. Fixed Band    │ Conformal Set vs. p > 0.50     │ Demonstrates error control         │
│ A10│ No Document Caching         │ Pairwise re-vectorization      │ Quantifies O(M^2) latency explosion│
│ A11│ Cold-Start vs. Warm-Cache   │ First run vs. Warm dict lookup │ Establishes deployment latency     │
│ A12│ Evidence Ablation Drop      │ Mask aligned spans             │ Validates explanation faithfulness │
└────┴─────────────────────────────┴────────────────────────────────┴────────────────────────────────────┘
```

---
*End of Document: experimental_protocol_v3_2.md*

