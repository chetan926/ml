# Novelty, Literature Analysis & External Validity Assessment (Version 3.6)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Document Version:** Version 3.6  
**Date:** November 10, 2026  
**Status:** METHODOLOGICALLY POSITIONED & SCHOLARLY AUDITED

---

## 1. Positioning Against State-of-the-Art Literature

Existing academic integrity detection tools and scholarly literature divide sharply into two divergent paradigms:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 PRIOR ART LANDSCAPE                    │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
┌──────────────────────────────┐                            ┌──────────────────────────────┐
│ Traditional Heuristic Tools  │                            │ Heavy Deep Neural Models     │
│ (Winnowing, Moss, JPlag)     │                            │ (Cross-Encoders, CodeBERT)   │
├──────────────────────────────┤                            ├──────────────────────────────┤
│ • Ultra-low latency          │                            │ • High semantic recall       │
│ • Zero paraphrase robustness │                            │ • O(M²) computational cost   │
│ • 0% recall on 119 zero-     │                            │ • Uncalibrated probabilities │
│   overlap positive text pairs│                            │ • Prohibitive energy waste   │
└──────────────┬───────────────┘                            └──────────────┬───────────────┘
               │                                                           │
               └─────────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │           PROPOSED FRAMEWORK              │
                       ├───────────────────────────────────────────┤
                       │ • Multi-View Fast Candidate Screening     │
                       │ • Pareto Budget-Constrained Escalation    │
                       │ • Calibrated Posteriors (Platt Scaling)   │
                       │ • Group Split Conformal Prediction (1-ε)  │
                       │ • Interpretable Multi-Modal Evidence      │
                       └───────────────────────────────────────────┘
```

### 1.1 Academic Text Plagiarism Detection
- **Lexical and Fingerprinting Methods:** Algorithms such as Winnowing (Schleimer et al., SIGMOD 2003) and Karp-Rabin string matching form the backbone of commercial systems (Turnitin, CopyLeaks). While computationally efficient ($\mathcal{O}(L)$ string hashing), our empirical audit proves their fundamental fragility: in [text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv), **119 positive text pairs share 0% lexical overlap**, rendering lexical fingerprinting completely blind to semantic paraphrase.
- **Neural Semantic Encoders:** Dense bi-encoders (Reimers & Gurevych, EMNLP 2019) capture semantic similarity but struggle with fine-grained lexical substitutions and structural reordering. Monolithic cross-encoders achieve peak accuracy but require quadratic evaluation ($\mathcal{O}(M^2)$ forward passes), creating an intractable bottleneck in institutional cohorts ($M=1,000 \implies 500,000$ comparisons).

### 1.2 Source Code Plagiarism and Clone Detection
- **Token-Based Systems (JPlag, Moss):** JPlag (Prechelt et al., 2002) and Moss (Aiken, 1994) rely on Greedy String Tiling (RKR-GST) and token winnowing. While effective against naive renaming (L1), they degrade severely under structural loop transformations (L4) and method refactoring (L5).
- **Tree-Based and Graph-Based Encoders:** ASTNN (Zhang et al., ICSE 2019) and CodeBERT (Feng et al., EMNLP 2020) capture syntactic hierarchy but incur massive memory footprints and AST construction overheads.

### 1.3 Conformal Prediction in Information Retrieval
Split conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021) has seen widespread adoption in computer vision and medical imaging, but remains virtually unapplied to academic integrity detection. Our framework represents the first formulation of **group-level split conformal prediction over clustered premise structures**, guaranteeing coverage over non-independent submission slices while explicitly characterizing finite-sample attainability bounds.

---

## 2. External Validity Assessment: SNLI as an Academic Integrity Proxy

In rigorous research preparing for top-tier Q1 journals (e.g., IEEE Transactions on Knowledge and Data Engineering, ACM Computing Surveys, or Information Sciences), methodological honesty regarding benchmark datasets is paramount.

### 2.1 The Nature of the SNLI Text Benchmark
The text benchmark utilized in this repository ([text_plagiarism_benchmark.csv](file:///D:/ml%20rearcch%20project%20ad/datasets/processed/text_plagiarism_benchmark.csv)) is constructed from the Stanford Natural Language Inference (SNLI) corpus (Bowman et al., EMNLP 2015), where:
- Entailment pairs are mapped to positive plagiarism candidates ($y=1$).
- Contradiction / neutral pairs are mapped to negative non-plagiarized candidates ($y=0$).
- All pairs sharing a reference premise sentence form a natural thematic group.

### 2.2 Threats to External Validity
Reviewers in high-impact journals will scrutinize whether results on SNLI transfer to real-world academic plagiarism. We formally document the following limitations:

1. **Syntactic and Structural Shift:**
   - Real student submissions consist of multi-paragraph essays, literature reviews, or thesis chapters ($500$–$10,000$ words).
   - SNLI instances consist of isolated sentence pairs ($10$–$35$ words). Document-level discourse flow, paragraph reordering, and section restructuring are absent.
2. **Topical and Stylistic Domain Gap:**
   - SNLI premises originate from Flickr30k image captions depicting everyday physical scenes ("A dog catching a frisbee in a park").
   - Authentic academic plagiarism involves dense technical jargon, formal discourse markers, mathematical equations, and citation syntax ("According to Smith et al. (2021), transformer attention scales quadratically...").
3. **Conceptual Entailment vs. Plagiarism Intent:**
   - Entailment denotes logical implication, which is an imperfect proxy for academic misconduct. Two students may independently describe the same experimental fact using identical semantic facts without committing plagiarism.

### 2.3 Scientific Justification for Proxy Benchmarking
Despite these domain limitations, SNLI remains a scientifically valid and necessary proxy testbed for initial algorithmic validation:
- **Extreme Paraphrase Stress-Testing:** SNLI contains human-generated paraphrases exhibiting extreme lexical dissimilarity while preserving semantic meaning, creating the rigorous **119 zero-overlap positive pairs** required to test multi-view fusion.
- **Clustered Group Structure:** The shared premise topology provides an authentic multi-candidate evaluation space with intrinsic intra-group dependencies, allowing rigorous testing of **group-level conformal prediction** and cross-split contamination elimination.
- **Reproducibility and Standardization:** SNLI is an open, standardized benchmark enabling independent verification without FERPA/GDPR privacy hurdles associated with proprietary student essay databases.

### 2.4 Mandatory Manuscript Positioning Guidelines
In any manuscript submitted for journal review, the authors must adhere to the following positioning:
> *"The text experiments in this study utilize a clustered semantic inference proxy corpus (derived from SNLI) to rigorously evaluate multi-view candidate screening and group-level conformal calibration under controlled conditions of extreme paraphrase and zero lexical overlap. While this benchmark demonstrates algorithmic efficacy under severe lexical shifts, full institutional deployment requires fine-tuning on domain-specific academic corpora containing multi-paragraph student essays and academic citations."*

