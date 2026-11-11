# Critical Research Limitations & Threats to Validity

**Project Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Document Type:** Formal Research Limitations Report  
**Target Audience:** Academic Faculty Evaluation Committee & Peer Reviewers  
**Date:** November 2026  

---

## 1. Introduction

Scientific integrity requires an unsparing, transparent accounting of research limitations, theoretical assumptions, and empirical boundary conditions. This document outlines the critical methodological, algorithmic, and dataset limitations of the **Typed-Decision Pre-Filtering Pipeline**, identifying avenues for future research.

---

## 2. Dataset Limitations & Proxy Conflation

### 2.1 Sentence-Level Inference Proxy vs. Authentic Academic Essay Plagiarism
- **The SNLI Benchmark Proxy:** As documented in [`experiments/results/dataset_provenance_audit.md`](file:///d:/ml%20rearcch%20project%20ad/experiments/results/dataset_provenance_audit.md), the natural language text dataset is derived from the Stanford Natural Language Inference (SNLI) corpus (Flickr30k captions), repackaged as the "MIT Plagiarism Detection Dataset" on Kaggle.
- **Threat to External Validity:**
  - In SNLI, reference sentences average $13.9$ words and suspicious sentences average $7.5$ words.
  - Authentic academic plagiarism operates on multi-sentence paragraphs, essay sections, or thesis chapters ($50$ to $1,000+$ words) exhibiting complex discourse structure, citations, and structural section borrowing.
  - While SNLI serves as an effective benchmark for evaluating sentence-level lexical vs. contextual semantic similarity under synonymy, it does not evaluate macro-level document structure, citation manipulation, or patchwriting.

### 2.2 The Zero-Overlap Lexical Ceiling
- In the SNLI validation partition, **10.57% of positive pairs share zero content words** (TF-IDF score = $0.0000$).
- *Implication:* Any lexical pre-filter with a non-zero threshold $\theta_{low} > 0$ will inevitably prune these pairs. This imposes an intrinsic theoretical ceiling of $\approx 89.4\%$ on lexical candidate recall for this dataset. Full essay benchmarks with longer text windows would exhibit higher baseline n-gram overlap even under paraphrasing.

### 2.3 Introductory Programming Scope in IR-Plag
- The IR-Plag Java Corpus consists of introductory undergraduate programming assignments (e.g., printing messages, matrix multiplication, calendar calculation, prime sieving, string palindrome, sorting).
- *Limitation:* Introductory assignments are compact ($20$ to $50$ lines of code). Modern university assignments in upper-division software engineering or systems courses involve multi-class architectures, design patterns, third-party libraries, and multi-file packages ($500$ to $5,000+$ lines). The efficacy of token-level pre-filtering on massive object-oriented codebases remains to be empirically verified.

---

## 3. Algorithmic & Methodological Limitations

### 3.1 Language-Specific Syntactic Lexing
- The source code feature extractor in [`src/code_preprocessing.py`](file:///d:/ml%20rearcch%20project%20ad/src/code_preprocessing.py) and [`src/code_plagiarism.py`](file:///d:/ml%20rearcch%20project%20ad/src/code_plagiarism.py) is tailored to the **Java programming language** (leveraging 50 Java reserved keywords, standard syntax delimiters, and comment conventions).
- *Generalizability Constraint:* Applying the pipeline to Python, C++, Go, or Rust requires language-specific tokenizers, AST parsers, and keyword lexicons. Cross-language plagiarism (e.g., a student translating a Java algorithm to Python) was not evaluated.

### 3.2 Resilience Breakdown Under Compound Obfuscation (Level 6 Attacks)
- While the SVM detector achieved **100.00% recall on Levels 2, 3, 4, and 5** (identifier renaming, statement reordering, loop transformations, method abstraction), detection recall dropped to **77.78% on Level 6** (compound multi-transform attacks).
- *Root Cause:* When an adversary simultaneously combines identifier renaming, control flow replacement (`for` $\leftrightarrow$ `while`), statement reordering, and modifier replacement, static token distributions and cyclomatic proxies are substantially altered.
- *Remediation Path:* Future extensions must incorporate graph-based structural representations, such as Control Flow Graph (CFG) embeddings or Program Dependence Graph (PDG) Graph Neural Networks (GNNs).

### 3.3 Scalability Beyond $O(N^2)$ Pairwise Comparisons
- Pre-filtering reduces heavy model forward passes by **29.97% on text and 36.67% on code**, delivering substantial compute savings.
- However, the pre-filter itself is still evaluated on all $\frac{N(N-1)}{2}$ pairs.
- For a cohort of $N = 500$ students, $\approx 124,750$ pairs must be evaluated by the cheap pre-filter.
- For massive cohorts ($N > 10,000$, yielding $\approx 50,000,000$ pairs), even lightweight pairwise pre-filtering becomes a computational bottleneck. In such hyper-scale settings, Locality-Sensitive Hashing (LSH) or sub-quadratic inverted index clustering must precede the pre-filtering stage.

---

## 4. Institutional Deployment & Ethical Boundaries

### 4.1 Risk of Algorithmic Bias on ESL / Non-Native Student Writing
- Students writing in English as a Second Language (ESL) often exhibit narrower vocabulary distributions, higher repetition of template phrases, and reliance on formal connective phrases.
- *Threat:* Lexical and semantic similarity pre-filters may produce elevated similarity scores for ESL submissions due to boilerplate overlap rather than academic dishonesty.
- *Safety Safeguard:* High-confidence pre-filtering thresholds ($\theta_{high}$) must be conservatively set, and escalation to deep contextual models must be maintained for all uncertain cases.

### 4.2 Ethical Stance: Decision Support vs. Automated Punishment
- Automated similarity scores **must never be used to issue automated punitive sanctions, disciplinary actions, or academic penalties against students**.
- The proposed pipeline is strictly engineered as an **evidence-generating academic decision-support tool**:
  - `LOW_RISK_PRUNED`: Saves institutional compute by bypassing unnecessary review.
  - `UNCERTAIN_ESCALATED`: Invocations of high-fidelity models provide granular similarity evidence.
  - `HIGH_RISK_FLAGGED`: Flags pairs for mandatory, context-sensitive human review by course instructors.
- Every student facing plagiarism scrutiny must be afforded procedural due process, access to the generated comparison evidence, and the right to explain their independent thought process.

---

## 5. Summary Matrix of Limitations & Future Directions

| Dimension | Current Implementation Limit | Proposed Future Direction |
| :--- | :--- | :--- |
| **Text Scale** | Sentence-level SNLI proxy pairs ($7$–$14$ words) | Evaluation on full-length academic essays (e.g., PAN Plagiarism Corpus) |
| **Code Obfuscation** | Level 6 compound attacks drop to 77.78% recall | Integration of Control Flow Graph (CFG) Graph Neural Networks |
| **Language Coverage** | English text and Java source code | Multi-lingual transformer embeddings and cross-language AST parsers |
| **Extreme Cohort Scale** | Pairwise comparison still requires $O(N^2)$ pre-filter checks | Multi-tier MinHash / LSH indexing before tri-band escalation |
| **Governance** | Similarity scoring only | Explainable AST diff generation and instructor-facing provenance reports |

