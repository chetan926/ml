# Literature Methodology Verification and Citation Audit

## 1. Selected Research Methodologies

| Paper ID | Authors & Year | Venue & DOI | Core Methodology | Adaptation & Role |
| :--- | :--- | :--- | :--- | :--- |
| **M1: SBERT Semantic** | Setu et al. (2025) | *J. Umm Al-Qura Univ. Eng. Archit.*, DOI: 10.1007/s43995-025-00108-1 | Dense contextual Sentence-BERT embeddings (all-MiniLM-L6-v2) with cosine distance. | Implemented as Method 1 (Semantic Baseline) and heavy comparator in Typed-Decision. |
| **M2: Setu Hybrid** | Setu et al. (2025) | *J. Umm Al-Qura Univ. Eng. Archit.*, DOI: 10.1007/s43995-025-00108-1 | Hybrid lexical-semantic similarity `max(TF-IDF, SBERT)`, categorizing direct copies vs. paraphrases. | Implemented as Condition E1 (Paper-Faithful Full Hybrid) and Condition E2 (Proposed Cost-Efficient Staged Hybrid). |
| **M3: SVM Code** | Gandhi et al. (2024) | *Frontiers in Computer Science*, DOI: 10.3389/fcomp.2024.1393723 | Multi-dimensional AST and token metric extraction classified by RBF Support Vector Machine. | Faithfully adapted from Python to Java on the IR-Plag benchmark. |
| **M4: Random Forest** | Mehsen & Joshi (2024) | *Int. J. Computing*, DOI: 10.47839/ijc.23.1.3438 | Machine learning classification using Random Forest ensemble over syntactic & token vectors. | Implemented on IR-Plag Java benchmark as ensemble baseline. |
| **M5: ES-Plag** | Sulistiani & Karnalim (2019) | *Comput. Appl. Eng. Educ.*, DOI: 10.1002/cae.22066 | Fast token cosine indexing with dynamic thresholding + Running-Karp-Rabin Greedy-String-Tiling (m=2) + penalty normalization. | Implemented in `src/code/` with Penalized Max and Penalized Average metrics. |
| **B1: TF-IDF** | Salton & Buckley (1988); Sajid (2025) | *IP&M* / *PLOS ONE*, DOI: 10.1371/journal.pone.0319551 | Sublinear TF-IDF vectorization with cosine distance. | Traditional lexical baseline and Tier-1 candidate screening filter. |
| **B2: Token Jaccard** | Faidhi & Robinson (1987) | *Comput. Educ.*, DOI: 10.1016/0360-1315(87)90042-X | Set-theoretic token overlap without structural weighting. | Traditional source code baseline and Tier-1 candidate screening filter. |

## 2. Research Contributions and Modality Isolation
All methodologies are evaluated strictly under modality-isolated conditions:
1. Natural Language Text Modality: MIT / SNLI proxy benchmark.
2. Source Code Modality: IR-Plag Java Corpus across Faidhi-Robinson obfuscation levels (L1 to L6).
3. Candidate Pre-Filtering Evaluation: Evaluated under Recall-Constrained Optimization (target recall >= 0.95 and >= 0.98) with random filtering controls.
