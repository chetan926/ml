# Recent (2024–2026) Research in Plagiarism Detection — Journal Papers Only

**Scope.** Peer-reviewed journal articles published 2024–2026 relevant to (a) text plagiarism detection and
(b) Java source-code plagiarism / code-clone detection. Per the standing constraint, **only journal papers**
are listed. Conference proceedings, workshop papers, preprints (e.g. arXiv), theses, and blogs were
examined during the literature sweep and are explicitly excluded below.

All venues below are scholarly journals carrying an ISSN and a DOI.

---

## A. Text Plagiarism Detection (Journal Papers, 2024–2026)

| # | Citation (journal) | DOI | Focus / method | Key finding relevant to this work |
|---|---|---|---|---|
| T1 | Amirzhanov, A., Turan, C., & Makhmutova, A. (2025). Plagiarism types and detection methods: a systematic survey of algorithms in text analysis. *Frontiers in Computer Science*, 7:1504725. | 10.3389/fcomp.2025.1504725 | Systematic survey of verbatim, paraphrase, translation, and idea-based text plagiarism; string-matching, ML, NLP, and deep-learning detectors. | Computational cost and scalability remain key challenges; lexical/shingle methods miss paraphrase and AI-generated plagiarism. |
| T2 | Sajid, M., Sanaullah, M., Fuzail, M., Malik, T. S., & Shuhidan, S. M. (2025). Comparative analysis of text-based plagiarism detection techniques. *PLoS ONE*, 20(4): e0319551. | 10.1371/journal.pone.0319551 | Comparative evaluation of text-based plagiarism detection techniques across common datasets; lexical, syntax, and deep-learning families. | Summarizes SOTA techniques and dataset usage; quality assessment restricted to DOI-bearing, Google Scholar-indexed primary research. |
| T3 | Singh, M., & Gupta, V. (2026). A novel semantic–syntactic hybrid plagiarism detection system based on word embeddings and similarity measures. *Knowledge and Information Systems*, 68:222. | 10.1007/s10115-026-02837-8 | Hybrid extrinsic text detector combining Word2Vec/GloVe/FastText/BERT semantic vectors with syntactic similarity features; evaluated on PAN-PC-11, PAN-12, PAN-14. | Combined semantic + syntactic scoring outperforms single-view embeddings; reports PlagDet, precision, recall, F-measure, granularity across thresholds. |
| T4 | Iqbal, H. R., Sharjeel, M., Shafi, J., et al. (2025). Urdu paraphrased text reuse and plagiarism detection using pre-trained large language models and deep hybrid neural networks. *Multimedia Tools and Applications*, 84, 43475–43497. | 10.1007/s11042-025-20862-7 | Deep hybrid CNN–LSTM architecture with pre-trained LLM input; word- and sentence-embedding variants on Urdu paraphrase corpora. | Sentence-embedding variant reached F1 ≈ 95.15 on simulated paraphrase cases; illustrates LLM-encoder strength on paraphrased text. |
| T5 | Pudasaini, S., Miralles-Pechuán, L., Lillis, D., et al. (2025). Survey on AI-generated plagiarism detection: The impact of large language models on academic integrity. *Journal of Academic Ethics*, 23, 1137–1170. | 10.1007/s10805-024-09576-x | Survey of AI-generated plagiarism, detection algorithms, datasets, evasion strategies, and LLM-based countermeasures. | Identifies LLM-generated text as an emerging evasion frontier requiring dedicated detection capability. |

---

## B. Java Source-Code Plagiarism / Code-Clone Detection (Journal Papers, 2024–2026)

| # | Citation (journal) | DOI | Focus / method | Key finding relevant to this work |
|---|---|---|---|---|
| C1 | Quradaa, F. H., Shahzad, S., Saeed, R., & Sufyan, M. M. (2024). A novel code representation for detecting Java code clones using high-level and abstract compiled code representations. *PLoS ONE*, 19(5): e0302333. | 10.1371/journal.pone.0302333 | Hybrid Java clone representation: AST (high-level) + Soot Baf/Jimple PDG bytecode IR (low-level); 15 ML classifiers; LightGBM best; BigCloneBench. | Combined high-level + low-level features linearly outperform single representations; ensemble classifiers (LightGBM) excel on semantic clones. |
| C2 | Glani, Y., & Ping, L. (2024). CodeGuard: enhancing accuracy in detecting clones within java source code. *Frontiers in Computer Science*, 6:1455860. | 10.3389/fcomp.2024.1455860 | Level-by-level abstraction for Type-II clones and flexible signature matching for Type-III ST/MT clones; xxHash64 chunk signatures with CloneVault indexing. | Strong Type-II/Type-III clone detection; chunk-signature indexing avoids whole-database search. |
| C3 | Rong, Y., & Zhou, Y. (2025). Quantifying cross-language code reuse via function-level clone detection. *Journal of King Saud University Computer and Information Sciences*, 37:327. | 10.1007/s44443-025-00362-2 | Unified AST + semantic-annotation representation and Siamese deep model for cross-language (Java–Python) clone detection; BigCloneBench + XLCoST. | Precision 96.7% / Recall 92.4% / F1 94.5% on Java BigCloneBench; reuse-ratio quantification supports plagiarism measurement. |
| C4 | Wu, Z. (2026). A comprehensive study of LLM-based code clone detection. *Applied and Computational Engineering*, 227, 46–56. | 10.54254/2755-2721/2026.GU32229 | Survey and empirical analysis of prompt engineering, fine-tuning, and RAG for LLM-based Java code clone detection. | LLM-based approaches (fine-tuning/RAG) raise clone-detection accuracy and stability; Java-focused empirical evaluation. |

---

## C. Excluded Source Types (per "journal only" constraint)

The following source types surfaced during the literature sweep and were **deliberately excluded** because they are not peer-reviewed journal articles:

- **Conference proceedings / workshops:** ICSSSM 2025 (Gawade et al., TF-IDF + Sentence-BERT hybrid), IMACSI 2025 (Koka et al., TF-IDF + Word2Vec ensemble), ITiCSE 2024 (Ebrahim & Joy, CodeBERT/UniXcoder/PLBART/GraphCodeBERT on SOCO), WSEAS 2024 (MLP + Code2Vec/TreeBert on 14M Java examples).
- **Preprints:** Kamat et al. (2024), arXiv:2412.06241 ("Plagiarism Detection Using Machine Learning") — not journal-published.
- **Blogs / industry benchmarks:** Codequiry (2026) MOSS/JPlag/AI-detector obfuscation benchmark — not a scholarly journal article.

---

**Verification statement.** Every entry in Sections A and B is a peer-reviewed journal article published in 2024, 2025, or 2026, identifiable by a journal name, volume/article number, year, and DOI. No conference paper, preprint, thesis, or blog is included.
