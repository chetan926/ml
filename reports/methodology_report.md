# Methodology and Algorithmic Workflow Report

**Project Title:** Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines  
**Document:** Methodology & Theoretical Architecture  
**Date:** November 2026  

---

## 1. Architectural Foundations

The core insight of **Typed-Decision Pre-Filtering** is that pairwise similarity evaluation across multi-modal academic submissions must not treat all candidate pairs uniformly. By incorporating input type routing and dividing candidate pairs into risk bands, computational resources are focused exclusively on ambiguous cases.

### Mathematical Formulation of Modality Routing
Given a candidate document pair $(D_A, D_B)$, the routing function $\mathcal{R}(D_A, D_B)$ maps to an input modality:

$$\mathcal{R}(D_A, D_B) = \begin{cases} 
\text{SourceCode}, & \text{if } \text{ContainsJavaSyntax}(D_A \cup D_B) \lor \frac{\sum_{w \in D_A \cup D_B} \mathbb{I}(w \in \mathcal{K}_{Java})}{|D_A \cup D_B|} > \gamma_{kw} \\
\text{NaturalText}, & \text{otherwise}
\end{cases}$$

where $\mathcal{K}_{Java}$ is the set of 50 reserved Java language keywords, and $\gamma_{kw} = 0.08$ is the empirical keyword prevalence threshold.

---

## 2. Text Plagiarism Methodologies

### 2.1 Method 1: Lexical TF-IDF Vectorization with Cosine Metric (Sajid et al., 2025)
Given vocabulary $\mathcal{V}$ of unigrams and bigrams:
$$w_{i, d} = (1 + \log(\text{tf}_{i, d})) \cdot \log\left(\frac{1 + N}{1 + \text{df}_i}\right)$$
For reference text $T_{ref}$ and suspicious text $T_{sus}$, with normalized L2-vectors $\mathbf{v}_{ref}$ and $\mathbf{v}_{sus}$:
$$S_{lex}(T_{ref}, T_{sus}) = \cos(\mathbf{v}_{ref}, \mathbf{v}_{sus}) = \frac{\mathbf{v}_{ref} \cdot \mathbf{v}_{sus}}{\|\mathbf{v}_{ref}\|_2 \|\mathbf{v}_{sus}\|_2}$$

### 2.2 Method 2: Contextual SBERT Semantic Embeddings (Setu et al., 2025)
Given text $T$, the transformer encoder $\mathcal{E}$ generates hidden states $\mathbf{H} = \mathcal{E}(T) \in \mathbb{R}^{L \times d_{model}}$. Mean pooling over attention mask $\mathbf{m} \in \{0, 1\}^L$ produces sequence representation $\mathbf{e}$:
$$\mathbf{e} = \frac{\sum_{j=1}^L \mathbf{H}_j \cdot m_j}{\sum_{j=1}^L m_j}, \quad \hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$
Semantic similarity is calculated as:
$$S_{sem}(T_{ref}, T_{sus}) = \hat{\mathbf{e}}_{ref} \cdot \hat{\mathbf{e}}_{sus}$$

---

## 3. Source Code Plagiarism Methodology (Gandhi et al., 2024 Adapted to Java)

For Java submissions $(C_A, C_B)$, seven invariant features are extracted:

1. **Raw Token Jaccard ($f_1$):**
   $$f_1 = \frac{|\mathcal{T}(C_A) \cap \mathcal{T}(C_B)|}{|\mathcal{T}(C_A) \cup \mathcal{T}(C_B)|}$$
2. **Anonymized Token Jaccard ($f_2$):**
   $$f_2 = \frac{|\mathcal{T}_{anon}(C_A) \cap \mathcal{T}_{anon}(C_B)|}{|\mathcal{T}_{anon}(C_A) \cup \mathcal{T}_{anon}(C_B)|}$$
   where student-defined identifiers are converted to canonical tokens $\text{ID}_k$.
3. **Keyword Cosine Similarity ($f_3$):**
   $$f_3 = \frac{\mathbf{k}_A \cdot \mathbf{k}_B}{\|\mathbf{k}_A\|_2 \|\mathbf{k}_B\|_2}$$
   where $\mathbf{k} \in \mathbb{R}^{50}$ represents the frequency distribution over Java keywords.
4. **Operator Jaccard Index ($f_4$):**
   Overlap of the 26 Java arithmetic, relational, and logical operators.
5. **Clean Line Ratio ($f_5$):**
   $$f_5 = \frac{\min(L_A, L_B)}{\max(L_A, L_B)}$$
   where comments and blank lines are stripped.
6. **Cyclomatic Complexity Proxy ($f_6$):**
   $$f_6 = 1.0 - \frac{|CC(C_A) - CC(C_B)|}{\max(CC(C_A), CC(C_B))}$$
   where $CC = 1 + \sum \text{branching tokens}$.
7. **Character Trigram Jaccard ($f_7$):**
   Overlap of whitespace-collapsed 3-character n-grams.

The feature vector $\mathbf{x} = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]^T$ is standardized and passed to an RBF Support Vector Machine:
$$P(\text{Plagiarized} \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f_{svm}(\mathbf{x}) + B)}$$

---

## 4. Proposed Tri-Band Decision Policy

For cheap pre-filter score $S_{filter}$ and validation-tuned thresholds $[\theta_{low}, \theta_{high}]$:

$$\mathcal{D}(D_A, D_B) = \begin{cases}
\text{Low Risk (Prune)}, & S_{filter} < \theta_{low} \implies \hat{y} = 0 \\
\text{Uncertain (Escalate)}, & \theta_{low} \le S_{filter} < \theta_{high} \implies \hat{y} = \mathbb{I}(S_{heavy} \ge \tau_{heavy}) \\
\text{High Risk (Direct Flag)}, & S_{filter} \ge \theta_{high} \implies \hat{y} = 1 \text{ (Triage to Human Review)}
\end{cases}$$

---

## 5. Algorithmic Pseudocode

```python
def typed_decision_pipeline(pair, theta_low, theta_high, tau_heavy):
    modality = route_modality(pair)
    
    if modality == 'text':
        s_filter = compute_tfidf_cosine(pair.ref, pair.sus)
        if s_filter < theta_low:
            return Decision(risk='LOW', pred=0, escalated=False)
        elif s_filter >= theta_high:
            return Decision(risk='HIGH', pred=1, escalated=False, flag_review=True)
        else:
            s_heavy = compute_sbert_cosine(pair.ref, pair.sus)
            pred = int(s_heavy >= tau_heavy)
            return Decision(risk='UNCERTAIN', pred=pred, escalated=True)
            
    elif modality == 'code':
        s_filter = compute_token_jaccard(pair.ref, pair.sus)
        if s_filter < theta_low:
            return Decision(risk='LOW', pred=0, escalated=False)
        elif s_filter >= theta_high:
            return Decision(risk='HIGH', pred=1, escalated=False, flag_review=True)
        else:
            features = extract_java_svm_features(pair.ref, pair.sus)
            prob = svm_classifier.predict_proba(features)
            pred = int(prob >= tau_heavy)
            return Decision(risk='UNCERTAIN', pred=pred, escalated=True)
```

