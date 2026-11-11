# Algorithms and Pseudocode Specifications
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 1.0 (Phase 2 Deliverable)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Complete Algorithmic Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## Algorithm 1: Multi-View Candidate Generation (Contribution 1)

```python
"""
ALGORITHM 1: Multi-View Candidate Generation
Input:
    Pair (D_A, D_B): Raw text or source code submissions
    Modality M in {'text', 'code'}
    Trained Candidate Scorer g_M(v) -> [0, 1]
    Validation-Tuned Cutoff theta_cand
Output:
    Decision in {'PRUNED', 'RETAINED'}
    Multi-View Feature Vector v in R^5
    Candidate Score s_cand in [0, 1]
"""

function GENERATE_CANDIDATE(D_A, D_B, M, g_M, theta_cand):
    if M == 'text':
        # View 1: Sublinear TF-IDF Cosine
        t_A = COMPUTE_SUBLINEAR_TFIDF(D_A)
        t_B = COMPUTE_SUBLINEAR_TFIDF(D_B)
        v_1 = COSINE_SIMILARITY(t_A, t_B)
        
        # View 2: Character 3-Gram Jaccard
        g_A = EXTRACT_CHAR_NGRAMS(D_A, n=3)
        g_B = EXTRACT_CHAR_NGRAMS(D_B, n=3)
        v_2 = JACCARD_SIMILARITY(g_A, g_B)
        
        # View 3: Word Multiset Jaccard
        w_A = TOKENIZE_WORDS(D_A)
        w_B = TOKENIZE_WORDS(D_B)
        v_3 = JACCARD_SIMILARITY(w_A, w_B)
        
        # View 4: Length Ratio Asymmetry
        v_4 = min(len(w_A), len(w_B)) / max(len(w_A), len(w_B), 1)
        
        # View 5: Fast Embedding Proxy
        e_A = MEAN_TOKEN_EMBEDDING(w_A)
        e_B = MEAN_TOKEN_EMBEDDING(w_B)
        v_5 = COSINE_SIMILARITY(e_A, e_B)
        
        v = [v_1, v_2, v_3, v_4, v_5]
        
    elif M == 'code':
        # View 1: Raw Lexical Token Jaccard
        tok_A = EXTRACT_RAW_TOKENS(D_A, strip_comments=True)
        tok_B = EXTRACT_RAW_TOKENS(D_B, strip_comments=True)
        u_1 = JACCARD_SIMILARITY(tok_A, tok_B)
        
        # View 2: Identifier-Normalized Token Jaccard
        norm_A = NORMALIZE_IDENTIFIERS(tok_A)
        norm_B = NORMALIZE_IDENTIFIERS(tok_B)
        u_2 = JACCARD_SIMILARITY(norm_A, norm_B)
        
        # View 3: AST Node Type N-Gram Overlap
        ast_A = PARSE_JAVA_AST(D_A)
        ast_B = PARSE_JAVA_AST(D_B)
        types_A = EXTRACT_NODE_TYPE_BIGRAMS(ast_A)
        types_B = EXTRACT_NODE_TYPE_BIGRAMS(ast_B)
        u_3 = JACCARD_SIMILARITY(types_A, types_B)
        
        # View 4: Operator Profile Cosine
        ops_A = COUNT_JAVA_OPERATORS(tok_A)
        ops_B = COUNT_JAVA_OPERATORS(tok_B)
        u_4 = COSINE_SIMILARITY(ops_A, ops_B)
        
        # View 5: Cyclomatic Complexity Ratio
        cc_A = COMPUTE_CYCLOMATIC_COMPLEXITY(ast_A)
        cc_B = COMPUTE_CYCLOMATIC_COMPLEXITY(ast_B)
        u_5 = 1.0 - abs(cc_A - cc_B) / max(cc_A, cc_B, 1)
        
        v = [u_1, u_2, u_3, u_4, u_5]
    
    # Compute fused candidate score
    s_cand = g_M.predict_proba(v)
    
    if s_cand < theta_cand:
        return 'PRUNED', v, s_cand
    else:
        return 'RETAINED', v, s_cand
end function
```

---

## Algorithm 2: Probability Calibration on Calibration Split (Contribution 2)

```python
"""
ALGORITHM 2: Probability Calibration
Input:
    Calibration Split D_cal = {(x_i, y_i)}_{i=1}^{N_cal}
    Heavy Classifier Decision Function f(x) -> z in R
    Method in {'platt', 'isotonic'}
Output:
    Calibrator model g_cal(z) -> p in [0, 1]
    Calibration metrics: ECE, Brier Score
"""

function FIT_PROBABILITY_CALIBRATOR(D_cal, f, method='isotonic'):
    Z_cal = []
    Y_cal = []
    
    # Step 1: Extract uncalibrated classifier outputs
    for (x_i, y_i) in D_cal:
        z_i = f(x_i)   # Raw decision margin or uncalibrated probability
        Z_cal.append(z_i)
        Y_cal.append(y_i)
    
    # Step 2: Fit selected calibrator
    if method == 'platt':
        # Logistic regression on uncalibrated score: p = 1 / (1 + exp(A*z + B))
        model = LogisticRegression(solver='lbfgs')
        model.fit(reshape(Z_cal, (-1, 1)), Y_cal)
        g_cal = function(z): return model.predict_proba([[z]])[0, 1]
        
    elif method == 'isotonic':
        # Non-parametric piecewise constant step function via PAV
        model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        model.fit(Z_cal, Y_cal)
        g_cal = function(z): return model.predict([z])[0]
    
    # Step 3: Compute validation calibration metrics (ECE, Brier)
    P_cal = [g_cal(z) for z in Z_cal]
    ece = COMPUTE_ECE(P_cal, Y_cal, num_bins=10)
    brier = MEAN_SQUARED_ERROR(P_cal, Y_cal)
    
    return g_cal, ece, brier
end function

function COMPUTE_ECE(probs, labels, num_bins=10):
    bin_boundaries = LINSPACE(0.0, 1.0, num_bins + 1)
    ece = 0.0
    N = len(probs)
    
    for m in range(num_bins):
        low = bin_boundaries[m]
        high = bin_boundaries[m + 1]
        
        # Select indices in bin
        bin_idx = [i for i in range(N) if low < probs[i] <= high]
        bin_size = len(bin_idx)
        
        if bin_size > 0:
            bin_acc = sum(labels[i] for i in bin_idx) / bin_size
            bin_conf = sum(probs[i] for i in bin_idx) / bin_size
            ece += (bin_size / N) * abs(bin_acc - bin_conf)
            
    return ece
end function
```

---

## Algorithm 3: Inductive Conformal Prediction & Adaptive Tri-Band Router (Contribution 2)

```python
"""
ALGORITHM 3: Inductive Conformal Prediction & Adaptive State Router
Input:
    Calibration Predictions and Labels: {(p_i, y_i)}_{i=1}^{N_cal}
    Significance Level epsilon in (0, 1) (e.g., 0.05 for 95% coverage)
    Query Pair (D_A, D_B) with calibrated probability p_query
Output:
    Conformal Set C_{1-eps} in {{0}, {1}, {0, 1}}
    Operational Triage State in {'LOW_RISK_PRUNED', 'UNCERTAIN_ESCALATED', 'HIGH_RISK_FLAGGED'}
    Predictive Entropy H(p)
"""

function CALIBRATE_CONFORMAL_THRESHOLD(P_cal, Y_cal, epsilon=0.05):
    N_cal = len(P_cal)
    scores = []
    
    # Compute non-conformity score: s_i = 1 - P(Y = y_i | x_i)
    for i in range(N_cal):
        p_true_label = P_cal[i] if Y_cal[i] == 1 else (1.0 - P_cal[i])
        s_i = 1.0 - p_true_label
        scores.append(s_i)
    
    scores.sort()
    # Conformal empirical quantile index
    rank_idx = int(ceil((N_cal + 1) * (1.0 - epsilon))) - 1
    rank_idx = min(rank_idx, N_cal - 1)
    q_hat = scores[rank_idx]
    
    return q_hat
end function

function ROUTE_ADAPTIVE_TRI_BAND(p_query, q_hat):
    # Construct conformal prediction set C
    C = set()
    # Test class 0
    if (1.0 - (1.0 - p_query)) <= q_hat:  # p_query <= q_hat
        C.add(0)
    # Test class 1
    if (1.0 - p_query) <= q_hat:          # (1.0 - p_query) <= q_hat <=> p_query >= 1 - q_hat
        C.add(1)
        
    # Compute predictive entropy
    p_safe = clip(p_query, 1e-7, 1.0 - 1e-7)
    entropy = -p_safe * log2(p_safe) - (1.0 - p_safe) * log2(1.0 - p_safe)
    
    # Map conformal set to operational states
    if C == {0}:
        state = 'LOW_RISK_PRUNED'
    elif C == {1}:
        state = 'HIGH_RISK_FLAGGED'
    else: # C == {0, 1}
        state = 'UNCERTAIN_ESCALATED'
        
    return state, C, entropy
end function
```

---

## Algorithm 4: Budget-Constrained Threshold Optimization via Pareto Grid Search (Contribution 3)

```python
"""
ALGORITHM 4: Budget-Constrained Threshold Optimization
Input:
    Validation Set D_val with pre-computed candidate scores, heavy scores, and y_val
    Target Candidate Recall R_target (e.g. 0.98)
    Maximum Latency Budget B_max (in ms/pair)
    Empirical Cost Constants: C_prep, C_Tier1, C_feat_heavy, C_infer_heavy, C_router
Output:
    Optimal Policy theta* = (theta_low*, theta_high*)
    Pareto Frontier Points: List of (Recall, Latency, F1, theta_low, theta_high)
"""

function OPTIMIZE_BUDGET_CONSTRAINED_THRESHOLDS(D_val, R_target, B_max, Costs):
    grid_low = LINSPACE(0.01, 0.40, 40)
    grid_high = LINSPACE(0.60, 0.95, 36)
    
    best_policy = None
    best_latency = float('inf')
    pareto_frontier = []
    
    N = len(D_val)
    pos_mask = [y == 1 for y in D_val.y]
    total_positives = sum(pos_mask)
    
    for theta_l in grid_low:
        for theta_h in grid_high:
            if theta_l >= theta_h:
                continue
                
            # Classify validation pairs into states
            escalated_count = 0
            flagged_count = 0
            pruned_count = 0
            retained_positives = 0
            
            for i in range(N):
                score = D_val.cand_scores[i]
                y_i = D_val.y[i]
                
                if score < theta_l:
                    pruned_count += 1
                elif score >= theta_h:
                    flagged_count += 1
                    if y_i == 1: retained_positives += 1
                else:
                    escalated_count += 1
                    if y_i == 1: retained_positives += 1
                    
            cand_recall = retained_positives / total_positives
            p_escalate = escalated_count / N
            
            # Compute total expected latency per pair
            expected_latency = (
                Costs.C_prep + Costs.C_Tier1 +
                p_escalate * (Costs.C_feat_heavy + Costs.C_infer_heavy) +
                Costs.C_router
            )
            
            # Evaluate constraints
            meets_recall = (cand_recall >= R_target)
            meets_budget = (expected_latency <= B_max)
            
            pareto_frontier.append({
                'theta_low': theta_l,
                'theta_high': theta_h,
                'candidate_recall': cand_recall,
                'expected_latency_ms': expected_latency,
                'p_escalate': p_escalate,
                'feasible': (meets_recall and meets_budget)
            })
            
            if meets_recall and meets_budget:
                if expected_latency < best_latency:
                    best_latency = expected_latency
                    best_policy = (theta_l, theta_h)
                    
    # If no policy strictly satisfies both, fallback to policy maximizing recall within budget
    if best_policy is None:
        valid_budget = [p for p in pareto_frontier if p['expected_latency_ms'] <= B_max]
        if valid_budget:
            best_p = max(valid_budget, key=lambda p: p['candidate_recall'])
            best_policy = (best_p['theta_low'], best_p['theta_high'])
        else:
            best_policy = (0.05, 0.80) # Default safe fallback
            
    return best_policy, pareto_frontier
end function
```

---

## Algorithm 5: Explainable Cross-Modal Evidence Extraction (Contribution 4)

```python
"""
ALGORITHM 5: Cross-Modal Evidence Extraction & Dossier Synthesis
Input:
    Submission Pair (D_A, D_B)
    Modality M in {'text', 'code'}
    Calibrated Plagiarism Probability p_hat
    Operational Triage State
    Uncertainty Entropy H(p)
Output:
    EvidenceDossier Object (Structured JSON / Markdown)
"""

function EXTRACT_EVIDENCE_DOSSIER(D_A, D_B, M, p_hat, state, entropy):
    dossier = {
        'modality': M,
        'calibrated_probability': p_hat,
        'triage_state': state,
        'uncertainty_entropy': entropy,
        'matching_spans': [],
        'transformations': {},
        'faithfulness_score': 0.0,
        'coverage_score': 0.0,
        'narrative_summary': ''
    }
    
    if M == 'text':
        tokens_A = TOKENIZE_WORDS(D_A)
        tokens_B = TOKENIZE_WORDS(D_B)
        
        # 1. Smith-Waterman Local Alignment Matrix
        H, trace = SMITH_WATERMAN_ALIGN(tokens_A, tokens_B, match=2.0, gap=1.0)
        alignments = TRACEBACK_ALIGNMENTS(H, trace)
        
        # 2. Extract Paraphrase Spans (Low Lexical, High SBERT)
        paraphrases = DETECT_PARAPHRASE_SPANS(D_A, D_B, alignments)
        
        dossier['matching_spans'] = alignments
        dossier['transformations']['paraphrase_substitutions'] = paraphrases
        dossier['coverage_score'] = len(set(flatten(alignments))) / len(tokens_B)
        
    elif M == 'code':
        ast_A = PARSE_JAVA_AST(D_A)
        ast_B = PARSE_JAVA_AST(D_B)
        
        # 1. Identifier Mapping Dictionary Extraction
        id_map = EXTRACT_VARIABLE_MAPPINGS(ast_A, ast_B)
        
        # 2. AST Subtree Isomorphism Matching
        matched_subtrees = FIND_ISOMORPHIC_SUBTREES(ast_A, ast_B)
        
        # 3. Statement Reordering Detection
        permutations = DETECT_STATEMENT_PERMUTATIONS(ast_A, ast_B)
        
        dossier['matching_spans'] = matched_subtrees
        dossier['transformations']['identifier_substitutions'] = id_map
        dossier['transformations']['statement_permutations'] = permutations
        dossier['coverage_score'] = len(matched_subtrees) / COUNT_NODES(ast_B)
    
    # 3. Compute Faithfulness via Ablation Drop
    D_B_ablated = MASK_EVIDENCE_REGIONS(D_B, dossier['matching_spans'])
    p_ablated = RE_EVALUATE_MODEL(D_A, D_B_ablated)
    dossier['faithfulness_score'] = max(0.0, p_hat - p_ablated)
    
    # 4. Synthesize Human-Readable Narrative
    dossier['narrative_summary'] = GENERATE_NARRATIVE(dossier)
    
    return dossier
end function
```

---

## Algorithm 6: Unified Master Execution Pipeline (Integration)

```python
"""
ALGORITHM 6: Unified Master Execution Pipeline
Input:
    Query Pair (D_A, D_B)
    Modality Detector
    Candidate Scorer g_cand
    Heavy Detector f_heavy
    Probability Calibrator g_cal
    Conformal Cutoff q_hat
    Budget Policy theta*
Output:
    Final Decision Label y_pred in {0, 1}
    Calibrated Probability p_hat
    Evidence Dossier
"""

function EXECUTE_PIPELINE(D_A, D_B):
    # Step 1: Detect Modality
    M = DETECT_MODALITY(D_A, D_B)  # 'text' or 'code'
    
    # Step 2: Multi-View Candidate Screening (Contribution 1)
    status, v, s_cand = GENERATE_CANDIDATE(D_A, D_B, M, g_cand[M], theta_cand=theta*[M].theta_low)
    
    if status == 'PRUNED':
        # Immediate return: 0 heavy compute consumed
        return {
            'decision': 0,
            'state': 'LOW_RISK_PRUNED',
            'probability': 0.01,
            'entropy': 0.05,
            'dossier': None,
            'heavy_invoked': False
        }
        
    if s_cand >= theta*[M].theta_high:
        # Direct Tier-1 Flag: Verbatim copy confirmed without heavy inference
        return {
            'decision': 1,
            'state': 'HIGH_RISK_FLAGGED',
            'probability': 0.98,
            'entropy': 0.10,
            'dossier': EXTRACT_EVIDENCE_DOSSIER(D_A, D_B, M, 0.98, 'HIGH_RISK_FLAGGED', 0.10),
            'heavy_invoked': False
        }
        
    # Step 3: Uncertain Case Escalation to Heavy Detector (Contribution 3)
    z_raw = f_heavy[M].predict_raw(D_A, D_B)
    
    # Step 4: Calibrate Probability & Conformal Route (Contribution 2)
    p_cal = g_cal[M](z_raw)
    state, C, entropy = ROUTE_ADAPTIVE_TRI_BAND(p_cal, q_hat[M])
    
    # Step 5: Final Classification Decision
    y_pred = 1 if p_cal >= 0.50 else 0
    
    # Step 6: Generate Explainable Dossier (Contribution 4)
    dossier = EXTRACT_EVIDENCE_DOSSIER(D_A, D_B, M, p_cal, state, entropy)
    
    return {
        'decision': y_pred,
        'state': state,
        'probability': p_cal,
        'entropy': entropy,
        'dossier': dossier,
        'heavy_invoked': True
    }
end function
```

