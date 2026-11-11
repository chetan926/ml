# Algorithms and Pseudocode Specifications (Version 3.0)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.0 (Phase 2.2 Methodological Audit Revision)  
**Author:** Senior Machine Learning Research Scientist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Revised Algorithmic Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## Algorithm 1: Purely Lightweight Multi-View Candidate Generation (Contribution 1)

```python
"""
ALGORITHM 1: Lightweight Multi-View Candidate Generation
Execution Stage: Tier 1 (Executed on 100% of candidate pairs)
Target Ingestion Latency: < 0.050 ms / pair (To be measured under warmed-up profiling)
Note: Zero heavy AST parsing at Tier 1. AST parsing is strictly reserved for Tier 2.

Input:
    Pair (D_A, D_B): Raw text or source code submissions
    Modality M in {'text', 'code'}
    Trained Candidate Scorer g_cand[M]: v -> s_cand in [0, 1]
    Routing Cutoffs (theta_low, theta_high)
Output:
    Triage State in {'LOW_RISK_PRUNED', 'UNCERTAIN_ESCALATED', 'HIGH_RISK_FLAGGED'}
    Multi-View Feature Vector v in R^4
    Candidate Score s_cand in [0, 1]
"""

function GENERATE_LIGHTWEIGHT_CANDIDATE(D_A, D_B, M, g_cand, theta_low, theta_high):
    if M == 'text':
        # View 1: Sublinear TF-IDF Cosine
        t_A = COMPUTE_SUBLINEAR_TFIDF(D_A)
        t_B = COMPUTE_SUBLINEAR_TFIDF(D_B)
        denom_tfidf = NORM(t_A) * NORM(t_B)
        v_1 = DOT_PRODUCT(t_A, t_B) / denom_tfidf if denom_tfidf > 0 else 0.0
        
        # View 2: Character 3-Gram Set Jaccard Overlap
        g_A = EXTRACT_CHAR_3GRAM_SET(D_A)
        g_B = EXTRACT_CHAR_3GRAM_SET(D_B)
        denom_char = len(g_A | g_B)
        v_2 = len(g_A & g_B) / denom_char if denom_char > 0 else 0.0
        
        # View 3: Word Set Jaccard Overlap
        w_A = TOKENIZE_WORD_SET(D_A)
        w_B = TOKENIZE_WORD_SET(D_B)
        denom_word = len(w_A | w_B)
        v_3 = len(w_A & w_B) / denom_word if denom_word > 0 else 0.0
        
        # View 4: Length Ratio Asymmetry
        v_4 = min(len(w_A), len(w_B)) / max(len(w_A), len(w_B), 1)
        
        v = [v_1, v_2, v_3, v_4]
        
    elif M == 'code':
        # View 1: Raw Lexical Token Set Jaccard
        tok_A = EXTRACT_REGEX_TOKEN_SET(D_A, strip_comments=True)
        tok_B = EXTRACT_REGEX_TOKEN_SET(D_B, strip_comments=True)
        denom_raw = len(tok_A | tok_B)
        u_1 = len(tok_A & tok_B) / denom_raw if denom_raw > 0 else 0.0
        
        # View 2: Identifier-Normalized Token Set Jaccard
        norm_A = NORMALIZE_IDENTIFIERS_REGEX_SET(tok_A)
        norm_B = NORMALIZE_IDENTIFIERS_REGEX_SET(tok_B)
        denom_norm = len(norm_A | norm_B)
        u_2 = len(norm_A & norm_B) / denom_norm if denom_norm > 0 else 0.0
        
        # View 3: Operator Frequency Profile Cosine
        ops_A = COUNT_JAVA_OPERATORS_REGEX(tok_A)
        ops_B = COUNT_JAVA_OPERATORS_REGEX(tok_B)
        denom_ops = NORM(ops_A) * NORM(ops_B)
        u_3 = DOT_PRODUCT(ops_A, ops_B) / denom_ops if denom_ops > 0 else 0.0
        
        # View 4: Token Length Ratio Asymmetry
        u_4 = min(len(tok_A), len(tok_B)) / max(len(tok_A), len(tok_B), 1)
        
        v = [u_1, u_2, u_3, u_4]
    
    # Compute fused candidate score
    s_cand = g_cand.predict_proba(v)
    
    # Tri-Band Decision Policy
    if s_cand < theta_low:
        state = 'LOW_RISK_PRUNED'
    elif s_cand >= theta_high:
        state = 'HIGH_RISK_FLAGGED'
    else:
        state = 'UNCERTAIN_ESCALATED'
        
    return state, v, s_cand
end function
```

---

## Algorithm 2: Selection-Aware Probability Calibration (Contribution 2)

```python
"""
ALGORITHM 2: Selection-Aware Probability Calibration
Execution Stage: Offline Training & Diagnostic Evaluation
Input:
    Selection-Filtered Split D_cal_prob_esc = {(x_i, y_i) in D_cal_prob | theta_low <= s_cand(x_i) < theta_high}
    Independent Diagnostic Split D_val_esc = {(x_i, y_i) in D_val | theta_low <= s_cand(x_i) < theta_high}
    Heavy Classifier Decision Function f_heavy(x) -> z in R
    Method in {'platt', 'isotonic'}
Output:
    Calibrator model g_cal(z) -> p in [0, 1]
    Independent Diagnostic metrics on D_val_esc: ECE, Brier Score
"""

function FIT_PROBABILITY_CALIBRATOR(D_cal_prob_esc, D_val_esc, f_heavy, method='isotonic'):
    Z_train = []
    Y_train = []
    
    # Step 1: Extract heavy margins on conditional escalation split D_cal_prob_esc
    for (x_i, y_i) in D_cal_prob_esc:
        z_i = f_heavy(x_i)
        Z_train.append(z_i)
        Y_train.append(y_i)
    
    # Step 2: Fit selected calibrator on conditional escalation data
    if method == 'platt':
        model = LogisticRegression(solver='lbfgs')
        model.fit(reshape(Z_train, (-1, 1)), Y_train)
        g_cal = function(z): return model.predict_proba([[z]])[0, 1]
        
    elif method == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        model.fit(Z_train, Y_train)
        g_cal = function(z): return model.predict([z])[0]
    
    # Step 3: Compute INDEPENDENT Diagnostic Metrics on separate D_val_esc (NOT on training data!)
    Z_val = [f_heavy(x_i) for (x_i, _) in D_val_esc]
    Y_val = [y_i for (_, y_i) in D_val_esc]
    P_val = [g_cal(z) for z in Z_val]
    
    independent_ece = COMPUTE_ECE(P_val, Y_val, num_bins=10)
    independent_brier = MEAN_SQUARED_ERROR(P_val, Y_val)
    
    return g_cal, independent_ece, independent_brier
end function

function COMPUTE_ECE(probs, labels, num_bins=10):
    bin_boundaries = LINSPACE(0.0, 1.0, num_bins + 1)
    ece = 0.0
    N = len(probs)
    if N == 0: return 0.0
    
    for m in range(num_bins):
        low = bin_boundaries[m]
        high = bin_boundaries[m + 1]
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

## Algorithm 3: Selection-Aware Conformal Prediction with Complete 4-Set Triage (Contribution 2)

```python
"""
ALGORITHM 3: Inductive Conformal Prediction & Complete Four-Set Router
Execution Stage: Runtime Routing for ESCALATED pairs
Input:
    Selection-Filtered Conformal Split D_cal_conf_esc
    Calibrated Probability Estimator g_cal(z) -> p_hat
    Significance Level epsilon in (0, 1) (e.g. 0.05 for 95% coverage)
    Query Pair x_query with calibrated probability p_query
Output:
    Conformal Set C in {{0}, {1}, {0, 1}, Ø}
    Operational Action in {'CLEARED_NON_PLAGIARIZED', 'CONFIRMED_PLAGIARIZED', 
                          'AMBIGUOUS_HUMAN_REVIEW', 'ABSTAIN_UNRESOLVED'}
    Predictive Entropy H(p)
"""

function CALIBRATE_CONFORMAL_QUANTILE(D_cal_conf_esc, g_cal, f_heavy, epsilon=0.05):
    N_cal = len(D_cal_conf_esc)
    scores = []
    
    for (x_i, y_i) in D_cal_conf_esc:
        z_i = f_heavy(x_i)
        p_i = g_cal(z_i)
        p_true = p_i if y_i == 1 else (1.0 - p_i)
        scores.append(1.0 - p_true)
        
    scores.sort()
    rank_idx = int(ceil((N_cal + 1) * (1.0 - epsilon))) - 1
    rank_idx = min(max(rank_idx, 0), N_cal - 1)
    q_hat = scores[rank_idx]
    
    return q_hat
end function

function ROUTE_CONFORMAL_FOUR_SET(p_query, q_hat):
    C = set()
    
    # Test label 0 conformity: 1 - P(Y = 0 | x) <= q_hat <=> p_query <= q_hat
    if p_query <= q_hat:
        C.add(0)
        
    # Test label 1 conformity: 1 - P(Y = 1 | x) <= q_hat <=> (1 - p_query) <= q_hat
    if (1.0 - p_query) <= q_hat:
        C.add(1)
        
    # Compute predictive entropy proxy
    p_safe = clip(p_query, 1e-7, 1.0 - 1e-7)
    entropy = -p_safe * log2(p_safe) - (1.0 - p_safe) * log2(1.0 - p_safe)
    
    # Complete 4-Set Operational Semantics
    if C == {0}:
        action = 'CLEARED_NON_PLAGIARIZED'
        human_review = False
    elif C == {1}:
        action = 'CONFIRMED_PLAGIARIZED'
        human_review = True
    elif C == {0, 1}:
        action = 'AMBIGUOUS_HUMAN_REVIEW'
        human_review = True
    elif C == set(): # EMPTY SET Ø: Abstention / Unresolved Prediction
        action = 'ABSTAIN_UNRESOLVED'
        human_review = True # Mandatory human escalation
        
    return C, action, entropy, human_review
end function
```

---

## Algorithm 4: Recall-Constrained Policy Optimization via Pareto Surface Search (Contribution 3)

```python
"""
ALGORITHM 4: Recall-Constrained Routing-Policy Optimization via Discretized Pareto Surface Search
Execution Stage: Offline Optimization on Validation Split D_val
Input:
    Validation Set D_val with Candidate Scores s_cand, Heavy Scores z_heavy, and Labels y_val
    Target Candidate Recall R_target (e.g. 0.98)
    Maximum Latency Budget B_max (in ms/pair)
    Target Direct Precision P_target (e.g. 0.90)
    Comprehensive Cost Model Constants: C_prep, C_Tier1, C_feat_heavy, C_infer_heavy, C_router, C_explain
Output:
    Optimal Policy theta* = (theta_low*, theta_high*) or 'STATUS: INFEASIBLE_BUDGET'
    Pareto Frontier Points: List of (Recall, Latency, DirectPrecision, F1, theta_low, theta_high)
"""

function OPTIMIZE_BUDGET_CONSTRAINED_THRESHOLDS(D_val, R_target, B_max, P_target, Costs):
    grid_low = LINSPACE(0.01, 0.40, 40)
    grid_high = LINSPACE(0.60, 0.95, 36)
    
    all_evaluated_policies = []
    feasible_policies = []
    
    N = len(D_val)
    pos_mask = [y == 1 for y in D_val.y]
    total_positives = sum(pos_mask)
    
    for theta_l in grid_low:
        for theta_h in grid_high:
            if theta_l >= theta_h:
                continue
                
            escalated_idx = []
            flagged_idx = []
            pruned_idx = []
            
            for i in range(N):
                score = D_val.s_cand[i]
                if score < theta_l:
                    pruned_idx.append(i)
                elif score >= theta_h:
                    flagged_idx.append(i)
                else:
                    escalated_idx.append(i)
                    
            retained_pos = sum(1 for i in (escalated_idx + flagged_idx) if D_val.y[i] == 1)
            cand_recall = retained_pos / total_positives if total_positives > 0 else 1.0
            
            direct_flag_tp = sum(1 for i in flagged_idx if D_val.y[i] == 1)
            direct_prec = direct_flag_tp / len(flagged_idx) if len(flagged_idx) > 0 else 1.0
            
            p_escalate = len(escalated_idx) / N
            # Evidence generation charged ONLY when human review is required!
            p_review = (len(flagged_idx) + sum(1 for i in escalated_idx if D_val.conformal_action[i] in ['CONFIRMED_PLAGIARIZED', 'AMBIGUOUS_HUMAN_REVIEW', 'ABSTAIN_UNRESOLVED'])) / N
            
            # Comprehensive Per-Pair Latency Model matching Master Pipeline
            expected_latency = (
                Costs.C_prep + Costs.C_Tier1 +
                p_escalate * (Costs.C_feat_heavy + Costs.C_infer_heavy + Costs.C_router) +
                p_review * Costs.C_explain
            )
            
            # Compute Combined F1 on D_val (Direct Flags + Escalated Heavy Decisions)
            tp = direct_flag_tp + sum(1 for i in escalated_idx if D_val.p_cal[i] >= 0.50 and D_val.y[i] == 1)
            fp = (len(flagged_idx) - direct_flag_tp) + sum(1 for i in escalated_idx if D_val.p_cal[i] >= 0.50 and D_val.y[i] == 0)
            fn = total_positives - tp
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
            
            # Constraint Verification
            meets_recall = (cand_recall >= R_target)
            meets_budget = (expected_latency <= B_max)
            meets_prec = (direct_prec >= P_target)
            
            is_feasible = (meets_recall and meets_budget and meets_prec)
            
            policy_record = {
                'theta_low': theta_l,
                'theta_high': theta_h,
                'candidate_recall': cand_recall,
                'direct_precision': direct_prec,
                'expected_latency_ms': expected_latency,
                'f1_score': f1,
                'p_escalate': p_escalate,
                'p_review': p_review,
                'is_feasible': is_feasible
            }
            all_evaluated_policies.append(policy_record)
            if is_feasible:
                feasible_policies.append(policy_record)
                
    # Filter Dominated Points to Form Non-Dominated Pareto Frontier
    pareto_frontier = EXTRACT_NON_DOMINATED_POINTS(all_evaluated_policies)
    
    # Infeasibility Check: Never use arbitrary hardcoded fallbacks!
    if len(feasible_policies) == 0:
        return 'STATUS: INFEASIBLE_BUDGET', pareto_frontier
        
    # Select minimum latency policy among all feasible policies
    best_policy = min(feasible_policies, key=lambda p: p['expected_latency_ms'])
    return (best_policy['theta_low'], best_policy['theta_high']), pareto_frontier
end function
```

---

## Algorithm 5: Cross-Modal Evidence Extraction & Dossier Synthesis (Contribution 4)

```python
"""
ALGORITHM 5: Cross-Modal Evidence Extraction & Dossier Synthesis
Execution Stage: Invoked ONLY when human_review_required == True
Input:
    Pair (D_A, D_B)
    Modality M in {'text', 'code'}
    Triage State & Conformal Action
    Calibrated Probability p_hat (or None for Tier-1 direct flags)
    Conformal Set C (or None for Tier-1 direct flags)
Output:
    EvidenceDossier Object (Structured JSON / Markdown)
"""

function EXTRACT_EVIDENCE_DOSSIER(D_A, D_B, M, state, action, p_hat, C, entropy):
    dossier = {
        'modality': M,
        'operational_state': state,
        'conformal_action': action,
        'calibrated_probability': p_hat,
        'conformal_set': list(C) if C is not None else None,
        'predictive_entropy': entropy,
        'matching_spans': [],
        'transformations': {},
        'faithfulness_score': 0.0,
        'coverage_score': 0.0,
        'narrative_summary': ''
    }
    
    if M == 'text':
        tokens_A = TOKENIZE_WORDS(D_A)
        tokens_B = TOKENIZE_WORDS(D_B)
        
        # Smith-Waterman Local Sequence Alignment (Affine Gap Recurrence)
        # H[i, j] = max(0, H[i-1, j-1] + s(a_i, b_j), E[i, j], F[i, j])
        alignments = SMITH_WATERMAN_AFFINE_ALIGN(tokens_A, tokens_B, match=2.0, mismatch=-1.0, u=2.0, v=0.5)
        paraphrases = DETECT_PARAPHRASE_SPANS(tokens_A, tokens_B, alignments)
        
        dossier['matching_spans'] = alignments
        dossier['transformations']['paraphrase_substitutions'] = paraphrases
        dossier['coverage_score'] = len(set(flatten(alignments))) / max(len(tokens_B), 1)
        
    elif M == 'code':
        ast_A = PARSE_JAVA_AST(D_A)
        ast_B = PARSE_JAVA_AST(D_B)
        
        # AST Subtree Isomorphism Matching & Identifier Map
        id_map = EXTRACT_VARIABLE_MAPPINGS(ast_A, ast_B)
        matched_subtrees = FIND_ISOMORPHIC_SUBTREES(ast_A, ast_B)
        permutations = DETECT_STATEMENT_PERMUTATIONS(ast_A, ast_B)
        
        dossier['matching_spans'] = matched_subtrees
        dossier['transformations']['identifier_substitutions'] = id_map
        dossier['transformations']['statement_permutations'] = permutations
        dossier['coverage_score'] = len(matched_subtrees) / max(COUNT_AST_NODES(ast_B), 1)
        
    # Faithfulness via Decision Probability Drop (Only for heavy-evaluated pairs!)
    if p_hat is not None and isinstance(p_hat, float):
        D_B_ablated = ABLATE_EVIDENCE_SPANS(D_B, dossier['matching_spans'])
        p_ablated = RE_EVALUATE_CALIBRATED_PROBABILITY(D_A, D_B_ablated)
        dossier['faithfulness_score'] = max(0.0, p_hat - p_ablated)
    else:
        dossier['faithfulness_score'] = None # Not applicable for Tier-1 direct flags
        
    # Synthesize Human-Readable Narrative
    dossier['narrative_summary'] = SYNTHESIZE_EXPLANATION_TEXT(dossier)
    
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
    Lightweight Candidate Scorer g_cand[M]
    Heavy Feature Extractor & Classifier f_heavy[M]
    Probability Calibrator g_cal[M] (Fitted on D_cal_prob_esc)
    Conformal Quantile q_hat[M] (Calibrated on D_cal_conf_esc)
    Optimal Routing Policy (theta_low[M], theta_high[M])
Output:
    Strictly Typed Execution Record
"""

function EXECUTE_MASTER_PIPELINE(D_A, D_B):
    # Step 1: Modality Detection (< 0.005 ms)
    M = DETECT_MODALITY(D_A, D_B)  # 'text' or 'code'
    
    # Step 2: Lightweight Tier-1 Candidate Screening (< 0.045 ms)
    triage_state, v_tier1, s_cand = GENERATE_LIGHTWEIGHT_CANDIDATE(
        D_A, D_B, M, g_cand[M], theta_low[M], theta_high[M]
    )
    
    # BRANCH A: Low-Risk Pruned (Bypasses Heavy Models & Conformal Routing)
    if triage_state == 'LOW_RISK_PRUNED':
        return {
            'tier1_candidate_score': s_cand,
            'tier1_routing_state': 'LOW_RISK_PRUNED',
            'heavy_model_raw_score': None,       # Bypassed
            'calibrated_probability': None,      # Bypassed
            'conformal_prediction_set': None,    # Bypassed
            'predictive_entropy': None,          # Bypassed
            'human_review_required': False,
            'evidence_dossier': None
        }
        
    # BRANCH B: High-Risk Flagged at Tier 1 (Bypasses Heavy Models & Conformal Routing)
    if triage_state == 'HIGH_RISK_FLAGGED':
        # Direct flag for human review based on Tier-1 near-verbatim overlap
        dossier = EXTRACT_EVIDENCE_DOSSIER(
            D_A, D_B, M, state='HIGH_RISK_FLAGGED', action='TIER1_DIRECT_FLAG', 
            p_hat=None, C=None, entropy=None
        )
        return {
            'tier1_candidate_score': s_cand,
            'tier1_routing_state': 'HIGH_RISK_FLAGGED',
            'heavy_model_raw_score': None,       # Bypassed
            'calibrated_probability': None,      # Bypassed
            'conformal_prediction_set': None,    # Bypassed
            'predictive_entropy': None,          # Bypassed
            'human_review_required': True,
            'evidence_dossier': dossier
        }
        
    # BRANCH C: Uncertain Candidate Escalated to Tier 2 Heavy Model
    # Extract heavy features (Sentence-BERT on text; 7-D AST parsing on code)
    z_raw = f_heavy[M].predict_raw_score(D_A, D_B)
    
    # Calibrate Posterior Probability
    p_cal = g_cal[M](z_raw)
    
    # Route via Inductive Conformal Prediction (Complete 4-Set Handling)
    C_set, action, entropy, review_req = ROUTE_CONFORMAL_FOUR_SET(p_cal, q_hat[M])
    
    # Generate Evidence Dossier ONLY when review is required!
    dossier = None
    if review_req:
        dossier = EXTRACT_EVIDENCE_DOSSIER(
            D_A, D_B, M, state='UNCERTAIN_ESCALATED', action=action, 
            p_hat=p_cal, C=C_set, entropy=entropy
        )
    
    return {
        'tier1_candidate_score': s_cand,
        'tier1_routing_state': 'UNCERTAIN_ESCALATED',
        'heavy_model_raw_score': z_raw,
        'calibrated_probability': p_cal,
        'conformal_prediction_set': C_set,
        'predictive_entropy': entropy,
        'human_review_required': review_req,
        'evidence_dossier': dossier
    }
end function
```

