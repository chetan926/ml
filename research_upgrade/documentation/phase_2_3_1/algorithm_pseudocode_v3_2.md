# Algorithms and Pseudocode Specifications (Version 3.2)
## Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

**Document Version:** 3.2 (Phase 2.3.1 Independent Audit Revision)  
**Author:** Independent Senior Machine Learning Researcher, Statistical Methodologist & Experimental Auditor  
**Date:** November 2026  
**Status:** Audited and Finalized Algorithmic Specification  
**Target Venue:** High-Impact Q1 Machine Learning / Academic Computing Journals

---

## Algorithm 1: Purely Lightweight Multi-View Candidate Generation (Contribution 1)

```python
"""
ALGORITHM 1: Purely Lightweight Multi-View Candidate Generation
Execution Stage: Tier 1 (Executed on 100% of candidate pairs)
Target Ingestion Latency: < 0.050 ms / pair (Measured under warm-cache profiling)
Guaranteed Constraint: ZERO heavy AST parsing at Tier 1. AST parsing is sequestered in Tier 2.

Input:
    Pair (D_A, D_B): Raw text strings or source code submissions
    Modality M in {'text', 'code'}
    Trained Candidate Scorer g_cand[M]: v -> s_cand in [0, 1]
    Routing Cutoffs (theta_low, theta_high)
Output:
    Triage State in {'LOW_RISK_REVIEW_DEFERRED', 'UNCERTAIN_ESCALATED', 'HIGH_RISK_CANDIDATE_FLAGGED'}
    Multi-View Feature Vector v in R^4
    Candidate Score s_cand in [0, 1]
"""

function GENERATE_LIGHTWEIGHT_CANDIDATE(D_A, D_B, M, g_cand, theta_low, theta_high):
    if M == 'text':
        # View 1: Word-Level Sublinear TF-IDF Cosine
        t_A = GET_OR_COMPUTE_SUBLINEAR_TFIDF(D_A)
        t_B = GET_OR_COMPUTE_SUBLINEAR_TFIDF(D_B)
        denom_tfidf = NORM(t_A) * NORM(t_B)
        v_1 = DOT_PRODUCT(t_A, t_B) / denom_tfidf if denom_tfidf > 0.0 else 0.0
        
        # View 2: Character 3-Gram Set Jaccard Overlap
        g_A = GET_OR_EXTRACT_CHAR_3GRAM_SET(D_A)
        g_B = GET_OR_EXTRACT_CHAR_3GRAM_SET(D_B)
        denom_char = len(g_A | g_B)
        v_2 = len(g_A & g_B) / denom_char if denom_char > 0 else 0.0
        
        # View 3: Word Token Set Jaccard Overlap
        w_A = GET_OR_TOKENIZE_WORD_SET(D_A)
        w_B = GET_OR_TOKENIZE_WORD_SET(D_B)
        denom_word = len(w_A | w_B)
        v_3 = len(w_A & w_B) / denom_word if denom_word > 0 else 0.0
        
        # View 4: Length Ratio Asymmetry
        v_4 = min(len(w_A), len(w_B)) / max(len(w_A), len(w_B), 1)
        
        v = [v_1, v_2, v_3, v_4]
        
    elif M == 'code':
        # View 1: Raw Lexical Token Set Jaccard (Comments & Whitespace Stripped)
        tok_A = GET_OR_EXTRACT_REGEX_TOKEN_SET(D_A, strip_comments=True)
        tok_B = GET_OR_EXTRACT_REGEX_TOKEN_SET(D_B, strip_comments=True)
        denom_tok = len(tok_A | tok_B)
        v_1 = len(tok_A & tok_B) / denom_tok if denom_tok > 0 else 0.0
        
        # View 2: Normalized Identifier Token Set Jaccard
        norm_A = GET_OR_EXTRACT_NORMALIZED_TOKEN_SET(D_A)
        norm_B = GET_OR_EXTRACT_NORMALIZED_TOKEN_SET(D_B)
        denom_norm = len(norm_A | norm_B)
        v_2 = len(norm_A & norm_B) / denom_norm if denom_norm > 0 else 0.0
        
        # View 3: Operator Frequency Profile Cosine (13 Java Operators)
        op_A = GET_OR_EXTRACT_OPERATOR_COUNTS(D_A)
        op_B = GET_OR_EXTRACT_OPERATOR_COUNTS(D_B)
        denom_op = NORM(op_A) * NORM(op_B)
        v_3 = DOT_PRODUCT(op_A, op_B) / denom_op if denom_op > 0.0 else 0.0
        
        # View 4: Cyclomatic Keyword Profile Differential (Regex Keywords, NO AST)
        k_A = GET_OR_COUNT_DECISION_KEYWORDS(D_A)
        k_B = GET_OR_COUNT_DECISION_KEYWORDS(D_B)
        v_4 = 1.0 - (abs(k_A - k_B) / max(k_A, k_B, 1))
        
        v = [v_1, v_2, v_3, v_4]
        
    # Evaluate Candidate Logistic Scorer
    s_cand = g_cand.predict_probability(v)
    
    # Strictly Disambiguated Triage Assignment (Issue 4)
    if s_cand < theta_low:
        triage_state = 'LOW_RISK_REVIEW_DEFERRED'
    elif s_cand >= theta_high:
        triage_state = 'HIGH_RISK_CANDIDATE_FLAGGED'
    else:
        triage_state = 'UNCERTAIN_ESCALATED'
        
    return triage_state, v, s_cand
end function
```

---

## Algorithm 2: Selection-Aware Probability Calibration (Contribution 2)

```python
"""
ALGORITHM 2: Selection-Aware Probability Calibration
Execution Stage: Offline Fitting on Escalated Calibration Split D_cal_prob_esc
Constraint: Must be fitted strictly on the conditional escalation population, 
            NOT the marginal population. Evaluated on independent evaluation data.

Input:
    Calibration Split D_cal_prob with (D_A, D_B, y)
    Modality M in {'text', 'code'}
    Candidate Scorer g_cand[M]
    Frozen Routing Thresholds (theta_low, theta_high)
    Heavy Classifier f_heavy[M]
    Calibrator Type in {'platt', 'isotonic'}
Output:
    Fitted Calibrator Object g_cal[M]: z -> p_hat in [0, 1]
"""

function FIT_SELECTION_AWARE_CALIBRATOR(D_cal_prob, M, g_cand, theta_low, theta_high, f_heavy, cal_type):
    escalated_margins = []
    escalated_labels = []
    
    # Step 1: Filter to Conditional Escalation Population using FROZEN thresholds
    for pair in D_cal_prob:
        triage_state, _, s_cand = GENERATE_LIGHTWEIGHT_CANDIDATE(
            pair.D_A, pair.D_B, M, g_cand, theta_low, theta_high
        )
        if triage_state == 'UNCERTAIN_ESCALATED':
            z = f_heavy.predict_raw_score(pair.D_A, pair.D_B)
            escalated_margins.append(z)
            escalated_labels.append(pair.y)
            
    assert len(escalated_margins) > 0, "No escalated instances found in calibration split!"
    
    # Step 2: Fit Calibrator on Escalated Margins
    if cal_type == 'platt':
        g_cal = FIT_LOGISTIC_REGRESSION(escalated_margins, escalated_labels)
    elif cal_type == 'isotonic':
        g_cal = FIT_ISOTONIC_REGRESSION_PAV(escalated_margins, escalated_labels)
        
    return g_cal
end function

function EVALUATE_CALIBRATION_DIAGNOSTICS(D_eval_esc, g_cal, f_heavy, n_bins=10):
    # Evaluated on independent evaluation data (e.g. D_test_esc), NEVER on the fitting set
    z_eval = [f_heavy.predict_raw_score(p.D_A, p.D_B) for p in D_eval_esc]
    p_hat = [g_cal.predict(z) for z in z_eval]
    y_true = [p.y for p in D_eval_esc]
    
    ece = COMPUTE_EXPECTED_CALIBRATION_ERROR(p_hat, y_true, n_bins)
    brier = MEAN([(p - y)**2 for p, y in zip(p_hat, y_true)])
    nll = -MEAN([y * log(max(p, 1e-15)) + (1 - y) * log(max(1 - p, 1e-15)) for p, y in zip(p_hat, y_true)])
    
    return {'ece': ece, 'brier_score': brier, 'nll': nll}
end function
```

---

## Algorithm 3: Modality-Specific Uncertainty Routing & Conformal Calibration (Contribution 2)

```python
"""
ALGORITHM 3: Modality-Specific Uncertainty Routing & Conformal Calibration
Execution Stage: Offline Calibration + Online Query Routing
Constraint:
    - TEXT: Retains formal split inductive conformal prediction on D_cal_conf_esc.
            Uses exact finite-sample order statistic k = ceil((n+1)(1-epsilon)).
    - CODE: Formally disclaims conformal guarantees due to 15 reference programs.
            Applies exploratory calibrated uncertainty routing via decision bands.

Input (Text Calibration):
    Independent Calibration Split D_cal_conf
    Frozen Routing Thresholds (theta_low, theta_high)
    Heavy Scorer f_heavy['text']
    Fitted Calibrator g_cal['text']
    Significance Level epsilon (e.g. 0.05 for 95% nominal coverage)
Output (Text):
    Conformal Non-Conformity Quantile q_hat_text
"""

function CALIBRATE_TEXT_CONFORMAL_QUANTILE(D_cal_conf, g_cand, theta_low, theta_high, f_heavy, g_cal, epsilon):
    non_conformity_scores = []
    
    # Step 1: Filter to Conditional Escalation Population using FROZEN thresholds
    for pair in D_cal_conf:
        triage_state, _, s_cand = GENERATE_LIGHTWEIGHT_CANDIDATE(
            pair.D_A, pair.D_B, 'text', g_cand, theta_low, theta_high
        )
        if triage_state == 'UNCERTAIN_ESCALATED':
            z = f_heavy.predict_raw_score(pair.D_A, pair.D_B)
            p_hat = g_cal.predict(z)
            # Alpha_i = 1 - P(Y = y_i | x_i)
            alpha_i = (1.0 - p_hat) if pair.y == 1 else p_hat
            non_conformity_scores.append(alpha_i)
            
    n = len(non_conformity_scores)
    assert n > 0, "No escalated instances found in conformal calibration split!"
    
    # Step 2: Compute Exact Finite-Sample Order Statistic (Issue 2)
    k = ceil((n + 1) * (1.0 - epsilon))
    if k > n:
        # Mathematical Condition: n < ceil((1-epsilon)/epsilon)
        # Cannot attain coverage without extrapolation; abstain from formal guarantee
        raise InadequateSampleSizeError("Sample size n is insufficient to guarantee 1-epsilon coverage!")
        
    sorted_scores = SORT_ASCENDING(non_conformity_scores)
    q_hat = sorted_scores[k - 1]  # 1-indexed k converted to 0-indexed position
    
    return q_hat
end function

function ROUTE_TEXT_CONFORMAL(p_query, q_hat):
    # Form formal conformal prediction set for text
    C = set()
    if p_query <= q_hat:
        C.add(0)
    if (1.0 - p_query) <= q_hat:
        C.add(1)
        
    p_safe = clip(p_query, 1e-7, 1.0 - 1e-7)
    entropy = -p_safe * log2(p_safe) - (1.0 - p_safe) * log2(1.0 - p_safe)
    
    # Standardized Output States (Issue 4)
    if C == {0}:
        action = 'LOW_RISK_REVIEW_DEFERRED'
        human_review = False
    elif C == {1}:
        action = 'HIGH_RISK_CANDIDATE_FLAGGED'
        human_review = True
    elif C == {0, 1}:
        action = 'AMBIGUOUS_HUMAN_REVIEW'
        human_review = True
    elif C == set():  # Empty set Ø
        action = 'ABSTAIN_UNRESOLVED'
        human_review = True
        
    return C, action, entropy, human_review
end function

function ROUTE_CODE_EXPLORATORY_UNCERTAINTY(p_query, tau_non_plag=0.20, tau_plag=0.80, tau_entropy=0.72):
    # Exploratory calibrated uncertainty routing (Conformal guarantees disclaimed)
    p_safe = clip(p_query, 1e-7, 1.0 - 1e-7)
    entropy = -p_safe * log2(p_safe) - (1.0 - p_safe) * log2(1.0 - p_safe)
    
    # Standardized Output States (Issue 4)
    if p_query < tau_non_plag and entropy <= tau_entropy:
        action = 'LOW_RISK_REVIEW_DEFERRED'
        human_review = False
    elif p_query >= tau_plag and entropy <= tau_entropy:
        action = 'HIGH_RISK_CANDIDATE_FLAGGED'
        human_review = True
    else:
        action = 'AMBIGUOUS_HUMAN_REVIEW'
        human_review = True
        
    # Conformal set is explicitly None for source code
    return None, action, entropy, human_review
end function
```

---

## Algorithm 4: Recall-Constrained Policy Optimization via Discretized Pareto Surface Search (Contribution 3)

```python
"""
ALGORITHM 4: Recall-Constrained Policy Optimization via Discretized Pareto Surface Search
Execution Stage: Offline Optimization on Validation Split D_val
Constraint: Evidence extraction time is charged strictly when review is required.

Input:
    Validation Set D_val with Candidate Scores s_cand, Heavy Predictions, and Labels y_val
    Target Candidate Recall R_target (e.g. 0.980)
    Maximum Latency Budget B_max (in ms/pair, e.g. 1.50)
    Target Direct Precision P_target (e.g. 0.900)
    Time Profile Constants T_prep, T_Tier1, T_feat_heavy, T_infer_heavy, T_router, T_explain
Output:
    Optimal Frozen Policy theta* = (theta_low*, theta_high*) or 'STATUS: INFEASIBLE_BUDGET'
    Non-Dominated Pareto Frontier Points
"""

function OPTIMIZE_BUDGET_CONSTRAINED_THRESHOLDS(D_val, R_target, B_max, P_target, T_profile):
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
                    
            # Candidate Recall
            retained_pos = sum(1 for i in (escalated_idx + flagged_idx) if D_val.y[i] == 1)
            cand_recall = retained_pos / total_positives if total_positives > 0 else 1.0
            
            # Direct Flag Precision
            direct_flag_tp = sum(1 for i in flagged_idx if D_val.y[i] == 1)
            direct_prec = direct_flag_tp / len(flagged_idx) if len(flagged_idx) > 0 else 1.0
            
            p_escalate = len(escalated_idx) / N
            p_review = (len(flagged_idx) + sum(1 for i in escalated_idx if D_val.review_required[i])) / N
            
            # Per-Pair Wall-Clock Latency Model (Milliseconds)
            expected_latency_ms = (
                T_profile.T_prep + T_profile.T_Tier1 +
                p_escalate * (T_profile.T_feat_heavy + T_profile.T_infer_heavy + T_profile.T_router) +
                p_review * T_profile.T_explain
            )
            
            # Constraint Verification
            meets_recall = (cand_recall >= R_target)
            meets_budget = (expected_latency_ms <= B_max)
            meets_prec = (direct_prec >= P_target)
            is_feasible = (meets_recall and meets_budget and meets_prec)
            
            policy_record = {
                'theta_low': theta_l,
                'theta_high': theta_h,
                'candidate_recall': cand_recall,
                'direct_precision': direct_prec,
                'expected_latency_ms': expected_latency_ms,
                'p_escalate': p_escalate,
                'p_review': p_review,
                'is_feasible': is_feasible
            }
            all_evaluated_policies.append(policy_record)
            if is_feasible:
                feasible_policies.append(policy_record)
                
    # Extract Non-Dominated Pareto Frontier
    pareto_frontier = EXTRACT_NON_DOMINATED_POINTS(all_evaluated_policies)
    
    # Infeasibility Check: Never use unvalidated fallbacks!
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
Execution Stage: Invoked STRICTLY when human_review_required == True
Constraint: Bypassed for cleared pairs to avoid unnecessary overhead.

Input:
    Pair (D_A, D_B)
    Modality M in {'text', 'code'}
    Triage State & Routing Action
    Calibrated Probability p_hat (or None for Tier-1 direct flags)
    Conformal Set C (or None for Tier-1 direct flags / code)
    Predictive Entropy (or None for Tier-1 direct flags)
Output:
    EvidenceDossier Object (Structured Data)
"""

function EXTRACT_EVIDENCE_DOSSIER(D_A, D_B, M, state, action, p_hat, C, entropy):
    dossier = {
        'modality': M,
        'operational_state': state,
        'routing_action': action,
        'calibrated_probability': p_hat,
        'conformal_set': list(C) if C is not None else None,
        'predictive_entropy': entropy,
        'matching_spans': [],
        'transformations': {},
        'faithfulness_score': None,
        'coverage_score': 0.0,
        'human_review_required': True,
        'narrative_summary': ''
    }
    
    if M == 'text':
        tokens_A = TOKENIZE_WORDS(D_A)
        tokens_B = TOKENIZE_WORDS(D_B)
        
        # Smith-Waterman Local Sequence Alignment (Affine Gap Recurrence)
        alignments = SMITH_WATERMAN_AFFINE_ALIGN(tokens_A, tokens_B, match=2.0, mismatch=-1.0, u=2.0, v=0.5)
        paraphrases = DETECT_PARAPHRASE_SPANS(tokens_A, tokens_B, alignments)
        
        dossier['matching_spans'] = alignments
        dossier['transformations']['paraphrase_substitutions'] = paraphrases
        dossier['coverage_score'] = len(set(flatten(alignments))) / max(len(tokens_B), 1)
        
    elif M == 'code':
        # AST Parsing executed strictly at Tier 2 for evidence generation
        ast_A = PARSE_JAVA_AST(D_A)
        ast_B = PARSE_JAVA_AST(D_B)
        
        id_map = EXTRACT_VARIABLE_MAPPINGS(ast_A, ast_B)
        matched_subtrees = FIND_ISOMORPHIC_SUBTREES(ast_A, ast_B)
        permutations = DETECT_STATEMENT_PERMUTATIONS(ast_A, ast_B)
        
        dossier['matching_spans'] = matched_subtrees
        dossier['transformations']['identifier_substitutions'] = id_map
        dossier['transformations']['statement_permutations'] = permutations
        dossier['coverage_score'] = len(matched_subtrees) / max(COUNT_AST_NODES(ast_B), 1)
        
    # Faithfulness via Decision Probability Drop (Heavy-evaluated pairs only!)
    if p_hat is not None and isinstance(p_hat, float):
        D_B_ablated = ABLATE_EVIDENCE_SPANS(D_B, dossier['matching_spans'])
        p_ablated = RE_EVALUATE_CALIBRATED_PROBABILITY(D_A, D_B_ablated)
        dossier['faithfulness_score'] = max(0.0, p_hat - p_ablated)
        
    dossier['narrative_summary'] = SYNTHESIZE_HUMAN_AUDIT_NARRATIVE(dossier)
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
    Conformal Quantile q_hat_text (Calibrated on text D_cal_conf_esc)
    Frozen Routing Policy (theta_low[M], theta_high[M])
Output:
    Strictly Typed Execution Record
"""

function EXECUTE_MASTER_PIPELINE(D_A, D_B):
    # Step 1: Fast Modality Detection (< 0.005 ms)
    M = DETECT_MODALITY(D_A, D_B)  # 'text' or 'code'
    
    # Step 2: Lightweight Tier-1 Candidate Screening (< 0.050 ms)
    triage_state, v_tier1, s_cand = GENERATE_LIGHTWEIGHT_CANDIDATE(
        D_A, D_B, M, g_cand[M], theta_low[M], theta_high[M]
    )
    
    # BRANCH A: Low-Risk Review Deferred (Bypasses Heavy Models & Routers)
    if triage_state == 'LOW_RISK_REVIEW_DEFERRED':
        return {
            'tier1_candidate_score': s_cand,
            'tier1_routing_state': 'LOW_RISK_REVIEW_DEFERRED',
            'heavy_model_raw_score': None,       # Bypassed
            'calibrated_probability': None,      # Bypassed
            'conformal_prediction_set': None,    # Bypassed
            'predictive_entropy': None,          # Bypassed
            'human_review_required': False,
            'final_adjudication_state': 'LOW_RISK_REVIEW_DEFERRED',
            'evidence_dossier': None
        }
        
    # BRANCH B: High-Risk Candidate Flagged at Tier 1 (Bypasses Heavy Inference)
    if triage_state == 'HIGH_RISK_CANDIDATE_FLAGGED':
        # Early human escalation triage signal; generates evidence dossier
        dossier = EXTRACT_EVIDENCE_DOSSIER(
            D_A, D_B, M, state='HIGH_RISK_CANDIDATE_FLAGGED', 
            action='TIER1_DIRECT_FLAG', p_hat=None, C=None, entropy=None
        )
        return {
            'tier1_candidate_score': s_cand,
            'tier1_routing_state': 'HIGH_RISK_CANDIDATE_FLAGGED',
            'heavy_model_raw_score': None,       # Bypassed
            'calibrated_probability': None,      # Bypassed
            'conformal_prediction_set': None,    # Bypassed
            'predictive_entropy': None,          # Bypassed
            'human_review_required': True,
            'final_adjudication_state': 'HIGH_RISK_CANDIDATE_FLAGGED',
            'evidence_dossier': dossier
        }
        
    # BRANCH C: Uncertain Candidate Escalated to Tier-2 Heavy Model
    z_raw = f_heavy[M].predict_raw_score(D_A, D_B)
    p_cal = g_cal[M].predict(z_raw)
    
    # Uncertainty Routing (Modality Specific)
    if M == 'text':
        C_set, action, entropy, review_req = ROUTE_TEXT_CONFORMAL(p_cal, q_hat_text)
    elif M == 'code':
        C_set, action, entropy, review_req = ROUTE_CODE_EXPLORATORY_UNCERTAINTY(p_cal)
        
    # Evidence Dossier extracted strictly when review is required
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
        'final_adjudication_state': action,
        'evidence_dossier': dossier
    }
end function
```

---

## Algorithm 7: Complete Sequential Execution and Calibration Protocol (Issue 1)

```python
"""
ALGORITHM 7: Complete Sequential Execution and Calibration Protocol
Execution Stage: Full Experimental Pipeline Preparation
Enforces strict partition separation and threshold freezing sequence.

Partitions:
    D_train: Base model training (candidate generator, heavy models)
    D_val: Routing threshold optimization
    D_cal_prob: Probability calibrator fitting
    D_cal_conf: Conformal quantile estimation (Text)
    D_test: Untouched final evaluation (evaluated once)
"""

function RUN_SEQUENTIAL_PIPELINE_CALIBRATION(D_train, D_val, D_cal_prob, D_cal_conf, D_test, M, R_target, B_max, P_target, epsilon):
    # Step 1: Train Candidate Generator & Heavy Classifiers
    g_cand = TRAIN_CANDIDATE_GENERATOR(D_train, M)
    f_heavy = TRAIN_HEAVY_CLASSIFIER(D_train, M)
    
    # Step 2: Select Routing Thresholds on Validation Data
    theta_opt, pareto = OPTIMIZE_BUDGET_CONSTRAINED_THRESHOLDS(
        D_val, R_target, B_max, P_target, T_PROFILES[M]
    )
    assert theta_opt != 'STATUS: INFEASIBLE_BUDGET', "Optimization infeasible!"
    
    # Step 3: FREEZE Routing Thresholds
    theta_low, theta_high = theta_opt
    
    # Step 4 & 5: Fit Probability Calibrator on Frozen Escalation Population
    g_cal = FIT_SELECTION_AWARE_CALIBRATOR(
        D_cal_prob, M, g_cand, theta_low, theta_high, f_heavy, cal_type='platt'
    )
    
    # Step 6 & 7: Conformal Calibration (Text Only)
    q_hat = None
    if M == 'text':
        q_hat = CALIBRATE_TEXT_CONFORMAL_QUANTILE(
            D_cal_conf, g_cand, theta_low, theta_high, f_heavy, g_cal, epsilon
        )
        
    # Step 8: FREEZE the Complete Pipeline
    frozen_pipeline = {
        'modality': M,
        'g_cand': g_cand,
        'f_heavy': f_heavy,
        'g_cal': g_cal,
        'q_hat': q_hat,
        'theta_low': theta_low,
        'theta_high': theta_high
    }
    
    # Step 9: Final Single-Pass Evaluation on Untouched Held-Out Test Set
    test_results = []
    for test_pair in D_test:
        rec = EXECUTE_MASTER_PIPELINE_INSTANCE(test_pair, frozen_pipeline)
        test_results.append(rec)
        
    return test_results, frozen_pipeline
end function
```

---
*End of Document: algorithm_pseudocode_v3_2.md*

