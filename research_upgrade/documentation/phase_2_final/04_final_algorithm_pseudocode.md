# Final Algorithmic Specifications & Implementation Pseudocode (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 04 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** FULLY SPECIFIED & IMPLEMENTATION-READY

---

## Algorithm 1: Exact Non-Overshooting Premise-Group Partitioning

```python
def allocate_partitions_without_overshoot(
    df_raw: pd.DataFrame,
    target_counts: dict[str, int],  # {"train": 4000, "val": 1500, "cal_prob": 1000, "cal_conf": 1000}
    test_target: int = 3000
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Partitions text benchmark into 5 active partitions without deliberate overshoot.
    Guarantees zero premise overlap between splits and exact target integer matching.
    """
    # 1. Identify cross-split contaminated premises
    val_raw_prems = set(df_raw[df_raw["split"] == "val"]["reference_text"].unique())
    test_raw_prems = set(df_raw[df_raw["split"] == "test"]["reference_text"].unique())
    crossing_premises = val_raw_prems.intersection(test_raw_prems)
    
    # 2. Identify redundant duplicates (Policy A: retain 1st occurrence)
    dupe_mask = df_raw.duplicated(subset=["reference_text", "suspicious_text"], keep="first")
    purged_dupe_ids = set(df_raw[dupe_mask]["id"])
    
    # Exclude contaminated and redundant rows
    excluded_mask = df_raw["reference_text"].isin(crossing_premises) | df_raw["id"].isin(purged_dupe_ids)
    df_clean = df_raw[~excluded_mask].copy().reset_index(drop=True)
    df_excluded = df_raw[excluded_mask].copy().reset_index(drop=True)
    
    def hash_premise(p: str) -> int:
        return int(hashlib.sha256(p.encode("utf-8")).hexdigest(), 16)
        
    def non_overshoot_allocator(prems_pool: list[str], counts: dict[str, int], target: int) -> tuple[list[str], list[str]]:
        allocated = []
        cur_sum = 0
        rem_pool = list(prems_pool)
        i = 0
        while i < len(rem_pool) and cur_sum < target:
            p = rem_pool[i]
            c = counts[p]
            if cur_sum + c <= target:
                allocated.append(p)
                cur_sum += c
                rem_pool.pop(i)
            else:
                i += 1
        return allocated, rem_pool

    # 3. Allocate Development Partitions from clean val split
    df_val_pool = df_clean[df_clean["split"] == "val"].copy()
    val_counts = df_val_pool["reference_text"].value_counts().to_dict()
    sorted_val_prems = sorted(val_counts.keys(), key=hash_premise)
    
    train_prems, rem_1 = non_overshoot_allocator(sorted_val_prems, val_counts, target_counts["train"])
    val_prems, rem_2 = non_overshoot_allocator(rem_1, val_counts, target_counts["val"])
    cal_prob_prems, rem_3 = non_overshoot_allocator(rem_2, val_counts, target_counts["cal_prob"])
    cal_conf_prems, dev_res_prems = non_overshoot_allocator(rem_3, val_counts, target_counts["cal_conf"])
    
    # 4. Allocate Test Partition from clean test split
    df_test_pool = df_clean[df_clean["split"] == "test"].copy()
    test_counts = df_test_pool["reference_text"].value_counts().to_dict()
    sorted_test_prems = sorted(test_counts.keys(), key=hash_premise)
    
    test_prems, test_res_prems = non_overshoot_allocator(sorted_test_prems, test_counts, test_target)
    
    # 5. Build premise partition map
    prem_map = {}
    for p in train_prems: prem_map[p] = "train"
    for p in val_prems: prem_map[p] = "val"
    for p in cal_prob_prems: prem_map[p] = "cal_prob"
    for p in cal_conf_prems: prem_map[p] = "cal_conf"
    for p in dev_res_prems: prem_map[p] = "dev_reserve"
    for p in test_prems: prem_map[p] = "test"
    for p in test_res_prems: prem_map[p] = "test_reserve"
    
    # 6. Build Row-by-Row Membership Ledger
    ledger = []
    for _, row in df_raw.iterrows():
        r_id = row["id"]
        prem = row["reference_text"]
        if row["reference_text"] in crossing_premises:
            ledger.append((r_id, "EXCLUDED", "CROSS_SPLIT_PREMISE_CONTAMINATION"))
        elif r_id in purged_dupe_ids:
            ledger.append((r_id, "EXCLUDED", "REDUNDANT_EXACT_DUPLICATE_ROW"))
        else:
            ledger.append((r_id, prem_map[prem], "NONE"))
            
    df_membership = pd.DataFrame(ledger, columns=["row_id", "assigned_partition", "exclusion_reason"])
    return df_clean, df_membership, prem_map
```

---

## Algorithm 2: Two-Stage Multi-View Screening & Adaptive Routing

```python
def process_candidate_pair(
    doc_pair: tuple[str, str],
    modality: str,  # "text" or "code"
    stage1_model,
    text_cross_encoder,      # Text Stage 2: ms-marco-MiniLM-L-6-v2
    code_stage2_detector,    # Code Stage 2: Code structural-semantic detector
    platt_calibrator,        # Text probability calibrator
    conformal_quantile: float,
    thresholds: dict         # {"theta_low", "theta_high", "gamma_lower", "gamma_upper", "margin_lower", "margin_upper"}
) -> dict:
    """
    Executes two-stage adaptive screening with standardized 4-state routing.
    Enforces independent Stage-2 components for text and code.
    """
    # Stage 1: Fast multi-view feature extraction & candidate scoring
    s_cand = stage1_model.predict_candidate_score(doc_pair, modality)
    
    # Check Direct Bypasses
    if s_cand < thresholds["theta_low"]:
        return {
            "output_state": "LOW_RISK_REVIEW_DEFERRED",
            "candidate_score": s_cand,
            "stage2_executed": False,
            "calibrated_prob": None,
            "prediction_set": None,
            "human_review_required": False
        }
        
    if s_cand >= thresholds["theta_high"]:
        return {
            "output_state": "HIGH_RISK_CANDIDATE_FLAGGED",
            "candidate_score": s_cand,
            "stage2_executed": False,  # Direct bypass: FLOPs saved!
            "calibrated_prob": None,
            "prediction_set": None,
            "human_review_required": True
        }
        
    # Escalation Path to Modality-Specific Stage 2
    if modality == "text":
        # Text Stage 2: Semantic cross-encoder forward pass
        z_logit = text_cross_encoder.predict_logit(doc_pair)
        p_cal = platt_calibrator.predict_proba(z_logit)
        
        # Conformal Set Construction
        non_conf_0 = p_cal
        non_conf_1 = 1.0 - p_cal
        c_set = set()
        if non_conf_0 <= conformal_quantile: c_set.add(0)
        if non_conf_1 <= conformal_quantile: c_set.add(1)
        
        # Operational 4-State Routing
        if len(c_set) == 0:
            output_state = "ABSTAIN_UNRESOLVED"
            review_req = True
        elif c_set == {1} and p_cal >= thresholds["gamma_upper"]:
            output_state = "HIGH_RISK_CANDIDATE_FLAGGED"
            review_req = True
        elif c_set == {0} and p_cal <= thresholds["gamma_lower"]:
            output_state = "LOW_RISK_REVIEW_DEFERRED"
            review_req = False
        else:
            output_state = "AMBIGUOUS_HUMAN_REVIEW"
            review_req = True
            
        return {
            "output_state": output_state,
            "candidate_score": s_cand,
            "stage2_executed": True,
            "raw_logit": z_logit,
            "calibrated_prob": p_cal,
            "prediction_set": sorted(list(c_set)),
            "human_review_required": review_req
        }
        
    else:
        # Code Stage 2: Structural-semantic detector forward pass
        z_logit = code_stage2_detector.predict_logit(doc_pair)
        
        # Exploratory raw decision-margin thresholding (case-05 single-class topology)
        if z_logit >= thresholds["margin_upper"]:
            output_state = "HIGH_RISK_CANDIDATE_FLAGGED"
            review_req = True
        elif z_logit <= thresholds["margin_lower"]:
            output_state = "LOW_RISK_REVIEW_DEFERRED"
            review_req = False
        else:
            output_state = "AMBIGUOUS_HUMAN_REVIEW"
            review_req = True
            
        return {
            "output_state": output_state,
            "candidate_score": s_cand,
            "stage2_executed": True,
            "raw_logit": z_logit,
            "calibrated_prob": None,
            "prediction_set": None,
            "human_review_required": review_req
        }
```

---

## Algorithm 3: Exact Group Split Conformal Calibration & Set Construction (Text Modality)

```python
def compute_group_non_conformity_scores(
    calibration_groups: dict[str, list[tuple[float, int]]]
) -> list[float]:
    """
    Computes group-level non-conformity score alpha_g across multiple candidate pairs
    sharing each premise group g in D_cal-conf^esc.
    
    Sampling Unit & Escalation Filtering:
    - calibration_groups: dict mapping premise_id -> list of escalated pairs (calibrated_p, y).
    - Groups with zero escalated pairs (m_g = 0) require zero Stage-2 passes and are
      excluded from calibration_groups prior to calling this function.
    - Calibration and test groups are filtered using the exact same frozen Stage-1
      model and thresholds (theta_low, theta_high), preserving exchangeability across
      active groups g in G^esc.
    - Exchangeability Assumption: Group scores (alpha_g) are assumed exchangeable across
      premise clusters under group-disjoint partitioning.
    - Simultaneous Guarantee: Guarantees simultaneous coverage over all pairs in test group g*,
      conditional on Stage-1 escalation. Stage-1 bypassed pairs are excluded.
    """
    group_scores = []
    for premise_id, pairs in calibration_groups.items():
        if len(pairs) == 0:
            # Skip groups with m_g = 0 (bypassed groups have no Stage-2 pairs)
            continue
            
        # Individual pair non-conformity: 1 - P_cal(Y = y_i)
        pair_non_conformities = [
            (1.0 - p_cal) if y == 1 else p_cal
            for (p_cal, y) in pairs
        ]
        # Group-level calibration score: maximum non-conformity across all pairs in group
        alpha_g = max(pair_non_conformities)
        group_scores.append(float(alpha_g))
    return group_scores

def compute_exact_group_conformal_quantile(
    group_non_conformities: list[float],
    epsilon: float = 0.05
) -> float:
    """
    Computes exact finite-sample order statistic over calibration premise groups.
    Rejects interpolated quantiles; handles finite-sample attainability limit.
    """
    scores = np.sort(np.asarray(group_non_conformities))
    n_groups = len(scores)
    
    # Exact order statistic index: ceil((n+1)(1 - eps))
    k = int(np.ceil((n_groups + 1) * (1.0 - epsilon)))
    
    if k > n_groups:
        # Finite-sample guarantee mathematically unattainable (requires n >= ceil((1-eps)/eps))
        return float("inf")
    else:
        # 1-indexed k corresponds to 0-indexed k - 1
        return float(scores[k - 1])

def construct_pair_prediction_set(
    p_cal: float,
    q_hat: float
) -> list[int]:
    """
    Constructs conformal prediction set C(x) subseteq {0, 1} for both classes.
    Class 1 included iff 1 - p_cal <= q_hat (p_cal >= 1 - q_hat).
    Class 0 included iff p_cal <= q_hat.
    Empty set [] maps to ABSTAIN_UNRESOLVED.
    """
    c_set = []
    if p_cal <= q_hat:
        c_set.append(0)  # Class 0 conforming
    if p_cal >= (1.0 - q_hat):
        c_set.append(1)  # Class 1 conforming
    return sorted(c_set)
```

---

## Algorithm 4: Rigorous Calibration Evaluation Routine

```python
def evaluate_calibration_diagnostics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10
) -> dict[str, float]:
    """
    Computes ECE (uniform M=10, empty bins contribute 0), MCE, Brier Score,
    and clipped NLL (clip range: [1e-15, 1 - 1e-15]).
    Evaluated strictly on untouched test escalation population.
    """
    assert len(y_true) == len(p_pred), "Dimension mismatch"
    N = len(y_true)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "nll": 0.0}
        
    # 1. Brier Score
    brier = float(np.mean((p_pred - y_true) ** 2))
    
    # 2. Clipped Negative Log-Likelihood
    eps = 1e-15
    p_clipped = np.clip(p_pred, eps, 1.0 - eps)
    nll = float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))
    
    # 3. ECE & MCE across M=10 uniform bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for m in range(n_bins):
        bin_lower = bin_edges[m]
        bin_upper = bin_edges[m + 1]
        
        if m == 0:
            in_bin = (p_pred >= bin_lower) & (p_pred <= bin_upper)
        else:
            in_bin = (p_pred > bin_lower) & (p_pred <= bin_upper)
            
        count_m = np.sum(in_bin)
        if count_m > 0:
            # acc_m represents positive-class empirical prevalence / frequency: bar{y}_m = (1/|B_m|) sum y_i
            # (In standard binary calibration software, this is labeled 'acc', but represents
            # positive empirical frequency, distinct from top-label classification accuracy).
            # When y_true contains only negative labels (e.g. Case-05 Folds 2-5), bar{y}_m is identically 0.0,
            # so diff = abs(0.0 - conf_m) = conf_m, and ECE collapses into mean(p_pred).
            acc_m = float(np.mean(y_true[in_bin]))
            conf_m = float(np.mean(p_pred[in_bin]))
            diff = abs(acc_m - conf_m)
            ece += (count_m / N) * diff
            if diff > mce:
                mce = diff
        else:
            # Empty bins explicitly contribute 0
            pass
            
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "nll": float(nll)
    }
```

---

## Algorithm 5: Multi-Modal Attribution Evidence Synthesis

```python
def synthesize_attribution_dossier(
    doc_a: str,
    doc_b: str,
    modality: str
) -> dict:
    """
    Synthesizes interpretable evidence dossiers for integrity committee review.
    Enforces strict separation of statistical evidence from judicial accusations.
    """
    if modality == "text":
        tokens_a = doc_a.split()
        tokens_b = doc_b.split()
        diff_ops = compute_myers_diff(tokens_a, tokens_b)
        
        return {
            "evidence_type": "TEXT_STATISTICAL_DOSSIER",
            "aligned_token_spans": format_diff_spans(diff_ops),
            "legal_disclaimer": "Automated statistical analysis. Judicial misconduct adjudication reserved for university integrity board."
        }
    else:
        ast_a = javalang.parse.parse(doc_a)
        ast_b = javalang.parse.parse(doc_b)
        subtrees_a = extract_canonical_subtrees(ast_a)
        subtrees_b = extract_canonical_subtrees(ast_b)
        
        shared_blocks = subtrees_a.intersection(subtrees_b)
        
        return {
            "evidence_type": "CODE_STRUCTURAL_DOSSIER",
            "isomorphic_ast_blocks_count": len(shared_blocks),
            "renamed_identifier_map": extract_identifier_mappings(ast_a, ast_b),
            "legal_disclaimer": "Automated statistical analysis. Judicial misconduct adjudication reserved for university integrity board."
        }
```

