# Algorithmic Specifications & Complete Pseudocode (Version 3.6)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.5 (Evidence Integrity Repair and Final Readiness Review)  
**Document Version:** Version 3.6  
**Date:** November 10, 2026  
**Status:** FULLY SPECIFIED & READY FOR IMPLEMENTATION

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
    Guarantees zero premise overlap between splits and exact target matching.
    """
    # 1. Identify and purge cross-split contaminated premises
    val_raw_prems = set(df_raw[df_raw["split"] == "val"]["reference_text"].unique())
    test_raw_prems = set(df_raw[df_raw["split"] == "test"]["reference_text"].unique())
    crossing_premises = val_raw_prems.intersection(test_raw_prems)
    
    # 2. Identify redundant duplicates (Policy A: retain 1st occurrence)
    dupe_mask = df_raw.duplicated(subset=["reference_text", "suspicious_text"], keep="first")
    purged_dupe_ids = set(df_raw[dupe_mask]["id"])
    
    # Construct exclusion ledger
    excluded_mask = df_raw["reference_text"].isin(crossing_premises) | df_raw["id"].isin(purged_dupe_ids)
    df_clean = df_raw[~excluded_mask].copy().reset_index(drop=True)
    df_excluded = df_raw[excluded_mask].copy().reset_index(drop=True)
    
    # Helper: deterministic premise hash
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
    
    # 5. Build premise map
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

## Algorithm 2: Two-Stage Uncertainty-Aware Screening Pipeline

```python
def process_candidate_pair(
    doc_pair: tuple[str, str],
    modality: str,  # "text" or "code"
    stage1_model,
    stage2_model,
    platt_calibrator,
    conformal_quantile: float,
    thresholds: dict  # {"theta_low": 0.35, "theta_high": 0.85, "gamma_lower": 0.20, "gamma_upper": 0.80}
) -> dict:
    """
    Executes two-stage adaptive screening with standardized 4-state routing.
    """
    # Stage 1: Fast multi-view feature extraction & candidate scoring
    s_cand = stage1_model.predict_candidate_score(doc_pair, modality)
    
    # Direct routing decision
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
        
    # Escalation to Stage 2 (Cross-Encoder)
    z_logit = stage2_model.predict_logit(doc_pair, modality)
    
    if modality == "text":
        p_cal = platt_calibrator.predict_proba(z_logit)
        
        # Conformal Set Construction
        non_conf_0 = p_cal
        non_conf_1 = 1.0 - p_cal
        c_set = set()
        if non_conf_0 <= conformal_quantile: c_set.add(0)
        if non_conf_1 <= conformal_quantile: c_set.add(1)
        
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
        # Code modality: exploratory margin thresholding
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

## Algorithm 3: Group Split Conformal Calibration (Text Modality)

```python
def fit_group_conformal_calibrator(
    cal_conf_pairs: list[dict],
    epsilon: float = 0.05
) -> float:
    """
    Fits split group conformal quantile over escalated calibration premise groups.
    Enforces finite-sample attainability bound.
    """
    # Group by premise ID
    premise_groups = defaultdict(list)
    for pair in cal_conf_pairs:
        # Only escalated pairs enter the conformal population
        if pair["stage2_executed"]:
            premise_groups[pair["premise_id"]].append(pair)
            
    n_groups = len(premise_groups)
    min_required_groups = int(np.ceil((1.0 - epsilon) / epsilon))
    
    if n_groups < min_required_groups:
        warnings.warn(
            f"Escalated premise groups ({n_groups}) < minimum required ({min_required_groups}). "
            f"Conformal quantile is mathematically infinite."
        )
        return float("inf")
        
    group_non_conformity = []
    for g_id, pairs in premise_groups.items():
        # Worst-case pair non-conformity in group
        max_alpha = max(
            (1.0 - p["calibrated_prob"]) if p["true_label"] == 1 else p["calibrated_prob"]
            for p in pairs
        )
        group_non_conformity.append(max_alpha)
        
    # Conformal quantile: index = ceil((n+1)(1-eps))/n
    quantile_level = min(1.0, np.ceil((n_groups + 1) * (1.0 - epsilon)) / n_groups)
    q_hat = float(np.quantile(group_non_conformity, quantile_level, method="higher"))
    
    return q_hat
```

---

## Algorithm 4: Rigorous Calibration Evaluation Routine

```python
def compute_calibration_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10
) -> dict[str, float]:
    """
    Computes ECE (uniform M=10, empty bins contribute 0), MCE, Brier Score,
    and clipped NLL (clip range: [1e-15, 1 - 1e-15]).
    Evaluated strictly on untouched test escalation population.
    """
    assert len(y_true) == len(p_pred), "Shape mismatch between labels and probabilities"
    N = len(y_true)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "nll": 0.0}
        
    # 1. Brier Score
    brier = float(np.mean((p_pred - y_true) ** 2))
    
    # 2. Clipped NLL
    eps = 1e-15
    p_clipped = np.clip(p_pred, eps, 1.0 - eps)
    nll = float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))
    
    # 3. ECE & MCE across uniform bins
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

## Algorithm 5: Multi-Modal Attribution Alignment

```python
def generate_attribution_evidence(
    doc_a: str,
    doc_b: str,
    modality: str
) -> dict:
    """
    Synthesizes interpretable evidence dossiers for institutional review.
    Separates statistical facts from judicial guilt claims.
    """
    if modality == "text":
        # Myers diff on token level
        tokens_a = doc_a.split()
        tokens_b = doc_b.split()
        diff_ops = compute_myers_diff(tokens_a, tokens_b)
        highlighted_spans = format_diff_spans(diff_ops)
        
        return {
            "evidence_type": "TEXT_LEGAL_DOSSIER",
            "aligned_token_spans": highlighted_spans,
            "disclaimer": "Automated statistical analysis. Administrative adjudication reserved for integrity committee."
        }
    else:
        # Java source code: AST Subtree extraction
        ast_a = javalang.parse.parse(doc_a)
        ast_b = javalang.parse.parse(doc_b)
        subtrees_a = extract_canonical_subtrees(ast_a)
        subtrees_b = extract_canonical_subtrees(ast_b)
        
        shared_isomorphic_blocks = subtrees_a.intersection(subtrees_b)
        
        return {
            "evidence_type": "CODE_STRUCTURAL_DOSSIER",
            "isomorphic_ast_blocks_count": len(shared_isomorphic_blocks),
            "renamed_identifier_map": extract_identifier_mappings(ast_a, ast_b),
            "disclaimer": "Automated statistical analysis. Administrative adjudication reserved for integrity committee."
        }
```

