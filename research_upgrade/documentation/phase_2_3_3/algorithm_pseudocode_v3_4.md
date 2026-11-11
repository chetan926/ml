# Algorithmic Pseudocode and Data Structures (Version 3.4)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2.3.3 (Final Dataset, Routing, and Integrity Verification)  
**Document Version:** Version 3.4  
**Date:** March 2026  
**Status:** FULL AUDIT CERTIFIED (Empirical Ground-Truth Verified)

---

## 1. Core Data Structures and Typed Enumerations

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
import numpy as np

class OutputState(Enum):
    LOW_RISK_REVIEW_DEFERRED = "LOW_RISK_REVIEW_DEFERRED"
    HIGH_RISK_CANDIDATE_FLAGGED = "HIGH_RISK_CANDIDATE_FLAGGED"
    AMBIGUOUS_HUMAN_REVIEW = "AMBIGUOUS_HUMAN_REVIEW"
    ABSTAIN_UNRESOLVED = "ABSTAIN_UNRESOLVED"

@dataclass
class CandidatePair:
    pair_id: str
    doc_a_id: str
    doc_b_id: str
    text_or_code_a: str
    text_or_code_b: str
    premise_or_task_id: str
    ground_truth_label: Optional[int] = None

@dataclass
class MultiViewFeatures:
    lexical_char_sim: float
    lexical_word_sim: float
    dense_semantic_sim: float
    structural_or_ling_sim: float
    fused_candidate_score: float

@dataclass
class PipelineResult:
    pair_id: str
    candidate_score: float
    stage2_executed: bool
    calibrated_probability: Optional[float]
    conformal_prediction_set: Optional[Set[int]]
    final_output_state: OutputState
    execution_latency_ms: float
    evidence_dossier_generated: bool
```

---

## 2. Algorithm 1: Multi-View Screening and Stage-1 Routing

```python
def extract_text_multi_view(doc_a: str, doc_b: str, models: Dict) -> MultiViewFeatures:
    """
    Extracts 4 complementary views for text candidate pairs.
    """
    # 1. Character n-gram TF-IDF cosine similarity
    s_char = models['char_tfidf'].compute_cosine(doc_a, doc_b)
    
    # 2. Word n-gram TF-IDF cosine similarity
    s_word = models['word_tfidf'].compute_cosine(doc_a, doc_b)
    
    # 3. Dense semantic cosine similarity (all-MiniLM-L6-v2)
    emb_a = models['sbert'].encode(doc_a)
    emb_b = models['sbert'].encode(doc_b)
    s_sem = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
    
    # 4. Surface linguistic complexity similarity
    s_ling = models['ling_extractor'].compute_similarity(doc_a, doc_b)
    
    # Linear fusion with validation-optimized weights
    v = np.array([s_char, s_word, s_sem, s_ling])
    w = models['fusion_weights']
    b = models['fusion_bias']
    s_cand = float(1.0 / (1.0 + np.exp(-(np.dot(w, v) + b))))
    
    return MultiViewFeatures(s_char, s_word, s_sem, s_ling, s_cand)

def route_stage1_candidate(features: MultiViewFeatures, theta_low: float, theta_high: float) -> str:
    """
    Routes candidate based on screening thresholds.
    """
    if features.fused_candidate_score < theta_low:
        return "DEFER"
    elif features.fused_candidate_score >= theta_high:
        return "DIRECT_BYPASS"
    else:
        return "ESCALATE"
```

---

## 3. Algorithm 2: Probability Calibration and Group Conformal Fitting

```python
def fit_calibration_and_conformal_text(
    d_cal_prob: List[CandidatePair],
    d_cal_conf: List[CandidatePair],
    theta_low: float,
    theta_high: float,
    epsilon: float,
    stage1_models: Dict,
    stage2_cross_encoder: object
) -> Tuple[object, float]:
    """
    Executes Steps 4, 5, and 6 of the canonical 7-step calibration lifecycle.
    """
    # Step 4: Filter strictly to escalated slice for probability calibration
    escalated_cal_prob = []
    y_cal_prob = []
    logits_cal_prob = []
    
    for pair in d_cal_prob:
        mv = extract_text_multi_view(pair.text_or_code_a, pair.text_or_code_b, stage1_models)
        if theta_low <= mv.fused_candidate_score < theta_high:
            logit = stage2_cross_encoder.predict_logit(pair.text_or_code_a, pair.text_or_code_b)
            escalated_cal_prob.append(pair)
            y_cal_prob.append(pair.ground_truth_label)
            logits_cal_prob.append(logit)
            
    # Step 5: Fit Platt scaling parameters (A*, B*)
    platt_calibrator = fit_platt_scaling(np.array(logits_cal_prob), np.array(y_cal_prob))
    
    # Step 6: Filter strictly to escalated slice for conformal calibration
    premise_to_escalated_pairs = {}
    for pair in d_cal_conf:
        mv = extract_text_multi_view(pair.text_or_code_a, pair.text_or_code_b, stage1_models)
        if theta_low <= mv.fused_candidate_score < theta_high:
            g = pair.premise_or_task_id
            if g not in premise_to_escalated_pairs:
                premise_to_escalated_pairs[g] = []
            premise_to_escalated_pairs[g].append(pair)
            
    # Compute group non-conformity scores (worst-case pair error in cluster)
    group_alpha_scores = []
    for g, pairs in premise_to_escalated_pairs.items():
        pair_alphas = []
        for pair in pairs:
            logit = stage2_cross_encoder.predict_logit(pair.text_or_code_a, pair.text_or_code_b)
            p_hat = platt_calibrator.predict_proba(logit)
            true_y = pair.ground_truth_label
            alpha_i = 1.0 - (p_hat if true_y == 1 else (1.0 - p_hat))
            pair_alphas.append(alpha_i)
        group_alpha_scores.append(max(pair_alphas))
        
    n = len(group_alpha_scores)
    # Check sample size attainability condition: n >= ceil((1 - epsilon) / epsilon)
    min_required_n = int(np.ceil((1.0 - epsilon) / epsilon))
    if n < min_required_n:
        # Cannot attain finite-sample guarantee; abstain from finite cutoff
        q_hat = float('inf')
    else:
        k = int(np.ceil((n + 1) * (1.0 - epsilon)))
        sorted_alphas = sorted(group_alpha_scores)
        q_hat = sorted_alphas[min(k - 1, n - 1)]
        
    return platt_calibrator, q_hat
```

---

## 4. Algorithm 3: End-to-End Decision Routing (Academic Text Modality)

```python
def evaluate_text_pair(
    pair: CandidatePair,
    theta_low: float,
    theta_high: float,
    gamma_lower: float,
    gamma_upper: float,
    platt_calibrator: object,
    q_hat: float,
    stage1_models: Dict,
    stage2_cross_encoder: object
) -> PipelineResult:
    """
    Evaluates an incoming candidate pair emitting one of the 4 standardized states.
    """
    mv = extract_text_multi_view(pair.text_or_code_a, pair.text_or_code_b, stage1_models)
    
    # 1. Low-risk screening deferral
    if mv.fused_candidate_score < theta_low:
        return PipelineResult(
            pair_id=pair.pair_id,
            candidate_score=mv.fused_candidate_score,
            stage2_executed=False,
            calibrated_probability=None,
            conformal_prediction_set=None,
            final_output_state=OutputState.LOW_RISK_REVIEW_DEFERRED,
            execution_latency_ms=0.045,
            evidence_dossier_generated=False
        )
        
    # 2. Direct high-risk candidate bypass
    if mv.fused_candidate_score >= theta_high:
        return PipelineResult(
            pair_id=pair.pair_id,
            candidate_score=mv.fused_candidate_score,
            stage2_executed=False,
            calibrated_probability=None,
            conformal_prediction_set=None,
            final_output_state=OutputState.HIGH_RISK_CANDIDATE_FLAGGED,
            execution_latency_ms=0.045,
            evidence_dossier_generated=True
        )
        
    # 3. Escalation to Stage-2 Cross-Encoder
    logit = stage2_cross_encoder.predict_logit(pair.text_or_code_a, pair.text_or_code_b)
    p_hat = platt_calibrator.predict_proba(logit)
    
    # Construct conformal prediction set C(x)
    conformal_set = set()
    if (1.0 - p_hat) <= q_hat:
        conformal_set.add(1)
    if (1.0 - (1.0 - p_hat)) <= q_hat:
        conformal_set.add(0)
        
    # Standardized 4-State Decision Logic
    if len(conformal_set) == 0:
        final_state = OutputState.ABSTAIN_UNRESOLVED
        generate_dossier = True
    elif conformal_set == {1} and p_hat >= gamma_upper:
        final_state = OutputState.HIGH_RISK_CANDIDATE_FLAGGED
        generate_dossier = True
    elif conformal_set == {0} and p_hat <= gamma_lower:
        final_state = OutputState.LOW_RISK_REVIEW_DEFERRED
        generate_dossier = False
    else:
        # Ambiguous: non-singleton {0, 1} or probability in uncertainty margin
        final_state = OutputState.AMBIGUOUS_HUMAN_REVIEW
        generate_dossier = True
        
    return PipelineResult(
        pair_id=pair.pair_id,
        candidate_score=mv.fused_candidate_score,
        stage2_executed=True,
        calibrated_probability=p_hat,
        conformal_prediction_set=conformal_set,
        final_output_state=final_state,
        execution_latency_ms=4.091,
        evidence_dossier_generated=generate_dossier
    )
```

---

## 5. Algorithm 4: Source Code Raw Margin Decision Routing (3-State Schema)

```python
def evaluate_code_pair(
    pair: CandidatePair,
    theta_low: float,
    theta_high: float,
    gamma_lower_code: float,
    gamma_upper_code: float,
    stage1_code_models: Dict,
    stage2_code_model: object
) -> PipelineResult:
    """
    Evaluates a Java code pair emitting one of 3 permitted states (conformal disclaimed).
    """
    s_cand = extract_code_multi_view(pair.text_or_code_a, pair.text_or_code_b, stage1_code_models)
    
    # Stage 1 Screening
    if s_cand < theta_low:
        return PipelineResult(
            pair_id=pair.pair_id,
            candidate_score=s_cand,
            stage2_executed=False,
            calibrated_probability=None,
            conformal_prediction_set=None,
            final_output_state=OutputState.LOW_RISK_REVIEW_DEFERRED,
            execution_latency_ms=0.058,
            evidence_dossier_generated=False
        )
    elif s_cand >= theta_high:
        return PipelineResult(
            pair_id=pair.pair_id,
            candidate_score=s_cand,
            stage2_executed=False,
            calibrated_probability=None,
            conformal_prediction_set=None,
            final_output_state=OutputState.HIGH_RISK_CANDIDATE_FLAGGED,
            execution_latency_ms=0.058,
            evidence_dossier_generated=True
        )
        
    # Stage 2 Raw Margin Evaluation
    z = stage2_code_model.compute_raw_margin(pair.text_or_code_a, pair.text_or_code_b)
    
    if z >= gamma_upper_code:
        final_state = OutputState.HIGH_RISK_CANDIDATE_FLAGGED
        generate_dossier = True
    elif z <= gamma_lower_code:
        final_state = OutputState.LOW_RISK_REVIEW_DEFERRED
        generate_dossier = False
    else:
        final_state = OutputState.AMBIGUOUS_HUMAN_REVIEW
        generate_dossier = True
        
    return PipelineResult(
        pair_id=pair.pair_id,
        candidate_score=s_cand,
        stage2_executed=True,
        calibrated_probability=None,
        conformal_prediction_set=None,
        final_output_state=final_state,
        execution_latency_ms=2.108,
        evidence_dossier_generated=generate_dossier
    )
```

---

## 6. Algorithm 5: Exact Binomial McNemar Test & Clustered Bootstrap

```python
from scipy.stats import binom

def exact_binomial_mcnemar(y_true: np.ndarray, y_pred_1: np.ndarray, y_pred_2: np.ndarray) -> Tuple[int, int, float]:
    """
    Computes exact two-sided Binomial McNemar p-value for paired binary classifiers.
    """
    corr_1 = (y_pred_1 == y_true)
    corr_2 = (y_pred_2 == y_true)
    
    # Discordant counts
    b = int(np.sum(corr_1 & (~corr_2))) # Proposed correct, baseline wrong
    c = int(np.sum((~corr_1) & corr_2)) # Proposed wrong, baseline correct
    
    n_disc = b + c
    if n_disc == 0:
        return b, c, 1.0
        
    k = min(b, c)
    # Two-sided exact binomial p-value under p=0.5
    p_val = min(1.0, 2.0 * float(binom.cdf(k, n_disc, 0.5)))
    return b, c, p_val

def clustered_bootstrap_ci(
    pairs_df: pd.DataFrame,
    cluster_col: str,
    metric_fn: callable,
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes percentile confidence interval via cluster-level resampling with replacement.
    """
    unique_clusters = pairs_df[cluster_col].unique()
    n_clusters = len(unique_clusters)
    boot_stats = []
    
    for _ in range(n_resamples):
        resampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        # Gather all pairs from selected clusters
        sample_df = pairs_df[pairs_df[cluster_col].isin(resampled_clusters)]
        boot_stats.append(metric_fn(sample_df))
        
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_stats, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0)))
    original_stat = float(metric_fn(pairs_df))
    
    return original_stat, ci_lower, ci_upper
```

