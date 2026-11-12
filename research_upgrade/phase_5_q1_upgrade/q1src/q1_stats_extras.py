"""
Phase 5 Statistical & Operating-Point Utilities.

New beyond Phase 4.1:
    1. Wilson score confidence intervals for empirical conformal coverage
       (answers the reviewer question: is 93.69% group coverage statistically
       consistent with the nominal 95% target?).
    2. Iso-recall threshold tuning: every compared method is tuned on
       VALIDATION data only to a common recall target, enabling the
       recall-matched precision/FPR comparison required for the paper's
       "proposed outperforms" claim.
    3. Validation-only operating-point grid search for the tri-band entropy
       router under an explicit Stage-2 budget cap.
"""

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score


def _fast_prf_mcc(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Microsecond-level (precision, recall, f1, mcc) for binary vectors —
    used inside grid-search loops; sklearn metric calls carry per-call
    validation overhead that dominates tight loops."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num / den if den > 0 else 0.0
    return float(prec), float(rec), float(f1), float(mcc)


def wilson_score_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (p_hat, ci_lower, ci_upper). Uses z=1.96 for alpha=0.05
    (normal quantile via statistics — no scipy dependency at call time).
    """
    from statistics import NormalDist

    if n <= 0:
        return 0.0, 0.0, 0.0
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = p_hat + z * z / (2.0 * n)
    half = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n)
    lower = (centre - half) / denom
    upper = (centre + half) / denom
    return float(p_hat), float(lower), float(upper)


def tune_threshold_for_f1(y_true: np.ndarray, scores: np.ndarray,
                          grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Tunes a decision threshold on validation to maximize F1. Returns (thr, best_f1)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if grid is None:
        grid = np.round(np.arange(0.02, 0.9801, 0.01), 4)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def tune_threshold_for_iso_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall_target: float,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Tunes a threshold on validation to achieve recall >= recall_target while
    maximizing precision. If infeasible, falls back to the highest-precision
    threshold among those maximizing recall.

    Returns (thr, achieved_val_recall).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if grid is None:
        grid = np.round(np.arange(0.005, 0.9981, 0.005), 4)

    feasible: list = []
    best_fallback = (0.5, 0.0)
    best_fallback_recall = -1.0
    for thr in grid:
        pred = (scores >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if rec >= recall_target:
            feasible.append((prec, float(thr), float(rec)))
        if rec > best_fallback_recall or (rec == best_fallback_recall and prec > best_fallback[1]):
            best_fallback_recall = float(rec)
            best_fallback = (float(thr), float(prec))
    if feasible:
        # Highest validation precision; ties broken toward the larger threshold (conservative).
        feasible.sort(key=lambda t: (-t[0], -t[1]))
        prec, thr, rec = feasible[0]
        return thr, float(rec)
    return best_fallback[0], best_fallback_recall


def tune_routing_operating_point(
    y_val: np.ndarray,
    p_val_calibrated: np.ndarray,
    entropy_val: np.ndarray,
    s2_scores_val: np.ndarray,
    stage2_budget_cap: float = 0.70,
    min_val_recall: float = 0.0,
) -> Dict[str, Any]:
    """
    Grid-searches the tri-band routing operating point (tau_low, tau_high,
    entropy threshold, stage-2 decision threshold) on VALIDATION ONLY.

    Objective: maximize validation MCC subject to Stage-2 escalation ratio
    <= stage2_budget_cap. Tie-break: fewer escalations.

    Returns the selected configuration dict (plus val metrics).
    """
    y_val = np.asarray(y_val).astype(int)
    p = np.asarray(p_val_calibrated, dtype=float)
    ent = np.asarray(entropy_val, dtype=float)
    s2 = np.asarray(s2_scores_val, dtype=float)
    n = len(y_val)

    best: Optional[Dict[str, Any]] = None
    tau_low_grid = np.round(np.arange(0.05, 0.451, 0.05), 4)
    tau_high_grid = np.round(np.arange(0.55, 0.951, 0.05), 4)
    entropy_grid = np.round(np.arange(0.30, 0.951, 0.05), 4)
    s2_thr_grid = np.round(np.arange(0.35, 0.651, 0.05), 4)

    for tl in tau_low_grid:
        for th in tau_high_grid:
            base_escalate = ((p >= tl) & (p <= th))
            for h in entropy_grid:
                escalate = base_escalate | (ent >= h)
                esc_ratio = float(np.mean(escalate))
                if esc_ratio > stage2_budget_cap:
                    continue
                decisive = ~escalate
                base_pred = (p >= 0.5).astype(int)
                for st in s2_thr_grid:
                    pred = np.where(escalate, (s2 >= st).astype(int), base_pred)
                    prec, rec, f1, mcc = _fast_prf_mcc(y_val, pred)
                    if rec < min_val_recall:
                        continue
                    key = (mcc, -esc_ratio)
                    if best is None or key > best["_key"]:
                        best = {
                            "_key": key,
                            "tau_low": float(tl),
                            "tau_high": float(th),
                            "entropy_threshold": float(h),
                            "stage2_decision_threshold": float(st),
                            "val_escalation_ratio": esc_ratio,
                            "val_mcc": float(mcc),
                            "val_f1": f1,
                            "val_accuracy": float(np.mean(pred == y_val)),
                            "val_precision": prec,
                            "val_recall": rec,
                        }
    if best is None:
        # No config satisfies the cap: escalate everything, decide by S2 at 0.5.
        pred = (s2 >= 0.5).astype(int)
        best = {
            "_key": (matthews_corrcoef(y_val, pred), -1.0),
            "tau_low": 0.0,
            "tau_high": 1.0,
            "entropy_threshold": 0.0,
            "stage2_decision_threshold": 0.5,
            "val_escalation_ratio": 1.0,
            "val_mcc": float(matthews_corrcoef(y_val, pred)),
            "val_f1": float(f1_score(y_val, pred, zero_division=0)),
        }
    best.pop("_key", None)
    return best


def tune_routing_for_iso_recall(
    y_val: np.ndarray,
    p_val_calibrated: np.ndarray,
    entropy_val: np.ndarray,
    s2_scores_val: np.ndarray,
    recall_target: float,
    budget_caps: Iterable[float] = (0.70, 0.85, 1.00),
) -> Dict[str, Any]:
    """
    Recall-matched operating-point tuning on VALIDATION ONLY: reaches
    recall >= recall_target with maximal validation precision, using the
    smallest Stage-2 escalation budget that makes the target feasible.
    """
    y_val = np.asarray(y_val).astype(int)
    p = np.asarray(p_val_calibrated, dtype=float)
    ent = np.asarray(entropy_val, dtype=float)
    s2 = np.asarray(s2_scores_val, dtype=float)
    n = len(y_val)

    for cap in budget_caps:
        best: Optional[Dict[str, Any]] = None
        for tl in np.round(np.arange(0.02, 0.501, 0.04), 4):
            for th in np.round(np.arange(0.50, 0.981, 0.04), 4):
                base_esc = ((p >= tl) & (p <= th))
                for h in np.round(np.arange(0.20, 1.001, 0.05), 4):
                    esc = base_esc | (ent >= h)
                    esc_ratio = float(np.mean(esc))
                    if esc_ratio > cap:
                        continue
                    base_pred = (p >= 0.5).astype(int)
                    for st in np.round(np.arange(0.20, 0.651, 0.05), 4):
                        pred = np.where(esc, (s2 >= st).astype(int), base_pred)
                        prec, rec, f1f, mcc = _fast_prf_mcc(y_val, pred)
                        if rec < recall_target:
                            continue
                        key = (prec, float(rec), -esc_ratio)
                        if best is None or key > best["_key"]:
                            best = {
                                "_key": key,
                                "tau_low": float(tl), "tau_high": float(th),
                                "entropy_threshold": float(h),
                                "stage2_decision_threshold": float(st),
                                "val_escalation_ratio": esc_ratio,
                                "val_recall": float(rec), "val_precision": float(prec),
                                "val_mcc": mcc,
                                "val_f1": f1f,
                            }
        if best is not None:
            best.pop("_key", None)
            best["budget_cap_used"] = cap
            return best

    # Infeasible: full escalation, S2 at low threshold maximizes recall.
    pred = (s2 >= 0.30).astype(int)
    return {
        "tau_low": 0.0, "tau_high": 1.0, "entropy_threshold": 0.0,
        "stage2_decision_threshold": 0.30,
        "val_escalation_ratio": 1.0,
        "val_recall": float(recall_score(y_val, pred, zero_division=0)),
        "val_precision": float(precision_score(y_val, pred, zero_division=0)),
        "val_mcc": float(matthews_corrcoef(y_val, pred)),
        "val_f1": float(f1_score(y_val, pred, zero_division=0)),
        "budget_cap_used": 1.0,
    }


def tune_routing_cost_minimizer(
    y_val: np.ndarray,
    p_val_calibrated: np.ndarray,
    entropy_val: np.ndarray,
    s2_scores_val: np.ndarray,
    epsilon_mcc: float = 0.01,
) -> Dict[str, Any]:
    """
    Cost-minimizing operating-point tuning on validation data: selects the
    configuration with the FEWEST Stage-2 escalations whose validation MCC stays
    within epsilon_mcc of the full-escalation (beta=1) reference MCC. Tie-break:
    higher MCC. This yields routing that provably preserves heavy-verifier
    accuracy (up to epsilon on validation) at minimal cost.
    """
    y_val = np.asarray(y_val).astype(int)
    p = np.asarray(p_val_calibrated, dtype=float)
    ent = np.asarray(entropy_val, dtype=float)
    s2 = np.asarray(s2_scores_val, dtype=float)
    n = len(y_val)

    ref_pred = (s2 >= 0.5).astype(int)
    _r_prec, ref_recall, _r_f1_d, ref_mcc = _fast_prf_mcc(y_val, ref_pred)
    ref_f1 = f1_score(y_val, ref_pred, zero_division=0)

    best: Optional[Dict[str, Any]] = None
    for tl in np.round(np.arange(0.02, 0.501, 0.04), 4):
        for th in np.round(np.arange(0.50, 0.981, 0.04), 4):
            base_esc = ((p >= tl) & (p <= th))
            for h in np.round(np.arange(0.20, 1.001, 0.05), 4):
                esc = base_esc | (ent >= h)
                esc_ratio = float(np.mean(esc))
                base_pred = (p >= 0.5).astype(int)
                for st in np.round(np.arange(0.25, 0.651, 0.05), 4):
                    pred = np.where(esc, (s2 >= st).astype(int), base_pred)
                    _prec, rec, f1f, mcc = _fast_prf_mcc(y_val, pred)
                    if mcc < ref_mcc - epsilon_mcc:
                        continue
                    key = (-esc_ratio, mcc)
                    if best is None or key > best["_key"]:
                        best = {
                            "_key": key,
                            "tau_low": float(tl), "tau_high": float(th),
                            "entropy_threshold": float(h),
                            "stage2_decision_threshold": float(st),
                            "val_escalation_ratio": esc_ratio,
                            "val_mcc": mcc, "val_recall": rec,
                            "val_f1": f1f,
                        }
    if best is None:
        best = {
            "_key": (-1.0, ref_mcc),
            "tau_low": 0.0, "tau_high": 1.0, "entropy_threshold": 0.0,
            "stage2_decision_threshold": 0.5, "val_escalation_ratio": 1.0,
            "val_mcc": ref_mcc, "val_recall": ref_recall,
            "val_f1": float(ref_f1),
        }
    best.pop("_key", None)
    best["reference_beta1_mcc"] = ref_mcc
    best["objective"] = f"minimize escalations s.t. val MCC >= beta1_MCC - {epsilon_mcc}"
    return best


def apply_routing(
    p_calibrated: np.ndarray,
    entropy: np.ndarray,
    s2_scores: np.ndarray,
    tau_low: float,
    tau_high: float,
    entropy_threshold: float,
    stage2_decision_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applies a tuned tri-band operating point. Returns
    (final_predictions, decision_states, routing_cost_summary).
    """
    p = np.asarray(p_calibrated, dtype=float)
    ent = np.asarray(entropy, dtype=float)
    s2 = np.asarray(s2_scores, dtype=float)
    n = len(p)

    states = np.empty(n, dtype=object)
    pred = np.zeros(n, dtype=int)
    escalate = ((p >= tau_low) & (p <= tau_high)) | (ent >= entropy_threshold)
    decisive = ~escalate

    states[decisive & (p < 0.5)] = "LOW_RISK_PRUNED"
    states[decisive & (p >= 0.5)] = "HIGH_RISK_FLAGGED"
    states[escalate] = "UNCERTAIN_ESCALATED"

    pred[decisive] = (p[decisive] >= 0.5).astype(int)
    pred[escalate] = (s2[escalate] >= stage2_decision_threshold).astype(int)

    n_pruned = int(np.sum(states == "LOW_RISK_PRUNED"))
    n_flagged = int(np.sum(states == "HIGH_RISK_FLAGGED"))
    n_esc = int(np.sum(escalate))
    avoided = n_pruned + n_flagged
    cost = {
        "total_pairs": n,
        "pruned_count": n_pruned,
        "flagged_direct_count": n_flagged,
        "escalated_count": n_esc,
        "stage2_calls_avoided_count": avoided,
        "stage2_calls_avoided_percent": round((avoided / n) * 100.0, 2),
    }
    return pred, states, cost
