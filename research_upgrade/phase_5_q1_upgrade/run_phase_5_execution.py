"""
Phase 5 Q1-Grade Master Research Execution Runner.

Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for
Cost-Efficient Academic Integrity Detection — Q1 Upgrade (Phase 5).

Deliverables produced under research_upgrade/phase_5_q1_upgrade/artifacts/:
    1. Headline comparison of the proposed framework against ALL implemented
       peers under identical corrected protocols (both modalities).
    2. Iso-recall (recall-matched) comparison.
    3. Iso-cost Pareto routing frontiers (shared heavy verifier).
    4. Simulated adversarial paraphrase robustness (text) and
       extra-benchmark identifier-renaming robustness (code).
    5. Group-conformal coverage with Wilson score CIs.
    6. Honest Stage-1 vs heavy Stage-2 latency/parameter cost analysis.
    7. Contribution-wise ablations, statistical significance tests, figures.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "/tmp"), "mpl5"))

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from research_upgrade.src.config import RANDOM_SEED, set_global_seed
from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
from research_upgrade.src.data.code_loader import CodeBenchmarkDataLoader
from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_paired_difference_bootstrap_ci,
    compute_mcnemar_significance,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import (
    TEXT_MULTIVIEW_FEATURES,
    CodeBaselineSuiteLOGO,
    TextBaselineSuite,
    evaluate_esplag_baseline_logo,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_text_pipeline import (
    PHASE5_TEXT_FEATURES,
    Phase5TextPipeline,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_code_evaluator import Phase5CodeEvaluator
from research_upgrade.phase_5_q1_upgrade.q1src.q1_attacks import (
    apply_text_attack,
    systematic_identifier_renaming,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    tune_routing_for_iso_recall,
    tune_routing_operating_point,
    tune_threshold_for_f1,
    tune_threshold_for_iso_recall,
    apply_routing,
    wilson_score_ci,
)
from research_upgrade.phase_5_q1_upgrade.q1src.lsa_semantic import LsaSemanticEncoder
from research_upgrade.phase_5_q1_upgrade.q1src.heavy_verifiers import (
    CrossStylePairVerifier,
    measure_lr_stage1_latency_ms,
)
from research_upgrade.src.features.text_lexical import TextLexicalExtractor
from research_upgrade.src.features.text_character import TextCharacterExtractor
from research_upgrade.src.features.text_structural import TextStructuralExtractor
from research_upgrade.src.features.feature_pipeline import CodeMultiViewPipeline


# --------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _mm(y, pred):
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
    }


def main() -> Dict[str, Any]:
    set_global_seed(RANDOM_SEED)
    root = Path(PROJECT_ROOT)
    out_dir = root / "research_upgrade" / "phase_5_q1_upgrade" / "artifacts"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_buf = io.StringIO()

    def log(msg: str):
        print(msg)
        log_buf.write(str(msg) + "\n")

    T0 = time.time()
    log("=" * 78)
    log("PHASE 5 Q1 UPGRADE — MASTER EXECUTION")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}  Seed: {RANDOM_SEED}")
    log("=" * 78)

    # ------------------------------------------------------------------ data
    log("\n[S1] Loading benchmark data + precomputed Phase-3 feature stores...")
    text_loader = TextBenchmarkDataLoader()
    raw_tr, raw_val, raw_te = (text_loader.get_partition(s) for s in ("train", "val", "test"))
    feats_dir = root / "research_upgrade" / "artifacts" / "phase_3" / "phase_3_3" / "features"
    F_tr = pd.read_parquet(feats_dir / "text_feats_train.parquet")
    F_val = pd.read_parquet(feats_dir / "text_feats_val.parquet")
    F_te = pd.read_parquet(feats_dir / "text_feats_test.parquet")
    code_loader = CodeBenchmarkDataLoader()
    raw_code_a = code_loader.get_dataset("policy_a")
    raw_code_b = code_loader.get_dataset("policy_b")
    C_a = pd.read_parquet(feats_dir / "code_feats_policy_a.parquet")
    C_b = pd.read_parquet(feats_dir / "code_feats_policy_b.parquet")

    # Alignment verification between raw partitions and feature stores
    align_text = list(raw_te["id"].values) == list(F_te["pair_id"].values)
    align_code = list(raw_code_a["id"].values) == list(C_a["pair_id"].values)
    log(f"      Text row alignment (raw vs features): {'EXACT' if align_text else 'ORDER-BASED'}; "
        f"Code: {'EXACT' if align_code else 'ORDER-BASED'}")
    log(f"      Text: train={len(F_tr)}, val={len(F_val)}, test={len(F_te)} | "
        f"Code Policy A={len(C_a)}, Policy B={len(C_b)}")

    results: Dict[str, Any] = {"meta": {"seed": RANDOM_SEED, "phase": "5",
                                          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}

    # ================================================================== TEXT
    log("\n[S2] TEXT — existing peers under corrected protocol (B-T1..B-T3)...")
    tb = TextBaselineSuite(random_state=RANDOM_SEED).fit(F_tr, F_val)
    text_baseline_res = tb.evaluate_test(F_te)

    log("\n[S3] TEXT — proposed Phase-5 framework (12-view S1 + heavy cross-style S2)...")
    pipe = Phase5TextPipeline(random_state=RANDOM_SEED, alpha=0.05, stage2_budget_cap=0.70)
    pipe.fit(F_tr, F_val, raw_tr, raw_val)
    t_res = pipe.evaluate_test(F_te, raw_te)
    tm = t_res["classification_metrics"]
    log(f"      Proposed: Acc={tm['accuracy']:.4f} Prec={tm['precision']:.4f} "
        f"Rec={tm['recall']:.4f} F1={tm['f1_score']:.4f} MCC={tm['matthews_corrcoef']:.4f} | "
        f"S2 avoided={t_res['routing_cost']['stage2_calls_avoided_percent']}%")
    log(f"      Operating point (val-tuned): {t_res['operating_point']}")
    log(f"      Heavy S2: {t_res['stage2_heavy_verifier']['trainable_parameters']} params, "
        f"{t_res['stage2_heavy_verifier']['measured_single_pair_latency_ms']:.4f} ms/pair")

    y_te_t = F_te["ground_truth_label"].values.astype(int)
    test_prems = raw_te["reference_text"].values

    # ------------------------------------------------- headline table (text)
    pred_s2_only_text = (t_res["s2_scores_all"] >= 0.5).astype(int)
    headline_rows = [{
        "Modality": "Text", "Method": "PROPOSED (C1–C4, heavy S2)",
        "Type": "Proposed", **_mm(y_te_t, t_res["final_predictions"]),
        "stage2_calls_avoided_pct": t_res["routing_cost"]["stage2_calls_avoided_percent"],
    }, {
        "Modality": "Text", "Method": "PROPOSED heavy verifier (beta=1.0, cost-unconstrained)",
        "Type": "Proposed (beta=1 config)", **_mm(y_te_t, pred_s2_only_text),
        "stage2_calls_avoided_pct": 0.0,
    }]
    for name, bres in text_baseline_res.items():
        headline_rows.append({
            "Modality": "Text", "Method": name, "Type": "Baseline",
            **{k: bres[k] for k in ("accuracy", "precision", "recall", "f1", "mcc")},
            "stage2_calls_avoided_pct": 0.0,
        })

    # ------------------------------------------------------- iso-recall text
    log("\n[S4] TEXT — iso-recall comparison (OP tuned on D_val only)...")
    y_val_t = F_val["ground_truth_label"].values.astype(int)
    val_scores = {
        "B-T1 TF-IDF Lexical (Sajid 2025)": F_val["text_word_tfidf_sim"].values,
        "B-T2 Dense Semantic MiniLM (Setu 2025)": F_val["text_semantic_minilm_sim"].values,
        "B-T3 Flat Multi-View Hybrid LR":
            tb.flat_model_.predict_proba(F_val[TEXT_MULTIVIEW_FEATURES].values)[:, 1],
    }
    test_scores = {
        "B-T1 TF-IDF Lexical (Sajid 2025)": F_te["text_word_tfidf_sim"].values,
        "B-T2 Dense Semantic MiniLM (Setu 2025)": F_te["text_semantic_minilm_sim"].values,
        "B-T3 Flat Multi-View Hybrid LR":
            tb.flat_model_.predict_proba(F_te[TEXT_MULTIVIEW_FEATURES].values)[:, 1],
    }
    # largest common feasible recall target (max of per-method best val recall, then min)
    target_scan = np.round(np.arange(0.60, 0.991, 0.01), 2)
    feas_targets: List[float] = []
    for cand in target_scan:
        ok = all(
            tune_threshold_for_iso_recall(y_val_t, s, cand)[1] >= cand for s in val_scores.values()
        )
        if ok:
            feas_targets.append(float(cand))
    iso_target_text = max(feas_targets) if feas_targets else 0.60
    log(f"      Common iso-recall target (max feasible on D_val): {iso_target_text:.2f}")

    iso_rows = []
    for name in val_scores:
        thr, val_rec = tune_threshold_for_iso_recall(y_val_t, val_scores[name], iso_target_text)
        pred = (test_scores[name] >= thr).astype(int)
        iso_rows.append({"Modality": "Text", "Method": name, "Recall_Target": iso_target_text,
                         "Val_Recall_Achieved": round(val_rec, 4), "Threshold": round(thr, 4),
                         **_mm(y_te_t, pred)})
    # proposed iso-recall operating point
    p_val_s1 = pipe.s1_model.predict_proba(
        pipe._val_df_enriched[PHASE5_TEXT_FEATURES].values)[:, 1]
    p_val_cal = pipe.calibrator.predict_positive_proba(p_val_s1)
    emb_v0 = pipe.lsa_encoder.encode(list(raw_val["reference_text"].values))
    emb_v1 = pipe.lsa_encoder.encode(list(raw_val["suspicious_text"].values))
    s2_val = pipe.s2_model.predict_proba(
        pipe._build_s2_inputs(pipe._val_df_enriched, emb_v0, emb_v1))
    ent_val = compute_predictive_entropy(p_val_cal)
    op_iso = tune_routing_for_iso_recall(
        y_val_t, p_val_cal, ent_val, s2_val, recall_target=iso_target_text)
    pred_iso, states_iso, cost_iso = apply_routing(
        t_res["calibrated_probs"], t_res["entropies"], t_res["s2_scores_all"],
        op_iso["tau_low"], op_iso["tau_high"], op_iso["entropy_threshold"],
        op_iso["stage2_decision_threshold"])
    iso_rows.insert(0, {
        "Modality": "Text", "Method": "PROPOSED (iso-recall OP)", "Recall_Target": iso_target_text,
        "Val_Recall_Achieved": round(op_iso["val_recall"], 4),
        "Threshold": f"tau=({op_iso['tau_low']},{op_iso['tau_high']},H={op_iso['entropy_threshold']})",
        **_mm(y_te_t, pred_iso),
        "S2_Escalation_Ratio_Test": round(cost_iso["escalated_count"] / len(y_te_t), 4),
    })
    # Apples-to-apples iso-recall: full escalation to heavy S2, threshold tuned on
    # val with the identical single-score protocol as the baselines.
    thr_s2_iso, val_rec_s2 = tune_threshold_for_iso_recall(y_val_t, s2_val, iso_target_text)
    pred_s2_iso = (t_res["s2_scores_all"] >= thr_s2_iso).astype(int)
    iso_rows.insert(1, {
        "Modality": "Text", "Method": "PROPOSED heavy verifier (beta=1.0, iso-recall thr)",
        "Recall_Target": iso_target_text,
        "Val_Recall_Achieved": round(val_rec_s2, 4), "Threshold": round(thr_s2_iso, 4),
        **_mm(y_te_t, pred_s2_iso),
        "S2_Escalation_Ratio_Test": 1.0,
    })
    pd.DataFrame(iso_rows).to_csv(out_dir / "PHASE_5_ISO_RECALL_COMPARISON_TEXT.csv", index=False)
    for r in iso_rows:
        log(f"      {r['Method']:44s} P={r['precision']:.4f} R={r['recall']:.4f} "
            f"F1={r['f1']:.4f} MCC={r['mcc']:.4f}")

    # -------------------------------------------- iso-cost Pareto (text)
    log("\n[S5] TEXT — iso-cost Pareto routing frontiers (shared heavy verifier)...")
    budgets = np.round(np.arange(0.05, 1.001, 0.05), 2)
    priority_proposed = t_res["entropies"]
    priority_baselines = {
        "B-T1 TF-IDF": -np.abs(F_te["text_word_tfidf_sim"].values - tb.thresholds_["tfidf"]),
        "B-T2 Semantic": -np.abs(F_te["text_semantic_minilm_sim"].values - tb.thresholds_["semantic"]),
        "B-T3 FlatHybrid": compute_predictive_entropy(
            tb.flat_model_.predict_proba(F_te[TEXT_MULTIVIEW_FEATURES].values)[:, 1]),
    }
    own_decision = {
        "B-T1 TF-IDF": (F_te["text_word_tfidf_sim"].values >= tb.thresholds_["tfidf"]).astype(int),
        "B-T2 Semantic": (F_te["text_semantic_minilm_sim"].values >= tb.thresholds_["semantic"]).astype(int),
        "B-T3 FlatHybrid": (tb.flat_model_.predict_proba(F_te[TEXT_MULTIVIEW_FEATURES].values)[:, 1]
                             >= tb.thresholds_["flat_hybrid"]).astype(int),
    }
    proposed_decision = (t_res["calibrated_probs"] >= 0.5).astype(int)
    s2_decisions = (t_res["s2_scores_all"] >= 0.5).astype(int)
    pareto_rows = []
    for b in budgets:
        # Proposed: escalate highest-entropy pairs to the shared heavy verifier
        k = np.quantile(priority_proposed, 1.0 - b)
        esc = priority_proposed >= k
        pred = proposed_decision.copy()
        pred[esc] = s2_decisions[esc]
        pareto_rows.append({"Method": "PROPOSED routing (calibrated entropy)",
                            "Budget": float(b), "S2_Calls": int(esc.sum()),
                            **_mm(y_te_t, pred)})
        for name in priority_baselines:
            k = np.quantile(priority_baselines[name], 1.0 - b)
            esc = priority_baselines[name] >= k
            pred = own_decision[name].copy()
            pred[esc] = s2_decisions[esc]
            pareto_rows.append({"Method": f"{name} routing", "Budget": float(b),
                                "S2_Calls": int(esc.sum()), **_mm(y_te_t, pred)})
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df.to_csv(out_dir / "PHASE_5_PARETO_FRONTIER.csv", index=False)
    log("      Frontiers computed for "
        f"{len(budgets)} budgets x {1 + len(priority_baselines)} methods.")

    # =================================================== TEXT ROBUSTNESS
    log("\n[S6] TEXT — simulated adversarial paraphrase robustness study...")
    lex = TextLexicalExtractor().fit(
        list(raw_tr["reference_text"].values) + list(raw_tr["suspicious_text"].values))
    cha = TextCharacterExtractor().fit(
        list(raw_tr["reference_text"].values) + list(raw_tr["suspicious_text"].values))
    stx = TextStructuralExtractor()
    lsa_att = pipe.lsa_encoder  # already fitted on train only

    OFFLINE_FEATS = [f for f in PHASE5_TEXT_FEATURES if f != "text_semantic_minilm_sim"]

    def extract_offline(refs, suss):
        rows = []
        for r, s in zip(refs, suss):
            d = {}
            d.update(lex.extract(r, s))
            d.update(cha.extract(r, s))
            d.update(stx.extract(r, s))
            d["text_lsa_semantic_sim"] = lsa_att.pair_cosine(r, s)
            rows.append(d)
        return pd.DataFrame(rows)

    # Build offline variant (trained pipeline, all features offline-recomputable)
    F_tr_off = extract_offline(raw_tr["reference_text"], raw_tr["suspicious_text"])
    F_val_off = extract_offline(raw_val["reference_text"], raw_val["suspicious_text"])
    F_te_off = extract_offline(raw_te["reference_text"], raw_te["suspicious_text"])

    y_tr_t = F_tr["ground_truth_label"].values.astype(int)

    # Adversarial augmentation (leak-free): one light-composed paraphrase copy of
    # every TRAIN positive, derived only from D_train. Hardens both stages of the
    # offline variant against paraphrase drift.
    pos_tr_mask = raw_tr["is_plagiarized"].values == 1
    aug_suss = apply_text_attack(list(raw_tr["suspicious_text"].values[pos_tr_mask]),
                                 "combined", seed=RANDOM_SEED + 7)
    F_aug_off = extract_offline(list(raw_tr["reference_text"].values[pos_tr_mask]), aug_suss)
    X_tr_off_aug = np.vstack([F_tr_off[OFFLINE_FEATS].values,
                              F_aug_off[OFFLINE_FEATS].values])
    y_tr_off_aug = np.concatenate([y_tr_t, np.ones(int(pos_tr_mask.sum()), dtype=int)])

    s1_off = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    s1_off.fit(X_tr_off_aug, y_tr_off_aug)
    calib_off = PlattScaler().fit(
        s1_off.predict_proba(F_val_off[OFFLINE_FEATS].values)[:, 1], y_val_t)

    emb_tr_off = (lsa_att.encode(list(raw_tr["reference_text"])),
                  lsa_att.encode(list(raw_tr["suspicious_text"])))
    Xtr_s2_off = np.hstack([
        LsaSemanticEncoder.interaction_features(*emb_tr_off), F_tr_off[OFFLINE_FEATS].values])
    emb_aug_off = (lsa_att.encode(list(raw_tr["reference_text"].values[pos_tr_mask])),
                   lsa_att.encode(list(aug_suss)))
    X_aug_s2_off = np.hstack([
        LsaSemanticEncoder.interaction_features(*emb_aug_off),
        F_aug_off[OFFLINE_FEATS].values])
    Xtr_s2_off_full = np.vstack([Xtr_s2_off, X_aug_s2_off])
    y_s2_off_full = np.concatenate([y_tr_t, np.ones(int(pos_tr_mask.sum()), dtype=int)])

    s2_off = CrossStylePairVerifier(random_state=RANDOM_SEED)
    t_s2off = time.time()
    s2_off.fit(Xtr_s2_off_full, y_s2_off_full)
    log(f"      Offline-variant heavy S2 (adversarially augmented, "
        f"N={len(Xtr_s2_off_full)}) fitted in {time.time() - t_s2off:.1f}s "
        f"({s2_off.n_params_} params)")

    # Operating point (val-tuned, budget-capped) for offline variant
    emb_v_off = (lsa_att.encode(list(raw_val["reference_text"])),
                 lsa_att.encode(list(raw_val["suspicious_text"])))
    s2_val_off = s2_off.predict_proba(np.hstack([
        LsaSemanticEncoder.interaction_features(*emb_v_off), F_val_off[OFFLINE_FEATS].values]))
    p_val_off = calib_off.predict_positive_proba(
        s1_off.predict_proba(F_val_off[OFFLINE_FEATS].values)[:, 1])
    ent_val_off = compute_predictive_entropy(p_val_off)
    op_off = tune_routing_operating_point(y_val_t, p_val_off, ent_val_off, s2_val_off,
                                          stage2_budget_cap=0.70, min_val_recall=0.65)

    def offline_route(feats_off_df, raw_refs, raw_suss):
        s1p = s1_off.predict_proba(feats_off_df[OFFLINE_FEATS].values)[:, 1]
        pc = calib_off.predict_positive_proba(s1p)
        ent = compute_predictive_entropy(pc)
        em = (lsa_att.encode(list(raw_refs)), lsa_att.encode(list(raw_suss)))
        s2p = s2_off.predict_proba(np.hstack([
            LsaSemanticEncoder.interaction_features(*em), feats_off_df[OFFLINE_FEATS].values]))
        pred, states, cost = apply_routing(pc, ent, s2p, op_off["tau_low"], op_off["tau_high"],
                                           op_off["entropy_threshold"],
                                           op_off["stage2_decision_threshold"])
        pred_full = (s2p >= 0.5).astype(int)  # beta=1.0: all pairs escalated
        return pred, cost, pred_full

    # Offline baselines (thresholds tuned on val off-feats)
    thr_lex, _ = tune_threshold_for_f1(y_val_t, F_val_off["text_word_tfidf_sim"].values)
    thr_lsa, _ = tune_threshold_for_f1(y_val_t, F_val_off["text_lsa_semantic_sim"].values)
    flat_off = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    flat_off.fit(F_tr_off[OFFLINE_FEATS].values, y_tr_t)
    p_flat_val = flat_off.predict_proba(F_val_off[OFFLINE_FEATS].values)[:, 1]
    thr_flat_off, _ = tune_threshold_for_f1(y_val_t, p_flat_val)

    attacks = ["synonym_light", "synonym_heavy", "word_order", "char_noise", "filler", "combined"]
    pos_mask = (F_te["ground_truth_label"].values == 1)
    pos_refs = raw_te["reference_text"].values[pos_mask]
    pos_suss = raw_te["suspicious_text"].values[pos_mask]
    neg_mask = ~pos_mask

    # Clean offline reference performance
    pred_off_clean, _, pred_off_clean_full = offline_route(
        F_te_off, raw_te["reference_text"], raw_te["suspicious_text"])
    clean_rec_off = recall_score(y_te_t, pred_off_clean, zero_division=0)

    rob_rows = [{
        "Attack": "none (clean offline reference)",
        "Method": "PROPOSED (offline variant, tuned OP, budget-capped)",
        **_mm(y_te_t, pred_off_clean),
        "positive_recall": round(float(clean_rec_off), 4),
    }, {
        "Attack": "none (clean offline reference)",
        "Method": "PROPOSED heavy verifier (beta=1.0)",
        **_mm(y_te_t, pred_off_clean_full),
        "positive_recall": round(float(recall_score(y_te_t, pred_off_clean_full,
                                                     zero_division=0)), 4),
    }]
    for name, feats_vec, thr in [
        ("TF-IDF threshold (offline)", F_te_off["text_word_tfidf_sim"].values, thr_lex),
        ("LSA semantic threshold (offline)", F_te_off["text_lsa_semantic_sim"].values, thr_lsa),
        ("Flat hybrid LR (offline)", flat_off.predict_proba(F_te_off[OFFLINE_FEATS].values)[:, 1],
         thr_flat_off),
    ]:
        pred = (feats_vec >= thr).astype(int)
        rob_rows.append({"Attack": "none (clean offline reference)", "Method": name,
                         **_mm(y_te_t, pred),
                         "positive_recall": round(float(
                             recall_score(y_te_t, pred, zero_division=0)), 4)})

    for attack in attacks:
        att_suss = apply_text_attack(list(pos_suss), attack, seed=RANDOM_SEED)
        feats_att_pos = extract_offline(list(pos_refs), list(att_suss))
        # admixture: attacked positives + untouched negatives (same offline extractor)
        feats_att = pd.concat(
            [feats_att_pos[OFFLINE_FEATS],
             F_te_off.loc[neg_mask, OFFLINE_FEATS].reset_index(drop=True)],
            ignore_index=True,
        )[OFFLINE_FEATS]
        refs_mix = list(pos_refs) + list(raw_te["reference_text"].values[neg_mask])
        suss_mix = list(att_suss) + list(raw_te["suspicious_text"].values[neg_mask])
        y_mix = np.concatenate([np.ones(int(pos_mask.sum()), dtype=int),
                                np.zeros(int(neg_mask.sum()), dtype=int)])
        pred_att, _, pred_att_full = offline_route(
            feats_att.reset_index(drop=True), refs_mix, suss_mix)
        pos_recall = float(np.mean(pred_att[: int(pos_mask.sum())] == 1))
        pos_recall_full = float(np.mean(pred_att_full[: int(pos_mask.sum())] == 1))
        rob_rows.append({"Attack": attack,
                         "Method": "PROPOSED (offline variant, tuned OP, budget-capped)",
                         **_mm(y_mix, pred_att),
                         "positive_recall": round(pos_recall, 4)})
        rob_rows.append({"Attack": attack,
                         "Method": "PROPOSED heavy verifier (beta=1.0)",
                         **_mm(y_mix, pred_att_full),
                         "positive_recall": round(pos_recall_full, 4)})
        # baselines recover feature cols from feats_att
        for name, col, thr in [("TF-IDF threshold (offline)", "text_word_tfidf_sim", thr_lex),
                                ("LSA semantic threshold (offline)", "text_lsa_semantic_sim", thr_lsa)]:
            pred_b = (feats_att[col].values >= thr).astype(int)
            rob_rows.append({"Attack": attack, "Method": name, **_mm(y_mix, pred_b),
                             "positive_recall": round(float(
                                 np.mean(pred_b[: int(pos_mask.sum())] == 1)), 4)})
        pred_f = (flat_off.predict_proba(feats_att[OFFLINE_FEATS].values)[:, 1] >= thr_flat_off).astype(int)
        rob_rows.append({"Attack": attack, "Method": "Flat hybrid LR (offline)",
                         **_mm(y_mix, pred_f),
                         "positive_recall": round(float(
                             np.mean(pred_f[: int(pos_mask.sum())] == 1)), 4)})
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out_dir / "PHASE_5_ROBUSTNESS_TEXT_ATTACKS.csv", index=False)
    for r in rob_rows:
        if not r["Attack"].startswith("none"):
            log(f"      [{r['Attack']:>14s}] {r['Method']:38s} PosRec={r['positive_recall']:.4f}")

    # ================================================================== CODE
    log("\n[S7] CODE — proposed Phase-5 framework under nested LOGO (heavy S2)...")
    # Leak-free adversarial augmentation: identifier-renamed copies of TRAINING
    # positives (memoized feature extraction; training data only).
    code_pipe_aug = CodeMultiViewPipeline()
    _aug_cache: Dict[int, np.ndarray] = {}

    def code_augment_fn(abs_idx: np.ndarray):
        Xa, ya = [], []
        for i in abs_idx:
            i = int(i)
            if int(C_a["ground_truth_label"].values[i]) != 1:
                continue
            if i not in _aug_cache:
                fe = code_pipe_aug.extract_pair(
                    str(raw_code_a["reference_code"].values[i]),
                    systematic_identifier_renaming(str(raw_code_a["suspicious_code"].values[i])))
                _aug_cache[i] = np.array([fe[c] for c in C_FEATS], dtype=float)
            Xa.append(_aug_cache[i])
            ya.append(1)
        if not Xa:
            return np.zeros((0, len(C_FEATS))), np.zeros(0, dtype=int)
        return np.asarray(Xa, dtype=np.float64), np.asarray(ya, dtype=int)

    from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import CODE_FEATURES as C_FEATS

    evaluator_a = Phase5CodeEvaluator(random_state=RANDOM_SEED, stage2_budget_cap=0.45,
                                      min_val_recall=0.80, augment_pairs_fn=code_augment_fn)
    code_a = evaluator_a.evaluate_logo(C_a, raw_code_a["case"], "Policy A")
    cam = code_a["classification_metrics"]
    sel_used = [v["selected"] for v in code_a["verifier_selection"]]
    log(f"      Policy A: Acc={cam['accuracy']:.4f} Prec={cam['precision']:.4f} "
        f"Rec={cam['recall']:.4f} F1={cam['f1_score']:.4f} MCC={cam['matthews_corrcoef']:.4f} | "
        f"S2 avoided={code_a['routing_cost']['stage2_calls_avoided_percent']}% | "
        f"verifiers: {sel_used}")
    evaluator_b = Phase5CodeEvaluator(random_state=RANDOM_SEED, stage2_budget_cap=0.45,
                                      min_val_recall=0.80)
    code_b = evaluator_b.evaluate_logo(C_b, raw_code_b["case"], "Policy B")
    cbm = code_b["classification_metrics"]
    evaluator_ls = Phase5CodeEvaluator(random_state=RANDOM_SEED, stage2_budget_cap=0.45,
                                       min_val_recall=0.80, augment_pairs_fn=code_augment_fn)
    code_strat = evaluator_ls.evaluate_stratified_leakage(C_a, "Policy A (stratified contrast)")
    leak_dmcc = (code_strat["classification_metrics"]["matthews_corrcoef"]
                 - cam["matthews_corrcoef"])
    log(f"      Policy B MCC={cbm['matthews_corrcoef']:.4f} | Stratified(leak) MCC="
        f"{code_strat['classification_metrics']['matthews_corrcoef']:.4f} "
        f"-> leakage inflation ΔMCC={leak_dmcc:+.4f}")

    log("\n[S8] CODE — existing peers under nested LOGO (B-C1..B-C5)...")
    cb = CodeBaselineSuiteLOGO(random_state=RANDOM_SEED)
    code_base_res = cb.evaluate(C_a, raw_code_a["case"], threshold_mode="f1")
    esplag_res = evaluate_esplag_baseline_logo(raw_code_a, raw_code_a["case"], RANDOM_SEED)
    code_base_res["B-C5 ES-Plag (2019)"] = esplag_res
    for name, r in code_base_res.items():
        extra = f" (filter pass: {r['pairs_passing_dynamic_filter']}/720)" \
            if name.startswith("B-C5") else ""
        log(f"      {name:34s} F1={r['f1']:.4f} MCC={r['mcc']:.4f}{extra}")

    y_code = C_a["ground_truth_label"].values.astype(int)
    for name, r in code_base_res.items():
        headline_rows.append({
            "Modality": "Code(PolicyA LOGO)", "Type": "Baseline", "Method": name,
            **{k: r[k] for k in ("accuracy", "precision", "recall", "f1", "mcc")},
            "stage2_calls_avoided_pct": 0.0,
        })
    pred_s2_only_code = (code_a["oof_s2_scores"] >= 0.5).astype(int)
    headline_rows.insert(1, {
        "Modality": "Code(PolicyA LOGO)", "Method": "PROPOSED (C1–C4, heavy S2)",
        "Type": "Proposed", **_mm(y_code, code_a["predictions_dataframe"]["final_prediction"].values),
        "stage2_calls_avoided_pct": code_a["routing_cost"]["stage2_calls_avoided_percent"],
    })
    headline_rows.insert(2, {
        "Modality": "Code(PolicyA LOGO)",
        "Method": "PROPOSED heavy verifier (beta=1.0, cost-unconstrained)",
        "Type": "Proposed (beta=1 config)", **_mm(y_code, pred_s2_only_code),
        "stage2_calls_avoided_pct": 0.0,
    })
    pd.DataFrame(headline_rows).to_csv(out_dir / "PHASE_5_HEADLINE_COMPARISON.csv", index=False)

    # code iso-recall @0.95
    cb_iso = CodeBaselineSuiteLOGO(random_state=RANDOM_SEED)
    code_base_iso = cb_iso.evaluate(C_a, raw_code_a["case"], threshold_mode="iso_recall",
                                    iso_target=0.95)
    iso_code_rows = [{"Modality": "Code", "Method": name, "Recall_Target": 0.95,
                      "Threshold": "per-fold train-tuned",
                      **{k: r[k] for k in ("accuracy", "precision", "recall", "f1", "mcc")}}
                     for name, r in code_base_iso.items()]
    iso_code_rows.insert(0, {"Modality": "Code", "Method": "PROPOSED (nested LOGO, tuned OP)",
                             "Recall_Target": 0.95, "Threshold": "per-fold inner-OOF tuned",
                             **_mm(y_code, code_a["predictions_dataframe"]["final_prediction"].values)})
    pd.DataFrame(iso_code_rows).to_csv(out_dir / "PHASE_5_ISO_RECALL_COMPARISON_CODE.csv",
                                       index=False)

    # code per-obfuscation-level recall
    lvl_rows = []
    lvls = C_a["obfuscation_level"].values
    pred_code = code_a["predictions_dataframe"]["final_prediction"].values
    for lvl in sorted(set(lvls)):
        if str(lvl) in ("None", "nan", ""):
            continue
        m = (lvls == lvl) & (y_code == 1)
        if m.sum() == 0:
            continue
        row = {"Level": lvl, "n_positive": int(m.sum()),
               "PROPOSED": round(float(np.mean(pred_code[m] == 1)), 4)}
        for name, r in code_base_res.items():
            row[name] = round(float(np.mean(r["predictions"][m] == 1)), 4)
        lvl_rows.append(row)
    pd.DataFrame(lvl_rows).to_csv(out_dir / "PHASE_5_CODE_OBFUSCATION_LEVELS.csv", index=False)

    # --------------------------------------------------- code renaming attack
    log("\n[S9] CODE — extra-benchmark systematic identifier-renaming attack...")
    code_pipe = CodeMultiViewPipeline()
    feat_cols = list(code_pipe.feature_names)
    pos_c = C_a["ground_truth_label"].values == 1
    att_feats = []
    refs_att = raw_code_a["reference_code"].values[pos_c]
    suss_att = raw_code_a["suspicious_code"].values[pos_c]
    cases_att = raw_code_a["case"].values[pos_c]
    t_r = time.time()
    for rcode, scode in zip(refs_att, suss_att):
        att_feats.append(code_pipe.extract_pair(str(rcode), str(systematic_identifier_renaming(str(scode)))))
    X_att = pd.DataFrame(att_feats)[evaluator_a.feature_cols].values
    pred_att_c = np.zeros(len(X_att), dtype=int)
    pred_att_c_full = np.zeros(len(X_att), dtype=int)
    for case_name in set(cases_att):
        idx = np.array([i for i, c in enumerate(cases_att) if c == case_name])
        out = evaluator_a.apply_attacked_features(case_name, X_att[idx], mode="routed")
        out_full = evaluator_a.apply_attacked_features(case_name, X_att[idx], mode="s2_only")
        pred_att_c[idx] = out["predictions"]
        pred_att_c_full[idx] = out_full["predictions"]
    att_pos_recall = float(np.mean(pred_att_c == 1))
    att_pos_recall_full = float(np.mean(pred_att_c_full == 1))
    clean_pos_pred = pred_code[pos_c]
    clean_pos_recall = float(np.mean(clean_pos_pred == 1))
    rename_rows = [
        {"Method": "PROPOSED (tuned OP)", "Condition": "clean positives",
         "positive_recall": round(clean_pos_recall, 4)},
        {"Method": "PROPOSED (tuned OP)",
         "Condition": "identifier-renamed positives (extra-benchmark attack)",
         "positive_recall": round(att_pos_recall, 4)},
        {"Method": "PROPOSED heavy verifier (beta=1.0)",
         "Condition": "identifier-renamed positives (extra-benchmark attack)",
         "positive_recall": round(att_pos_recall_full, 4)},
    ]
    # baselines on attacked features
    y_att = np.ones(len(X_att), dtype=int)
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import (
        CODE_FEATURES as CF, CODE_SVM_FEATURES, CODE_RF_FEATURES)
    logo_att = LeaveOneGroupOut()
    svm_pred_att = np.zeros(len(X_att), dtype=int)
    rf_pred_att = np.zeros(len(X_att), dtype=int)
    jac_pred_att = np.zeros(len(X_att), dtype=int)
    rkr_pred_att = np.zeros(len(X_att), dtype=int)
    full_X = C_a[CF].values
    Xdf_att = pd.DataFrame(X_att, columns=CF)
    for tr_idx, te_idx in logo_att.split(full_X, y_code, groups=raw_code_a["case"].values):
        held = raw_code_a["case"].values[te_idx][0]
        sub = np.array([i for i, c in enumerate(cases_att) if c == held])
        if len(sub) == 0:
            continue
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED)
        svm.fit(C_a[CODE_SVM_FEATURES].values[tr_idx], y_code[tr_idx])
        svm_pred_att[sub] = svm.predict(Xdf_att[CODE_SVM_FEATURES].values[sub])
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        rf.fit(C_a[CODE_RF_FEATURES].values[tr_idx], y_code[tr_idx])
        rf_pred_att[sub] = rf.predict(Xdf_att[CODE_RF_FEATURES].values[sub])
        thrj, _ = tune_threshold_for_f1(y_code[tr_idx], C_a["code_token_jaccard"].values[tr_idx])
        thrr, _ = tune_threshold_for_f1(y_code[tr_idx], C_a["code_rkrgst_sim"].values[tr_idx])
        jac_pred_att[sub] = (Xdf_att["code_token_jaccard"].values[sub] >= thrj).astype(int)
        rkr_pred_att[sub] = (Xdf_att["code_rkrgst_sim"].values[sub] >= thrr).astype(int)
    for name, arr in [("B-C1 Token Jaccard", jac_pred_att), ("B-C2 RKRGST", rkr_pred_att),
                       ("B-C3 Gandhi RBF-SVM (2024)", svm_pred_att),
                       ("B-C4 Mehsen RF (2024)", rf_pred_att)]:
        rename_rows.append({"Method": name, "Condition": "identifier-renamed positives",
                            "positive_recall": round(float(np.mean(arr == 1)), 4)})
    pd.DataFrame(rename_rows).to_csv(out_dir / "PHASE_5_CODE_RENAMING_ATTACK.csv", index=False)
    log(f"      Renaming attack computed in {time.time() - t_r:.1f}s | "
        f"PROPOSED pos-recall clean={clean_pos_recall:.4f} -> attacked={att_pos_recall:.4f}")

    # ==================================================== STATISTICAL TESTS
    log("\n[S10] Statistical significance tests (McNemar + clustered paired bootstrap)...")
    stat_tests: Dict[str, Any] = {"text": {}, "code": {}}
    pred_prop_t = t_res["final_predictions"]
    for name, bres in text_baseline_res.items():
        stat_tests["text"][name] = {
            "mcnemar": compute_mcnemar_significance(y_te_t, pred_prop_t, bres["predictions"]),
            "paired_bootstrap": compute_paired_difference_bootstrap_ci(
                test_prems, y_te_t, pred_prop_t, bres["predictions"],
                n_resamples=1000, random_state=RANDOM_SEED),
        }
    pred_prop_c = code_a["predictions_dataframe"]["final_prediction"].values
    for name, r in code_base_res.items():
        stat_tests["code"][name] = {
            "mcnemar": compute_mcnemar_significance(y_code, pred_prop_c, r["predictions"].astype(int)),
            "paired_bootstrap": compute_paired_difference_bootstrap_ci(
                raw_code_a["case"].values, y_code, pred_prop_c, r["predictions"].astype(int),
                n_resamples=1000, random_state=RANDOM_SEED),
        }
        stat_tests["code"][f"{name} | vs PROPOSED beta=1"] = {
            "mcnemar": compute_mcnemar_significance(y_code, pred_s2_only_code, r["predictions"].astype(int)),
            "paired_bootstrap": compute_paired_difference_bootstrap_ci(
                raw_code_a["case"].values, y_code, pred_s2_only_code, r["predictions"].astype(int),
                n_resamples=1000, random_state=RANDOM_SEED),
        }
    with open(out_dir / "PHASE_5_STATISTICAL_TESTS.json", "w") as f:
        json.dump(stat_tests, f, indent=2, cls=NumpyEncoder)
    for name in text_baseline_res:
        d = stat_tests["text"][name]["paired_bootstrap"]["delta_f1"]
        log(f"      Text ΔF1 vs {name[:30]:30s}: {d['mean_delta']:+.4f} CI=[{d['ci_lower']:+.4f},{d['ci_upper']:+.4f}]")
    for name in code_base_res:
        d = stat_tests["code"][name]["paired_bootstrap"]["delta_f1"]
        log(f"      Code ΔF1 vs {name[:30]:30s}: {d['mean_delta']:+.4f} CI=[{d['ci_lower']:+.4f},{d['ci_upper']:+.4f}]")

    # ----------------------------------------------- conformal + Wilson CIs
    conf = t_res["group_conformal_summary"]
    conf_row = {
        "alpha": 0.05, "nominal_coverage": 0.95,
        "empirical_group_coverage": conf["empirical_group_coverage"],
        "group_wilson_ci_95": conf["wilson_ci_95"]["group_coverage"],
        "empirical_pair_coverage": conf["empirical_pair_coverage"],
        "pair_wilson_ci_95": conf["wilson_ci_95"]["pair_coverage"],
        "average_set_size": conf["average_prediction_set_size"],
        "calibration_quantile": conf["calibration_quantile"],
    }
    pd.DataFrame([conf_row]).to_csv(out_dir / "PHASE_5_CONFORMAL_COVERAGE.csv", index=False)
    gc = conf_row["group_wilson_ci_95"]
    log(f"      Group coverage {conf_row['empirical_group_coverage']:.4f} "
        f"Wilson95=[{gc['wilson_lower']:.4f}, {gc['wilson_upper']:.4f}] vs nominal 0.95")

    # -------------------------------------------------------------- costs
    s1_lat_t = measure_lr_stage1_latency_ms(
        pipe.s1_model, t_res["_enriched_test"][PHASE5_TEXT_FEATURES].values[:1])
    s2_lat_t = t_res["stage2_heavy_verifier"]["measured_single_pair_latency_ms"]
    # Batched amortized latencies (compute-dominated regime)
    X_b = t_res["_enriched_test"][PHASE5_TEXT_FEATURES].values[:512]
    pipe.s1_model.predict_proba(X_b)
    t0 = time.perf_counter()
    for _ in range(50):
        pipe.s1_model.predict_proba(X_b)
    s1_batch_ms = ((time.perf_counter() - t0) / 50) * 1000.0 / len(X_b)
    emb_b0 = pipe.lsa_encoder.encode(list(raw_te["reference_text"].values[:512]))
    emb_b1 = pipe.lsa_encoder.encode(list(raw_te["suspicious_text"].values[:512]))
    Xb_s2 = pipe._build_s2_inputs(t_res["_enriched_test"].iloc[:512], emb_b0, emb_b1)
    pipe.s2_model.predict_proba(Xb_s2)
    t0 = time.perf_counter()
    for _ in range(10):
        pipe.s2_model.predict_proba(Xb_s2)
    s2_batch_ms = ((time.perf_counter() - t0) / 10) * 1000.0 / len(Xb_s2)

    code_latency = evaluator_a.measure_stage2_latency(C_a)
    params_t = t_res['stage2_heavy_verifier']['trainable_parameters']
    cost_rows = [
        {"Domain": "Text", "Stage": "Stage-1 LR screener", "architectural_complexity": 12,
         "single_pair_latency_ms": round(s1_lat_t, 6),
         "batched_ms_per_pair": round(s1_batch_ms, 6)},
        {"Domain": "Text", "Stage": "Stage-2 heavy cross-style MLP verifier",
         "architectural_complexity": params_t,
         "single_pair_latency_ms": round(s2_lat_t, 6),
         "batched_ms_per_pair": round(s2_batch_ms, 6)},
        {"Domain": "Code", "Stage": "Stage-1 LR screener", "architectural_complexity": 13,
         "single_pair_latency_ms": code_latency["stage1_single_pair_ms"],
         "batched_ms_per_pair": code_latency["stage1_batched_ms_per_pair"]},
        {"Domain": "Code", "Stage": f"Stage-2 heavy verifier"
            f" ({code_latency['selected_fold0_verifier']}; trees/nodes="
            f"{code_latency.get('s2_fitted_trees')}/{code_latency.get('s2_fitted_nodes')})",
         "architectural_complexity": code_latency.get("s2_fitted_nodes"),
         "single_pair_latency_ms": code_latency["stage2_single_pair_ms"],
         "batched_ms_per_pair": code_latency["stage2_batched_ms_per_pair"]},
    ]
    cost_df = pd.DataFrame(cost_rows)
    cost_df["latency_ratio_vs_stage1_batched"] = [
        1.0, round(s2_batch_ms / max(s1_batch_ms, 1e-9), 1), 1.0,
        code_latency["latency_ratio_s2_over_s1_batched"]]
    cost_df["complexity_ratio_vs_stage1"] = [
        1.0, round(params_t / 12, 1), 1.0,
        round((code_latency.get("s2_fitted_nodes") or 1) / 13, 1)]
    cost_df.to_csv(out_dir / "PHASE_5_COST_ANALYSIS.csv", index=False)
    log(f"      Measured batched latency S2/S1: text={s2_batch_ms / max(s1_batch_ms, 1e-9):.1f}x, "
        f"code={code_latency['latency_ratio_s2_over_s1_batched']:.1f}x | params/nodes ratio: "
        f"text {params_t}/12, code ~{code_latency.get('s2_fitted_nodes')}/13")

    # ------------------------------------------------------------- ablations
    log("\n[S11] Contribution-wise ablations...")
    ab_rows = []
    view_groups = {
        "Word lexical only": ["text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio"],
        "Char n-gram only": ["text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio"],
        "MiniLM semantic only": ["text_semantic_minilm_sim"],
        "LSA semantic only": ["text_lsa_semantic_sim"],
        "Structural only": ["text_length_ratio", "text_token_count_delta",
                              "text_sentence_delta", "text_ttr_sim"],
    }
    enriched_te = t_res["_enriched_test"]
    for name, cols in view_groups.items():
        m = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
        m.fit(pipe._train_df_enriched[cols].values, y_tr_t)
        pred = m.predict(enriched_te[cols].values)
        ab_rows.append({"Domain": "Text", "Configuration": name,
                        **_mm(y_te_t, pred)})
    m_all = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    m_all.fit(pipe._train_df_enriched[PHASE5_TEXT_FEATURES].values, y_tr_t)
    p_all = m_all.predict_proba(enriched_te[PHASE5_TEXT_FEATURES].values)[:, 1]
    ab_rows.append({"Domain": "Text", "Configuration": "C1 multi-view fusion (12 views)",
                    **_mm(y_te_t, (p_all >= 0.5).astype(int))})
    p_all_cal = pipe.calibrator.predict_positive_proba(p_all)
    ab_rows.append({"Domain": "Text", "Configuration": "C1 + C2 calibration",
                    **_mm(y_te_t, (p_all_cal >= 0.5).astype(int))})
    ab_rows.append({"Domain": "Text", "Configuration": "C1+C2+C3+C4 full framework (heavy S2)",
                    **_mm(y_te_t, pred_prop_t)})
    ab_rows.append({"Domain": "Code", "Configuration": "C1 multi-view fusion LR (OOF-LOGO ref: Phase 4.1)",
                    "accuracy": 0.8486, "precision": 0.8157, "recall": 0.8801,
                    "f1": 0.8467, "mcc": 0.6996})
    ab_rows.append({"Domain": "Code", "Configuration": "C1+C2+C3+C4 full framework (heavy S2, LOGO)",
                    **_mm(y_code, pred_prop_c)})
    ab_df = pd.DataFrame(ab_rows)
    ab_df.to_csv(out_dir / "PHASE_5_ABLATION.csv", index=False)

    # ---------------------------------------------------------------- figures
    log("\n[S12] Figures...")
    # Fig 1: calibration curve (text)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    p = t_res["calibrated_probs"]
    bins = np.linspace(0, 1, 11)
    cpts, apts = [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() > 0:
            cpts.append(p[m].mean()); apts.append(y_te_t[m].mean())
    ax.plot(cpts, apts, "o-", label=f"Platt-calibrated S1 (ECE={t_res['calibration_metrics']['expected_calibration_error']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Figure 1: Calibration Reliability (Phase 5)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "figure_1_calibration.png"); plt.close(fig)

    # Fig 2: risk-coverage
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    covs, risks = [], []
    for cov in np.linspace(0.1, 1.0, 20):
        k = int(len(p) * cov)
        idx = np.argsort(-np.abs(p - 0.5))[:k]
        covs.append(cov)
        risks.append(1 - accuracy_score(y_te_t[idx], (p[idx] >= 0.5).astype(int)))
    ax.plot(covs, risks, lw=2.5)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk")
    ax.set_title("Figure 2: Selective Risk–Coverage (Phase 5)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "figure_2_risk_coverage.png"); plt.close(fig)

    # Fig 3: Pareto multi-frontier
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    styles = {"PROPOSED routing (calibrated entropy)": ("o-", "#c00000", 2.8),
              "B-T1 TF-IDF routing": ("s--", "#7f7f7f", 1.5),
              "B-T2 Semantic routing": ("^--", "#1f77b4", 1.5),
              "B-T3 FlatHybrid routing": ("d--", "#2ca02c", 1.5)}
    for name, (mk, c, lw) in styles.items():
        sub = pareto_df[pareto_df["Method"] == name].sort_values("Budget")
        ax.plot(sub["Budget"] * 100, sub["mcc"], mk, color=c, lw=lw, ms=5, label=name)
    ax.set_xlabel("Stage-2 heavy-verifier budget (% of pairs escalated)")
    ax.set_ylabel("MCC")
    ax.set_title("Figure 3: Iso-Cost Routing Frontiers — Shared Heavy Verifier")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "figure_3_pareto_isocost.png"); plt.close(fig)

    # Fig 4: ablation
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    sub = ab_df[ab_df["Domain"] == "Text"]
    ax.barh(sub["Configuration"], sub["mcc"], color="#1f4e79")
    ax.set_xlabel("MCC"); ax.set_title("Figure 4: Contribution-wise MCC Progression (Text)")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "figure_4_ablation_mcc.png"); plt.close(fig)

    # Fig 5: robustness
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    rab = rob_df[rob_df["Attack"] != "none (clean offline reference)"]
    meths = list(dict.fromkeys(rab["Method"]))
    x = np.arange(len(attacks))
    w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        vals = [rab[(rab["Method"] == m) & (rab["Attack"] == a)]["positive_recall"].iloc[0]
                for a in attacks]
        ax.bar(x + j * w, vals, w, label=m, alpha=0.9)
    clean_line = rob_df[(rob_df["Attack"].str.startswith("none")) &
                        (rob_df["Method"].str.startswith("PROPOSED"))]["positive_recall"].iloc[0]
    ax.axhline(clean_line, color="red", ls=":", lw=1.5, label=f"clean proposed recall ({clean_line:.3f})")
    ax.set_xticks(x + w); ax.set_xticklabels(attacks, rotation=20)
    ax.set_ylabel("Recall on attacked positives")
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 5: Adversarial Paraphrase Robustness (Text)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(fig_dir / "figure_5_robustness_attacks.png"); plt.close(fig)

    # Fig 6: code obfuscation levels
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    if lvl_rows:
        lvl_df = pd.DataFrame(lvl_rows).set_index("Level").drop(columns=["n_positive"])
        x = np.arange(len(lvl_df.index))
        w = 0.8 / len(lvl_df.columns)
        for j, m in enumerate(lvl_df.columns):
            ax.bar(x + j * w, lvl_df[m].values, w, label=m, alpha=0.9)
        ax.set_xticks(x + w * 2); ax.set_xticklabels(lvl_df.index)
        ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
        ax.set_title("Figure 6: Recall across IR-Plag Obfuscation Levels L1–L6 (Code)")
        ax.legend(fontsize=7.5, ncol=2); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
    fig.savefig(fig_dir / "figure_6_code_obfuscation_levels.png"); plt.close(fig)

    # ------------------------------------------------------------- full JSON
    results["text"] = {
        "proposed": {k: t_res[k] for k in ("classification_metrics", "calibration_metrics",
                                            "group_conformal_summary", "routing_cost",
                                            "operating_point", "stage2_heavy_verifier")},
        "proposed_beta1_metrics": _mm(y_te_t, pred_s2_only_text),
        "baselines": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("scores", "prob_scores", "predictions")}
                       for k, v in text_baseline_res.items()},
        "iso_recall": iso_rows,
        "pareto_frontier": pareto_rows,
        "robustness_attacks": rob_rows,
    }
    results["code"] = {
        "proposed_policy_a": {k: code_a[k] for k in ("classification_metrics",
                                                       "calibration_metrics", "routing_cost",
                                                       "fold_details")},
        "proposed_policy_a_beta1_metrics": _mm(y_code, pred_s2_only_code),
        "adversarial_augmentation": "identifier-renamed training positives (leak-free)",
        "proposed_policy_b_mcc": cbm["matthews_corrcoef"],
        "proposed_stratified_leakage": code_strat["classification_metrics"],
        "leakage_delta_mcc": round(float(leak_dmcc), 4),
        "baselines": {k: {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, np.ndarray)}
                       for k, v in code_base_res.items()},
        "iso_recall_baselines": iso_code_rows,
        "obfuscation_levels": lvl_rows,
        "renaming_attack": rename_rows,
        "latency": code_latency,
    }
    results["conformal"] = conf_row
    results["costs"] = cost_df.to_dict(orient="records")
    results["ablations"] = ab_rows
    with open(out_dir / "PHASE_5_RESEARCH_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    t_res["predictions_dataframe"].to_csv(out_dir / "PHASE_5_PREDICTIONS_TEXT_TEST.csv", index=False)
    code_a["predictions_dataframe"].to_csv(out_dir / "PHASE_5_PREDICTIONS_CODE_OOF.csv", index=False)

    log(f"\nPhase 5 complete in {time.time() - T0:.1f}s. Artifacts: {out_dir}")
    with open(out_dir / "PHASE_5_EXECUTION_LOG.txt", "w") as f:
        f.write(log_buf.getvalue())
    return results


if __name__ == "__main__":
    main()
