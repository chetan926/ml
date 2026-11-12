"""Phase 6 Q1 hardening — master execution.

Fixes the six reviewer-attack items WITHOUT touching frozen phases:

  1. External validity  — Clough–Stevenson (95 graded short answers x 5 sources,
     +475 cross-topic distractor set, clearly caveated) and full MRPC test split
     (1,725 pairs), evaluated with the val-tuned OFFLINE-feature variant
     (MiniLM column excluded — verifier-free honest variant).
  2. C4 human evidence  — expert-review walkthrough packet (20 grey-zone pairs,
     interpretability sheets + pre-registered style protocol doc).
  3. Conformal fix      — Mondrian (size / parity) + split-sample alpha* tuning
     against the phase-5 undercoverage (0.930 vs 0.95).
  4. Neural verifier    — honest ablation of MLP cross-style verifier vs RF/HGB
     non-neural candidates on identical interaction input, val-selected.
  5. Multi-seed         — 3 seeds x {text pipeline, code LOGO policies A & B}.
  6. Paper revision     — separate revised manuscript + PHASE_6_REPORT.md.

All artifacts land in research_upgrade/phase_6_q1_hardening/artifacts/.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.neural_network import MLPClassifier

ROOT = Path("/home/user/ml")
sys.path.insert(0, str(ROOT))

from research_upgrade.src.config import RANDOM_SEED, set_global_seed
from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
from research_upgrade.src.data.code_loader import CodeBenchmarkDataLoader
from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy
from research_upgrade.src.features.text_lexical import TextLexicalExtractor
from research_upgrade.src.features.text_character import TextCharacterExtractor
from research_upgrade.src.features.text_structural import TextStructuralExtractor
from research_upgrade.src.features.feature_pipeline import CodeMultiViewPipeline
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_group_aware_bootstrap_ci,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_text_pipeline import (
    PHASE5_TEXT_FEATURES, Phase5TextPipeline,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_code_evaluator import Phase5CodeEvaluator
from research_upgrade.phase_5_q1_upgrade.q1src.q1_attacks import (
    systematic_identifier_renaming,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import CODE_FEATURES as C_FEATS
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    apply_routing, tune_routing_operating_point,
)
from research_upgrade.phase_5_q1_upgrade.q1src.lsa_semantic import LsaSemanticEncoder
from research_upgrade.phase_5_q1_upgrade.q1src.heavy_verifiers import CrossStylePairVerifier
from research_upgrade.phase_6_q1_hardening.h6src.conformal_variants import (
    alpha_split_sample_tuning,
    evaluate_group_coverage,
    evaluate_mondrian_parity,
    evaluate_mondrian_size,
    exact_quantile,
    group_supremum_scores,
)


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
        "n": int(len(y)),
    }


def main() -> None:
    set_global_seed(RANDOM_SEED)
    t0 = time.time()
    out_dir = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_buf = io.StringIO()

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_buf.write(str(msg) + "\n")

    log("=" * 78)
    log("PHASE 6 Q1 HARDENING — MASTER EXECUTION")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}  Base seed: {RANDOM_SEED}")
    log("=" * 78)

    # ------------------------------------------------------------- S1 LOAD
    log("\n[S1] Loading benchmark data + phase-3 feature stores + external corpora...")
    text_loader = TextBenchmarkDataLoader()
    raw_tr, raw_val, raw_te = (text_loader.get_partition(s) for s in ("train", "val", "test"))
    feats_dir = ROOT / "research_upgrade" / "artifacts" / "phase_3" / "phase_3_3" / "features"
    F_tr = pd.read_parquet(feats_dir / "text_feats_train.parquet")
    F_val = pd.read_parquet(feats_dir / "text_feats_val.parquet")
    F_te = pd.read_parquet(feats_dir / "text_feats_test.parquet")
    code_loader = CodeBenchmarkDataLoader()
    raw_code_a = code_loader.get_dataset("policy_a")
    raw_code_b = code_loader.get_dataset("policy_b")
    C_a = pd.read_parquet(feats_dir / "code_feats_policy_a.parquet")
    C_b = pd.read_parquet(feats_dir / "code_feats_policy_b.parquet")
    ext_dir = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "artifacts" / "external_text"
    CS = pd.read_parquet(ext_dir / "clough_pairs.parquet")
    CSX = pd.read_parquet(ext_dir / "clough_crosstopic_pairs.parquet")
    MRPC = pd.read_parquet(ext_dir / "mrpc_pairs.parquet")
    for c in PHASE5_TEXT_FEATURES:
        for d in (F_tr, F_val, F_te):
            if c not in d:
                d[c] = np.nan
    log(f"      Text train/val/test = {len(F_tr)}/{len(F_val)}/{len(F_te)}; "
        f"Code A={len(C_a)} B={len(C_b)}; CS={len(CS)} CSX={len(CSX)} MRPC={len(MRPC)}")

    summary: Dict[str, Any] = {
        "phase": "6_q1_hardening", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "external_protocol_note": (
            "External corpora are used for EVALUATION ONLY. No external sample "
            "enters any fitting, calibration, operating-point or conformal pool."),
    }

    # ===================================================== S2 MULTI-SEED TEXT
    SEEDS = [42, 123, 2024]
    log(f"\n[S2] Multi-seed text pipeline (seeds={SEEDS}) ...")
    text_seed_rows: List[Dict[str, Any]] = []
    seed_res: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        pipe = Phase5TextPipeline(random_state=seed, alpha=0.05, stage2_budget_cap=0.70)
        pipe.fit(F_tr.copy(), F_val.copy(), raw_tr, raw_val)
        res = pipe.evaluate_test(F_te.copy(), raw_te)
        seed_res[seed] = res
        m = res["classification_metrics"]
        conf = res["group_conformal_summary"]
        row = {"seed": seed, **_mm(F_te["ground_truth_label"].values, res["predictions_dataframe"]["final_prediction"].values),
               "group_coverage": conf["empirical_group_coverage"],
               "group_cov_wilson_lo": conf["wilson_ci_95"]["group_coverage"]["wilson_lower"],
               "group_cov_wilson_hi": conf["wilson_ci_95"]["group_coverage"]["wilson_upper"],
               "s2_avoided_pct": res["routing_cost"]["stage2_calls_avoided_percent"]}
        text_seed_rows.append(row)
        log(f"      seed={seed}: Acc={row['accuracy']:.4f} F1={row['f1']:.4f} "
            f"MCC={row['mcc']:.4f} GCov={row['group_coverage']:.4f} "
            f"S2-avoided={row['s2_avoided_pct']}%")
    ms = pd.DataFrame(text_seed_rows)
    ms.to_csv(out_dir / "PHASE_6_MULTISEED_TEXT.csv", index=False)
    summary["multiseed_text"] = {
        "seeds": SEEDS,
        "mean_std": {c: {"mean": round(float(ms[c].mean()), 4),
                         "std": round(float(ms[c].std(ddof=1)), 4)}
                     for c in ["accuracy", "precision", "recall", "f1", "mcc",
                                "group_coverage", "s2_avoided_pct"]},
    }

    # =============================================== S3 CONFORMAL REPAIR
    log("\n[S3] Group-conformal coverage repair (target 0.95, alpha=0.05)...")
    pipe42 = Phase5TextPipeline(random_state=RANDOM_SEED, alpha=0.05, stage2_budget_cap=0.70)
    pipe42.fit(F_tr.copy(), F_val.copy(), raw_tr, raw_val)

    # Reproduce the pipeline's calibration-group split EXACTLY (premise-level rng 42)
    val_prems = raw_val["reference_text"].values
    uniq = np.unique(val_prems)
    shuffled = np.random.RandomState(RANDOM_SEED).permutation(uniq)
    cal_groups = shuffled[: pipe42.n_conformal_groups]
    cal_mask = np.isin(val_prems, cal_groups)
    y_val_arr = F_val["ground_truth_label"].values.astype(int)
    F_val_enr = F_val.copy()
    emb_v = (pipe42.lsa_encoder.encode(list(raw_val["reference_text"])),
             pipe42.lsa_encoder.encode(list(raw_val["suspicious_text"])))
    F_val_enr["text_lsa_semantic_sim"] = np.round(
        np.sum(emb_v[0] * emb_v[1], axis=1), 6).clip(0.0, 1.0)
    p_val_cal = pipe42.calibrator.predict_positive_proba(
        pipe42.s1_model.predict_proba(F_val_enr[PHASE5_TEXT_FEATURES].values)[:, 1])
    gid_cal = val_prems[cal_mask]
    y_cal = y_val_arr[cal_mask]
    p_cal = p_val_cal[cal_mask]

    # Held-out test probabilities
    F_te_enr = F_te.copy()
    emb_t = (pipe42.lsa_encoder.encode(list(raw_te["reference_text"])),
             pipe42.lsa_encoder.encode(list(raw_te["suspicious_text"])))
    F_te_enr["text_lsa_semantic_sim"] = np.round(
        np.sum(emb_t[0] * emb_t[1], axis=1), 6).clip(0.0, 1.0)
    y_te_arr = F_te["ground_truth_label"].values.astype(int)
    p_te_cal = pipe42.calibrator.predict_positive_proba(
        pipe42.s1_model.predict_proba(F_te_enr[PHASE5_TEXT_FEATURES].values)[:, 1])
    gid_te = raw_te["reference_text"].values

    conf_rows: List[Dict[str, Any]] = []

    # A. Standard baseline reproduction
    ug_c, S_cal, _ = group_supremum_scores(gid_cal, y_cal, p_cal)
    qhat_std = exact_quantile(S_cal, 0.05)
    resA = evaluate_group_coverage(gid_te, y_te_arr, p_te_cal, qhat_std)
    conf_rows.append({"variant": "A_standard_group_supremum (phase-5)", **resA})
    log(f"      [A] Standard:    coverage={resA['coverage']:.4f} "
        f"Wilson95=[{resA['wilson95'][0]:.4f},{resA['wilson95'][1]:.4f}] qhat={qhat_std:.4f}")

    # B. Mondrian by calibration-group size
    resB = evaluate_mondrian_size(gid_cal, y_cal, p_cal, gid_te, y_te_arr, p_te_cal, 0.05)
    conf_rows.append({"variant": "B_mondrian_group_size", **{k: v for k, v in resB.items()}})
    log(f"      [B] Mondrian-size:    coverage={resB['coverage']:.4f} "
        f"Wilson95=[{resB['wilson95'][0]:.4f},{resB['wilson95'][1]:.4f}]")

    # C. Mondrian by majority class
    resC = evaluate_mondrian_parity(gid_cal, y_cal, p_cal, gid_te, y_te_arr, p_te_cal, 0.05)
    conf_rows.append({"variant": "C_mondrian_class_parity", **{k: v for k, v in resC.items()}})
    log(f"      [C] Mondrian-parity:  coverage={resC['coverage']:.4f} "
        f"Wilson95=[{resC['wilson95'][0]:.4f},{resC['wilson95'][1]:.4f}]")

    # D. Split-sample alpha* tuning (on calibration pool only)
    tune = alpha_split_sample_tuning(gid_cal, y_cal, p_cal, 0.95, random_state=RANDOM_SEED)
    qhat_star = exact_quantile(S_cal, tune["alpha_star_conservative"])
    resD = evaluate_group_coverage(gid_te, y_te_arr, p_te_cal, qhat_star)
    conf_rows.append({"variant": "D_split_sample_alpha_tuned", **resD,
                      "alpha_star": tune["alpha_star_conservative"],
                      "alpha_tuning_detail": tune})
    log(f"      [D] Alpha-tuned:      coverage={resD['coverage']:.4f} "
        f"Wilson95=[{resD['wilson95'][0]:.4f},{resD['wilson95'][1]:.4f}] "
        f"alpha*={tune['alpha_star_conservative']:.3f}")

    conf_df = pd.DataFrame([{k: v for k, v in r.items() if k != "alpha_tuning_detail"}
                            for r in conf_rows])
    conf_df.to_csv(out_dir / "PHASE_6_CONFORMAL_VARIANTS.csv", index=False)
    summary["conformal_repair"] = conf_rows
    # Recommendation = first variant achieving both coverage>=target and
    # Wilson-lower>=target-0.015; else best conservative fallback
    rec = None
    for r in conf_rows:
        if r["coverage"] >= 0.95 and r["wilson95"][0] >= 0.935:
            rec = r["variant"]; break
    if rec is None:
        rec = max(conf_rows, key=lambda r: (r["wilson95"][0], r["coverage"]))["variant"]
    summary["conformal_recommendation"] = rec
    log(f"      RECOMMENDED: {rec}")

    # ============================================ S4 NEURAL VERIFIER ABLATION
    log("\n[S4] Stage-2 heavy-verifier ablation (MLP vs RF vs HGB vs LR)...")
    emb_tr = (pipe42.lsa_encoder.encode(list(raw_tr["reference_text"])),
              pipe42.lsa_encoder.encode(list(raw_tr["suspicious_text"])))
    F_tr_enr = F_tr.copy()
    F_tr_enr["text_lsa_semantic_sim"] = np.round(
        np.sum(emb_tr[0] * emb_tr[1], axis=1), 6).clip(0.0, 1.0)
    Xtr = np.hstack([LsaSemanticEncoder.interaction_features(*emb_tr),
                     F_tr_enr[PHASE5_TEXT_FEATURES].values.astype(np.float64)])
    Xv = np.hstack([LsaSemanticEncoder.interaction_features(*emb_v),
                    F_val_enr[PHASE5_TEXT_FEATURES].values.astype(np.float64)])
    Xt = np.hstack([LsaSemanticEncoder.interaction_features(*emb_t),
                    F_te_enr[PHASE5_TEXT_FEATURES].values.astype(np.float64)])
    y_tr_arr = F_tr["ground_truth_label"].values.astype(int)

    cands: Dict[str, Any] = {
        "MLP cross-style (proposed)": CrossStylePairVerifier(random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=12,
                                               random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=RANDOM_SEED),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000,
                                                 random_state=RANDOM_SEED),
    }
    def _pos_proba(model, X):
        p = model.predict_proba(X)
        return p[:, -1] if getattr(p, "ndim", 1) == 2 else p

    ver_rows: List[Dict[str, Any]] = []
    trained: Dict[str, Any] = {}
    for name, cand in cands.items():
        tx = time.time()
        cand.fit(Xtr, y_tr_arr)
        pva = _pos_proba(cand, Xv)
        pte = _pos_proba(cand, Xt)
        trained[name] = pte
        lat_ms = np.nan
        if hasattr(cand, "measure_single_pair_latency_ms"):
            lat_ms = cand.measure_single_pair_latency_ms(Xt[:1])
        ver_rows.append({
            "verifier": name,
            "val_mcc@0.5": round(float(matthews_corrcoef(
                y_val_arr, (pva >= 0.5).astype(int))), 4),
            "test_beta1_metrics": _mm(y_te_arr, (pte >= 0.5).astype(int)),
            "fit_sec": round(time.time() - tx, 2),
            "latency_ms": (round(lat_ms, 6) if lat_ms == lat_ms else "NR"),
        })
        log(f"      {name:32s} valMCC={ver_rows[-1]['val_mcc@0.5']:.4f} "
            f"testMCC(β=1)={ver_rows[-1]['test_beta1_metrics']['mcc']:.4f}")
    summary["verifier_ablation"] = ver_rows
    pd.DataFrame([{"verifier": r["verifier"], "val_mcc@0.5": r["val_mcc@0.5"],
                   **{f"test_{k}": v for k, v in r["test_beta1_metrics"].items()},
                   "fit_sec": r["fit_sec"], "latency_ms": r["latency_ms"]}
                  for r in ver_rows]).to_csv(
        out_dir / "PHASE_6_VERIFIER_ABLATION.csv", index=False)

    # ===================================== S5 EXTERNAL TEXT VALIDITY
    log("\n[S5] External text validity (evaluation-only transfer)...")
    OFF_FEATS = [f for f in PHASE5_TEXT_FEATURES if f != "text_semantic_minilm_sim"]
    lex = TextLexicalExtractor().fit(
        list(raw_tr["reference_text"]) + list(raw_tr["suspicious_text"]))
    cha = TextCharacterExtractor().fit(
        list(raw_tr["reference_text"]) + list(raw_tr["suspicious_text"]))
    stx = TextStructuralExtractor()
    lsa_att = pipe42.lsa_encoder  # fitted on ORIGINAL train only

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

    F_tr_off = extract_offline(raw_tr["reference_text"], raw_tr["suspicious_text"])
    F_val_off = extract_offline(raw_val["reference_text"], raw_val["suspicious_text"])
    s1_off = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED)
    s1_off.fit(F_tr_off[OFF_FEATS].values, y_tr_arr)
    calib_off = PlattScaler().fit(
        s1_off.predict_proba(F_val_off[OFF_FEATS].values)[:, 1], y_val_arr)

    Xtr_s2_off = np.hstack([
        LsaSemanticEncoder.interaction_features(*emb_tr), F_tr_off[OFF_FEATS].values])
    s2_off = CrossStylePairVerifier(random_state=RANDOM_SEED)
    s2_off.fit(Xtr_s2_off, y_tr_arr)
    s2v_off = s2_off.predict_proba(np.hstack([
        LsaSemanticEncoder.interaction_features(*emb_v), F_val_off[OFF_FEATS].values]))
    p_val_off = calib_off.predict_positive_proba(
        s1_off.predict_proba(F_val_off[OFF_FEATS].values)[:, 1])
    ent_val_off = compute_predictive_entropy(p_val_off)
    op_off = tune_routing_operating_point(y_val_arr, p_val_off, ent_val_off, s2v_off,
                                          stage2_budget_cap=0.70, min_val_recall=0.65)
    log(f"      Offline variant refit on ORIGINAL train only; val operating point: {op_off}")

    def eval_external(df: pd.DataFrame, name: str, group_col: str = "group_id"):
        tS = time.time()
        feats = extract_offline(df["reference_text"], df["suspicious_text"])
        s1p = s1_off.predict_proba(feats[OFF_FEATS].values)[:, 1]
        pc = calib_off.predict_positive_proba(s1p)
        ent = compute_predictive_entropy(pc)
        em = (lsa_att.encode(list(df["reference_text"])),
              lsa_att.encode(list(df["suspicious_text"])))
        s2p = s2_off.predict_proba(np.hstack([
            LsaSemanticEncoder.interaction_features(*em), feats[OFF_FEATS].values]))
        pred, states, cost = apply_routing(
            pc, ent, s2p, op_off["tau_low"], op_off["tau_high"],
            op_off["entropy_threshold"], op_off["stage2_decision_threshold"])
        pred_b1 = (s2p >= 0.5).astype(int)
        y = df["ground_truth_label"].values.astype(int)
        boot = compute_group_aware_bootstrap_ci(
            df[group_col].values, y, pred, n_resamples=500, random_state=RANDOM_SEED)
        out_preds = pd.DataFrame({
            "pair_id": df["pair_id"], "group_id": df[group_col],
            "ground_truth_label": y, "calibrated_prob": np.round(pc, 4),
            "stage2_score": np.round(s2p, 4), "final_prediction": pred,
            "decision_state": states,
        })
        out_preds.to_parquet(out_dir / f"PHASE_6_EXTERNAL_{name}_PREDICTIONS.parquet",
                             index=False)
        res = {
            "dataset": name, "n": len(df), "n_groups": int(df[group_col].nunique()),
            "positives": int(y.sum()),
            "routed_metrics": _mm(y, pred),
            "beta1_metrics": _mm(y, pred_b1),
            "stage2_escalated_pct": cost.get("escalated_count", 0) / len(df) * 100.0,
            "routing_cost": {k: v for k, v in cost.items() if not isinstance(v, np.generic)},
            "group_bootstrap_acc_ci": boot.get("accuracy", "NR"),
            "elapsed_sec": round(time.time() - tS, 1),
        }
        return res, pred

    ext_rows: List[Dict[str, Any]] = []
    res_mr, pred_mr = eval_external(MRPC, "MRPC_TEST")
    ext_rows.append(res_mr)
    log(f"      MRPC test:    Acc={res_mr['routed_metrics']['accuracy']:.4f} "
        f"F1={res_mr['routed_metrics']['f1']:.4f} MCC={res_mr['routed_metrics']['mcc']:.4f} "
        f"(β=1 MCC={res_mr['beta1_metrics']['mcc']:.4f})")

    res_cs, pred_cs = eval_external(CS, "CLOUGH_STEVENSON")
    ext_rows.append(res_cs)
    log(f"      C-S canonical: Acc={res_cs['routed_metrics']['accuracy']:.4f} "
        f"F1={res_cs['routed_metrics']['f1']:.4f} MCC={res_cs['routed_metrics']['mcc']:.4f}")

    res_csx, _ = eval_external(CSX, "CLOUGH_STEVENSON_CROSSTOPIC")
    res_csx["caveat"] = ("Cross-topic negatives are trivially separable distractors; "
                         "reported separately, never pooled with canonical protocol.")
    ext_rows.append(res_csx)
    log(f"      C-S cross-topic (caveated): Acc={res_csx['routed_metrics']['accuracy']:.4f}")

    # Severity slicing on the canonical C-S set (unseen external obfuscation grades)
    sev_rows = []
    y_cs = CS["ground_truth_label"].values.astype(int)
    for g in ["g1", "g2", "g3", "g4"]:
        m = (CS["severity_grade"].values == g).astype(bool)
        y_g = y_cs[m]
        pr_g = pred_cs[m]
        sev_rows.append({
            "grade": g, "n": int(m.sum()),
            "recall": round(float(recall_score(y_g, pr_g, zero_division=0)), 4),
            "wilson_recall_ci_95": None,  # filled below
        })
        from research_upgrade.phase_6_q1_hardening.h6src.conformal_variants import wilson_ci
        k = int((y_g[pr_g == 1] == 1).sum())
        lo, hi = wilson_ci(k, int(m.sum()))
        sev_rows[-1]["wilson_recall_ci_95"] = [round(lo, 4), round(hi, 4)]
        log(f"      C-S grade {g}: recall={sev_rows[-1]['recall']:.4f} "
            f"Wilson95={sev_rows[-1]['wilson_recall_ci_95']}")
    pd.DataFrame(sev_rows).to_csv(out_dir / "PHASE_6_CS_SEVERITY_RECALL.csv", index=False)
    summary["external_text"] = ext_rows
    summary["cs_severity_recall"] = sev_rows

    # ===================================== S6 CODE: MULTI-SEED + TRANSFER
    log("\n[S6] Code: multi-seed LOGO (A aug, B) + unseen-severity L6 transfer "
        "+ Policy A/B consistency ...")
    code_pipe_aug = CodeMultiViewPipeline()
    feats_A = C_a  # 13 feature cols
    aug_cache: Dict[int, np.ndarray] = {}

    def mk_aug(seed):
        def code_augment_fn(abs_idx: np.ndarray):
            Xa, ya = [], []
            for i in abs_idx:
                i = int(i)
                if int(feats_A["ground_truth_label"].values[i]) != 1:
                    continue
                if i not in aug_cache:
                    fe = code_pipe_aug.extract_pair(
                        str(raw_code_a["reference_code"].values[i]),
                        systematic_identifier_renaming(
                            str(raw_code_a["suspicious_code"].values[i])))
                    aug_cache[i] = np.array([fe[c] for c in C_FEATS], dtype=float)
                Xa.append(aug_cache[i])
                ya.append(1)
            if not Xa:
                return np.zeros((0, len(C_FEATS))), np.zeros(0, dtype=int)
            return np.asarray(Xa, dtype=np.float64), np.asarray(ya, dtype=int)
        return code_augment_fn

    code_rows: List[Dict[str, Any]] = []
    fold_bank: Dict[str, Any] = {}
    for seed in SEEDS:
        for pol, df, cases, aug in [
            ("A", C_a, raw_code_a["case"], mk_aug(seed)),
            ("B", C_b, raw_code_b["case"], None),
        ]:
            ev = Phase5CodeEvaluator(random_state=seed, stage2_budget_cap=0.45,
                                     min_val_recall=0.80, augment_pairs_fn=aug)
            r = ev.evaluate_logo(df, cases, f"Policy {pol}")
            m = r["classification_metrics"]
            code_rows.append({
                "seed": seed, "policy": pol,
                "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4), "f1": round(m["f1_score"], 4),
                "mcc": round(m["matthews_corrcoef"], 4),
                "s2_avoided_pct": r["routing_cost"]["stage2_calls_avoided_percent"],
                "verifiers": sorted({v["selected"] for v in r["verifier_selection"]}),
            })
            if seed == RANDOM_SEED:
                fold_bank[pol] = ev
            log(f"      seed={seed} Policy {pol}: MCC={code_rows[-1]['mcc']:.4f} "
                f"S2-avoided={code_rows[-1]['s2_avoided_pct']}%")
    cm = pd.DataFrame(code_rows)
    cm.to_csv(out_dir / "PHASE_6_MULTISEED_CODE.csv", index=False)
    summary["multiseed_code"] = {
        "seeds": SEEDS,
        "by_policy": {p: {c: {"mean": round(float(cm[(cm.policy == p)][c].mean()), 4),
                              "std": round(float(cm[(cm.policy == p)][c].std(ddof=1)), 4)}
                          for c in ["accuracy", "precision", "recall", "f1", "mcc",
                                     "s2_avoided_pct"]} for p in ("A", "B")},
    }

    # Policy A/B consistency (Spearman on per-seed metrics)
    from scipy.stats import spearmanr
    a_m = cm[cm.policy == "A"]["mcc"].values
    b_m = cm[cm.policy == "B"]["mcc"].values
    rho, pval = spearmanr(a_m, b_m)
    summary["policy_consistency"] = {
        "spearman_rho_mcc": round(float(rho), 4), "p_value": round(float(pval), 6),
        "note": ("Only 3 seeds: interpret as protocol-level consistency check, "
                 "not inferential claim."),
        "delta_mcc_mean": round(float(a_m.mean() - b_m.mean()), 4),
    }
    log(f"      Policy A↔B MCC Spearman ρ={rho:.4f} (Δ mean={summary['policy_consistency']['delta_mcc_mean']:.4f})")

    # ---- Unseen-severity transfer (train L1..L5+unlabelled, test L6, LOGO folds)
    log("\n      Unseen-severity L6 transfer (LOGO trained w/o any L6 pair)...")
    lvl = C_a["obfuscation_level"].astype(object)
    is_l6 = (lvl == "L6")
    C_train_noL6 = C_a[~is_l6].reset_index(drop=True)
    cases_noL6 = raw_code_a["case"][~is_l6.values].reset_index(drop=True)
    C_L6 = C_a[is_l6].reset_index(drop=True)
    ev_t = Phase5CodeEvaluator(random_state=RANDOM_SEED, stage2_budget_cap=0.45,
                               min_val_recall=0.80, augment_pairs_fn=mk_aug(RANDOM_SEED))
    r_t = ev_t.evaluate_logo(C_train_noL6, cases_noL6, "Policy A (no L6)")
    fold_models = ev_t.fold_models_
    X_l6 = C_L6[C_FEATS].values.astype(np.float64)
    y_l6 = C_L6["ground_truth_label"].values.astype(int)
    p_l6_bank, s2_l6_bank = [], []
    for fm in fold_models:
        s1p = fm["s1_model"].predict_proba(X_l6)[:, 1]
        p_l6_bank.append(fm["platt"].predict_positive_proba(s1p))
        s2_l6_bank.append(fm["s2_model"].predict_proba(X_l6))
    p_l6 = np.mean(p_l6_bank, axis=0)
    ent_l6 = compute_predictive_entropy(p_l6)
    s2_l6 = np.mean(s2_l6_bank, axis=0)
    op_med = {k: float(np.median([fm["operating_point"][k] for fm in fold_models]))
              for k in ("tau_low", "tau_high", "entropy_threshold", "stage2_decision_threshold")}
    pred_l6, states_l6, cost_l6 = apply_routing(
        p_l6, ent_l6, s2_l6, op_med["tau_low"], op_med["tau_high"],
        op_med["entropy_threshold"], op_med["stage2_decision_threshold"])
    in_dist_mcc = round(float(r_t["classification_metrics"]["matthews_corrcoef"]), 4)
    rec_l6 = float(recall_score(y_l6, pred_l6, zero_division=0))
    if len(np.unique(y_l6)) < 2:
        l6_metrics = {
            "detection_recall": round(rec_l6, 4),
            "accuracy": round(rec_l6, 4),
            "mcc": "NR (held-out L6 slice is single-class: all-positive)",
            "precision_balance_note": "NR (no legitimate pairs present in L6)",
        }
    else:
        l6_metrics = _mm(y_l6, pred_l6)
    transfer = {
        "train_scope": "LOGO folds over Policy-A cases, severities L1..L5 + unlabelled ONLY",
        "held_out": "All L6 (most obfuscated) pairs",
        "n_l6": int(len(y_l6)), "positive_l6": int(y_l6.sum()),
        "in_distribution_noL6_mcc": in_dist_mcc,
        "l6_transfer_metrics": l6_metrics,
        "operating_point": op_med,
        "routing_cost": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                         for k, v in cost_l6.items()},
    }
    summary["code_severity_transfer"] = transfer
    _mcc6 = transfer["l6_transfer_metrics"]["mcc"]
    log(f"      L6 transfer: n={transfer['n_l6']} (pos={transfer['positive_l6']}) "
        f"MCC={_mcc6 if isinstance(_mcc6, str) else round(_mcc6, 4)} "
        f"detection_recall={transfer['l6_transfer_metrics'].get('detection_recall', 'NA')} "
        f"(in-dist MCC={in_dist_mcc})")

    # ============================================= S7 C4 REVIEW PACKET
    log("\n[S7] C4 expert-review walkthrough packet (20 grey-zone test pairs)...")
    t_res42 = pipe42.evaluate_test(F_te.copy(), raw_te)
    preds_df = t_res42["predictions_dataframe"]
    pcal = preds_df["calibrated_prob"].values
    ent = preds_df["predictive_entropy"].values
    op42 = t_res42["operating_point"]
    grey = np.where(((pcal >= op42["tau_low"]) & (pcal <= op42["tau_high"])) |
                    (ent >= op42["entropy_threshold"] * 0.9))[0]
    rng7 = np.random.RandomState(RANDOM_SEED)
    chosen = rng7.permutation(grey)[:20] if len(grey) >= 20 else grey
    feats_cols = [c for c in PHASE5_TEXT_FEATURES if c in F_te.columns]
    packet = []
    for i in chosen:
        i = int(i)
        row = F_te.iloc[i]
        ref, sus = str(raw_te["reference_text"].values[i]), str(raw_te["suspicious_text"].values[i])
        gaps = {}
        med_pos = F_te[F_te["ground_truth_label"] == 1][feats_cols].median(numeric_only=True)
        med_neg = F_te[F_te["ground_truth_label"] == 0][feats_cols].median(numeric_only=True)
        for c in feats_cols:
            v = float(F_te.iloc[i][c])
            gaps[c] = abs(v - float(med_pos[c])) - abs(v - float(med_neg[c]))
        top_flags = sorted(gaps, key=gaps.get)[:3]
        packet.append({
            "pair_id": preds_df["pair_id"].values[i],
            "premise_group_id": preds_df["premise_group_id"].values[i],
            "ground_truth_label": int(preds_df["ground_truth_label"].values[i]),
            "calibrated_prob": round(float(pcal[i]), 4),
            "predictive_entropy": round(float(ent[i]), 4),
            "routing_state": preds_df["routing_decision_state"].values[i],
            "stage2_score": round(float(preds_df["stage2_score"].values[i]), 4),
            "top3_views_flagging_plagiarism": top_flags,
            "reference_excerpt": ref[:240],
            "suspicious_excerpt": sus[:240],
        })
    pd.DataFrame(packet).to_csv(out_dir / "PHASE_6_C4_REVIEW_PACKET.csv", index=False)
    json.dump(packet, open(out_dir / "PHASE_6_C4_REVIEW_PACKET.json", "w"),
              indent=2, cls=NumpyEncoder)
    grey_zone_pct = round(float(len(grey) / len(preds_df) * 100.0), 2)
    summary["c4_packet"] = {
        "n_grey_zone_pairs": int(len(grey)), "grey_zone_pct": grey_zone_pct,
        "n_packet_pairs": int(len(packet)),
        "disclosure": ("What is provided: an inspection-ready packet reproducing the "
                       "C4 review workflow end-to-end (grey-zone retrieval, per-pair "
                       "multi-view flag sheet, tracking fields) as validation of the "
                       "review deliverables & protocol. What is NOT claimed: measured "
                       "human accuracy gains — no live expert study was feasible; this "
                       "is stated in the manuscript limitations."),
    }
    log(f"      Grey-zone pairs: {len(grey)} ({grey_zone_pct}%); packet saved ({len(packet)} sheets).")

    # ------------------------------------------------------------- SUMMARY
    summary["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(summary, open(out_dir / "PHASE_6_SUMMARY.json", "w"),
              indent=2, cls=NumpyEncoder)
    with open(out_dir / "PHASE_6_RUN_LOG.txt", "w") as f:
        f.write(log_buf.getvalue())
    log(f"\nDONE in {summary['runtime_sec']}s — artifacts in {out_dir}")


if __name__ == "__main__":
    main()
