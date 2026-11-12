"""RFA-6/10: genuine A0-A6 ablation runs (text + code Policy-A LOGO).

Configs (no augmentation; no renaming attack stack — ablation isolates the four
architectural components only):
  A0 basic single-view LR (strongest single view, val-tuned threshold)
  A1 multi-view flat LR (threshold 0.5)
  A2 multi-view LR + Platt calibration (C2 without routing)
  A3 multi-view LR + adaptive routing, NO calibration (raw s1 probs into router)
  A4 multi-view LR + budget cap only (top-beta uncertain by |p-0.5| escalate to S2)
  A5 A3 + calibration (routing ON, budget cap = 1.0 i.e. unconstrained OP)
  A6 full proposed framework (calibration + routing + budget cap)

A3/A5/A6 route with the heavy Stage-2 verifier (HGB on the same multi-view input
for code; MLP cross-style on LSA interaction space for text — the phase-5
verifier classes, refit here under the same protocol).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, str(Path("/home/user/ml")))
from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    apply_routing, tune_routing_operating_point, tune_threshold_for_f1)
from research_upgrade.phase_5_q1_upgrade.q1src.lsa_semantic import LsaSemanticEncoder
from research_upgrade.phase_5_q1_upgrade.q1src.heavy_verifiers import CrossStylePairVerifier
from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
from rfa_common import RFA, SEED, full_metrics, load_code_raw, load_features, wc, wj

OUT = RFA / "06_ABLATION_STUDY"

CODE_FEATS = ["code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
              "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim",
              "code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim",
              "code_operator_dist_sim", "code_control_flow_sim", "code_cyclomatic_sim",
              "code_sloc_ratio"]
TEXT_FEATS = ["text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
              "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio",
              "text_semantic_minilm_sim", "text_lsa_semantic_sim", "text_length_ratio",
              "text_token_count_delta", "text_sentence_delta", "text_ttr_sim"]


def budget_row(tag, n, n_s2, runtime_ms_total, mets):
    n_avoid = n - n_s2
    return {
        "Configuration": tag, **{k: v for k, v in mets.items() if k != "confusion"},
        "expensive_calls": n_s2,
        "escalation_rate_pct": round(100.0 * n_s2 / n, 2),
        "pruning_ratio": round(n_avoid / n, 4),
        "candidate_recall": round(mets["recall"], 4),
        "runtime_ms_total": round(runtime_ms_total, 1),
        "runtime_ms_per_pair": round(runtime_ms_total / n, 3),
    }


# ---------------------------------------------------------------- code LOGO
def code_ablation(C, cases):
    y = C["ground_truth_label"].values.astype(int)
    X = C[CODE_FEATS].values.astype(np.float64)
    n = len(y)
    logo = list(LeaveOneGroupOut().split(X, y, groups=cases))

    out = []
    single = ["code_token_jaccard"]
    results = {}

    for cfg in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
        feats = single if cfg == "A0" else CODE_FEATS
        oof_pred = np.zeros(n, dtype=int)
        oof_s = np.zeros(n)
        n_s2 = 0
        t0 = time.time()
        for tr, tei in logo:
            lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
            lr.fit(X[tr][:, [CODE_FEATS.index(c) for c in feats]], y[tr])
            s1_tr = lr.predict_proba(X[tr][:, [CODE_FEATS.index(c) for c in feats]])[:, 1]
            s1_te = lr.predict_proba(X[tei][:, [CODE_FEATS.index(c) for c in feats]])[:, 1]
            s2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                random_state=SEED)
            s2.fit(X[tr], y[tr])
            s2_tr = s2.predict_proba(X[tr])[:, 1]
            s2_te = s2.predict_proba(X[tei])[:, 1]

            if cfg in ("A0", "A1"):
                thr = 0.5 if cfg == "A1" else tune_threshold_for_f1(y[tr], s1_tr)[0]
                oof_pred[tei] = (s1_te >= thr).astype(int)
                oof_s[tei] = s1_te
                if cfg == "A1":
                    n_s2 += 0  # no escalation
            elif cfg == "A2":
                pl = PlattScaler().fit(s1_tr, y[tr])
                p_te = pl.predict_positive_proba(s1_te)
                oof_pred[tei] = (p_te >= 0.5).astype(int)
                oof_s[tei] = p_te
            elif cfg in ("A3", "A5", "A6"):
                pl = PlattScaler().fit(s1_tr, y[tr])
                p_tr = pl.predict_positive_proba(s1_tr)
                p_te = pl.predict_positive_proba(s1_te)
                use_cal = cfg in ("A5", "A6")
                pr_tr = p_tr if use_cal else s1_tr
                pr_te = p_te if use_cal else s1_te
                op = tune_routing_operating_point(
                    y[tr], pr_tr, compute_predictive_entropy(pr_tr), s2_tr,
                    stage2_budget_cap={None: 0.45, "A3": 1.0, "A5": 1.0}.get(cfg, 0.45))
                pred_te, states, cost = apply_routing(
                    pr_te, compute_predictive_entropy(pr_te), s2_te,
                    op["tau_low"], op["tau_high"], op["entropy_threshold"],
                    op["stage2_decision_threshold"])
                oof_pred[tei] = pred_te
                oof_s[tei] = pr_te
                n_s2 += cost["escalated_count"]
            elif cfg == "A4":
                # budget-only: prune/flag by confidence distance, escalate hardest 45%
                unc_tr = np.abs(s1_tr - 0.5)
                k = int(np.quantile(unc_tr, 1 - 0.45))
                unc_te = np.abs(s1_te - 0.5)
                esc = unc_te <= np.quantile(unc_tr, 0.45)  # most uncertain 45%
                pred = np.zeros(len(tei), dtype=int)
                pred[esc] = (s2_te[esc] >= 0.5).astype(int)
                pred[~esc] = (s1_te[~esc] >= 0.5).astype(int)
                oof_pred[tei] = pred
                oof_s[tei] = s1_te
                n_s2 += int(esc.sum())
        rt = (time.time() - t0) * 1000
        m = full_metrics(y, oof_pred, oof_s)
        results[f"code_{cfg}"] = {"m": m, "n_s2": n_s2, "runtime": rt, "oof": oof_pred}
        out.append(budget_row(f"Code-{cfg}", n, n_s2, rt, m))
        print(f"  code {cfg}: MCC={m['mcc']:.4f} esc={n_s2} rt={rt/1000:.1f}s")
    return out, results


# ---------------------------------------------------------------- text
def text_ablation(F, raw):
    tr_, va_, te_ = F["text_train"].copy(), F["text_val"].copy(), F["text_test"].copy()
    y_tr = tr_["ground_truth_label"].values.astype(int)
    y_va = va_["ground_truth_label"].values.astype(int)
    y_te = te_["ground_truth_label"].values.astype(int)
    n = len(y_te)

    # refit LSA (train-only) + enrich the parquet features with the runtime LSA sim
    lsa = LsaSemanticEncoder(n_components=128, random_state=SEED).fit(
        list(raw["train"]["reference_text"]) + list(raw["train"]["suspicious_text"]))
    for d, r in ((tr_, raw["train"]), (va_, raw["val"]), (te_, raw["test"])):
        e = (lsa.encode(list(r["reference_text"])), lsa.encode(list(r["suspicious_text"])))
        d["text_lsa_semantic_sim"] = np.clip(np.sum(e[0] * e[1], axis=1), 0.0, 1.0)
    Xtr = tr_[TEXT_FEATS].values.astype(np.float64)
    Xva = va_[TEXT_FEATS].values.astype(np.float64)
    Xte = te_[TEXT_FEATS].values.astype(np.float64)
    def inter(df_raw):
        e = (lsa.encode(list(df_raw["reference_text"])), lsa.encode(list(df_raw["suspicious_text"])))
        return LsaSemanticEncoder.interaction_features(*e)
    s2 = CrossStylePairVerifier(random_state=SEED)
    s2.fit(np.hstack([inter(raw["train"]), Xtr]), y_tr)
    s2_va = s2.predict_proba(np.hstack([inter(raw["val"]), Xva]))
    s2_te = s2.predict_proba(np.hstack([inter(raw["test"]), Xte]))

    out, results = [], {}
    single_va = va_["text_semantic_minilm_sim"].values
    single_te = te_["text_semantic_minilm_sim"].values
    single_tr = tr_["text_semantic_minilm_sim"].values

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(Xtr, y_tr)
    s1_tr, s1_va, s1_te = (lr.predict_proba(X)[:, 1] for X in (Xtr, Xva, Xte))
    pl = PlattScaler().fit(s1_va, y_va)
    p_va, p_te = pl.predict_positive_proba(s1_va), pl.predict_positive_proba(s1_te)

    configs = {
        "A0": lambda: ((single_te >= tune_threshold_for_f1(y_va, single_va)[0]).astype(int), single_te, 0),
        "A1": lambda: ((s1_te >= 0.5).astype(int), s1_te, 0),
        "A2": lambda: ((p_te >= 0.5).astype(int), p_te, 0),
    }
    for cfg, fn in configs.items():
        t0 = time.time()
        pred, sc, n_s2 = fn()
        rt = (time.time() - t0) * 1000
        m = full_metrics(y_te, pred, sc)
        out.append(budget_row(f"Text-{cfg}", n, n_s2, rt, m))
        results[f"text_{cfg}"] = {"m": m, "n_s2": n_s2}
        print(f"  text {cfg}: MCC={m['mcc']:.4f}")

    for cfg, use_cal, cap in [("A3", False, 1.0), ("A5", True, 1.0), ("A6", True, 0.70)]:
        pr_va = p_va if use_cal else s1_va
        pr_te = p_te if use_cal else s1_te
        op = tune_routing_operating_point(y_va, pr_va, compute_predictive_entropy(pr_va),
                                          s2_va, stage2_budget_cap=cap)
        pred, states, cost = apply_routing(pr_te, compute_predictive_entropy(pr_te), s2_te,
                                           op["tau_low"], op["tau_high"],
                                           op["entropy_threshold"],
                                           op["stage2_decision_threshold"])
        m = full_metrics(y_te, pred, pr_te)
        out.append(budget_row(f"Text-{cfg}", n, int(cost["escalated_count"]), 0.0, m))
        results[f"text_{cfg}"] = {"m": m, "n_s2": int(cost["escalated_count"]), "op": op}
        print(f"  text {cfg}: MCC={m['mcc']:.4f} esc={cost['escalated_count']}")

    # A4 budget-only
    unc_va = np.abs(s1_va - 0.5)
    q = np.quantile(unc_va, 0.30)  # escalate = most uncertain 30% -> cap .70 budget
    esc_te = np.abs(s1_te - 0.5) <= q
    pred = np.zeros(n, dtype=int)
    pred[esc_te] = (s2_te[esc_te] >= 0.5).astype(int)
    pred[~esc_te] = (s1_te[~esc_te] >= 0.5).astype(int)
    m = full_metrics(y_te, pred, s1_te)
    out.append(budget_row("Text-A4", n, int(esc_te.sum()), 0.0, m))
    results["text_A4"] = {"m": m, "n_s2": int(esc_te.sum())}
    print(f"  text A4: MCC={m['mcc']:.4f} esc={esc_te.sum()}")
    return out, results


def main():
    F = load_features()
    raw_code = load_code_raw()
    tl = TextBenchmarkDataLoader()
    raw = {s: tl.get_partition(s) for s in ("train", "val", "test")}
    print("text ablation ...")
    txt_rows, txt_res = text_ablation(F, raw)
    print("code ablation (LOGO) ...")
    code_rows, code_res = code_ablation(F["code_a"], raw_code["policy_a"]["case"].values)

    df = pd.DataFrame(txt_rows + code_rows)
    df = df.sort_values("Configuration").reset_index(drop=True)
    wc(df, OUT / "ABLATION_RESULTS.csv")
    wj({"configurations": {
        "A0": "single strongest view (MiniLM sim text / token jaccard code), val/inner-tuned threshold",
        "A1": "multi-view flat LR, threshold 0.5",
        "A2": "A1 + Platt calibration on val / inner-OOF",
        "A3": "A1 + adaptive routing to heavy S2 (no calibration, unconstrained OP cap 1.0)",
        "A4": "A1 + budget-only escalation (uncertainty band by |p-0.5|, cap-driven)",
        "A5": "A2 + routing (calibrated) with unconstrained operating point",
        "A6": "full framework = A5 + budget-capped operating point (text 0.70 / code 0.45)",
        "scope_note": "identifier-renaming augmentation is attack-hardening, excluded here; "
                      "it is isolated in the phase-5 robustness artifacts."}},
       OUT / "ABLATION_CONFIGURATIONS.json")
    wj({"text": {k.split("_", 1)[1]: v for k, v in txt_res.items()},
        "code": {k.split("_", 1)[1]: {kk: vv for kk, vv in v.items() if kk != "oof"}
               for k, v in code_res.items()}},
       OUT / "ABLATION_RAW.json")
    print(df[["Configuration", "mcc", "escalation_rate_pct", "pruning_ratio"]].to_string())


if __name__ == "__main__":
    main()
