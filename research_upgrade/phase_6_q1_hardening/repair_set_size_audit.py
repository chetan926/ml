"""Audit set-size informativeness of the recommended conformal repair (α*=0.01).

Recalibrates the phase-5 CorrectedGroupConformalPredictor at the tuned alpha on
the SAME val calibration groups and reports the full prediction-set statistics
(avg size, singleton/ambiguous/empty %, pair + group coverage with Wilson CIs)
on the held-out test set. Ensures the repair is not merely an uninformative
universal-set fallback.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/ml")
sys.path.insert(0, str(ROOT))

from research_upgrade.src.config import RANDOM_SEED, set_global_seed
from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
from research_upgrade.phase_4_1_corrections.src.calibration.group_conformal import (
    CorrectedGroupConformalPredictor,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_text_pipeline import (
    PHASE5_TEXT_FEATURES, Phase5TextPipeline,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import wilson_score_ci


def main():
    set_global_seed(RANDOM_SEED)
    out_dir = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "artifacts"
    text_loader = TextBenchmarkDataLoader()
    raw_tr, raw_val, raw_te = (text_loader.get_partition(s) for s in ("train", "val", "test"))
    feats_dir = ROOT / "research_upgrade" / "artifacts" / "phase_3" / "phase_3_3" / "features"
    F_tr = pd.read_parquet(feats_dir / "text_feats_train.parquet")
    F_val = pd.read_parquet(feats_dir / "text_feats_val.parquet")
    F_te = pd.read_parquet(feats_dir / "text_feats_test.parquet")

    pipe = Phase5TextPipeline(random_state=RANDOM_SEED, alpha=0.05, stage2_budget_cap=0.70)
    pipe.fit(F_tr.copy(), F_val.copy(), raw_tr, raw_val)

    val_prems = raw_val["reference_text"].values
    shuffled = np.random.RandomState(RANDOM_SEED).permutation(np.unique(val_prems))
    cal_mask = np.isin(val_prems, shuffled[: pipe.n_conformal_groups])
    y_val = F_val["ground_truth_label"].values.astype(int)

    F_val_enr = F_val.copy()
    emb_v = (pipe.lsa_encoder.encode(list(raw_val["reference_text"])),
             pipe.lsa_encoder.encode(list(raw_val["suspicious_text"])))
    F_val_enr["text_lsa_semantic_sim"] = np.round(np.sum(emb_v[0] * emb_v[1], axis=1), 6).clip(0, 1)
    p_val_cal = pipe.calibrator.predict_positive_proba(
        pipe.s1_model.predict_proba(F_val_enr[PHASE5_TEXT_FEATURES].values)[:, 1])

    F_te_enr = F_te.copy()
    emb_t = (pipe.lsa_encoder.encode(list(raw_te["reference_text"])),
             pipe.lsa_encoder.encode(list(raw_te["suspicious_text"])))
    F_te_enr["text_lsa_semantic_sim"] = np.round(np.sum(emb_t[0] * emb_t[1], axis=1), 6).clip(0, 1)
    y_te = F_te["ground_truth_label"].values.astype(int)
    p_te_cal = pipe.calibrator.predict_positive_proba(
        pipe.s1_model.predict_proba(F_te_enr[PHASE5_TEXT_FEATURES].values)[:, 1])
    gid_te = raw_te["reference_text"].values

    from research_upgrade.phase_6_q1_hardening.h6src.conformal_variants import (
        evaluate_mondrian_size, evaluate_mondrian_parity,
    )
    rows = {}
    for tag, alpha in [("standard_alpha_0.05", 0.05), ("repaired_alpha_0.01", 0.01)]:
        cp = CorrectedGroupConformalPredictor(alpha=alpha)
        cp.calibrate(val_prems[cal_mask], y_val[cal_mask], p_val_cal[cal_mask])
        s = cp.evaluate_test_coverage(gid_te, y_te, p_te_cal)
        k = int(round(s["empirical_group_coverage"] * len(np.unique(gid_te))))
        _, lo, hi = wilson_score_ci(k, len(np.unique(gid_te)))
        s["group_cov_wilson95"] = [round(lo, 4), round(hi, 4)]
        rows[tag] = s
        print(f"[{tag}] group_cov={s['empirical_group_coverage']} "
              f"pair_cov={s['empirical_pair_coverage']} "
              f"avg_set={s['average_prediction_set_size']} "
              f"singleton={s['singleton_set_percentage']}% "
              f"ambiguous={s['ambiguous_pair_percentage']}% "
              f"qhat={s['calibration_quantile']}")

    # Mondrian variants at the tuned alpha (group-supremum coverage by stratum)
    mond = {}
    for alpha in (0.05, 0.01):
        rb = evaluate_mondrian_size(val_prems[cal_mask], y_val[cal_mask],
                                    p_val_cal[cal_mask], gid_te, y_te, p_te_cal, alpha)
        rc = evaluate_mondrian_parity(val_prems[cal_mask], y_val[cal_mask],
                                      p_val_cal[cal_mask], gid_te, y_te, p_te_cal, alpha)
        mond[f"alpha_{alpha}"] = {
            "mondrian_size": {k: v for k, v in rb.items()},
            "mondrian_parity": {k: v for k, v in rc.items()},
        }
        print(f"mondrian@alpha={alpha}: size={rb['coverage']} parity={rc['coverage']}")
    rows["mondrian_variants_at_tuned_alpha"] = mond

    json.dump(rows, open(out_dir / "PHASE_6_CONFORMAL_SET_SIZE_AUDIT.json", "w"), indent=2)


if __name__ == "__main__":
    main()
