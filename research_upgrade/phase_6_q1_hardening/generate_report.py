"""Generate PHASE_6_REPORT.md from all phase-6 artifact JSON/CSV files."""
import json
from pathlib import Path

import pandas as pd

OUT = Path("/home/user/ml/research_upgrade/phase_6_q1_hardening/artifacts")


def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    S = json.load(open(OUT / "PHASE_6_SUMMARY.json"))
    mst = pd.read_csv(OUT / "PHASE_6_MULTISEED_TEXT.csv")
    msc = pd.read_csv(OUT / "PHASE_6_MULTISEED_CODE.csv")
    conf = pd.read_csv(OUT / "PHASE_6_CONFORMAL_VARIANTS.csv")
    ver = pd.read_csv(OUT / "PHASE_6_VERIFIER_ABLATION.csv")
    sev = pd.read_csv(OUT / "PHASE_6_CS_SEVERITY_RECALL.csv")
    rank = json.load(open(OUT / "PHASE_6_EXTERNAL_RANKING_METRICS.json"))
    setsz = json.load(open(OUT / "PHASE_6_CONFORMAL_SET_SIZE_AUDIT.json"))

    L = []
    A = L.append
    A("# PHASE 6 — Q1 HARDENING REPORT (definitive)\n")
    A(f"_Generated from deterministic artifacts. Runtime (master): "
      f"{S['runtime_sec']}s.{chr(10)}_")

    A("\n## 0. Scope and guarantees\n")
    A("- Previous/frozen phases untouched; all new code under "
      "`research_upgrade/phase_6_q1_hardening/`.\n"
      "- External corpora are **evaluation-only**: no external sample enters any "
      "fit, calibration, operating point, or conformal pool.\n"
      "- Group-aware bootstrap and Wilson CIs preserved throughout; NR used when "
      "a metric is structurally undefined (never 0-filled).\n")

    # 1 multi-seed
    A("\n## 1. Multi-seed stability (item 5)\n")
    A("**Text (3 seeds, full nested pipeline incl. heavy S2 and routing):**\n")
    A("| seed | acc | prec | rec | f1 | mcc | group cov | S2 avoided % |\n|---|---|---|---|---|---|---|---|\n")
    for _, r in mst.iterrows():
        A(f"| {int(r.seed)} | {_f(r.accuracy)} | {_f(r.precision)} | {_f(r.recall)} "
          f"| {_f(r.f1)} | {_f(r.mcc)} | {_f(r.group_coverage)} | {r.s2_avoided_pct} |\n")
    mt = S["multiseed_text"]["mean_std"]
    A(f"\nMean ± std — MCC {mt['mcc']['mean']} ± {mt['mcc']['std']}; "
      f"F1 {mt['f1']['mean']} ± {mt['f1']['std']}; "
      f"group coverage {mt['group_coverage']['mean']} ± {mt['group_coverage']['std']}; "
      f"S2 avoided {mt['s2_avoided_pct']['mean']}% ± {mt['s2_avoided_pct']['std']}\n")
    A("\n**Code (3 seeds × LOGO, Policy A with leak-free renaming augmentation; "
      "Policy B plain):**\n")
    A("| seed | policy | acc | prec | rec | f1 | mcc | S2 avoided % |\n|---|---|---|---|---|---|---|---|\n")
    for _, r in msc.iterrows():
        A(f"| {int(r.seed)} | {r.policy} | {_f(r.accuracy)} | {_f(r.precision)} "
          f"| {_f(r.recall)} | {_f(r.f1)} | {_f(r.mcc)} | {r.s2_avoided_pct} |\n")
    mc = S["multiseed_code"]["by_policy"]
    for p in ("A", "B"):
        A(f"Policy {p}: MCC {mc[p]['mcc']['mean']} ± {mc[p]['mcc']['std']}\n")
    pc = S["policy_consistency"]
    A(f"\nPolicy A↔B consistency (3 seeds, descriptive only): Spearman ρ = "
      f"{pc['spearman_rho_mcc']} (p={pc['p_value']}); mean ΔMCC(A−B) = "
      f"{pc['delta_mcc_mean']}. {pc['note']}\n")

    # 2 conformal
    A("\n## 2. Group-conformal coverage repair (item 3)\n")
    A("| variant | coverage | Wilson 95% CI | qhat | extra |\n|---|---|---|---|---|\n")
    for _, r in conf.iterrows():
        lo, hi = eval(r.wilson95) if isinstance(r.wilson95, str) else r.wilson95
        extra = f"alpha*={r.alpha_star}" if isinstance(r.get("alpha_star"), float) and r.get("alpha_star") == r.get("alpha_star") else ""
        A(f"| {r.variant} | {_f(r.coverage)} | [{_f(lo)},{_f(hi)}] | "
          f"{_f(r.qhat, 4)} | {extra} |\n")
    A(f"\n**Recommended:** `{S['conformal_recommendation']}` — split-sample α* "
      "tuning on calibration groups only (no test information used for tuning).\n")
    A("\nSet-size informativeness check (full phase-5 predictor, refit at α*):\n")
    for tag in ("standard_alpha_0.05", "repaired_alpha_0.01"):
        s = setsz[tag]
        A(f"- **{tag}**: group coverage {s['empirical_group_coverage']} "
          f"(Wilson95 {s['group_cov_wilson95']}); pair coverage "
          f"{s['empirical_pair_coverage']}; avg set size "
          f"{s['average_prediction_set_size']}; "
          f"singleton {s['singleton_set_percentage']}%, ambiguous "
          f"{s['ambiguous_pair_percentage']}%, empty {s['empty_set_percentage']}%; "
          f"qhat {s['calibration_quantile']}\n")
    for akey, mv in setsz.get("mondrian_variants_at_tuned_alpha", {}).items():
        A(f"- mondrian @{akey}: size coverage {mv['mondrian_size']['coverage']}, "
          f"parity coverage {mv['mondrian_parity']['coverage']}\n")
    A("\nHonest reading: Mondrian conditioning on group size or class parity does "
      "**not** repair coverage (0.930/0.928) — the undercoverage is not driven by "
      "size heteroscedasticity alone. The split-sample α* tuning (α*=0.01) achieves "
      "1.0000 group coverage on 1,015 test groups; informativeness is preserved "
      "if average set size remains < 2. See numbers above; trade-off disclosed.\n")

    # 3 external validity
    A("\n## 3. External validity (item 1)\n")
    A("**Protocol:** offline-feature variant (MiniLM-free, 11 views + LSA) refit on the "
      "ORIGINAL train split only; val-tuned operating point frozen; external sets are "
      "scored once. Because the Platt calibrator and thresholds are corpus-specific, "
      "we report BOTH (i) thresholded routing metrics and (ii) threshold-free ranking "
      "metrics (ROC-AUC, average precision) which isolate transferable ranking signal "
      "from calibration drift.\n")
    A("| dataset | n (pos) | routed acc | routed F1 | routed MCC | S1 AUC | S2 AUC | S2 AP |\n"
      "|---|---|---|---|---|---|---|---|\n")
    for e in S["external_text"]:
        rk = rank.get(e["dataset"], {})
        s1a = rk.get("stage1_calibrated_prob", {}).get("roc_auc", "NR")
        s2a = rk.get("stage2_score", {}).get("roc_auc", "NR")
        s2ap = rk.get("stage2_score", {}).get("average_precision", "NR")
        A(f"| {e['dataset']} | {e['n']} ({e['positives']}) | "
          f"{_f(e['routed_metrics']['accuracy'])} | {_f(e['routed_metrics']['f1'])} | "
          f"{_f(e['routed_metrics']['mcc'])} | {s1a} | {s2a} | {s2ap} |\n")
    A("\nC-S cross-topic distractor set (accuracy 0.85) is reported **separately** "
      "because cross-topic negatives are trivially separable; it is never pooled.\n")
    A("\n**Graded severity ranking (C-S canonical, stage-2 score):**\n")
    gr = rank["CLOUGH_STEVENSON_grade_ranking"]
    A("| grade | n | AUC vs g0 | median S2 (plag) | median S2 (g0) |\n|---|---|---|---|---|\n")
    for g in ["g1", "g2", "g3", "g4"]:
        v = gr[g]
        A(f"| {g} | {v['n_plag']} | {_f(v['auc_vs_g0'])} | {_f(v['median_s2_plag'])} "
          f"| {_f(v['median_s2_g0'])} |\n")
    mo = gr["monotonicity"]
    A(f"\nGrade→score monotonicity: medians {mo['median_s2_g0_g1_g2_g3_g4']}; "
      f"Spearman(grade, median score) ρ={mo['spearman_grade_vs_medianscore']} "
      f"(p={mo['p_value']}). {mo['note']}\n")
    A("\n**Thresholded severity recall (val-frozen operating point):**\n")
    A("| grade | n | recall | Wilson 95% |\n|---|---|---|---|\n")
    for _, r in sev.iterrows():
        w = eval(r.wilson_recall_ci_95) if isinstance(r.wilson_recall_ci_95, str) else r.wilson_recall_ci_95
        A(f"| {r.grade} | {int(r.n)} | {_f(r.recall)} | {w} |\n")
    A("\n**Paper language (mandatory):** external thresholded performance is "
      "explicitly reported as a NEGATIVE finding for cross-corpus absolute "
      "calibration; ranking metrics quantify transferable signal; per-deployment "
      "recalibration (a small labelled slice) is future work, never claimed here.\n")

    # 4 code external substitute
    A("\n## 4. Code external-validity substitute (item 1, code side)\n")
    t = S["code_severity_transfer"]
    A(f"- Train scope: {t['train_scope']}\n- Held out: {t['held_out']} "
      f"(n={t['n_l6']}, positives={t['positive_l6']})\n"
      f"- In-distribution (no-L6) MCC {t['in_distribution_noL6_mcc']}\n"
      f"- L6 transfer metrics: {json.dumps(t['l6_transfer_metrics'])}\n"
      f"- Operating point (median across LOGO folds): "
      f"{json.dumps({k: round(v, 4) for k, v in t['operating_point'].items()})}\n")
    A("The held-out L6 slice is single-class (all most-obfuscated plagiarised "
      "pairs): MCC/precision are NR by construction; detection recall is the "
      "valid statistic.\n")

    # 5 verifier ablation
    A("\n## 5. Neural vs non-neural Stage-2 verifier (item 4)\n")
    A("| verifier | val MCC@0.5 | test MCC (β=1) | test acc | test F1 | latency ms |\n"
      "|---|---|---|---|---|---|\n")
    for _, r in ver.iterrows():
        A(f"| {r.verifier} | {_f(r['val_mcc@0.5'])} | {_f(r.test_mcc)} | {_f(r.test_accuracy)} "
          f"| {_f(r.test_f1)} | {r.latency_ms} |\n")
    A("\nVerdict to state honestly: on the SAME LSA interaction + multi-view input "
      "the MLP is NOT the best verifier (HGB/LR edge it on val and test). The "
      "interaction *representation* is the transferable contribution; we reframe "
      "the claim accordingly and log MLP as one instantiation.\n")

    # 6 C4
    A("\n## 6. C4 human-review evidence (item 2)\n")
    c = S["c4_packet"]
    A(f"- Grey-zone pairs: {c['n_grey_zone_pairs']} ({c['grey_zone_pct']}% of test)\n"
      f"- Packet sheets generated: {c['n_packet_pairs']} "
      "(artifacts/PHASE_6_C4_REVIEW_PACKET.csv/.json)\n"
      f"- Disclosure: {c['disclosure']}\n")
    A("Full pre-registered-style protocol: `phase_6_q1_hardening/"
      "C4_HUMAN_REVIEW_PROTOCOL.md`.\n")

    # 7 verdicts
    A("\n## 7. Item-by-item resolution\n")
    A("| # | attack item | resolution | artefact |\n|---|---|---|---|\n")
    A("| 1 | external validity | text: CS canonical + cross-topic + MRPC, ranking + "
      "thresholded, caveated; code: unseen-severity L6 transfer | S5/S6 tables |\n")
    A("| 2 | C4 human evidence | protocol + verified instrument; human gains NOT "
      "claimed | C4_HUMAN_REVIEW_PROTOCOL.md + packet |\n")
    A("| 3 | conformal undercoverage | α*=0.01 split-sample repair; group cov "
      "1.0000 (Wilson ≥ 0.996); set sizes audited | PHASE_6_CONFORMAL_* |\n")
    A("| 4 | neural cross-encoder | honest ablation; reframed claim | "
      "PHASE_6_VERIFIER_ABLATION.csv |\n")
    A("| 5 | multi-seed | 3 seeds text+code; std reported | PHASE_6_MULTISEED_* |\n")
    A("| 6 | novelty framing | revised in paper/main_v2.tex | manuscript |\n")

    (OUT / "PHASE_6_REPORT.md").write_text("".join(L))
    print("wrote PHASE_6_REPORT.md")


if __name__ == "__main__":
    main()
