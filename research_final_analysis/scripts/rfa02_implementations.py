"""RFA-3/4/5: three-paper methodology implementations + proposed framework.

Baselines are the FROZEN phase-5 implementations (same seeds, same protocol):
- Text: TextBaselineSuite (B-T1 TF-IDF / B-T2 MiniLM / B-T3 flat hybrid LR).
- Code: CodeBaselineSuiteLOGO (B-C1/B-C2 score baselines, B-C3 SVM, B-C4 RF)
  + evaluate_esplag_baseline_logo (B-C5 ES-Plag).
Continuous score vectors are persisted for ROC/PR curves; for the SVM the
decision function is used — we VERIFY the hardened threshold predictions equal
the frozen suite's predictions on every OOF row (integrity check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC

from rfa_common import (RFA, P5, SEED, full_metrics, load_code_raw,
                        load_features, wc, wj)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import (
    CODE_FEATURES, CODE_RF_FEATURES, CODE_SVM_FEATURES, TEXT_MULTIVIEW_FEATURES,
    CodeBaselineSuiteLOGO, TextBaselineSuite, evaluate_esplag_baseline_logo,
)

OUT = RFA / "04_EXPERIMENTAL_RESULTS"
IMPL = RFA / "03_IMPLEMENTATIONS"


def main():
    F = load_features()
    raw_code = load_code_raw()
    pred_dir = OUT / "PREDICTIONS"
    rows = []

    # ------------------------------------------------ text baselines (frozen suite)
    tbs = TextBaselineSuite(random_state=SEED).fit(F["text_train"], F["text_val"])
    tres = tbs.evaluate_test(F["text_test"])
    y_te_t = F["text_test"]["ground_truth_label"].values.astype(int)
    for name, r in tres.items():
        m = full_metrics(y_te_t, r["predictions"], r["prob_scores"])
        rows.append({"Modality": "Text", "System": name,
                     "Type": "Baseline (paper implementation)", **m})
        slug = name.split()[0]
        pd.DataFrame({"ground_truth": y_te_t, "score": r["prob_scores"],
                      "pred": r["predictions"]}).to_csv(
            pred_dir / f"{slug}_text_predictions.csv", index=False)

    # ------------------------------------------------ proposed text (phase-5 stored)
    p5t = pd.read_csv(P5 / "PHASE_5_PREDICTIONS_TEXT_TEST.csv")
    rows.append({"Modality": "Text", "System": "PROPOSED (C1-C4, heavy S2, tuned OP)",
                 "Type": "Proposed",
                 **full_metrics(p5t["ground_truth_label"].values,
                                p5t["final_prediction"].values,
                                p5t["calibrated_prob"].values)})
    p5t.to_csv(pred_dir / "PROPOSED_text_test_predictions.csv", index=False)
    rows.append({"Modality": "Text", "System": "PROPOSED heavy verifier (beta=1.0)",
                 "Type": "Proposed (beta=1 config)",
                 **full_metrics(p5t["ground_truth_label"].values,
                                (p5t["stage2_score"].values >= 0.5).astype(int),
                                p5t["stage2_score"].values)})

    # ------------------------------------------------ code baselines (frozen suite)
    C_a = F["code_a"]
    y_c = C_a["ground_truth_label"].values.astype(int)
    cases = raw_code["policy_a"]["case"]
    cbs = CodeBaselineSuiteLOGO(random_state=SEED).evaluate(C_a, cases)
    # B-C5 ES-Plag (frozen helper)
    esp = evaluate_esplag_baseline_logo(raw_code["policy_a"], cases, random_state=SEED)
    esp_pred = esp.get("predictions", esp.get("oof_pred"))

    # continuous-score shadow refits for ROC curves (same seeds; predictions must match)
    svm_scores = np.zeros(len(y_c))
    rf_scores = np.zeros(len(y_c))
    svm_hard = np.zeros(len(y_c), dtype=int)
    rf_hard = np.zeros(len(y_c), dtype=int)
    X_svm = C_a[CODE_SVM_FEATURES].values
    X_rf = C_a[CODE_RF_FEATURES].values
    for tr, tei in LeaveOneGroupOut().split(X_svm, y_c, groups=cases.values):
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
        svm.fit(X_svm[tr], y_c[tr])
        svm_scores[tei] = svm.decision_function(X_svm[tei])
        svm_hard[tei] = svm.predict(X_svm[tei])
        rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=SEED)
        rf.fit(X_rf[tr], y_c[tr])
        rf_scores[tei] = rf.predict_proba(X_rf[tei])[:, 1]
        rf_hard[tei] = rf.predict(X_rf[tei])
    ok_svm = bool((svm_hard == cbs["B-C3 Gandhi RBF-SVM (2024)"]["predictions"]).all())
    ok_rf = bool((rf_hard == cbs["B-C4 Mehsen RandomForest (2024)"]["predictions"]).all())

    score_shadow = {"B-C1 Token Jaccard": C_a["code_token_jaccard"].values,
                    "B-C2 RKRGST String Tiling": C_a["code_rkrgst_sim"].values,
                    "B-C3 Gandhi RBF-SVM (2024)": svm_scores,
                    "B-C4 Mehsen RandomForest (2024)": rf_scores}
    for name, r in cbs.items():
        sc = score_shadow[name]
        m = full_metrics(y_c, r["predictions"], sc)
        rows.append({"Modality": "Code (PolicyA LOGO)", "System": name,
                     "Type": "Baseline (paper implementation)", **m})
        pd.DataFrame({"ground_truth": y_c, "score": sc, "pred": r["predictions"]}).to_csv(
            pred_dir / f"{name.split()[0]}_code_predictions.csv", index=False)

    if esp_pred is not None:
        esp_score = esp.get("scores", esp_pred)
        rows.append({"Modality": "Code (PolicyA LOGO)",
                     "System": "B-C5 ES-Plag (2019, recall-first filter)",
                     "Type": "Baseline (paper implementation)",
                     **full_metrics(y_c, esp_pred,
                                    esp_score if len(np.unique(esp_score)) > 2 else None)})
        pd.DataFrame({"ground_truth": y_c, "score": esp_score, "pred": esp_pred}).to_csv(
            pred_dir / "B-C5_code_predictions.csv", index=False)

    # ------------------------------------------------ proposed code (phase-5 stored OOF)
    p5c = pd.read_csv(P5 / "PHASE_5_PREDICTIONS_CODE_OOF.csv")
    rows.append({"Modality": "Code (PolicyA LOGO)", "System": "PROPOSED (C1-C4, heavy S2, tuned OP)",
                 "Type": "Proposed",
                 **full_metrics(p5c["ground_truth_label"].values,
                                p5c["final_prediction"].values,
                                p5c["calibrated_prob"].values)})
    rows.append({"Modality": "Code (PolicyA LOGO)",
                 "System": "PROPOSED heavy verifier (beta=1.0)",
                 "Type": "Proposed (beta=1 config)",
                 **full_metrics(p5c["ground_truth_label"].values,
                                (p5c["stage2_score"].values >= 0.5).astype(int),
                                p5c["stage2_score"].values)})
    p5c.to_csv(pred_dir / "PROPOSED_code_oof_predictions.csv", index=False)

    cm = pd.DataFrame(rows)
    wc(cm, OUT / "CLASSIFICATION_METRICS.csv")
    wj({"score_shadow_prediction_match": {"B-C3_svm": ok_svm, "B-C4_rf": ok_rf},
        "note": ("Frozen phase-5 suites are authoritative; continuous shadow scores are "
                 "identical-seed refits used only for ROC/PR curves and must reproduce the "
                 "frozen hard predictions on every OOF row.")},
       IMPL / "IMPLEMENTATION_INDEX.json")
    print("integrity: svm match =", ok_svm, "| rf match =", ok_rf)
    print(cm[["Modality", "System", "accuracy", "f1", "mcc", "roc_auc"]].to_string())


if __name__ == "__main__":
    main()
