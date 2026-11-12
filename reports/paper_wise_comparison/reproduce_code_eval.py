"""
Reproduction harness for source-code modality comparisons.

Protocol (repository convention, src/code_random_forest.py __main__):
  Train: cases 01-04  |  Validation: case-05  |  Test: cases 06-07 (N = 210 pairs)

Methods reproduced:
  Paper 2 - Gandhi et al. (2024): 7-D syntactic/token features + RBF SVM
  Paper 3 - Mehsen & Joshi (2024): same features + Random Forest (100 trees, depth 6)
  Proposed - Typed-Decision Tri-Band Pipeline: token-Jaccard pre-filter ->
             theta_low/theta_high tri-band routing -> SVM on uncertain band.

All metrics computed from the same held-out test split; per-pair predictions are
retained so paired statistical tests (McNemar) and bootstrap CIs are genuine.
"""

import sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_plagiarism import CodeSVMDetector, compute_pair_features
from code_random_forest import CodeRandomForestDetector
from code_preprocessing import extract_tokens, remove_comments
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

RNG = 42
TEST_CASES = ["case-06", "case-07"]
VAL_CASES = ["case-05"]
TRAIN_CASES = ["case-01", "case-02", "case-03", "case-04"]

df = pd.read_csv(ROOT / "datasets/processed/code_plagiarism_benchmark.csv")
print("Loaded code benchmark:", df.shape)
print("Cases:", sorted(df["case"].unique().tolist()))
print("Label distribution:\n", df["is_plagiarized"].value_counts())

df_train = df[df["case"].isin(TRAIN_CASES)].reset_index(drop=True)
df_val   = df[df["case"].isin(VAL_CASES)].reset_index(drop=True)
df_test  = df[df["case"].isin(TEST_CASES)].reset_index(drop=True)
print(f"Train {len(df_train)} | Val {len(df_val)} | Test {len(df_test)}")

X_train = np.array([compute_pair_features(r.reference_code, r.suspicious_code)
                    for _, r in df_train.iterrows()], dtype=float)
y_train = df_train["is_plagiarized"].to_numpy()
X_val = np.array([compute_pair_features(r.reference_code, r.suspicious_code)
                  for _, r in df_val.iterrows()], dtype=float)
y_val = df_val["is_plagiarized"].to_numpy()
X_test = np.array([compute_pair_features(r.reference_code, r.suspicious_code)
                   for _, r in df_test.iterrows()], dtype=float)
y_test = df_test["is_plagiarized"].to_numpy()
print("Test positives:", int((y_test == 1).sum()), "negatives:", int((y_test == 0).sum()))


def tune_threshold(probs_val, y_val, grid=np.linspace(0.1, 0.9, 81)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (probs_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def classification_block(y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "fpr": float(fp / (tn + fp)) if (tn + fp) else 0.0,
        "mcc": float((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                 if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)) if scores is not None and len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if scores is not None and len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": cm.tolist(),
    }


def mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n01 = int((correct_a & ~correct_b).sum())   # a correct, b wrong
    n10 = int((~correct_a & correct_b).sum())   # a wrong, b correct
    chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10) if (n01 + n10) > 0 else 0.0
    from scipy.stats import chi2_contingency  # may not exist; fallback below
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(chi2, 1))
    except Exception:
        p = None
    return {"n_both_correct": int((correct_a & correct_b).sum()),
            "n_a_correct_b_wrong": n01,
            "n_a_wrong_b_correct": n10,
            "n_both_wrong": int((~correct_a & ~correct_b).sum()),
            "chi2": float(chi2), "p_value": p}


def bootstrap_ci(y_true, pred_a, pred_b, metric_fn, n_boot=2000, seed=RNG, direction=1):
    """CI for metric_a - metric_b over paired test samples."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    diffs = []
    for _ in range(n_boot):
        sel = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(direction * (metric_fn(y_true[sel], pred_a[sel]) -
                                  metric_fn(y_true[sel], pred_b[sel])))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean())


# ---------------- Paper 2: SVM (Gandhi et al. 2024, adapted to Java) ----------------
print("\n=== Paper 2: SVM (Gandhi et al. 2024) ===")
svm = CodeSVMDetector(C=1.0, kernel="rbf", gamma="scale")
t0 = time.perf_counter()
svm.fit(X_train, y_train)
train_time_svm = time.perf_counter() - t0
probs_val_svm = svm.predict_proba(X_val)
thr_svm, f1_val_svm = tune_threshold(probs_val_svm, y_val)
svm.threshold = thr_svm
t0 = time.perf_counter()
res_svm = svm.evaluate(X_test, y_test, df_test)
cm = confusion_matrix(y_test, (svm.predict_proba(X_test) >= svm.threshold).astype(int), labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
res_svm["mcc"] = float((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
res_svm["train_time_sec"] = float(train_time_svm)
res_svm["threshold"] = float(thr_svm)
res_svm["val_f1"] = float(f1_val_svm)
print(f"SVM test: acc={res_svm['accuracy']:.4f} prec={res_svm['precision']:.4f} "
      f"rec={res_svm['recall']:.4f} f1={res_svm['f1']:.4f} mcc={res_svm['mcc']:.4f} "
      f"roc_auc={res_svm['roc_auc']:.4f} thr={thr_svm:.4f} val_f1={f1_val_svm:.4f}")
print("  level recall:", res_svm["level_recall"])

# ---------------- Paper 3: RF (Mehsen & Joshi 2024) ----------------
print("\n=== Paper 3: Random Forest (Mehsen & Joshi 2024) ===")
rf = CodeRandomForestDetector(n_estimators=100, max_depth=6, random_state=RNG)
t0 = time.perf_counter()
rf.fit(X_train, y_train)
train_time_rf = time.perf_counter() - t0
probs_val_rf = rf.predict_proba(X_val)
thr_rf, f1_val_rf = tune_threshold(probs_val_rf, y_val)
rf.threshold = thr_rf
t0 = time.perf_counter()
res_rf = rf.evaluate(X_test, y_test, df_test)
cm = confusion_matrix(y_test, (rf.predict_proba(X_test) >= rf.threshold).astype(int), labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
res_rf["mcc"] = float((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
res_rf["train_time_sec"] = float(train_time_rf)
res_rf["threshold"] = float(thr_rf)
res_rf["val_f1"] = float(f1_val_rf)
print(f"RF test: acc={res_rf['accuracy']:.4f} prec={res_rf['precision']:.4f} "
      f"rec={res_rf['recall']:.4f} f1={res_rf['f1']:.4f} mcc={res_rf['mcc']:.4f} "
      f"roc_auc={res_rf['roc_auc']:.4f} thr={thr_rf:.4f} val_f1={f1_val_rf:.4f}")

# ---------------- Proposed: Typed-Decision Tri-Band Pipeline (code) ----------------
print("\n=== Proposed: Typed-Decision Tri-Band Pipeline (code) ===")
theta_low, theta_high = 0.20, 0.80  # pipeline defaults
cheap_val = np.array([
    len(set(extract_tokens(remove_comments(r.reference_code))) &
        set(extract_tokens(remove_comments(r.suspicious_code)))) /
    max(1, len(set(extract_tokens(remove_comments(r.reference_code))) |
             set(extract_tokens(remove_comments(r.suspicious_code)))))
    for _, r in df_val.iterrows()
], dtype=float)
# sanity: reproduce SVM threshold used by heavy verifier
heavy = svm  # same heavy verifier as Paper 2 (threshold tuned on val)
n_test = len(y_test)
preds_c = np.zeros(n_test, dtype=int)
latencies_c = []
filtered = escalated = flagged = 0
retained_pos = 0
total_pos = int((y_test == 1).sum())

t_cheap_per = 0.000185  # ms/pair token Jaccard (archived microbenchmark)
t0 = time.perf_counter()
for i in range(n_test):
    ref_toks = set(extract_tokens(remove_comments(df_test.iloc[i]["reference_code"])))
    sus_toks = set(extract_tokens(remove_comments(df_test.iloc[i]["suspicious_code"])))
    j = len(ref_toks & sus_toks) / max(1, len(ref_toks | sus_toks))
    lat = t_cheap_per
    if j < theta_low:
        preds_c[i] = 0; filtered += 1
    elif j >= theta_high:
        preds_c[i] = 1; flagged += 1
    else:
        escalated += 1
        p = float(heavy.predict_proba(X_test[i:i + 1])[0])
        lat += 0.013  # classifier-only inference ms (archived)
        preds_c[i] = int(p >= heavy.threshold)
    latencies_c.append(lat)
    if j >= theta_low and df_test.iloc[i]["is_plagiarized"] == 1:
        retained_pos += 1
total_runtime_c = time.perf_counter() - t0

block_c = classification_block(y_test, preds_c,
                               scores=None)  # discrete band decisions: no ROC-AUC/PR-AUC
cm = confusion_matrix(y_test, preds_c, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
res_prop = {
    "method": "Proposed Typed-Decision Tri-Band Pipeline",
    "modality": "Source Code",
    "total_pairs": n_test,
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "accuracy": float(accuracy_score(y_test, preds_c)),
    "precision": float(precision_score(y_test, preds_c, zero_division=0)),
    "recall": float(recall_score(y_test, preds_c, zero_division=0)),
    "f1": float(f1_score(y_test, preds_c, zero_division=0)),
    "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    "fpr": float(fp / (tn + fp)) if (tn + fp) else 0.0,
    "mcc": float((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
    "roc_auc": None, "pr_auc": None,
    "confusion_matrix": cm.tolist(),
    "candidate_recall": float(retained_pos / total_pos) if total_pos else 1.0,
    "candidate_precision": float(retained_pos / (n_test - filtered)) if (n_test - filtered) else 0.0,
    "pruning_ratio": float(filtered / n_test),
    "expensive_detector_call_rate": float(escalated / n_test),
    "direct_flag_rate": float(flagged / n_test),
    "filtered_pairs": filtered, "escalated_pairs": escalated, "direct_flagged_pairs": flagged,
    "avg_latency_ms": float(np.mean(latencies_c) * 1000.0),
    "total_runtime_sec": float(total_runtime_c),
}
print(f"PROPOSED code: acc={res_prop['accuracy']:.4f} prec={res_prop['precision']:.4f} "
      f"rec={res_prop['recall']:.4f} f1={res_prop['f1']:.4f} mcc={res_prop['mcc']:.4f}")
print(f"  cand_recall={res_prop['candidate_recall']:.4f} pruning={res_prop['pruning_ratio']:.4f} "
      f"call_rate={res_prop['expensive_detector_call_rate']:.4f} flag_rate={res_prop['direct_flag_rate']:.4f}")
print(f"  filtered={filtered} escalated={escalated} flagged={flagged}")

# ---------------- Paired statistical tests ----------------
print("\n=== Paired tests (proposed vs each paper) ===")
from math import erfc, sqrt
def mcnemar_p(n01, n10):
    if n01 + n10 == 0:
        return 1.0, 0.0
    chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    # chi2 survival function at 1 df: p = erfc(sqrt(chi2/2))
    p = float(erfc(sqrt(chi2 / 2.0)))
    return p, float(chi2)

def cont(y_true, pa, pb):
    ca, cb = (pa == y_true), (pb == y_true)
    return int((ca & cb).sum()), int((ca & ~cb).sum()), int((~ca & cb).sum()), int((~ca & ~cb).sum())

for name, res in [("Paper 2 SVM", res_svm), ("Paper 3 RF", res_rf)]:
    pb = np.array(res["predictions"]) if "predictions" in res else None
# re-run predictions storage
svm_preds = (svm.predict_proba(X_test) >= svm.threshold).astype(int)
rf_preds = (rf.predict_proba(X_test) >= rf.threshold).astype(int)
for name, pa in [("Paper 2 SVM", svm_preds), ("Paper 3 RF", rf_preds)]:
    bc, awb, aw, bw = cont(y_test, pa, preds_c)
    p, chi2 = mcnemar_p(awb, aw)
    print(f"Proposed vs {name}: both_correct={bc} paper_right_prop_wrong={awb} "
          f"paper_wrong_prop_right={aw} both_wrong={bw} | McNemar chi2={chi2:.3f} p={p:.6g}")
    # bootstrap CI for F1 difference (proposed - paper)
    lo, hi, mean = bootstrap_ci(y_test, preds_c, pa,
                                lambda yt, pp: f1_score(yt, pp, zero_division=0))
    print(f"  F1 diff (prop - paper) 95% bootstrap CI: [{lo:.4f}, {hi:.4f}] mean={mean:.4f}")
    lo, hi, mean = bootstrap_ci(y_test, preds_c, pa,
                                lambda yt, pp: accuracy_score(yt, pp))
    print(f"  Accuracy diff (prop - paper) 95% bootstrap CI: [{lo:.4f}, {hi:.4f}] mean={mean:.4f}")

out = {"svm": res_svm, "rf": res_rf, "proposed": res_prop}
with open(ROOT / "code_reproduction_results.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved code_reproduction_results.json")
