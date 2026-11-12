"""Run all three papers on Dataset 1 (repo text benchmark) and save results."""
import os, re, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix)

BASE = "/home/user/ml"
D1_PATH = os.path.join(BASE, "datasets", "processed", "text_plagiarism_benchmark.csv")
OUT_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(D1_PATH)
y = df["is_plagiarized"].values.astype(int)
train_mask = df["split"] == "val"
test_mask = df["split"] == "test"
train_a = df.loc[train_mask, "reference_text"].tolist()
train_b = df.loc[train_mask, "suspicious_text"].tolist()
train_y = y[train_mask]
test_a = df.loc[test_mask, "reference_text"].tolist()
test_b = df.loc[test_mask, "suspicious_text"].tolist()
test_y = y[test_mask]

def clean(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def wset(s):
    return set(clean(s).split())

def cng(s):
    s = clean(s)
    return set(s[i:i+3] for i in range(len(s)-2)) if len(s) >= 3 else set()

def _dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)

def cosine_batch(A, B):
    ad = _dense(A); bd = _dense(B)
    an = ad / np.maximum(np.sqrt((ad**2).sum(1, keepdims=True)), 1e-12)
    bn = bd / np.maximum(np.sqrt((bd**2).sum(1, keepdims=True)), 1e-12)
    return np.diag(an @ bn.T)

def compute_metrics(y_true, y_pred, y_score, latency_ms):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "Specificity": round(float(tn / (tn + fp)), 6) if (tn + fp) > 0 else 0.0,
        "ROC-AUC": round(float(roc_auc_score(y_true, y_score)), 6),
        "PR-AUC": round(float(average_precision_score(y_true, y_score)), 6),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Avg_Latency_ms": round(float(latency_ms), 6),
    }

results = []
N_TEST = len(test_y)
PAPER_FULL = {
    "Amirzhanov_2025_Frontiers": "Amirzhanov et al. (2025), Frontiers in Computer Science, DOI 10.3389/fcomp.2025.1504725",
    "Sajid_2025_PLoS": "Sajid et al. (2025), PLoS ONE, DOI 10.1371/journal.pone.0319551",
    "SinghGupta_2026_KnowledgeInfoSys": "Singh & Gupta (2026), Knowledge and Information Systems, DOI 10.1007/s10115-026-02837-8",
}

# ---------------- Paper 1: Amirzhanov ----------------
print("=== Amirzhanov ===")
t0 = time.time()
vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)
all_texts = [clean(t) for t in train_a] + [clean(t) for t in train_b]
vec.fit(all_texts)
fit_t = time.time() - t0
t0 = time.time()
va = vec.transform([clean(t) for t in test_a])
vb = vec.transform([clean(t) for t in test_b])
sims = cosine_batch(va, vb)
pred_t = time.time() - t0
yp = (sims >= 0.50).astype(int)
m = compute_metrics(test_y, yp, sims, (pred_t / N_TEST) * 1000)
m.update({"Dataset": "Dataset1_repo_text_benchmark", "Paper": "Amirzhanov_2025_Frontiers",
          "Paper_Full": PAPER_FULL["Amirzhanov_2025_Frontiers"],
          "N_Test_Pairs": N_TEST, "FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4)})
results.append(m)
print(f"  Acc={m['Accuracy']} F1={m['F1-Score']} ROC={m['ROC-AUC']} MCC={m['MCC']}")

# ---------------- Paper 2: Sajid ----------------
print("=== Sajid ===")
t0 = time.time()
vec2 = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)
vec2.fit(all_texts)

def feat_mat(ta, tb, vec):
    n = len(ta)
    ca = [clean(t) for t in ta]; cb = [clean(t) for t in tb]
    va = vec.transform(ca); vb = vec.transform(cb)
    tfc = cosine_batch(va, vb)
    ws_a = [wset(t) for t in ca]; ws_b = [wset(t) for t in cb]
    cs_a = [cng(t) for t in ca]; cs_b = [cng(t) for t in cb]
    jw = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
    jc = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
    oc = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
    ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) / max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
    so = np.array([len(ws_a[i] & ws_b[i]) / min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
    em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
    return np.column_stack([jw, jc, oc, ldr, so, em, tfc])

Xtr = feat_mat(train_a, train_b, vec2)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(Xtr, train_y)
fit_t = time.time() - t0
t0 = time.time()
Xte = feat_mat(test_a, test_b, vec2)
proba = clf.predict_proba(Xte)[:, 1]
yp = (proba >= 0.5).astype(int)
pred_t = time.time() - t0
m = compute_metrics(test_y, yp, proba, (pred_t / N_TEST) * 1000)
m.update({"Dataset": "Dataset1_repo_text_benchmark", "Paper": "Sajid_2025_PLoS",
          "Paper_Full": PAPER_FULL["Sajid_2025_PLoS"],
          "N_Test_Pairs": N_TEST, "FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4)})
results.append(m)
print(f"  Acc={m['Accuracy']} F1={m['F1-Score']} ROC={m['ROC-AUC']} MCC={m['MCC']}")

# ---------------- Paper 3: Singh & Gupta ----------------
print("=== Singh & Gupta ===")
t0 = time.time()
syn = cosine_batch(vec.transform([clean(t) for t in test_a]),
                   vec.transform([clean(t) for t in test_b]))
sem = np.zeros(N_TEST)
for i in range(N_TEST):
    ca = cng(test_a[i]); cb = cng(test_b[i])
    inter = len(ca & cb); union = len(ca | cb)
    sem[i] = inter / union if union > 0 else 0.0
fit_t = time.time() - t0
alpha = 0.5
hybrid = alpha * syn + (1.0 - alpha) * sem
t0 = time.time()
yp = (hybrid >= 0.50).astype(int)
pred_t = time.time() - t0
m = compute_metrics(test_y, yp, hybrid, (pred_t / N_TEST) * 1000)
m.update({"Dataset": "Dataset1_repo_text_benchmark", "Paper": "SinghGupta_2026_KnowledgeInfoSys",
          "Paper_Full": PAPER_FULL["SinghGupta_2026_KnowledgeInfoSys"],
          "N_Test_Pairs": N_TEST, "FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4)})
results.append(m)
print(f"  Acc={m['Accuracy']} F1={m['F1-Score']} ROC={m['ROC-AUC']} MCC={m['MCC']}")

# ---------------- Save ----------------
df2 = pd.DataFrame(results)
cols = ["Dataset", "Paper", "Paper_Full", "N_Test_Pairs",
        "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
        "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
        "Avg_Latency_ms", "FitTime_sec", "PredTime_sec"]
df2 = df2[cols]
out = os.path.join(OUT_DIR, "dataset1_three_papers.csv")
df2.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(df2[["Dataset", "Paper", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC", "Avg_Latency_ms"]].to_string(index=False))
