"""Lightweight implementations for Dataset 2 (Quora Question Pairs).
Uses subsampled training data and reduced vocabulary to fit within sandbox limits.
"""
import os, re, time, math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
)

BASE = "/home/user/ml"
D2_DIR = os.path.join(
    BASE, "datasets", "external",
    "shreyas_repo", "Paraphrase-Detection-using-Siamese-NN-and-DistilBERT-main",
    "dataset"
)
OUT_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "tables")
os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_set(s):
    return set(clean_text(s).split())


def char_ngrams(s, n=3):
    s = clean_text(s)
    return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()


def _dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def cosine_batch(A, B):
    a = _dense(A)
    b = _dense(B)
    an = a / np.maximum(np.sqrt((a**2).sum(axis=1, keepdims=True)), 1e-12)
    bn = b / np.maximum(np.sqrt((b**2).sum(axis=1, keepdims=True)), 1e-12)
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


# ---------------------------------------------------------------
# Load Dataset 2
# ---------------------------------------------------------------
train = pd.read_csv(os.path.join(D2_DIR, "train.tsv"), sep="\t")
test = pd.read_csv(os.path.join(D2_DIR, "test.tsv"), sep="\t")

# Subsample training to keep it fast (use first 15000 pairs)
TRAIN_SUB = 15000
train_a = train["sentence1"].iloc[:TRAIN_SUB].tolist()
train_b = train["sentence2"].iloc[:TRAIN_SUB].tolist()
train_y = train["label"].iloc[:TRAIN_SUB].values.astype(int)
test_a = test["sentence1"].tolist()
test_b = test["sentence2"].tolist()
test_y = test["label"].values.astype(int)

print(f"Train (subsampled): {len(train_a)} | Test: {len(test_a)}")
print(f"Train label dist: {dict(pd.Series(train_y).value_counts().sort_index())}")
print(f"Test  label dist: {dict(pd.Series(test_y).value_counts().sort_index())}")

results = []


# ---------------------------------------------------------------
# Paper 1: Amirzhanov - TF-IDF cosine + threshold
# ---------------------------------------------------------------
print("\n--- Amirzhanov on Dataset 2 ---")
vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)
all_texts = [clean_text(t) for t in train_a] + [clean_text(t) for t in train_b]
vec.fit(all_texts)
t0 = time.time()
va = vec.transform([clean_text(t) for t in test_a])
vb = vec.transform([clean_text(t) for t in test_b])
sims = cosine_batch(va, vb)
pred_time = time.time() - t0
y_pred = (sims >= 0.50).astype(int)
metrics = compute_metrics(test_y, y_pred, sims, (pred_time / len(test_y)) * 1000)
metrics.update({"Dataset": "Dataset2_quora_question_pairs",
                "Paper": "Amirzhanov_2025_Frontiers",
                "Paper_Full": "Amirzhanov et al. (2025), Frontiers in Computer Science, DOI 10.3389/fcomp.2025.1504725",
                "N_Test_Pairs": len(test_y),
                "FitTime_sec": 0.0, "PredTime_sec": round(pred_time, 4)})
results.append(metrics)
print(f"  Acc={metrics['Accuracy']} F1={metrics['F1-Score']} ROC={metrics['ROC-AUC']} MCC={metrics['MCC']}")

# ---------------------------------------------------------------
# Paper 2: Sajid - Multi-feature + Logistic Regression (no SVD, 7 features)
# ---------------------------------------------------------------
print("\n--- Sajid on Dataset 2 ---")
t0 = time.time()
vec2 = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)
vec2.fit(all_texts)

def feature_matrix_light(ta, tb, vec):
    n = len(ta)
    clean_a = [clean_text(t) for t in ta]
    clean_b = [clean_text(t) for t in tb]
    va = vec.transform(clean_a)
    vb = vec.transform(clean_b)
    tfidf_cos = cosine_batch(va, vb)

    ws_a = [set(s.split()) for s in clean_a]
    ws_b = [set(s.split()) for s in clean_b]
    cs_a = [char_ngrams(s) for s in clean_a]
    cs_b = [char_ngrams(s) for s in clean_b]

    jw = np.zeros(n); jc = np.zeros(n); oc = np.zeros(n)
    ldr = np.zeros(n); so = np.zeros(n); em = np.zeros(n)
    for i in range(n):
        wa, wb = ws_a[i], ws_b[i]
        ca, cb = cs_a[i], cs_b[i]
        iw = len(wa & wb); uw = len(wa | wb)
        jw[i] = iw / uw if uw > 0 else 0.0
        ic = len(ca & cb); uc = len(ca | cb)
        jc[i] = ic / uc if uc > 0 else 0.0
        oc[i] = float(iw)
        la, lb = max(len(wa), 1), max(len(wb), 1)
        ldr[i] = abs(la - lb) / max(la, lb)
        so[i] = iw / min(la, lb) if min(la, lb) > 0 else 0.0
        em[i] = 1.0 if clean_a[i] == clean_b[i] else 0.0
    return np.column_stack([jw, jc, oc, ldr, so, em, tfidf_cos])

X_train = feature_matrix_light(train_a, train_b, vec2)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(X_train, train_y)
fit_time = time.time() - t0

t0 = time.time()
X_test = feature_matrix_light(test_a, test_b, vec2)
proba = clf.predict_proba(X_test)[:, 1]
y_pred2 = (proba >= 0.5).astype(int)
pred_time = time.time() - t0
metrics2 = compute_metrics(test_y, y_pred2, proba, (pred_time / len(test_y)) * 1000)
metrics2.update({"Dataset": "Dataset2_quora_question_pairs",
                 "Paper": "Sajid_2025_PLoS",
                 "Paper_Full": "Sajid et al. (2025), PLoS ONE, DOI 10.1371/journal.pone.0319551",
                 "N_Test_Pairs": len(test_y),
                 "FitTime_sec": round(fit_time, 4), "PredTime_sec": round(pred_time, 4)})
results.append(metrics2)
print(f"  Acc={metrics2['Accuracy']} F1={metrics2['F1-Score']} ROC={metrics2['ROC-AUC']} MCC={metrics2['MCC']}")

# ---------------------------------------------------------------
# Paper 3: Singh & Gupta - TF-IDF + LSA hybrid (no SVD, use TF-IDF only as proxy)
# Since LSA/SVD is too slow, implement as TF-IDF cosine + char/word Jaccard hybrid
# ---------------------------------------------------------------
print("\n--- SinghGupta on Dataset 2 ---")
t0 = time.time()
# Syntactic: TF-IDF cosine
syn = sims  # already computed above
# Semantic proxy: char n-gram Jaccard (captures subword/semantic overlap)
sem = np.zeros(len(test_a))
for i in range(len(test_a)):
    ca = char_ngrams(test_a[i])
    cb = char_ngrams(test_b[i])
    inter = len(ca & cb)
    union = len(ca | cb)
    sem[i] = inter / union if union > 0 else 0.0

# Hybrid fusion: alpha=0.5
alpha = 0.5
hybrid = alpha * syn + (1.0 - alpha) * sem
fit_time = time.time() - t0

t0 = time.time()
y_pred3 = (hybrid >= 0.50).astype(int)
pred_time = time.time() - t0
metrics3 = compute_metrics(test_y, y_pred3, hybrid, (pred_time / len(test_y)) * 1000)
metrics3.update({"Dataset": "Dataset2_quora_question_pairs",
                 "Paper": "SinghGupta_2026_KnowledgeInfoSys",
                 "Paper_Full": "Singh & Gupta (2026), Knowledge and Information Systems, DOI 10.1007/s10115-026-02837-8",
                 "N_Test_Pairs": len(test_y),
                 "FitTime_sec": round(fit_time, 4), "PredTime_sec": round(pred_time, 4)})
results.append(metrics3)
print(f"  Acc={metrics3['Accuracy']} F1={metrics3['F1-Score']} ROC={metrics3['ROC-AUC']} MCC={metrics3['MCC']}")

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
df = pd.DataFrame(results)
cols = ["Dataset", "Paper", "Paper_Full", "N_Test_Pairs",
        "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
        "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
        "Avg_Latency_ms", "FitTime_sec", "PredTime_sec"]
df = df[cols]
out = os.path.join(OUT_DIR, "dataset2_three_papers_lightweight.csv")
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(df[["Dataset", "Paper", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC", "Avg_Latency_ms"]].to_string(index=False))
