"""
Implement the three journal papers on two text plagiarism datasets.

Papers (all peer-reviewed journals, 2024-2026):
  P1 - Amirzhanov et al. (2025), Frontiers in Computer Science
       Systematic survey of plagiarism types and algorithms.
       Implementation: string-matching / lexical tradition (TF-IDF cosine + threshold).
  P2 - Sajid et al. (2025), PLoS ONE
       Comparative analysis of feature extraction methods.
       Implementation: multi-feature + classical ML (Logistic Regression).
  P3 - Singh & Gupta (2026), Knowledge and Information Systems
       Extrinsic semantic-syntactic hybrid detection.
       Implementation: TF-IDF (syntactic) + LSA/TruncatedSVD (semantic) hybrid fusion.

Datasets:
  D1 - text_plagiarism_benchmark.csv (repo), 19,666 rows, binary target = is_plagiarized
  D2 - Quora Question Pairs (GitHub shreyas1104), 49,401 train / 8,000 test rows

Output: CSV tables with quantitative metrics for each paper x dataset combination.
"""

import os
import re
import time
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/home/user/ml"
D1_PATH = os.path.join(BASE, "datasets", "processed", "text_plagiarism_benchmark.csv")
D2_DIR = os.path.join(
    BASE, "datasets", "external",
    "shreyas_repo", "Paraphrase-Detection-using-Siamese-NN-and-DistilBERT-main",
    "dataset"
)
OUT_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_set(s):
    return set(clean_text(s).split())


# ---------------------------------------------------------------------------
# Dense helper
# ---------------------------------------------------------------------------
def _dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _row_normalize(mat):
    d = _dense(mat)
    norms = np.sqrt((d ** 2).sum(axis=1, keepdims=True))
    return d / np.maximum(norms, 1e-12)


def cosine_batch(A, B):
    """Row-wise cosine between two equal-length sparse/dense matrices."""
    a = _row_normalize(A)
    b = _row_normalize(B)
    return np.diag(a @ b.T)


# ---------------------------------------------------------------------------
# Batch feature extraction (vectorized, no per-Python-loop TF-IDF)
# ---------------------------------------------------------------------------
def extract_feature_matrix(text_list_a, text_list_b, tfidf, svd):
    """Return (N, 8) feature matrix for all pairs at once."""
    n = len(text_list_a)
    clean_a = [clean_text(t) for t in text_list_a]
    clean_b = [clean_text(t) for t in text_list_b]

    # Precompute word sets and char n-gram sets
    word_sets_a = [set(s.split()) for s in clean_a]
    word_sets_b = [set(s.split()) for s in clean_b]
    char_sets_a = [
        set(s[j:j+3] for j in range(len(s) - 2)) if len(s) >= 3 else set()
        for s in clean_a
    ]
    char_sets_b = [
        set(s[j:j+3] for j in range(len(s) - 2)) if len(s) >= 3 else set()
        for s in clean_b
    ]

    # TF-IDF transforms (batched)
    va = tfidf.transform(clean_a)
    vb = tfidf.transform(clean_b)

    # Syntactic: TF-IDF cosine
    tfidf_cos = cosine_batch(va, vb)

    # LSA semantic
    va_lsa = svd.transform(va)
    vb_lsa = svd.transform(vb)
    va_lsa_n = va_lsa / np.maximum(np.linalg.norm(va_lsa, axis=1, keepdims=True), 1e-12)
    vb_lsa_n = vb_lsa / np.maximum(np.linalg.norm(vb_lsa, axis=1, keepdims=True), 1e-12)
    lsa_cos = np.diag(va_lsa_n @ vb_lsa_n.T)

    # Lexical / overlap features from precomputed sets
    jaccard_word = np.zeros(n)
    jaccard_char = np.zeros(n)
    overlap_count = np.zeros(n)
    len_diff_ratio = np.zeros(n)
    shorter_overlap = np.zeros(n)
    exact_match = np.zeros(n)

    for i in range(n):
        wa = word_sets_a[i]
        wb = word_sets_b[i]
        ca = char_sets_a[i]
        cb = char_sets_b[i]

        inter_w = len(wa & wb)
        union_w = len(wa | wb)
        jaccard_word[i] = inter_w / union_w if union_w > 0 else 0.0

        inter_c = len(ca & cb)
        union_c = len(ca | cb)
        jaccard_char[i] = inter_c / union_c if union_c > 0 else 0.0

        overlap_count[i] = float(inter_w)
        la = max(len(wa), 1)
        lb = max(len(wb), 1)
        len_diff_ratio[i] = abs(la - lb) / max(la, lb)
        shorter_overlap[i] = inter_w / min(la, lb) if min(la, lb) > 0 else 0.0
        exact_match[i] = 1.0 if clean_a[i] == clean_b[i] else 0.0

    X = np.column_stack([
        jaccard_word, jaccard_char, overlap_count, len_diff_ratio,
        shorter_overlap, exact_match, tfidf_cos, lsa_cos
    ])
    return X


# ---------------------------------------------------------------------------
# Paper 1: Amirzhanov et al. (2025) - String-matching / lexical tradition
# ---------------------------------------------------------------------------
class AmirzhanovDetector:
    """String-matching tradition: TF-IDF cosine + threshold (per Amirzhanov survey)."""

    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)

    def fit(self, texts_a, texts_b):
        all_texts = [clean_text(t) for t in texts_a] + [clean_text(t) for t in texts_b]
        self.vectorizer.fit(all_texts)

    def predict_batch(self, text_list_a, text_list_b):
        va = self.vectorizer.transform([clean_text(t) for t in text_list_a])
        vb = self.vectorizer.transform([clean_text(t) for t in text_list_b])
        sims = cosine_batch(va, vb)
        return sims


# ---------------------------------------------------------------------------
# Paper 2: Sajid et al. (2025) - Multi-feature + classical ML
# ---------------------------------------------------------------------------
class SajidDetector:
    """Multi-feature + Logistic Regression (per Sajid comparative analysis)."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)
        self.svd = TruncatedSVD(n_components=25, random_state=42)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self.fitted = False

    def fit(self, texts_a, texts_b, y):
        all_texts = [clean_text(t) for t in texts_a] + [clean_text(t) for t in texts_b]
        tfidf_mat = self.tfidf.fit_transform(all_texts)
        self.svd.fit(tfidf_mat)
        X = extract_feature_matrix(texts_a, texts_b, self.tfidf, self.svd)
        self.clf.fit(X, y)
        self.fitted = True

    def predict_batch(self, text_list_a, text_list_b, threshold=0.5):
        if not self.fitted:
            raise RuntimeError("Not fitted")
        X = extract_feature_matrix(text_list_a, text_list_b, self.tfidf, self.svd)
        proba = self.clf.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return y_pred, proba


# ---------------------------------------------------------------------------
# Paper 3: Singh & Gupta (2026) - Semantic-syntactic hybrid
# ---------------------------------------------------------------------------
class SinghGuptaDetector:
    """Semantic-syntactic hybrid: TF-IDF cosine + LSA cosine, weighted fusion."""

    def __init__(self, alpha=0.5, threshold=0.50):
        self.alpha = alpha
        self.threshold = threshold
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=5000)
        self.svd = TruncatedSVD(n_components=25, random_state=42)

    def fit(self, texts_a, texts_b):
        all_texts = [clean_text(t) for t in texts_a] + [clean_text(t) for t in texts_b]
        tfidf_mat = self.tfidf.fit_transform(all_texts)
        self.svd.fit(tfidf_mat)

    def score_batch(self, text_list_a, text_list_b):
        va = self.tfidf.transform([clean_text(t) for t in text_list_a])
        vb = self.tfidf.transform([clean_text(t) for t in text_list_b])
        syn = cosine_batch(va, vb)

        va_lsa = self.svd.transform(va)
        vb_lsa = self.svd.transform(vb)
        va_lsa = np.asarray(va_lsa)
        vb_lsa = np.asarray(vb_lsa)
        va_lsa_n = va_lsa / np.maximum(np.linalg.norm(va_lsa, axis=1, keepdims=True), 1e-12)
        vb_lsa_n = vb_lsa / np.maximum(np.linalg.norm(vb_lsa, axis=1, keepdims=True), 1e-12)
        sem = np.diag(va_lsa_n @ vb_lsa_n.T)

        hybrid = self.alpha * syn + (1.0 - self.alpha) * sem
        return hybrid, syn, sem


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_score, latency_ms):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_score)
    except Exception:
        pr = float("nan")
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "Accuracy": round(float(acc), 6),
        "Precision": round(float(prec), 6),
        "Recall": round(float(rec), 6),
        "F1-Score": round(float(f1), 6),
        "Specificity": round(float(spec), 6),
        "ROC-AUC": round(float(roc), 6) if not math.isnan(roc) else "N/A",
        "PR-AUC": round(float(pr), 6) if not math.isnan(pr) else "N/A",
        "MCC": round(float(mcc), 6),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Avg_Latency_ms": round(float(latency_ms), 6),
    }


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def run_method(detector_name, detector, train_a, train_b, train_y,
               test_a, test_b, test_y):
    t0 = time.time()
    if hasattr(detector, 'clf'):
        detector.fit(train_a, train_b, train_y)
    else:
        detector.fit(train_a, train_b)
    fit_time = time.time() - t0

    t0 = time.time()
    if detector_name == "Amirzhanov":
        sims = detector.predict_batch(test_a, test_b)
        y_pred = (sims >= detector.threshold).astype(int)
        y_score = sims
    elif detector_name == "Sajid":
        y_pred, y_score = detector.predict_batch(test_a, test_b)
    elif detector_name == "SinghGupta":
        hybrid, syn, sem = detector.score_batch(test_a, test_b)
        y_pred = (hybrid >= detector.threshold).astype(int)
        y_score = hybrid
    else:
        raise ValueError(f"Unknown detector: {detector_name}")
    pred_time = time.time() - t0

    n_pairs = len(test_y)
    avg_latency_ms = (pred_time / n_pairs) * 1000.0

    metrics = compute_metrics(test_y, y_pred, y_score, avg_latency_ms)
    metrics["FitTime_sec"] = round(fit_time, 4)
    metrics["PredTime_sec"] = round(pred_time, 4)
    metrics["NPairs"] = int(n_pairs)
    return metrics, y_pred, y_score


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
def load_dataset1():
    df = pd.read_csv(D1_PATH)
    y = df["is_plagiarized"].values.astype(int)  # 1 = plagiarized, 0 = not
    train_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    train_a = df.loc[train_mask, "reference_text"].tolist()
    train_b = df.loc[train_mask, "suspicious_text"].tolist()
    train_y = y[train_mask]
    test_a = df.loc[test_mask, "reference_text"].tolist()
    test_b = df.loc[test_mask, "suspicious_text"].tolist()
    test_y = y[test_mask]
    return (train_a, train_b, train_y), (test_a, test_b, test_y)


def load_dataset2():
    train = pd.read_csv(os.path.join(D2_DIR, "train.tsv"), sep="\t")
    test = pd.read_csv(os.path.join(D2_DIR, "test.tsv"), sep="\t")
    train_a = train["sentence1"].tolist()
    train_b = train["sentence2"].tolist()
    train_y = train["label"].values.astype(int)
    test_a = test["sentence1"].tolist()
    test_b = test["sentence2"].tolist()
    test_y = test["label"].values.astype(int)
    return (train_a, train_b, train_y), (test_a, test_b, test_y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = []

    datasets = {
        "Dataset1_repo_text_benchmark": load_dataset1,
        "Dataset2_quora_question_pairs": load_dataset2,
    }

    methods = {
        "Amirzhanov_2025_Frontiers": ("Amirzhanov", AmirzhanovDetector(threshold=0.50)),
        "Sajid_2025_PLoS": ("Sajid", SajidDetector()),
        "SinghGupta_2026_KnowledgeInfoSys": ("SinghGupta", SinghGuptaDetector(alpha=0.5, threshold=0.50)),
    }

    paper_full = {
        "Amirzhanov_2025_Frontiers": "Amirzhanov et al. (2025), Frontiers in Computer Science, DOI 10.3389/fcomp.2025.1504725",
        "Sajid_2025_PLoS": "Sajid et al. (2025), PLoS ONE, DOI 10.1371/journal.pone.0319551",
        "SinghGupta_2026_KnowledgeInfoSys": "Singh & Gupta (2026), Knowledge and Information Systems, DOI 10.1007/s10115-026-02837-8",
    }

    for dname, loader in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dname}")
        print(f"{'='*70}")
        (train_a, train_b, train_y), (test_a, test_b, test_y) = loader()
        print(f"Train: {len(train_a)} pairs | Test: {len(test_a)} pairs")
        print(f"Train label dist: {dict(pd.Series(train_y).value_counts().sort_index())}")
        print(f"Test  label dist: {dict(pd.Series(test_y).value_counts().sort_index())}")

        for mname, (dtype, detector) in methods.items():
            print(f"\n--- Running {mname} on {dname} ---")
            metrics, y_pred, y_score = run_method(
                dtype, detector, train_a, train_b, train_y, test_a, test_b, test_y
            )
            row = {
                "Dataset": dname,
                "Paper": mname,
                "Paper_Full": paper_full[mname],
                "N_Test_Pairs": metrics.pop("NPairs"),
            }
            row.update(metrics)
            results.append(row)
            print(f"  Acc={metrics['Accuracy']}  Prec={metrics['Precision']}  Rec={metrics['Recall']}  "
                  f"F1={metrics['F1-Score']}  ROC={metrics['ROC-AUC']}  MCC={metrics['MCC']}  "
                  f"Latency={metrics['Avg_Latency_ms']}ms/pair")

    # Save results
    results_df = pd.DataFrame(results)
    cols = ["Dataset", "Paper", "Paper_Full", "N_Test_Pairs",
            "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
            "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
            "Avg_Latency_ms", "FitTime_sec", "PredTime_sec"]
    results_df = results_df[cols]

    out_csv = os.path.join(OUT_DIR, "three_papers_two_datasets_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    for dname in datasets:
        for mname in methods:
            sub = results_df[(results_df["Dataset"] == dname) & (results_df["Paper"] == mname)]
            if len(sub) > 0:
                fname = f"{dname}_{mname}.csv"
                sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
                print(f"Saved {fname}")

    print("\n=== SUMMARY ===")
    print(results_df[["Dataset", "Paper", "Accuracy", "Precision", "Recall",
                      "F1-Score", "ROC-AUC", "MCC", "Avg_Latency_ms"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    main()
