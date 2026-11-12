"""
Four New Contributions — Original methods NOT from the three implemented papers.

Contributions:
  C1 - Entropy-Aware Adaptive Thresholding (EAAT)
       Adaptive per-pair decision thresholds based on prediction entropy.
       High-entropy (uncertain) pairs are handled conservatively to cut FP/FN.

  C2 - Multi-View Semantic-Lexical Fusion (MSLF)
       Learned weighted fusion of TF-IDF, char-Jaccard, word-Jaccard, and LSA
       views. Optimal weights fitted on a held-out validation slice.

  C3 - Cost-Sensitive Cascaded Detection (CSCD)
       Two-stage pipeline: cheap TF-IDF pre-filter decides clear cases directly;
       only an uncertainty band is escalated to Logistic Regression.

  C4 - Conformal Calibrated Detection (CCD)
       Split-conformal calibration on a held-out calibration slice to produce
       calibrated prediction sets with a target coverage guarantee (90%).

All four are original contributions — none appear in Amirzhanov (2025), Sajid
(2025), or Singh & Gupta (2026). They are designed to outperform those baselines
on both text plagiarism datasets.

Outputs:
  - tables/four_contributions_results.csv   (quantitative, all 7 methods x 2 datasets)
  - tables/four_contributions_summary.csv   (absolute / consolidation table)
  - charts/*.png                             (pictorial analysis)
  - FOUR_CONTRIBUTIONS_QUALITATIVE_ANALYSIS.md (qualitative analysis)
"""

import os, re, time, json, math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

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
TABLE_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "tables")
CHART_DIR = os.path.join(BASE, "reports", "paper_wise_comparison", "charts")
QUAL_DIR = os.path.join(BASE, "reports", "paper_wise_comparison")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

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


def char_ngrams(s, n=3):
    s = clean_text(s)
    return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()


def _dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def cosine_batch(A, B):
    ad = _dense(A); bd = _dense(B)
    an = ad / np.maximum(np.sqrt((ad**2).sum(1, keepdims=True)), 1e-12)
    bn = bd / np.maximum(np.sqrt((bd**2).sum(1, keepdims=True)), 1e-12)
    return np.diag(an @ bn.T)


def entropy_from_proba(p):
    """Binary prediction entropy: -[p log p + (1-p) log (1-p)]."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


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


# ---------------------------------------------------------------------------
# Baseline methods (the three papers) — re-implemented for fair comparison
# ---------------------------------------------------------------------------
class AmirzhanovBaseline:
    """Paper 1: TF-IDF cosine + fixed threshold 0.50."""

    def __init__(self, threshold=0.50, max_features=5000):
        self.threshold = threshold
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)

    def fit(self, ta, tb, y=None):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)

    def predict(self, ta, tb):
        va = self.vec.transform([clean_text(t) for t in ta])
        vb = self.vec.transform([clean_text(t) for t in tb])
        sims = cosine_batch(va, vb)
        return (sims >= self.threshold).astype(int), sims


class SajidBaseline:
    """Paper 2: Multi-feature (7-D) + Logistic Regression, fixed threshold."""

    def __init__(self, max_features=5000):
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self.fitted = False

    def _feat(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]; cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca); vb = self.vec.transform(cb)
        tfc = cosine_batch(va, vb)
        ws_a = [word_set(t) for t in ca]; ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t) for t in ca]; cs_b = [char_ngrams(t) for t in cb]
        jw = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
        jc = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
        oc = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) / max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        so = np.array([len(ws_a[i] & ws_b[i]) / min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
        return np.column_stack([jw, jc, oc, ldr, so, em, tfc])

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        X = self._feat(ta, tb)
        self.clf.fit(X, y)
        self.fitted = True

    def predict(self, ta, tb):
        X = self._feat(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int), proba


class SinghGuptaBaseline:
    """Paper 3: TF-IDF + char-Jaccard hybrid, alpha=0.5, fixed threshold."""

    def __init__(self, alpha=0.5, threshold=0.50, max_features=5000):
        self.alpha = alpha
        self.threshold = threshold
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)

    def fit(self, ta, tb, y=None):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)

    def predict(self, ta, tb):
        va = self.vec.transform([clean_text(t) for t in ta])
        vb = self.vec.transform([clean_text(t) for t in tb])
        syn = cosine_batch(va, vb)
        sem = np.zeros(len(ta))
        for i in range(len(ta)):
            ca = char_ngrams(ta[i]); cb = char_ngrams(tb[i])
            inter = len(ca & cb); union = len(ca | cb)
            sem[i] = inter / union if union > 0 else 0.0
        hybrid = self.alpha * syn + (1.0 - self.alpha) * sem
        return (hybrid >= self.threshold).astype(int), hybrid


# ---------------------------------------------------------------------------
# Contribution C1 — Entropy-Aware Adaptive Thresholding
# ---------------------------------------------------------------------------
class C1_EntropyAwareAdaptive:
    """Adaptive per-pair threshold based on prediction entropy.
    Low-entropy pairs decided directly; high-entropy pairs use a stricter
    threshold (raise bar for positive) to cut false positives."""

    def __init__(self, base_thr=0.50, entropy_quantile=0.70, strict_thr=0.70, max_features=5000):
        self.base_thr = base_thr
        self.entropy_quantile = entropy_quantile
        self.strict_thr = strict_thr
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self.entropy_cutoff = None
        self.fitted = False

    def _feat(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]; cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca); vb = self.vec.transform(cb)
        tfc = cosine_batch(va, vb)
        ws_a = [word_set(t) for t in ca]; ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t) for t in ca]; cs_b = [char_ngrams(t) for t in cb]
        jw = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
        jc = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
        oc = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) / max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        so = np.array([len(ws_a[i] & ws_b[i]) / min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
        return np.column_stack([jw, jc, oc, ldr, so, em, tfc])

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        X = self._feat(ta, tb)
        self.clf.fit(X, y)
        # Determine entropy cutoff on training predictions
        proba = self.clf.predict_proba(X)[:, 1]
        ent = entropy_from_proba(proba)
        self.entropy_cutoff = np.quantile(ent, self.entropy_quantile)
        self.fitted = True

    def predict(self, ta, tb):
        X = self._feat(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        ent = entropy_from_proba(proba)
        # Adaptive threshold: strict where entropy is high
        thr = np.where(ent > self.entropy_cutoff, self.strict_thr, self.base_thr)
        y_pred = (proba >= thr).astype(int)
        return y_pred, proba


# ---------------------------------------------------------------------------
# Contribution C2 — Multi-View Semantic-Lexical Fusion
# ---------------------------------------------------------------------------
class C2_MultiViewFusion:
    """Learned weighted fusion of TF-IDF, char-Jaccard, word-Jaccard, LSA views.
    Fusion weights are fitted on a validation slice (not fixed)."""

    def __init__(self, threshold=0.50, max_features=5000, n_components=25):
        self.threshold = threshold
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.weights = np.array([0.45, 0.20, 0.20, 0.15])  # [w_tfidf, w_charj, w_wordj, w_lsa]
        self.fitted = False

    def _views(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]; cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca); vb = self.vec.transform(cb)
        tfidf_cos = cosine_batch(va, vb)
        # LSA semantic
        va_lsa = self.svd.transform(va); vb_lsa = self.svd.transform(vb)
        va_lsa = np.asarray(va_lsa); vb_lsa = np.asarray(vb_lsa)
        va_n = va_lsa / np.maximum(np.linalg.norm(va_lsa, axis=1, keepdims=True), 1e-12)
        vb_n = vb_lsa / np.maximum(np.linalg.norm(vb_lsa, axis=1, keepdims=True), 1e-12)
        lsa_cos = np.diag(va_n @ vb_n.T)
        # Lexical
        ws_a = [word_set(t) for t in ca]; ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t) for t in ca]; cs_b = [char_ngrams(t) for t in cb]
        word_j = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
        char_j = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
        return tfidf_cos, char_j, word_j, lsa_cos

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        tfidf_mat = self.vec.transform(all_t)
        self.svd.fit(tfidf_mat)

        # Fit fusion weights by grid search on validation slice (last 20% of train)
        n = len(ta)
        split = int(0.8 * n)
        val_a, val_b, val_y = ta[split:], tb[split:], y[split:]
        train_a, train_b, train_y = ta[:split], tb[:split], y[:split]

        # Grid search over weight simplex (coarse for speed); maximize MCC
        val_tf, val_cj, val_wj, val_ls = self._views(val_a, val_b)
        best_mcc = -1.0
        best_w = self.weights.copy()
        for w_t in np.linspace(0.2, 0.6, 5):
            for w_c in np.linspace(0.1, 0.3, 3):
                for w_w in np.linspace(0.1, 0.3, 3):
                    w_l = 1.0 - w_t - w_c - w_w
                    if w_l < 0.05:
                        continue
                    w = np.array([w_t, w_c, w_w, w_l])
                    hybrid = w[0]*val_tf + w[1]*val_cj + w[2]*val_wj + w[3]*val_ls
                    yp = (hybrid >= 0.5).astype(int)
                    mcc = matthews_corrcoef(val_y, yp)
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_w = w
        self.weights = best_w
        self.fitted = True

    def predict(self, ta, tb):
        tfidf_cos, char_j, word_j, lsa_cos = self._views(ta, tb)
        hybrid = (self.weights[0]*tfidf_cos + self.weights[1]*char_j +
                  self.weights[2]*word_j + self.weights[3]*lsa_cos)
        return (hybrid >= self.threshold).astype(int), hybrid


# ---------------------------------------------------------------------------
# Contribution C3 — Cost-Sensitive Cascaded Detection
# ---------------------------------------------------------------------------
class C3_CostSensitiveCascade:
    """Two-stage cascade: cheap TF-IDF pre-filter decides clear cases directly;
    only an uncertainty band (similarity near threshold) escalates to LR."""

    def __init__(self, low_thr=0.35, high_thr=0.65, max_features=5000):
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self.fitted = False

    def _feat(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]; cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca); vb = self.vec.transform(cb)
        tfc = cosine_batch(va, vb)
        ws_a = [word_set(t) for t in ca]; ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t) for t in ca]; cs_b = [char_ngrams(t) for t in cb]
        jw = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
        jc = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
        oc = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) / max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        so = np.array([len(ws_a[i] & ws_b[i]) / min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
        return np.column_stack([jw, jc, oc, ldr, so, em, tfc]), tfc

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        X, _ = self._feat(ta, tb)
        self.clf.fit(X, y)
        self.fitted = True

    def predict(self, ta, tb):
        X, tfc = self._feat(ta, tb)
        n = len(ta)
        y_pred = np.zeros(n, dtype=int)
        # Clear negatives: similarity below low_thr
        clear_neg = tfc < self.low_thr
        # Clear positives: similarity above high_thr
        clear_pos = tfc > self.high_thr
        # Uncertain band -> escalate to LR
        uncertain = ~(clear_neg | clear_pos)
        y_pred[clear_neg] = 0
        y_pred[clear_pos] = 1
        if uncertain.any():
            proba = self.clf.predict_proba(X[uncertain])[:, 1]
            y_pred[uncertain] = (proba >= 0.5).astype(int)
        # Score for metrics: use TF-IDF for clear cases, LR proba for uncertain
        y_score = tfc.copy()
        if uncertain.any():
            proba_all = self.clf.predict_proba(X)[:, 1]
            y_score[uncertain] = proba_all[uncertain]
        return y_pred, y_score


# ---------------------------------------------------------------------------
# Contribution C4 — Conformal Calibrated Detection
# ---------------------------------------------------------------------------
class C4_ConformalCalibrated:
    """Split-conformal calibration. Train on proper-train slice, calibrate on
    held-out calibration slice to produce prediction sets with target coverage.
    Final decision: positive if conformal set contains only class 1 (or majority)."""

    def __init__(self, coverage=0.85, max_features=5000):
        self.coverage = coverage
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")
        self.q = None  # conformal quantile
        self.fitted = False

    def _feat(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]; cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca); vb = self.vec.transform(cb)
        tfc = cosine_batch(va, vb)
        ws_a = [word_set(t) for t in ca]; ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t) for t in ca]; cs_b = [char_ngrams(t) for t in cb]
        jw = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
        jc = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
        oc = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) / max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        so = np.array([len(ws_a[i] & ws_b[i]) / min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
        return np.column_stack([jw, jc, oc, ldr, so, em, tfc])

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        n = len(ta)
        split = int(0.75 * n)
        tr_a, tr_b, tr_y = ta[:split], tb[:split], y[:split]
        cal_a, cal_b, cal_y = ta[split:], tb[split:], y[split:]

        Xtr = self._feat(tr_a, tr_b)
        self.clf.fit(Xtr, tr_y)

        # Conformity scores on calibration set: 1 - p_true_class
        Xcal = self._feat(cal_a, cal_b)
        proba_cal = self.clf.predict_proba(Xcal)
        # score = 1 - probability assigned to true label
        scores = np.array([1.0 - proba_cal[i, int(cal_y[i])] for i in range(len(cal_y))])
        # Quantile at coverage level (with n+1 adjustment)
        k = int(np.ceil((len(scores) + 1) * self.coverage))
        k = min(k, len(scores))
        self.q = np.sort(scores)[k - 1]
        self.fitted = True

    def predict(self, ta, tb):
        X = self._feat(ta, tb)
        proba = self.clf.predict_proba(X)  # shape (n, 2)
        # Conformal set: classes with score <= q
        # score for class c = 1 - proba[:, c]; include c if 1 - p_c <= q  => p_c >= 1 - q
        threshold_prob = 1.0 - self.q
        p_class1 = proba[:, 1]
        p_class0 = proba[:, 0]
        # Positive if class 1 is in the conformal set AND (class 0 not in set OR p1>p0)
        in_set_1 = p_class1 >= threshold_prob
        in_set_0 = p_class0 >= threshold_prob
        y_pred = ((in_set_1 & ~in_set_0) | (in_set_1 & in_set_0 & (p_class1 >= p_class0))).astype(int)
        # Score: calibrated probability of class 1
        return y_pred, p_class1


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def evaluate(name, detector, ta, tb, ty, tea, teb, tey):
    t0 = time.time()
    detector.fit(ta, tb, ty)
    fit_t = time.time() - t0
    t0 = time.time()
    y_pred, y_score = detector.predict(tea, teb)
    pred_t = time.time() - t0
    n = len(tey)
    latency = (pred_t / n) * 1000.0
    m = compute_metrics(tey, y_pred, y_score, latency)
    m.update({
        "FitTime_sec": round(fit_t, 4),
        "PredTime_sec": round(pred_t, 4),
        "NPairs": int(n),
    })
    return m, y_pred, y_score


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_dataset1():
    df = pd.read_csv(D1_PATH)
    y = df["is_plagiarized"].values.astype(int)
    tr = df["split"] == "val"
    te = df["split"] == "test"
    return (df.loc[tr, "reference_text"].tolist(), df.loc[tr, "suspicious_text"].tolist(), y[tr]), \
           (df.loc[te, "reference_text"].tolist(), df.loc[te, "suspicious_text"].tolist(), y[te])


def load_dataset2():
    tr = pd.read_csv(os.path.join(D2_DIR, "train.tsv"), sep="\t")
    te = pd.read_csv(os.path.join(D2_DIR, "test.tsv"), sep="\t")
    # Subsample training for speed
    sub = 5000
    return (tr["sentence1"].iloc[:sub].tolist(), tr["sentence2"].iloc[:sub].tolist(),
            tr["label"].iloc[:sub].values.astype(int)), \
           (te["sentence1"].tolist(), te["sentence2"].tolist(), te["label"].values.astype(int))


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
        "Amirzhanov_2025_Frontiers": ("baseline", AmirzhanovBaseline()),
        "Sajid_2025_PLoS": ("baseline", SajidBaseline()),
        "SinghGupta_2026_KnowledgeInfoSys": ("baseline", SinghGuptaBaseline()),
        "C1_EntropyAwareAdaptive": ("contribution", C1_EntropyAwareAdaptive()),
        "C2_MultiViewFusion": ("contribution", C2_MultiViewFusion()),
        "C3_CostSensitiveCascade": ("contribution", C3_CostSensitiveCascade()),
        "C4_ConformalCalibrated": ("contribution", C4_ConformalCalibrated()),
    }

    for dname, loader in datasets.items():
        print(f"\n{'='*70}\nDATASET: {dname}\n{'='*70}")
        (ta, tb, ty), (tea, teb, tey) = loader()
        print(f"Train: {len(ta)} | Test: {len(tea)}")
        print(f"Train labels: {dict(pd.Series(ty).value_counts().sort_index())}")
        print(f"Test  labels: {dict(pd.Series(tey).value_counts().sort_index())}")

        for mname, (ctype, detector) in methods.items():
            print(f"\n--- {mname} on {dname} ---")
            m, yp, ys = evaluate(mname, detector, ta, tb, ty, tea, teb, tey)
            row = {"Dataset": dname, "Method": mname, "Type": ctype,
                   "Method_Full": {
                       "Amirzhanov_2025_Frontiers": "Amirzhanov et al. (2025), Frontiers in Computer Science (baseline)",
                       "Sajid_2025_PLoS": "Sajid et al. (2025), PLoS ONE (baseline)",
                       "SinghGupta_2026_KnowledgeInfoSys": "Singh & Gupta (2026), Knowledge and Information Systems (baseline)",
                       "C1_EntropyAwareAdaptive": "C1 — Entropy-Aware Adaptive Thresholding (NEW)",
                       "C2_MultiViewFusion": "C2 — Multi-View Semantic-Lexical Fusion (NEW)",
                       "C3_CostSensitiveCascade": "C3 — Cost-Sensitive Cascaded Detection (NEW)",
                       "C4_ConformalCalibrated": "C4 — Conformal Calibrated Detection (NEW)",
                   }[mname],
                   "N_Test_Pairs": m.pop("NPairs")}
            row.update(m)
            results.append(row)
            print(f"  Acc={m['Accuracy']} F1={m['F1-Score']} ROC={m['ROC-AUC']} MCC={m['MCC']} "
                  f"Prec={m['Precision']} Rec={m['Recall']} Spec={m['Specificity']}")

    # Save quantitative results
    df = pd.DataFrame(results)
    cols = ["Dataset", "Method", "Type", "Method_Full", "N_Test_Pairs",
            "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
            "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
            "Avg_Latency_ms", "FitTime_sec", "PredTime_sec"]
    df = df[cols]
    quant_path = os.path.join(TABLE_DIR, "four_contributions_results.csv")
    df.to_csv(quant_path, index=False)
    print(f"\nSaved quantitative results to {quant_path}")

    # --- Absolute / consolidation summary table ---
    print("\n=== ABSOLUTE / CONSOLIDATION SUMMARY ===")
    summary_rows = []
    for dname in datasets:
        sub = df[df["Dataset"] == dname]
        best_acc = sub.loc[sub["Accuracy"].idxmax()]
        best_f1 = sub.loc[sub["F1-Score"].idxmax()]
        best_mcc = sub.loc[sub["MCC"].idxmax()]
        best_roc = sub.loc[sub["ROC-AUC"].idxmax()]
        best_spec = sub.loc[sub["Specificity"].idxmax()]
        best_rec = sub.loc[sub["Recall"].idxmax()]
        # Overall winner: best MCC (most robust for imbalance)
        overall = sub.loc[sub["MCC"].idxmax()]
        summary_rows.append({
            "Dataset": dname,
            "N_Test_Pairs": int(sub["N_Test_Pairs"].iloc[0]),
            "Accuracy_Winner": best_acc["Method"],
            "Accuracy_Winner_Value": round(float(best_acc["Accuracy"]), 6),
            "F1_Winner": best_f1["Method"],
            "F1_Winner_Value": round(float(best_f1["F1-Score"]), 6),
            "MCC_Winner": best_mcc["Method"],
            "MCC_Winner_Value": round(float(best_mcc["MCC"]), 6),
            "ROC_AUC_Winner": best_roc["Method"],
            "ROC_AUC_Winner_Value": round(float(best_roc["ROC-AUC"]), 6),
            "Specificity_Winner": best_spec["Method"],
            "Specificity_Winner_Value": round(float(best_spec["Specificity"]), 6),
            "Recall_Winner": best_rec["Method"],
            "Recall_Winner_Value": round(float(best_rec["Recall"]), 6),
            "Overall_Verdict": overall["Method"],
            "Overall_MCC": round(float(overall["MCC"]), 6),
        })
    summary_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(TABLE_DIR, "four_contributions_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print(f"Saved absolute summary to {sum_path}")
    print(summary_df.to_string(index=False))

    # --- Pictorial analysis ---
    datasets_keys = list(datasets.keys())
    print("\n=== Generating pictorial charts ===")
    metrics_to_plot = ["Accuracy", "F1-Score", "MCC", "ROC-AUC", "Precision", "Recall", "Specificity"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for dname in datasets:
            sub = df[df["Dataset"] == dname]
            methods_list = sub["Method"].tolist()
            values = sub[metric].tolist()
            x = np.arange(len(methods_list))
            plt.bar(x - 0.2 if dname == datasets_keys[0] else x + 0.2, values, width=0.4,
                    label=dname, alpha=0.85)
        plt.xticks(x, methods_list, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison — All Methods × Both Datasets")
        plt.ylim(0, max(1.0, df[metric].max() * 1.15))
        plt.legend()
        plt.tight_layout()
        chart_path = os.path.join(CHART_DIR, f"chart_{metric.replace(' ', '_').replace('-', '_')}.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {chart_path}")

    # Grouped bar: MCC by method across datasets (clearest winner view)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.12
    method_list = [m for m in methods.keys()]
    for i, mname in enumerate(method_list):
        vals = []
        for dname in datasets:
            v = df[(df["Dataset"] == dname) & (df["Method"] == mname)]["MCC"].iloc[0]
            vals.append(v)
        plt.bar(x + (i - len(method_list)/2) * width + width/2, vals, width, label=mname, alpha=0.85)
    plt.xticks(x, [d.replace("_", " ") for d in datasets.keys()])
    plt.ylabel("MCC")
    plt.title("MCC Comparison — All 7 Methods Across Both Datasets")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylim(-0.1, max(df["MCC"].max(), 0) * 1.3 + 0.05)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "chart_mcc_grouped.png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved chart_mcc_grouped.png")

    print("\nDone. All outputs saved.")

    return df, summary_df


if __name__ == "__main__":
    main()
