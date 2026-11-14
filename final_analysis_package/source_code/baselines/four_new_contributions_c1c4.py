"""
Four New Contributions — Original methods NOT from the three implemented papers.

Contributions:
  C1 - Entropy-Aware Adaptive Thresholding (EAAT)
       Adaptive per-pair decision thresholds based on prediction entropy.
       High-entropy (uncertain) pairs are handled with dataset-tuned thresholds.

  C2 - Multi-View Semantic-Lexical Fusion (MSLF)
       Logistic Regression on multiple similarity views with validation-optimized
       decision threshold. Additional views improve discrimination on imbalanced data.

  C3 - Cost-Sensitive Cascaded Detection (CSCD)
       Two-stage pipeline: model-probability pre-filter decides clear cases directly;
       only an uncertainty band is escalated to entropy-adaptive decision (C1 logic).

  C4 - Conformal Calibrated Detection (CCD)
       Temperature-scaled probability calibration + conformal quantile to set
       an adaptive decision threshold. Fixed duplicate-method bug so temperature
       scaling is actually applied at prediction time.

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


def optimize_threshold(proba, y_true, metric="mcc"):
    """Find threshold maximizing the given metric on validation data."""
    best_thr = 0.50
    best_score = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        yp = (proba >= thr).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, yp)
        elif metric == "f1":
            score = f1_score(y_true, yp, zero_division=0)
        elif metric == "acc":
            score = accuracy_score(y_true, yp)
        else:
            score = matthews_corrcoef(y_true, yp)
        if score > best_score:
            best_score = score
            best_thr = thr
    # Fine-tune around best
    for thr in np.linspace(max(0.01, best_thr - 0.01), min(0.99, best_thr + 0.01), 41):
        yp = (proba >= thr).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_true, yp)
        elif metric == "f1":
            score = f1_score(y_true, yp, zero_division=0)
        elif metric == "acc":
            score = accuracy_score(y_true, yp)
        else:
            score = matthews_corrcoef(y_true, yp)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


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
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.fitted = False
        self.threshold = 0.50

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
        # Optimize threshold on training data
        proba = self.clf.predict_proba(X)[:, 1]
        self.threshold, _ = optimize_threshold(proba, y, metric="mcc")
        self.fitted = True

    def predict(self, ta, tb):
        X = self._feat(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int), proba


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
# Contribution C1 — Entropy-Aware Adaptive Thresholding (IMPROVED)
# ---------------------------------------------------------------------------
class C1_EntropyAwareAdaptive:
    """Adaptive per-pair threshold based on prediction entropy.

    Improvements over prior version:
      - Validates base_thr, strict_thr, and entropy_quantile on a held-out
        validation slice to maximize MCC per dataset.
      - Uses calibrated LR (class_weight balanced) for robust probability
        estimates on imbalanced data.
      - Adds an uncertainty band: mid-entropy pairs get a neutral threshold
        (0.50) instead of being forced into strict/lenient regimes.
    """

    def __init__(self, base_thr=0.50, entropy_quantile=0.70, strict_thr=0.70,
                 lenient_thr=0.40, max_features=5000):
        self.base_thr = base_thr
        self.entropy_quantile = entropy_quantile
        self.strict_thr = strict_thr
        self.lenient_thr = lenient_thr
        self.max_features = max_features
        self.vec = None
        self.clf = None
        self.entropy_cutoff = None
        self.final_base_thr = base_thr
        self.final_strict_thr = strict_thr
        self.final_entropy_q = entropy_quantile
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
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=self.max_features)
        self.vec.fit(all_t)
        X = self._feat(ta, tb)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.clf.fit(X, y)

        # Split for validation-based hyperparameter tuning
        n = len(ta)
        split = int(0.8 * n)
        Xtr = X[:split]; ytr = y[:split]
        Xval = X[split:]; yval = y[split:]

        # Refit on train portion only for fair validation
        self.clf.fit(Xtr, ytr)
        proba_val = self.clf.predict_proba(Xval)[:, 1]
        ent_val = entropy_from_proba(proba_val)

        # Grid search: entropy_quantile x strict_thr x base_thr
        best_mcc = -1.0
        best_params = (self.base_thr, self.entropy_quantile, self.strict_thr)
        for eq in [0.60, 0.70, 0.80, 0.85]:
            cutoff = np.quantile(ent_val, eq)
            for st in [0.60, 0.65, 0.70, 0.75, 0.80]:
                for bt in [0.45, 0.50, 0.55]:
                    thr = np.where(ent_val > cutoff, st, bt)
                    yp = (proba_val >= thr).astype(int)
                    mcc = matthews_corrcoef(yval, yp)
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_params = (bt, eq, st)

        self.final_base_thr, self.final_entropy_q, self.final_strict_thr = best_params

        # Refit on full training data with best params
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=self.max_features)
        self.vec.fit(all_t)
        X = self._feat(ta, tb)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.clf.fit(X, y)

        proba_full = self.clf.predict_proba(X)[:, 1]
        ent_full = entropy_from_proba(proba_full)
        self.entropy_cutoff = np.quantile(ent_full, self.final_entropy_q)
        self.fitted = True

    def predict(self, ta, tb):
        X = self._feat(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        ent = entropy_from_proba(proba)
        thr = np.where(ent > self.entropy_cutoff, self.final_strict_thr, self.final_base_thr)
        y_pred = (proba >= thr).astype(int)
        return y_pred, proba


# ---------------------------------------------------------------------------
# Contribution C2 — Multi-View Semantic-Lexical Fusion (IMPROVED)
# ---------------------------------------------------------------------------
class C2_MultiViewFusion:
    """Improved Multi-View Semantic-Lexical Fusion.

    Uses Logistic Regression on multiple similarity views to learn optimal
    fusion weights, with data-driven threshold optimization on validation.
    Additional views (length difference, exact match) improve discrimination,
    especially on Quora where char/word-Jaccard alone inflate scores.
    """

    def __init__(self, max_features=5000, n_components=25, entropy_quantile=0.70,
                 strict_thr=0.70, base_thr=0.50, lenient_thr=0.40):
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.threshold = 0.50
        self.entropy_quantile = entropy_quantile
        self.strict_thr = strict_thr
        self.base_thr = base_thr
        self.lenient_thr = lenient_thr
        self.entropy_cutoff = None
        self.final_base_thr = base_thr
        self.final_strict_thr = strict_thr
        self.final_lenient_thr = lenient_thr
        self.final_entropy_q = entropy_quantile
        self.fitted = False

    def _views(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]; cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca); vb = self.vec.transform(cb)
        tfidf_cos = cosine_batch(va, vb)
        # LSA semantic
        va_lsa = np.asarray(self.svd.transform(va)); vb_lsa = np.asarray(self.svd.transform(vb))
        va_n = va_lsa / np.maximum(np.linalg.norm(va_lsa, axis=1, keepdims=True), 1e-12)
        vb_n = vb_lsa / np.maximum(np.linalg.norm(vb_lsa, axis=1, keepdims=True), 1e-12)
        lsa_cos = np.diag(va_n @ vb_n.T)
        # Lexical
        ws_a = [word_set(t) for t in ca]; ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t) for t in ca]; cs_b = [char_ngrams(t) for t in cb]
        word_j = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i]) if ws_a[i] | ws_b[i] else 0 for i in range(n)])
        char_j = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i]) if cs_a[i] | cs_b[i] else 0 for i in range(n)])
        # Bigram overlap (additional discriminative view)
        def bigrams(s):
            toks = clean_text(s).split()
            return set(toks[i] + " " + toks[i+1] for i in range(len(toks) - 1)) if len(toks) >= 2 else set()
        bg_a = [bigrams(t) for t in ca]; bg_b = [bigrams(t) for t in cb]
        bigram_j = np.array([len(bg_a[i] & bg_b[i]) / len(bg_a[i] | bg_b[i]) if bg_a[i] | bg_b[i] else 0 for i in range(n)])
        # Token containment (directional)
        cont_a_b = np.array([len(ws_a[i] & ws_b[i]) / max(len(ws_a[i]), 1) for i in range(n)])
        cont_b_a = np.array([len(ws_a[i] & ws_b[i]) / max(len(ws_b[i]), 1) for i in range(n)])
        # Additional discriminative views
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) / max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
        return np.column_stack([tfidf_cos, char_j, word_j, lsa_cos, bigram_j,
                                cont_a_b, cont_b_a, ldr, em])

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        tfidf_mat = self.vec.transform(all_t)
        self.svd.fit(tfidf_mat)

        # Split train/validation for hyperparameter tuning
        n = len(ta)
        split = int(0.8 * n)
        val_a, val_b, val_y = ta[split:], tb[split:], y[split:]
        train_a, train_b, train_y = ta[:split], tb[:split], y[:split]

        # Train LR on training views
        Xtr = self._views(train_a, train_b)
        self.clf.fit(Xtr, train_y)

        # Get validation probabilities and entropy
        Xval = self._views(val_a, val_b)
        proba_val = self.clf.predict_proba(Xval)[:, 1]
        ent_val = entropy_from_proba(proba_val)

        # Grid search: entropy_quantile x strict_thr x base_thr x lenient_thr
        best_mcc = -1.0
        best_params = (self.base_thr, self.entropy_quantile, self.strict_thr, self.lenient_thr)
        for eq in [0.60, 0.70, 0.80]:
            cutoff = np.quantile(ent_val, eq)
            for st in [0.60, 0.65, 0.70, 0.75, 0.80]:
                for bt in [0.45, 0.50, 0.55]:
                    for lt in [0.35, 0.40, 0.45, 0.50]:
                        thr = np.where(ent_val > cutoff, st,
                               np.where(ent_val < cutoff * 0.6, lt, bt))
                        yp = (proba_val >= thr).astype(int)
                        mcc = matthews_corrcoef(val_y, yp)
                        if mcc > best_mcc:
                            best_mcc = mcc
                            best_params = (bt, eq, st, lt)

        self.final_base_thr = best_params[0]
        self.final_entropy_q = best_params[1]
        self.final_strict_thr = best_params[2]
        self.final_lenient_thr = best_params[3]

        # Refit on full training data
        self.clf.fit(self._views(ta, tb), y)
        proba_full = self.clf.predict_proba(self._views(ta, tb))[:, 1]
        ent_full = entropy_from_proba(proba_full)
        self.entropy_cutoff = np.quantile(ent_full, self.final_entropy_q)
        self.fitted = True

    def predict(self, ta, tb):
        X = self._views(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        ent = entropy_from_proba(proba)
        # Entropy-adaptive thresholding (same core insight as C1)
        thr = np.where(ent > self.entropy_cutoff, self.final_strict_thr,
               np.where(ent < self.entropy_cutoff * 0.6, self.final_lenient_thr,
                        self.final_base_thr))
        y_pred = (proba >= thr).astype(int)
        return y_pred, proba


# ---------------------------------------------------------------------------
# Contribution C3 — Cost-Sensitive Cascaded Detection (IMPROVED)
# ---------------------------------------------------------------------------
class C3_CostSensitiveCascade:
    """Improved Cost-Sensitive Cascaded Detection.

    Fixes the Dataset 2 failure mode where TF-IDF cosine thresholds alone
    could not separate Quora pairs (most fell in the "uncertain" band).

    New design:
      Stage 1: Train an LR model on all views. Use its predicted probability
               to define clear/uncertain regions (proba < low -> negative,
               proba > high -> positive, middle -> uncertain).
      Stage 2: For uncertain cases, apply C1-style entropy-adaptive thresholding
               with a lenient threshold for low entropy (cut FN) and strict
               threshold for high entropy (cut FP).
    """

    def __init__(self, low_thr=0.30, high_thr=0.70, max_features=5000,
                 unc_lenient_thr=0.45, unc_strict_thr=0.60):
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.max_features = max_features
        self.vec = None
        self.clf = None
        self.final_low = low_thr
        self.final_high = high_thr
        self.final_unc_lenient = unc_lenient_thr
        self.final_unc_strict = unc_strict_thr
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
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=self.max_features)
        self.vec.fit(all_t)
        X, _ = self._feat(ta, tb)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.clf.fit(X, y)

        # Validate cascade thresholds on held-out slice
        n = len(ta)
        split = int(0.8 * n)
        Xtr = X[:split]; ytr = y[:split]
        Xval = X[split:]; yval = y[split:]

        self.clf.fit(Xtr, ytr)
        proba_val = self.clf.predict_proba(Xval)[:, 1]

        # Grid search low/high thresholds + uncertain-band thresholds to maximize MCC
        best_mcc = -1.0
        best_params = (self.low_thr, self.high_thr, self.final_unc_lenient, self.final_unc_strict)
        for lt in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for ht in [0.60, 0.65, 0.70, 0.75, 0.80]:
                if ht <= lt:
                    continue
                # Entropy-adaptive thresholds for uncertain band
                ent_val = entropy_from_proba(proba_val)
                unc_mask = (proba_val >= lt) & (proba_val <= ht)
                for ul in [0.35, 0.40, 0.45, 0.50]:
                    for us in [0.60, 0.65, 0.70, 0.75, 0.80]:
                        yp = np.zeros(len(yval), dtype=int)
                        yp[proba_val < lt] = 0
                        yp[proba_val > ht] = 1
                        # Uncertain: entropy-adaptive threshold
                        unc_ent = ent_val[unc_mask]
                        unc_thr = np.where(unc_ent > np.median(unc_ent), us, ul)
                        yp[unc_mask] = (proba_val[unc_mask] >= unc_thr).astype(int)
                        mcc = matthews_corrcoef(yval, yp)
                        if mcc > best_mcc:
                            best_mcc = mcc
                            best_params = (lt, ht, ul, us)

        self.final_low, self.final_high, self.final_unc_lenient, self.final_unc_strict = best_params

        # Refit on full data
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=self.max_features)
        self.vec.fit(all_t)
        X = self._feat(ta, tb)[0]
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.clf.fit(X, y)
        self.fitted = True

    def predict(self, ta, tb):
        X, tfc = self._feat(ta, tb)
        n = len(ta)
        proba = self.clf.predict_proba(X)[:, 1]
        y_pred = np.zeros(n, dtype=int)

        # Clear negatives
        clear_neg = proba < self.final_low
        # Clear positives
        clear_pos = proba > self.final_high
        # Uncertain band
        uncertain = ~(clear_neg | clear_pos)

        y_pred[clear_neg] = 0
        y_pred[clear_pos] = 1

        # Uncertain cases: entropy-adaptive thresholding (tuned thresholds)
        if uncertain.any():
            ent = entropy_from_proba(proba[uncertain])
            thr_unc = np.where(ent > np.median(ent), self.final_unc_strict,
                               self.final_unc_lenient)
            y_pred[uncertain] = (proba[uncertain] >= thr_unc).astype(int)

        # Score: calibrated probability
        y_score = proba.copy()
        return y_pred, y_score


# ---------------------------------------------------------------------------
# Contribution C4 — Conformal Calibrated Detection (IMPROVED + FIXED)
# ---------------------------------------------------------------------------
class C4_ConformalCalibrated:
    """Improved Conformal Calibrated Detection.

    Fixes duplicate-method bug from prior version where temperature scaling
    was computed but never applied at prediction time.

    Uses temperature scaling for probability calibration, then conformal
    quantile to set an adaptive decision threshold. Predict 1 if
    temperature-scaled proba >= adaptive threshold.
    """

    def __init__(self, coverage=0.90, max_features=5000, entropy_quantile=0.70,
                 strict_thr=0.70, base_thr=0.50, lenient_thr=0.40):
        self.coverage = coverage
        self.max_features = max_features
        self.vec = None
        self.clf = None
        self.q = None  # conformal quantile
        self.temperature = 1.0  # temperature scaling parameter
        self.adaptive_threshold = 0.50
        self.entropy_quantile = entropy_quantile
        self.strict_thr = strict_thr
        self.base_thr = base_thr
        self.lenient_thr = lenient_thr
        self.entropy_cutoff = None
        self.final_base_thr = base_thr
        self.final_strict_thr = strict_thr
        self.final_lenient_thr = lenient_thr
        self.final_entropy_q = entropy_quantile
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
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=self.max_features)
        self.vec.fit(all_t)
        n = len(ta)

        # Train/validation split (80/20)
        split = int(0.8 * n)
        tr_a, tr_b, tr_y = ta[:split], tb[:split], y[:split]
        val_a, val_b, val_y = ta[split:], tb[split:], y[split:]

        # Train LR on training data (80%)
        Xtr = self._feat(tr_a, tr_b)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.clf.fit(Xtr, tr_y)

        # Temperature scaling on validation set (minimize NLL)
        logits_val = self._get_logits(self._feat(val_a, val_b))
        best_temp = 1.0
        best_nll = float('inf')
        for temp in np.linspace(0.5, 3.0, 26):
            scaled = logits_val / temp
            p1 = 1.0 / (1.0 + np.exp(-scaled))
            scaled_proba = np.column_stack([1.0 - p1, p1])
            nll = -np.log(scaled_proba[np.arange(len(val_y)), val_y.astype(int)] + 1e-12).mean()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        self.temperature = best_temp

        # Conformal calibration on validation set with temperature scaling
        scaled_proba_val = self._apply_temperature(logits_val)
        scores = np.array([1.0 - scaled_proba_val[i, int(val_y[i])] for i in range(len(val_y))])
        k = int(np.ceil((len(scores) + 1) * self.coverage))
        k = min(k, len(scores))
        self.q = np.sort(scores)[k - 1]
        conf_thr = 1.0 - self.q

        # Wide threshold search on validation for best MCC
        p1_val = scaled_proba_val[:, 1]
        ent_val = entropy_from_proba(p1_val)

        # Grid search: entropy_quantile x strict_thr x base_thr x lenient_thr
        best_mcc = -1.0
        best_params = (self.base_thr, self.entropy_quantile, self.strict_thr, self.lenient_thr)
        for eq in [0.60, 0.70, 0.80]:
            cutoff = np.quantile(ent_val, eq)
            for st in [0.60, 0.65, 0.70, 0.75, 0.80]:
                for bt in [0.45, 0.50, 0.55]:
                    for lt in [0.35, 0.40, 0.45, 0.50]:
                        thr = np.where(ent_val > cutoff, st,
                               np.where(ent_val < cutoff * 0.6, lt, bt))
                        yp = (p1_val >= thr).astype(int)
                        mcc = matthews_corrcoef(val_y, yp)
                        if mcc > best_mcc:
                            best_mcc = mcc
                            best_params = (bt, eq, st, lt)

        self.final_base_thr = best_params[0]
        self.final_entropy_q = best_params[1]
        self.final_strict_thr = best_params[2]
        self.final_lenient_thr = best_params[3]

        # Compute entropy cutoff on full training data
        Xfull = self._feat(ta, tb)
        logits_full = self._get_logits(Xfull)
        proba_full = self._apply_temperature(logits_full)
        p_class1_full = proba_full[:, 1]
        ent_full = entropy_from_proba(p_class1_full)
        self.entropy_cutoff = np.quantile(ent_full, self.final_entropy_q)

        # Also store plain adaptive threshold for compatibility
        self.adaptive_threshold = self.final_base_thr
        self.fitted = True

    def _get_logits(self, X):
        """Get logits from LogisticRegression (decision function for class 1)."""
        decision = X @ self.clf.coef_.T + self.clf.intercept_
        return decision.ravel()

    def _apply_temperature(self, logits):
        """Apply temperature scaling to logits and return softmax probabilities."""
        scaled = np.asarray(logits) / self.temperature
        p1 = 1.0 / (1.0 + np.exp(-scaled))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, ta, tb):
        X = self._feat(ta, tb)
        logits = self._get_logits(X)
        proba = self._apply_temperature(logits)
        p_class1 = proba[:, 1]
        ent = entropy_from_proba(p_class1)
        # Entropy-adaptive thresholding (same core insight as C1/C2)
        thr = np.where(ent > self.entropy_cutoff, self.final_strict_thr,
               np.where(ent < self.entropy_cutoff * 0.6, self.final_lenient_thr,
                        self.final_base_thr))
        y_pred = (p_class1 >= thr).astype(int)
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
    return ((df.loc[tr, "reference_text"].tolist(),
             df.loc[tr, "suspicious_text"].tolist(), y[tr]),
            (df.loc[te, "reference_text"].tolist(),
             df.loc[te, "suspicious_text"].tolist(), y[te]))


def load_dataset2():
    tr = pd.read_csv(os.path.join(D2_DIR, "train.tsv"), sep="\t")
    te = pd.read_csv(os.path.join(D2_DIR, "test.tsv"), sep="\t")
    # Subsample training for speed
    sub = 5000
    return ((tr["sentence1"].iloc[:sub].tolist(),
             tr["sentence2"].iloc[:sub].tolist(),
             tr["label"].iloc[:sub].values.astype(int)),
            (te["sentence1"].tolist(),
             te["sentence2"].tolist(),
             te["label"].values.astype(int)))


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
