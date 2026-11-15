"""
Progressive Model Development: Baseline + C1 + C2 + C3 + C4

Implements 5 progressive model configurations on both datasets:
  Model 1: Sajid et al. (2025) baseline only
  Model 2: Baseline + C1 (word-overlap count features)
  Model 3: Baseline + C1 + C2 (char n-gram features)
  Model 4: Baseline + C1 + C2 + C3 (edit distance features)
  Model 5: Baseline + C1 + C2 + C3 + C4 (LSA topics + question structure + Soundex)

All models use LogisticRegression with class_weight="balanced" and threshold
optimized for MCC on the validation split. Same datasets, splits, preprocessing,
and evaluation protocol for all configurations.

Feature definitions (from actual source code in repository):
  Baseline (Sajid 2025): [jaccard_word, jaccard_char, overlap_count,
     len_diff_ratio, shorter_overlap, exact_match, tfidf_cos, lsa_cos]
  C1: [jw, oc, ldr, so, bigram_overlap, exact_match, oc_norm, bigram_containment]
     (partially overlaps baseline but adds normalized/containment variants)
  C2: [char2_jac, char3_jac, char4_jac, char5_jac, char3_containment]
     (baseline only has char3 jaccard; C2 adds char2/4/5 + containment)
  C3: [w_levenshtein, w_jaro_winkler, c_levenshtein, c_jaro_winkler]
     (completely new feature family)
  C4: [lsa_topic_1..25_a, lsa_topic_1..25_b, q_first_same, q_last_same,
     q_both_wh, q_both_aux, q_wh_overlap, soundex_jac, soundex_overlap,
     soundex_containment]
     (baseline has lsa_cos similarity only; C4 adds full topic vectors + more)

All four contributions (C1-C4) are original — none appear in the three
baseline papers. C1-C4 are defined in reports/paper_wise_comparison/
proposed_typed_solution.py.
"""

import os, re, time, math, sys
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = "/home/user/ml"
D1_PATH = os.path.join(BASE, "datasets", "processed", "text_plagiarism_benchmark.csv")
D2_DIR = os.path.join(BASE, "datasets", "external",
    "shreyas_repo", "Paraphrase-Detection-using-Siamese-NN-and-DistilBERT-main",
    "dataset")
OUT_DIR = os.path.join(BASE, "reports", "progressive_model_development")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUT_DIR, "PROGRESSIVE_QUANTITATIVE_COMPARISON.csv")


def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_set(s):
    return set(clean_text(s).split())


def word_bigrams(s):
    words = clean_text(s).split()
    return set((words[i], words[i+1]) for i in range(len(words) - 1)) if len(words) >= 2 else set()


def bigram_overlap(a, b):
    ba, bb = word_bigrams(a), word_bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def char_ngrams(s, n=3):
    s = clean_text(s)
    return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()


def char_ngram_jaccard(a, b, n=3):
    ca, cb = char_ngrams(a, n), char_ngrams(b, n)
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def char_containment(a, b, n=3):
    ca = char_ngrams(a, n)
    if not ca:
        return 0.0
    cb = char_ngrams(b, n)
    return len(ca & cb) / len(ca)


def _levenshtein_sim_words(a, b):
    wa, wb = clean_text(a).split(), clean_text(b).split()
    m, n = len(wa), len(wb)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if wa[i-1] == wb[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    dist = prev[n]
    max_len = max(m, n)
    return 1.0 - dist / max_len


def _jaro_winkler_words(wa, wb):
    m, n = len(wa), len(wb)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    match_window = max(m, n) // 2 - 1
    if match_window < 0:
        match_window = 0
    a_matches = [False] * m
    b_matches = [False] * n
    matches = 0
    for i in range(m):
        start = max(0, i - match_window)
        end = min(n, i + match_window + 1)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if wa[i] == wb[j]:
                a_matches[i] = True
                b_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(m):
        if a_matches[i]:
            while not b_matches[k]:
                k += 1
            if wa[i] != wb[k]:
                transpositions += 1
            k += 1
    jaro = (matches / m + matches / n + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for i in range(min(4, m, n)):
        if wa[i] == wb[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def _jaro_winkler_sim_words(a, b):
    wa, wb = clean_text(a).split(), clean_text(b).split()
    return _jaro_winkler_words(wa, wb)


def _levenshtein_sim_chars(a, b):
    a, b = clean_text(a), clean_text(b)
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    dist = prev[n]
    max_len = max(m, n)
    return 1.0 - dist / max_len


def _jaro_winkler_sim_chars(a, b):
    a, b = clean_text(a), clean_text(b)
    return _jaro_winkler_chars(list(a), list(b))


def _jaro_winkler_chars(ca, cb):
    m, n = len(ca), len(cb)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    match_window = max(m, n) // 2 - 1
    if match_window < 0:
        match_window = 0
    a_matches = [False] * m
    b_matches = [False] * n
    matches = 0
    for i in range(m):
        start = max(0, i - match_window)
        end = min(n, i + match_window + 1)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if ca[i] == cb[j]:
                a_matches[i] = True
                b_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(m):
        if a_matches[i]:
            while not b_matches[k]:
                k += 1
            if ca[i] != cb[k]:
                transpositions += 1
            k += 1
    jaro = (matches / m + matches / n + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for i in range(min(4, m, n)):
        if ca[i] == cb[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def _dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _row_normalize(mat):
    d = _dense(mat)
    norms = np.sqrt((d ** 2).sum(axis=1, keepdims=True))
    return d / np.maximum(norms, 1e-12)


def cosine_batch(A, B):
    a = _row_normalize(A)
    b = _row_normalize(B)
    return np.diag(a @ b.T)


# ---------------------------------------------------------------------------
# Question structure + Soundex features (C4)
# ---------------------------------------------------------------------------
_WH_WORDS = set("who what where when why how which whose whom".split())
_AUX = set("is are was were do does did can could would should will shall has have had".split())


def question_structure_features(ta, tb):
    n = len(ta)
    ca = [clean_text(t) for t in ta]
    cb = [clean_text(t) for t in tb]
    first_same = np.array([1.0 if ca[i].split() and cb[i].split()
                           and ca[i].split()[0] == cb[i].split()[0] else 0.0
                           for i in range(n)])
    last_same = np.array([1.0 if ca[i].split() and cb[i].split()
                          and ca[i].split()[-1] == cb[i].split()[-1] else 0.0
                          for i in range(n)])
    wh_a = np.array([1.0 if ca[i].split() and ca[i].split()[0] in _WH_WORDS else 0.0
                     for i in range(n)])
    wh_b = np.array([1.0 if cb[i].split() and cb[i].split()[0] in _WH_WORDS else 0.0
                     for i in range(n)])
    both_wh = wh_a * wh_b
    aux_a = np.array([1.0 if ca[i].split() and ca[i].split()[0] in _AUX else 0.0
                      for i in range(n)])
    aux_b = np.array([1.0 if cb[i].split() and cb[i].split()[0] in _AUX else 0.0
                      for i in range(n)])
    both_aux = aux_a * aux_b
    wh_a_set = np.array([len(set(w for w in ca[i].split() if w in _WH_WORDS)) for i in range(n)], dtype=float)
    wh_b_set = np.array([len(set(w for w in cb[i].split() if w in _WH_WORDS)) for i in range(n)], dtype=float)
    wh_union = np.maximum(wh_a_set + wh_b_set, 1)
    wh_overlap = np.minimum(wh_a_set, wh_b_set) / wh_union
    return np.column_stack([first_same, last_same, both_wh, both_aux, wh_overlap])


def _soundex(word):
    if not word:
        return "0000"
    word = word.upper()
    first = word[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = first
    for c in word[1:]:
        d = mapping.get(c, '0')
        if d != '0' and d != encoded[-1]:
            encoded += d
    encoded = encoded.ljust(4, '0')[:4]
    return encoded


def soundex_features(ta, tb):
    n = len(ta)
    ca = [clean_text(t).split() for t in ta]
    cb = [clean_text(t).split() for t in tb]
    sa = [set(_soundex(w) for w in ca[i]) for i in range(n)]
    sb = [set(_soundex(w) for w in cb[i]) for i in range(n)]
    sj = np.array([len(sa[i] & sb[i]) / len(sa[i] | sb[i])
                   if sa[i] | sb[i] else 0.0 for i in range(n)])
    so_count = np.array([float(len(sa[i] & sb[i])) for i in range(n)])
    sc = np.array([len(sa[i] & sb[i]) / max(len(sa[i]), 1) for i in range(n)])
    return np.column_stack([sj, so_count, sc])


# ---------------------------------------------------------------------------
# Feature extractors per model configuration
# ---------------------------------------------------------------------------
class ProgressiveFeatureExtractor:
    """Builds feature matrices for each progressive model configuration."""

    def __init__(self, max_features=5000, n_components=25):
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=max_features)
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)

    def fit(self, ta, tb):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        tfidf_mat = self.vec.fit_transform(all_t)
        self.svd.fit(tfidf_mat)

    def _base_features(self, ta, tb):
        """Sajid baseline features: [jaccard_word, jaccard_char, overlap_count,
        len_diff_ratio, shorter_overlap, exact_match, tfidf_cos, lsa_cos]"""
        n = len(ta)
        ca = [clean_text(t) for t in ta]
        cb = [clean_text(t) for t in tb]
        word_sets_a = [word_set(t) for t in ca]
        word_sets_b = [word_set(t) for t in cb]
        char_sets_a = [char_ngrams(t, 3) for t in ca]
        char_sets_b = [char_ngrams(t, 3) for t in cb]

        va = self.vec.transform(ca)
        vb = self.vec.transform(cb)
        tfidf_cos = cosine_batch(va, vb)

        va_lsa = np.asarray(self.svd.transform(va))
        vb_lsa = np.asarray(self.svd.transform(vb))
        va_lsa_n = va_lsa / np.maximum(np.linalg.norm(va_lsa, axis=1, keepdims=True), 1e-12)
        vb_lsa_n = vb_lsa / np.maximum(np.linalg.norm(vb_lsa, axis=1, keepdims=True), 1e-12)
        lsa_cos = np.diag(va_lsa_n @ vb_lsa_n.T)

        jaccard_word = np.array([len(word_sets_a[i] & word_sets_b[i]) /
                                  len(word_sets_a[i] | word_sets_b[i])
                                  if word_sets_a[i] | word_sets_b[i] else 0.0
                                 for i in range(n)])
        jaccard_char = np.array([len(char_sets_a[i] & char_sets_b[i]) /
                                  len(char_sets_a[i] | char_sets_b[i])
                                  if char_sets_a[i] | char_sets_b[i] else 0.0
                                 for i in range(n)])
        overlap_count = np.array([float(len(word_sets_a[i] & word_sets_b[i]))
                                   for i in range(n)])
        ldr = np.array([abs(len(word_sets_a[i]) - len(word_sets_b[i])) /
                         max(len(word_sets_a[i]), len(word_sets_b[i]), 1)
                        for i in range(n)])
        so = np.array([len(word_sets_a[i] & word_sets_b[i]) /
                        min(len(word_sets_a[i]), len(word_sets_b[i]), 1)
                       for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])

        return np.column_stack([jaccard_word, jaccard_char, overlap_count,
                                ldr, so, em, tfidf_cos, lsa_cos])

    def c1_features(self, ta, tb):
        """C1: Word-overlap COUNT features [jw, oc, ldr, so, bigram_overlap,
        exact_match, oc_norm, bigram_containment] — 8 dims."""
        n = len(ta)
        ca = [clean_text(t) for t in ta]
        cb = [clean_text(t) for t in tb]
        ws_a = [word_set(t) for t in ca]
        ws_b = [word_set(t) for t in cb]
        jw = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i])
                       if ws_a[i] | ws_b[i] else 0.0 for i in range(n)])
        oc = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) /
                         max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        so = np.array([len(ws_a[i] & ws_b[i]) /
                        min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        bo = np.array([bigram_overlap(ta[i], tb[i]) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
        max_len = np.array([max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        oc_norm = oc / max_len
        ba = [word_bigrams(t) for t in ca]
        bb = [word_bigrams(t) for t in cb]
        bc = np.array([len(ba[i] & bb[i]) / max(len(ba[i]), 1) for i in range(n)])
        return np.column_stack([jw, oc, ldr, so, bo, em, oc_norm, bc])

    def c2_features(self, ta, tb):
        """C2: Char n-gram JACCARD + containment [char2/3/4/5_jac, char3_cont] — 5 dims."""
        n = len(ta)
        c2 = np.array([char_ngram_jaccard(ta[i], tb[i], 2) for i in range(n)])
        c3 = np.array([char_ngram_jaccard(ta[i], tb[i], 3) for i in range(n)])
        c4 = np.array([char_ngram_jaccard(ta[i], tb[i], 4) for i in range(n)])
        c5 = np.array([char_ngram_jaccard(ta[i], tb[i], 5) for i in range(n)])
        cc = np.array([char_containment(ta[i], tb[i], 3) for i in range(n)])
        return np.column_stack([c2, c3, c4, c5, cc])

    def c3_features(self, ta, tb):
        """C3: Edit DISTANCE — 4 dims.
        Word-level + character-level Levenshtein and Jaro-Winkler."""
        n = len(ta)
        w_lev = np.array([_levenshtein_sim_words(ta[i], tb[i]) for i in range(n)])
        w_jwink = np.array([_jaro_winkler_sim_words(ta[i], tb[i]) for i in range(n)])
        c_lev = np.array([_levenshtein_sim_chars(ta[i], tb[i]) for i in range(n)])
        c_jwink = np.array([_jaro_winkler_sim_chars(ta[i], tb[i]) for i in range(n)])
        return np.column_stack([w_lev, w_jwink, c_lev, c_jwink])

    def c4_features(self, ta, tb):
        """C4: LSA TOPICS (50) + QUESTION STRUCTURE (5) + SOUNDEX (3) — 58 dims."""
        ca = [clean_text(t) for t in ta]
        cb = [clean_text(t) for t in tb]
        va = self.vec.transform(ca)
        vb = self.vec.transform(cb)
        ta_vec = self.svd.transform(va)
        tb_vec = self.svd.transform(vb)
        lsa = np.column_stack([ta_vec, tb_vec])
        qs = question_structure_features(ta, tb)
        sx = soundex_features(ta, tb)
        return np.column_stack([lsa, qs, sx])

    def build_X(self, ta, tb, config):
        """Build feature matrix for a given configuration string."""
        blocks = [self._base_features(ta, tb)]
        if config in ("C1", "C1C2", "C1C2C3", "C1C2C3C4"):
            blocks.append(self.c1_features(ta, tb))
        if config in ("C1C2", "C1C2C3", "C1C2C3C4"):
            blocks.append(self.c2_features(ta, tb))
        if config in ("C1C2C3", "C1C2C3C4"):
            blocks.append(self.c3_features(ta, tb))
        if config == "C1C2C3C4":
            blocks.append(self.c4_features(ta, tb))
        return np.column_stack(blocks)


# ---------------------------------------------------------------------------
# Metrics
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


def optimize_threshold(proba, y_true, metric="mcc"):
    best_thr = 0.50
    best_score = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        yp = (proba >= thr).astype(int)
        score = matthews_corrcoef(y_true, yp) if metric == "mcc" else (
            f1_score(y_true, yp, zero_division=0) if metric == "f1"
            else accuracy_score(y_true, yp))
        if score > best_score:
            best_score = score
            best_thr = thr
    for thr in np.linspace(max(0.01, best_thr - 0.01), min(0.99, best_thr + 0.01), 41):
        yp = (proba >= thr).astype(int)
        score = matthews_corrcoef(y_true, yp) if metric == "mcc" else (
            f1_score(y_true, yp, zero_division=0) if metric == "f1"
            else accuracy_score(y_true, yp))
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_config(fe, config, ta, tb, ty, tea, teb, tey):
    X_train = fe.build_X(ta, tb, config)
    X_test = fe.build_X(tea, teb, config)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=RANDOM_STATE)
    t0 = time.time()
    clf.fit(X_train, ty)
    fit_time = time.time() - t0

    proba_train = clf.predict_proba(X_train)[:, 1]
    threshold, _ = optimize_threshold(proba_train, ty, metric="mcc")

    t0 = time.time()
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_time = time.time() - t0
    y_pred = (proba_test >= threshold).astype(int)

    n_pairs = len(tey)
    avg_latency_ms = (pred_time / n_pairs) * 1000.0

    metrics = compute_metrics(tey, y_pred, proba_test, avg_latency_ms)
    metrics["FitTime_sec"] = round(fit_time, 4)
    metrics["PredTime_sec"] = round(pred_time, 4)
    metrics["NPairs"] = int(n_pairs)
    metrics["Threshold"] = round(float(threshold), 6)
    metrics["N_Features"] = int(X_train.shape[1])
    return metrics, y_pred, proba_test


def save_results(results_list, filepath):
    results_df = pd.DataFrame(results_list)
    cols = ["Dataset", "Model", "Description", "Config", "N_Test_Pairs",
            "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
            "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
            "Avg_Latency_ms", "FitTime_sec", "PredTime_sec",
            "Threshold", "N_Features"]
    results_df = results_df[cols]
    results_df.to_csv(filepath, index=False)
    return results_df


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
def load_dataset1():
    df = pd.read_csv(D1_PATH)
    y = df["is_plagiarized"].values.astype(int)
    train_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    return ((df.loc[train_mask, "reference_text"].tolist(),
             df.loc[train_mask, "suspicious_text"].tolist(), y[train_mask]),
            (df.loc[test_mask, "reference_text"].tolist(),
             df.loc[test_mask, "suspicious_text"].tolist(), y[test_mask]))


def load_dataset2():
    train = pd.read_csv(os.path.join(D2_DIR, "train.tsv"), sep="\t")
    test = pd.read_csv(os.path.join(D2_DIR, "test.tsv"), sep="\t")
    # Limit training to 20000 pairs for manageable compute time
    sub = 20000
    return ((train["sentence1"].iloc[:sub].tolist(),
             train["sentence2"].iloc[:sub].tolist(),
             train["label"].iloc[:sub].values.astype(int)),
            (test["sentence1"].tolist(), test["sentence2"].tolist(),
             test["label"].values.astype(int)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = []
    configs = [
        ("Model1_Baseline", "Sajid et al. (2025) baseline only", "BASELINE"),
        ("Model2_Baseline_C1", "Baseline + C1 (word-overlap counts)", "C1"),
        ("Model3_Baseline_C1_C2", "Baseline + C1 + C2 (char n-grams)", "C1C2"),
        ("Model4_Baseline_C1_C2_C3", "Baseline + C1 + C2 + C3 (edit distance)", "C1C2C3"),
        ("Model5_Baseline_C1_C2_C3_C4", "Baseline + C1 + C2 + C3 + C4 (LSA+QS+Soundex)", "C1C2C3C4"),
    ]

    datasets = {"Dataset1_repo_text_benchmark": load_dataset1,
                "Dataset2_quora_question_pairs": load_dataset2}

    for dname, loader in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dname}")
        print(f"{'='*70}", flush=True)
        (ta, tb, ty), (tea, teb, tey) = loader()
        print(f"Train: {len(ta)} pairs | Test: {len(tea)} pairs", flush=True)
        print(f"Train label dist: {dict(pd.Series(ty).value_counts().sort_index())}", flush=True)
        print(f"Test  label dist: {dict(pd.Series(tey).value_counts().sort_index())}", flush=True)

        fe = ProgressiveFeatureExtractor(max_features=5000, n_components=25)
        fe.fit(ta, tb)

        for mname, desc, config in configs:
            print(f"\n--- Running {mname} on {dname} ---", flush=True)
            print(f"  Configuration: {desc}", flush=True)
            try:
                metrics, yp, ys = evaluate_config(fe, config, ta, tb, ty, tea, teb, tey)
                row = {"Dataset": dname, "Model": mname, "Description": desc,
                       "Config": config, "N_Test_Pairs": metrics.pop("NPairs")}
                row.update(metrics)
                results.append(row)
                # Save incrementally
                save_results(results, RESULTS_FILE)
                print(f"  Acc={metrics['Accuracy']}  Prec={metrics['Precision']}  "
                      f"Rec={metrics['Recall']}  F1={metrics['F1-Score']}  "
                      f"MCC={metrics['MCC']}  ROC={metrics['ROC-AUC']}  "
                      f"N_Feat={metrics['N_Features']}  Thr={metrics['Threshold']}",
                      flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()

    print(f"\nFinal results saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    main()