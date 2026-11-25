# Kaggle/Colab Code Cells — ALL IN ONE (Complete)

Copy each cell below into a separate cell in Kaggle or Google Colab, then run top to bottom.

This single file combines:
- **Progressive Model Development** (Models 1-5, C1-C4 contributions, ablation, baselines)
- **Single-Paper Implementations** (Sajid 2025, Amirzhanov 2025, SinghGupta 2026)

**Kaggle file paths:** Your uploaded files are at:
- `/kaggle/input/the-ml-data-sets/dataset1_text_plagiarism_benchmark.csv`
- `/kaggle/input/the-ml-data-sets/dataset2_quora_train.tsv`
- `/kaggle/input/the-ml-data-sets/dataset2_quora_test.tsv`

**Total cell count: 29 cells**

---

## SECTION A — SHARED INFRASTRUCTURE (Cells 1-6)

### Cell 1 — Install dependencies

```python
# !pip install -q pandas scikit-learn numpy matplotlib seaborn
```

---

### Cell 2 — Imports and global setup

```python
import os, re, time, math
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

---

### Cell 3 — Text preprocessing utilities

```python
def clean_text(s):
    """Lowercase, strip non-alphanumeric, collapse whitespace."""
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
```

---

### Cell 4 — Levenshtein & Jaro-Winkler similarity (C3 / edit distance)

```python
def _levenshtein_sim_words(a, b):
    """Word-level Levenshtein similarity."""
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
    """Character-level Levenshtein similarity."""
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

def _jaro_winkler_sim_chars(a, b):
    a, b = clean_text(a), clean_text(b)
    return _jaro_winkler_chars(list(a), list(b))
```

---

### Cell 5 — Cosine similarity helpers

```python
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
```

---

### Cell 6 — C4 features: Question structure + Soundex

```python
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
```

---

## SECTION B — PROGRESSIVE MODEL DEVELOPMENT (Cells 7-19)

### Cell 7 — Progressive feature extractor class

```python
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
        """Sajid baseline features: 8 dims."""
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
        """C1: Word-overlap COUNT features — 8 dims."""
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
        """C2: Char n-gram JACCARD + containment — 5 dims."""
        n = len(ta)
        c2 = np.array([char_ngram_jaccard(ta[i], tb[i], 2) for i in range(n)])
        c3 = np.array([char_ngram_jaccard(ta[i], tb[i], 3) for i in range(n)])
        c4 = np.array([char_ngram_jaccard(ta[i], tb[i], 4) for i in range(n)])
        c5 = np.array([char_ngram_jaccard(ta[i], tb[i], 5) for i in range(n)])
        cc = np.array([char_containment(ta[i], tb[i], 3) for i in range(n)])
        return np.column_stack([c2, c3, c4, c5, cc])

    def c3_features(self, ta, tb):
        """C3: Edit DISTANCE — 4 dims."""
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
```

---

### Cell 8 — Metrics, threshold optimization, and evaluation

```python
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
```

---

### Cell 9 — Baseline implementations (three papers) — progressive versions

```python
class SajidBaseline:
    """Sajid et al. (2025) — 8 features: word Jaccard, char trigram Jaccard,
    overlap count, length diff ratio, shorter overlap, exact match,
    TF-IDF cosine, LSA cosine."""
    def __init__(self, fe, threshold=0.50):
        self.fe = fe
        self.threshold = threshold
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=RANDOM_STATE)

    def fit(self, ta, tb, y):
        X = self.fe._base_features(ta, tb)
        self.clf.fit(X, y)
        proba = self.clf.predict_proba(X)[:, 1]
        self.threshold, _ = optimize_threshold(proba, y, metric="mcc")

    def predict(self, ta, tb):
        X = self.fe._base_features(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int), proba

class AmirzhanovBaseline:
    """Amirzhanov et al. (2025) — TF-IDF cosine similarity threshold classifier."""
    def __init__(self, fe, threshold=0.50):
        self.fe = fe
        self.threshold = threshold

    def fit(self, ta, tb, y=None):
        pass  # no training needed

    def predict(self, ta, tb):
        sims = cosine_batch(
            self.fe.vec.transform([clean_text(t) for t in ta]),
            self.fe.vec.transform([clean_text(t) for t in tb])
        )
        return (sims >= self.threshold).astype(int), sims

class SinghGuptaBaseline:
    """Singh & Gupta (2026) — Hybrid: alpha * word_Jaccard + (1-alpha) * normalized_overlap."""
    def __init__(self, fe, alpha=0.5, threshold=0.50):
        self.fe = fe
        self.alpha = alpha
        self.threshold = threshold

    def fit(self, ta, tb, y=None):
        pass

    def predict(self, ta, tb):
        c1 = self.fe.c1_features(ta, tb)
        jw = c1[:, 0]          # word Jaccard
        oc = c1[:, 1]          # overlap count
        hybrid = self.alpha * jw + (1 - self.alpha) * np.minimum(oc / 10.0, 1.0)
        return (hybrid >= self.threshold).astype(int), hybrid
```

---

### Cell 10 — Load datasets (Kaggle paths)

```python
def load_dataset1(d1_path):
    """Dataset 1: Text Plagiarism Benchmark (CSV with val/test split)."""
    df = pd.read_csv(d1_path)
    y = df["is_plagiarized"].values.astype(int)
    train_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    return ((df.loc[train_mask, "reference_text"].tolist(),
             df.loc[train_mask, "suspicious_text"].tolist(), y[train_mask]),
            (df.loc[test_mask, "reference_text"].tolist(),
             df.loc[test_mask, "suspicious_text"].tolist(), y[test_mask]))

def load_dataset2(train_path, test_path, train_subset=5000):
    """Dataset 2: Quora Question Pairs (TSV files)."""
    train = pd.read_csv(train_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    return ((train["sentence1"].iloc[:train_subset].tolist(),
             train["sentence2"].iloc[:train_subset].tolist(),
             train["label"].iloc[:train_subset].values.astype(int)),
            (test["sentence1"].tolist(), test["sentence2"].tolist(),
             test["label"].values.astype(int)))

# ---- Kaggle file paths ----
D1_PATH = "/kaggle/input/the-ml-data-sets/dataset1_text_plagiarism_benchmark.csv"
D2_TRAIN = "/kaggle/input/the-ml-data-sets/dataset2_quora_train.tsv"
D2_TEST = "/kaggle/input/the-ml-data-sets/dataset2_quora_test.tsv"

# Check files exist
for f in [D1_PATH, D2_TRAIN, D2_TEST]:
    if os.path.exists(f):
        print(f"  Found: {f}")
    else:
        print(f"  NOT FOUND: {f}")
        print(f"  Please upload the file via Kaggle sidebar (Add Input → Upload)")

# ---- Load Dataset 1 ----
print("=== Loading Dataset 1 ===")
(ta1, tb1, ty1), (tea1, teb1, tey1) = load_dataset1(D1_PATH)
print(f"D1 loaded: Train={len(ta1)} pairs | Test={len(tea1)} pairs")
print(f"D1 train labels: {dict(pd.Series(ty1).value_counts().sort_index())}")
print(f"D1 test labels:  {dict(pd.Series(tey1).value_counts().sort_index())}")

# ---- Load Dataset 2 ----
print()
print("=== Loading Dataset 2 ===")
(ta2, tb2, ty2), (tea2, teb2, tey2) = load_dataset2(D2_TRAIN, D2_TEST, train_subset=5000)
print(f"D2 loaded: Train={len(ta2)} pairs | Test={len(tea2)} pairs")
print(f"D2 train labels: {dict(pd.Series(ty2).value_counts().sort_index())}")
print(f"D2 test labels:  {dict(pd.Series(tey2).value_counts().sort_index())}")
```

---

### Cell 11 — Run all 5 progressive models on a dataset

```python
def run_all_models_on_dataset(dname, fe, ta, tb, ty, tea, teb, tey):
    """Run Models 1-5 (progressive) on a dataset."""
    configs = [
        ("Model1_Baseline",   "Sajid et al. (2025) baseline only",   "BASELINE"),
        ("Model2_Baseline_C1","Baseline + C1 (word-overlap counts)", "C1"),
        ("Model3_Baseline_C1_C2","Baseline + C1 + C2 (char n-grams)", "C1C2"),
        ("Model4_Baseline_C1_C2_C3","Baseline + C1 + C2 + C3 (edit distance)", "C1C2C3"),
        ("Model5_Baseline_C1_C2_C3_C4","Baseline + C1 + C2 + C3 + C4 (LSA+QS+Soundex)", "C1C2C3C4"),
    ]
    results = []
    print(f"Train: {len(ta)} pairs | Test: {len(tea)} pairs\n")
    for mname, desc, config in configs:
        print(f"--- {mname} on {dname} ---")
        print(f"  Configuration: {desc}", flush=True)
        try:
            metrics, yp, ys = evaluate_config(fe, config, ta, tb, ty, tea, teb, tey)
            row = {"Dataset": dname, "Model": mname, "Description": desc,
                   "Config": config, "N_Test_Pairs": metrics.pop("NPairs")}
            row.update(metrics)
            results.append(row)
            print(f"  Acc={metrics['Accuracy']}  Prec={metrics['Precision']}  "
                  f"Rec={metrics['Recall']}  F1={metrics['F1-Score']}  "
                  f"MCC={metrics['MCC']}  ROC={metrics['ROC-AUC']}  "
                  f"N_Feat={metrics['N_Features']}  Thr={metrics['Threshold']}\n",
                  flush=True)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
    return results
```

---

### Cell 12 — Run baselines (three papers) on a dataset

```python
def run_baselines_on_dataset(dname, fe, ta, tb, ty, tea, teb, tey):
    """Run the three paper baselines on a dataset."""
    baselines = [
        ("Sajid_2025_PLoS",   "Sajid et al. (2025) baseline",   SajidBaseline(fe)),
        ("Amirzhanov_2025_Frontiers", "Amirzhanov et al. (2025)", AmirzhanovBaseline(fe)),
        ("SinghGupta_2026_KnowledgeInfoSys", "Singh & Gupta (2026)", SinghGuptaBaseline(fe)),
    ]
    results = []
    print(f"Train: {len(ta)} pairs | Test: {len(tea)} pairs\n")
    for mname, desc, detector in baselines:
        print(f"--- {mname} on {dname} ---")
        print(f"  Configuration: {desc}", flush=True)
        try:
            t0 = time.time()
            detector.fit(ta, tb, ty)
            fit_t = time.time() - t0
            t0 = time.time()
            y_pred, y_score = detector.predict(tea, teb)
            pred_t = time.time() - t0
            n = len(tey)
            latency = (pred_t / n) * 1000.0
            m = compute_metrics(tey, y_pred, y_score, latency)
            m.update({"FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4),
                      "NPairs": int(n), "N_Features": 8})
            row = {"Dataset": dname, "Model": mname, "Description": desc,
                   "Config": "BASELINE", "N_Test_Pairs": m.pop("NPairs")}
            row.update(m)
            results.append(row)
            print(f"  Acc={m['Accuracy']}  Prec={m['Precision']}  "
                  f"Rec={m['Recall']}  F1={m['F1-Score']}  "
                  f"MCC={m['MCC']}  ROC={m['ROC-AUC']}\n", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
    return results
```

---

### Cell 13 — Execute all models on both datasets

```python
all_results = []

# ---- Dataset 1 ----
print("=" * 70)
print("DATASET 1: Text Plagiarism Benchmark")
print("=" * 70)
fe1 = ProgressiveFeatureExtractor(max_features=5000, n_components=25)
fe1.fit(ta1, tb1)

# Progressive models
r1 = run_all_models_on_dataset("Dataset1_repo_text_benchmark", fe1, ta1, tb1, ty1, tea1, teb1, tey1)
all_results.extend(r1)

# Baselines
print("\n--- Baselines on Dataset 1 ---")
r1b = run_baselines_on_dataset("Dataset1_repo_text_benchmark", fe1, ta1, tb1, ty1, tea1, teb1, tey1)
all_results.extend(r1b)

# ---- Dataset 2 ----
print()
print("=" * 70)
print("DATASET 2: Quora Question Pairs")
print("=" * 70)
fe2 = ProgressiveFeatureExtractor(max_features=5000, n_components=25)
fe2.fit(ta2, tb2)

# Progressive models
r2 = run_all_models_on_dataset("Dataset2_quora_question_pairs", fe2, ta2, tb2, ty2, tea2, teb2, tey2)
all_results.extend(r2)

# Baselines
print("\n--- Baselines on Dataset 2 ---")
r2b = run_baselines_on_dataset("Dataset2_quora_question_pairs", fe2, ta2, tb2, ty2, tea2, teb2, tey2)
all_results.extend(r2b)

print("\n" + "=" * 70)
print("ALL MODELS COMPLETED")
print("=" * 70)
```

---

### Cell 14 — Display and save quantitative results table

```python
results_df = pd.DataFrame(all_results)
cols = ["Dataset", "Model", "Description", "Config", "N_Test_Pairs",
        "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
        "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
        "Avg_Latency_ms", "FitTime_sec", "PredTime_sec",
        "Threshold", "N_Features"]
results_df = results_df[cols]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
display(results_df)

# Save results
results_df.to_csv("PROGRESSIVE_QUANTITATIVE_COMPARISON.csv", index=False)
print("\nResults saved to PROGRESSIVE_QUANTITATIVE_COMPARISON.csv")
```

---

### Cell 15 — Ablation analysis (MCC drop per contribution)

```python
def run_ablation_analysis(dname, fe, ta, tb, ty, tea, teb, tey):
    """Full model vs ablated variants (remove one contribution)."""
    full_config = "C1C2C3C4"
    ablations = [
        ("No_C1", "C2C3C4", "Remove C1 (word-overlap counts)"),
        ("No_C2", "C1C3C4", "Remove C2 (char n-grams)"),
        ("No_C3", "C1C2C4", "Remove C3 (edit distance)"),
        ("No_C4", "C1C2C3", "Remove C4 (LSA+QS+Soundex)"),
    ]
    results = []

    # Full model
    print(f"--- {dname}: Full Model (C1+C2+C3+C4) ---")
    metrics_full, _, _ = evaluate_config(fe, full_config, ta, tb, ty, tea, teb, tey)
    full_mcc = metrics_full["MCC"]
    full_acc = metrics_full["Accuracy"]
    full_f1 = metrics_full["F1-Score"]
    print(f"  Full MCC={full_mcc}  Acc={full_acc}  F1={full_f1}  N_Feat={metrics_full['N_Features']}\n")

    # Ablated variants
    for abl_name, abl_config, desc in ablations:
        print(f"--- {dname}: {desc} ---")
        try:
            metrics_abl, _, _ = evaluate_config(fe, abl_config, ta, tb, ty, tea, teb, tey)
            mcc_drop = full_mcc - metrics_abl["MCC"]
            acc_drop = full_acc - metrics_abl["Accuracy"]
            f1_drop = full_f1 - metrics_abl["F1-Score"]
            print(f"  Ablated MCC={metrics_abl['MCC']}  Acc={metrics_abl['Accuracy']}  "
                  f"F1={metrics_abl['F1-Score']}  N_Feat={metrics_abl['N_Features']}")
            print(f"  MCC Drop={mcc_drop:+.6f}  Acc Drop={acc_drop:+.6f}  F1 Drop={f1_drop:+.6f}\n")
            results.append({
                "Dataset": dname, "Variant": abl_name, "Removed": abl_name.replace("No_", "C"),
                "Full_MCC": full_mcc, "Ablated_MCC": metrics_abl["MCC"],
                "MCC_Drop": round(mcc_drop, 6),
                "Full_Acc": full_acc, "Ablated_Acc": metrics_abl["Accuracy"],
                "Acc_Drop": round(acc_drop, 6),
                "Full_F1": full_f1, "Ablated_F1": metrics_abl["F1-Score"],
                "F1_Drop": round(f1_drop, 6),
                "N_Features": metrics_abl["N_Features"],
            })
        except Exception as e:
            print(f"  ERROR: {e}\n")
    return results

# Run ablation on both datasets
ablation_results = []
print("=" * 70)
print("ABLATION ANALYSIS — MCC Drop When Each Contribution Removed")
print("=" * 70)

print("\n--- DATASET 1 ---")
r_abl1 = run_ablation_analysis("Dataset1_repo_text_benchmark", fe1, ta1, tb1, ty1, tea1, teb1, tey1)
ablation_results.extend(r_abl1)

print("\n--- DATASET 2 ---")
r_abl2 = run_ablation_analysis("Dataset2_quora_question_pairs", fe2, ta2, tb2, ty2, tea2, teb2, tey2)
ablation_results.extend(r_abl2)

abl_df = pd.DataFrame(ablation_results)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
display(abl_df)

# Save ablation results
abl_df.to_csv("PROGRESSIVE_ABLATION_RESULTS.csv", index=False)
print("\nAblation results saved to PROGRESSIVE_ABLATION_RESULTS.csv")
```

---

### Cell 16 — Confusion Matrix plots (all models × both datasets)

```python
def plot_confusion_matrices(dname, fe, ta, tb, ty, tea, teb, tey):
    """Plot confusion matrices for all 5 progressive models on a dataset."""
    configs = [
        ("Model1_Baseline",   "Sajid et al. (2025) baseline",   "BASELINE"),
        ("Model2_Baseline_C1","Baseline + C1 (word-overlap)",   "C1"),
        ("Model3_Baseline_C1_C2","Baseline + C1 + C2",           "C1C2"),
        ("Model4_Baseline_C1_C2_C3","Baseline + C1 + C2 + C3",   "C1C2C3"),
        ("Model5_Baseline_C1_C2_C3_C4","Baseline + C1 + C2 + C3 + C4", "C1C2C3C4"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"Confusion Matrices — {dname}", fontsize=16, fontweight='bold')

    for idx, (mname, desc, config) in enumerate(configs):
        metrics, y_pred, y_proba = evaluate_config(fe, config, ta, tb, ty, tea, teb, tey)
        cm = confusion_matrix(tey, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Plagiarized', 'Plagiarized'],
                    yticklabels=['Not Plagiarized', 'Plagiarized'],
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{mname}\nMCC={metrics['MCC']:.3f}", fontsize=10)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    fname = f"confusion_matrices_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")

print("=" * 70)
print("CONFUSION MATRICES — All Models × Both Datasets")
print("=" * 70)

print("\n--- Dataset 1 ---")
plot_confusion_matrices("Dataset1_repo_text_benchmark", fe1, ta1, tb1, ty1, tea1, teb1, tey1)

print("\n--- Dataset 2 ---")
plot_confusion_matrices("Dataset2_quora_question_pairs", fe2, ta2, tb2, ty2, tea2, teb2, tey2)
```

---

### Cell 17 — ROC Curve plots (all models × both datasets)

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(dname, fe, ta, tb, ty, tea, teb, tey):
    """Plot ROC curves for all 5 progressive models on a dataset."""
    configs = [
        ("Model1_Baseline",   "Sajid et al. (2025) baseline",   "BASELINE"),
        ("Model2_Baseline_C1","Baseline + C1 (word-overlap)",   "C1"),
        ("Model3_Baseline_C1_C2","Baseline + C1 + C2",           "C1C2"),
        ("Model4_Baseline_C1_C2_C3","Baseline + C1 + C2 + C3",   "C1C2C3"),
        ("Model5_Baseline_C1_C2_C3_C4","Baseline + C1 + C2 + C3 + C4", "C1C2C3C4"),
    ]
    plt.figure(figsize=(10, 8))
    for mname, desc, config in configs:
        metrics, y_pred, y_proba = evaluate_config(fe, config, ta, tb, ty, tea, teb, tey)
        fpr, tpr, _ = roc_curve(tey, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{mname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curves — {dname}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curves_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")

print("=" * 70)
print("ROC CURVES — All Models × Both Datasets")
print("=" * 70)

print("\n--- Dataset 1 ---")
plot_roc_curves("Dataset1_repo_text_benchmark", fe1, ta1, tb1, ty1, tea1, teb1, tey1)

print("\n--- Dataset 2 ---")
plot_roc_curves("Dataset2_quora_question_pairs", fe2, ta2, tb2, ty2, tea2, teb2, tey2)
```

---

### Cell 18 — Qualitative analysis output

```python
def print_qualitative_analysis(results_df, ablation_df):
    """Print qualitative observations per model per dataset."""
    print("=" * 70)
    print("QUALITATIVE ANALYSIS — Precision/Recall Trade-offs per Model")
    print("=" * 70)
    for dname in results_df["Dataset"].unique():
        sub = results_df[results_df["Dataset"] == dname]
        print(f"\n{'='*70}")
        print(f"DATASET: {dname}")
        print(f"{'='*70}")
        for _, row in sub.iterrows():
            mname = row["Model"]
            acc = row["Accuracy"]
            prec = row["Precision"]
            rec = row["Recall"]
            f1 = row["F1-Score"]
            mcc = row["MCC"]
            n_feat = row["N_Features"]
            if prec > rec + 0.05:
                profile = "precision-oriented (conservative)"
            elif rec > prec + 0.05:
                profile = "recall-oriented (liberal)"
            else:
                profile = "balanced precision/recall"
            print(f"\n  {mname} ({row['Description']})")
            print(f"    Features: {n_feat}  |  Acc={acc}  Prec={prec}  Rec={rec}  "
                  f"F1={f1}  MCC={mcc}")
            print(f"    Profile: {profile}")
    print(f"\n{'='*70}")
    print("ABLATION INSIGHTS — Contribution Importance")
    print(f"{'='*70}")
    for dname in ablation_df["Dataset"].unique():
        sub = ablation_df[ablation_df["Dataset"] == dname]
        print(f"\n  {dname}:")
        for _, row in sub.iterrows():
            removed = row["Removed"]
            drop = row["MCC_Drop"]
            sig = "HIGH" if abs(drop) >= 0.05 else ("MODERATE" if abs(drop) >= 0.01 else "LOW")
            direction = "drop" if drop > 0 else "gain"
            print(f"    Removing {removed}: MCC {direction} of {abs(drop):.6f} ({sig} impact)")

print_qualitative_analysis(results_df, abl_df)
```

---

### Cell 19 — Cost analysis (MCC per ms, pre-filtering efficiency)

```python
def print_cost_analysis(results_df):
    """Print cost efficiency analysis."""
    print("=" * 70)
    print("COST ANALYSIS — Pre-Filtering Efficiency")
    print("=" * 70)
    for dname in results_df["Dataset"].unique():
        sub = results_df[results_df["Dataset"] == dname]
        print(f"\n  {dname}:")
        for _, row in sub.iterrows():
            mcc = row["MCC"]
            latency = row["Avg_Latency_ms"]
            mcc_per_ms = mcc / latency if latency > 0 else 0
            print(f"    {row['Model']}: Latency={latency:.4f} ms/pair  "
                  f"MCC={mcc}  MCC/ms={mcc_per_ms:.3f}")
    print("\nPre-filtering insight: Model 1 (baseline, 8 features) has the lowest")
    print("latency. It can serve as an inexpensive pre-filter to catch obvious")
    print("non-plagiarized pairs, while Model 5 (83 features) verifies ambiguous")
    print("cases. This cascade design reduces average computational cost per")
    print("submission while maintaining high overall accuracy.")

print_cost_analysis(results_df)
```

---

### Cell 20 — Save all outputs

```python
import os
os.makedirs("kaggle_outputs", exist_ok=True)

# Save results CSV
results_df.to_csv("kaggle_outputs/PROGRESSIVE_QUANTITATIVE_COMPARISON.csv", index=False)
abl_df.to_csv("kaggle_outputs/PROGRESSIVE_ABLATION_RESULTS.csv", index=False)

print("All outputs saved to kaggle_outputs/")
print("Files:")
for f in os.listdir("kaggle_outputs"):
    print(f"  - {f}")
```

---

### Cell 20b — Comprehensive quantitative analysis

```python
def print_comprehensive_quantitative_analysis(results_df, abl_df):
    """
    Comprehensive quantitative analysis:
    - Per-dataset results summary
    - Model progression (MCC/Accuracy across Models 1-5)
    - Contribution impact (ablation summary)
    - Baseline comparison (progressive models vs single-paper baselines)
    - Best model per dataset
    - Feature count vs performance trade-off
    """
    print("=" * 70)
    print("COMPREHENSIVE QUANTITATIVE ANALYSIS")
    print("=" * 70)

    # ---- 1. Per-dataset results summary ----
    print("\n" + "-" * 70)
    print("1. RESULTS SUMMARY — All Models × Both Datasets")
    print("-" * 70)
    for dname in results_df["Dataset"].unique():
        sub = results_df[results_df["Dataset"] == dname]
        print(f"\n  DATASET: {dname}")
        print(f"    {'Model':<35} {'Config':<12} {'MCC':>8} {'Acc':>8} {'F1':>8} {'ROC':>8} {'Feat':>6}")
        print(f"    {'-'*35} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for _, row in sub.iterrows():
            print(f"    {row['Model']:<35} {row['Config']:<12} "
                  f"{row['MCC']:>8} {row['Accuracy']:>8} {row['F1-Score']:>8} "
                  f"{row['ROC-AUC']:>8} {row['N_Features']:>6}")

    # ---- 2. Model progression (progressive models only) ----
    print("\n" + "-" * 70)
    print("2. MODEL PROGRESSION — MCC & Accuracy Across Models 1-5")
    print("-" * 70)
    prog_models = ["Model1_Baseline", "Model2_Baseline_C1",
                   "Model3_Baseline_C1_C2", "Model4_Baseline_C1_C2_C3",
                   "Model5_Baseline_C1_C2_C3_C4"]
    for dname in results_df["Dataset"].unique():
        sub = results_df[(results_df["Dataset"] == dname) &
                         (results_df["Model"].isin(prog_models))]
        print(f"\n  DATASET: {dname}")
        prev_mcc = None
        for _, row in sub.iterrows():
            mcc = row["MCC"]
            acc = row["Accuracy"]
            delta = ""
            if prev_mcc is not None:
                delta = f" (ΔMCC={mcc - prev_mcc:+.4f})"
            print(f"    {row['Model']:<35} MCC={mcc}  Acc={acc}{delta}")
            prev_mcc = mcc

    # ---- 3. Contribution impact (ablation) ----
    print("\n" + "-" * 70)
    print("3. CONTRIBUTION IMPACT — Ablation MCC Drop")
    print("-" * 70)
    for dname in abl_df["Dataset"].unique():
        sub = abl_df[abl_df["Dataset"] == dname]
        print(f"\n  DATASET: {dname}")
        full_mcc = sub["Full_MCC"].iloc[0]
        print(f"    Full Model (C1+C2+C3+C4) MCC: {full_mcc}")
        for _, row in sub.iterrows():
            removed = row["Removed"]
            drop = row["MCC_Drop"]
            sig = "HIGH (≥0.05)" if abs(drop) >= 0.05 else (
                "MODERATE (≥0.01)" if abs(drop) >= 0.01 else "LOW")
            print(f"    Remove {removed}: MCC drop = {drop:+.6f} ({sig})")

    # ---- 4. Baseline comparison ----
    print("\n" + "-" * 70)
    print("4. BASELINE COMPARISON — Progressive vs Single-Paper Baselines")
    print("-" * 70)
    baseline_models = ["Sajid_2025_PLoS", "Amirzhanov_2025_Frontiers",
                       "SinghGupta_2026_KnowledgeInfoSys"]
    for dname in results_df["Dataset"].unique():
        sub = results_df[(results_df["Dataset"] == dname) &
                         (results_df["Model"].isin(baseline_models))]
        prog_sub = results_df[(results_df["Dataset"] == dname) &
                              (results_df["Model"] == "Model5_Baseline_C1_C2_C3_C4")]
        print(f"\n  DATASET: {dname}")
        print(f"    {'Baseline':<35} {'MCC':>8} {'Acc':>8} {'Feat':>6}")
        print(f"    {'-'*35} {'-'*8} {'-'*8} {'-'*6}")
        for _, row in sub.iterrows():
            print(f"    {row['Model']:<35} {row['MCC']:>8} {row['Accuracy']:>8} "
                  f"{row['N_Features']:>6}")
        if len(prog_sub) > 0:
            p = prog_sub.iloc[0]
            print(f"    {'Model 5 (Full Progressive)':<35} {p['MCC']:>8} "
                  f"{p['Accuracy']:>8} {p['N_Features']:>6}")

    # ---- 5. Best model per dataset ----
    print("\n" + "-" * 70)
    print("5. BEST MODEL PER DATASET (by MCC)")
    print("-" * 70)
    for dname in results_df["Dataset"].unique():
        sub = results_df[results_df["Dataset"] == dname]
        best = sub.loc[sub["MCC"].idxmax()]
        print(f"    {dname}: {best['Model']} — MCC={best['MCC']}  "
              f"Acc={best['Accuracy']}  Features={best['N_Features']}")

    # ---- 6. Feature count vs performance ----
    print("\n" + "-" * 70)
    print("6. FEATURE COUNT vs PERFORMANCE TRADE-OFF")
    print("-" * 70)
    for dname in results_df["Dataset"].unique():
        sub = results_df[(results_df["Dataset"] == dname) &
                         (results_df["Model"].isin(prog_models))]
        print(f"\n  DATASET: {dname}")
        for _, row in sub.iterrows():
            mcc_per_feat = row["MCC"] / row["N_Features"] if row["N_Features"] > 0 else 0
            print(f"    {row['Model']:<35} Features={row['N_Features']:>4}  "
                  f"MCC={row['MCC']}  MCC/Feature={mcc_per_feat:.4f}")

    print("\n" + "=" * 70)
    print("END OF COMPREHENSIVE QUANTITATIVE ANALYSIS")
    print("=" * 70)

# Run comprehensive quantitative analysis
print_comprehensive_quantitative_analysis(results_df, abl_df)

# Save detailed analysis to CSV
quant_analysis_rows = []
for dname in results_df["Dataset"].unique():
    sub = results_df[results_df["Dataset"] == dname]
    for _, row in sub.iterrows():
        quant_analysis_rows.append({
            "Dataset": dname,
            "Model": row["Model"],
            "Config": row["Config"],
            "MCC": row["MCC"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1-Score": row["F1-Score"],
            "ROC-AUC": row["ROC-AUC"],
            "N_Features": row["N_Features"],
            "Avg_Latency_ms": row["Avg_Latency_ms"],
        })
quant_df = pd.DataFrame(quant_analysis_rows)
quant_df.to_csv("QUANTITATIVE_ANALYSIS.csv", index=False)
print("\nQuantitative analysis saved to QUANTITATIVE_ANALYSIS.csv")
```

---

## SECTION C — SINGLE-PAPER IMPLEMENTATIONS (Cells 21-25)

Standalone self-contained implementations of each paper, evaluated independently on both datasets.

### Cell 21 — Paper 1: Sajid et al. (2025) — Standalone implementation

```python
class SajidStandalone:
    """
    Sajid et al. (2025) — PLoS ONE baseline (self-contained).
    8 features: word Jaccard, char trigram Jaccard, overlap count,
    length diff ratio, shorter overlap, exact match, TF-IDF cosine,
    LSA cosine. LogisticRegression classifier.
    """
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=5000)
        self.svd = TruncatedSVD(n_components=25, random_state=RANDOM_STATE)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=RANDOM_STATE)

    def _compute_features(self, ta, tb):
        n = len(ta)
        ca = [clean_text(t) for t in ta]
        cb = [clean_text(t) for t in tb]
        ws_a = [word_set(t) for t in ca]
        ws_b = [word_set(t) for t in cb]
        cs_a = [char_ngrams(t, 3) for t in ca]
        cs_b = [char_ngrams(t, 3) for t in cb]

        va = self.vec.transform(ca)
        vb = self.vec.transform(cb)
        tfidf_cos = cosine_batch(va, vb)

        va_lsa = np.asarray(self.svd.transform(va))
        vb_lsa = np.asarray(self.svd.transform(vb))
        va_lsa_n = va_lsa / np.maximum(np.linalg.norm(va_lsa, axis=1, keepdims=True), 1e-12)
        vb_lsa_n = vb_lsa / np.maximum(np.linalg.norm(vb_lsa, axis=1, keepdims=True), 1e-12)
        lsa_cos = np.diag(va_lsa_n @ vb_lsa_n.T)

        jaccard_word = np.array([len(ws_a[i] & ws_b[i]) / len(ws_a[i] | ws_b[i])
                                  if ws_a[i] | ws_b[i] else 0.0 for i in range(n)])
        jaccard_char = np.array([len(cs_a[i] & cs_b[i]) / len(cs_a[i] | cs_b[i])
                                  if cs_a[i] | cs_b[i] else 0.0 for i in range(n)])
        overlap_count = np.array([float(len(ws_a[i] & ws_b[i])) for i in range(n)])
        ldr = np.array([abs(len(ws_a[i]) - len(ws_b[i])) /
                         max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        so = np.array([len(ws_a[i] & ws_b[i]) /
                        min(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
        em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])

        return np.column_stack([jaccard_word, jaccard_char, overlap_count,
                                ldr, so, em, tfidf_cos, lsa_cos])

    def fit(self, ta, tb, y):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)
        self.svd.fit(self.vec.transform(all_t))

        X = self._compute_features(ta, tb)
        self.clf.fit(X, y)
        proba = self.clf.predict_proba(X)[:, 1]
        self.threshold, _ = optimize_threshold(proba, y, metric="mcc")

    def predict(self, ta, tb):
        X = self._compute_features(ta, tb)
        proba = self.clf.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int), proba

def run_sajid(dname, ta, tb, ty, tea, teb, tey):
    print(f"=== Sajid et al. (2025) — {dname} ===")
    print(f"Train: {len(ta)} pairs | Test: {len(tea)} pairs\n")

    model = SajidStandalone()
    t0 = time.time()
    model.fit(ta, tb, ty)
    fit_t = time.time() - t0

    t0 = time.time()
    y_pred, y_score = model.predict(tea, teb)
    pred_t = time.time() - t0

    n = len(tey)
    latency = (pred_t / n) * 1000.0
    m = compute_metrics(tey, y_pred, y_score, latency)
    m.update({"FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4),
              "NPairs": int(n), "N_Features": 8})

    print(f"Accuracy={m['Accuracy']}  Precision={m['Precision']}  "
          f"Recall={m['Recall']}  F1={m['F1-Score']}  MCC={m['MCC']}  "
          f"ROC={m['ROC-AUC']}  N_Feat={m['N_Features']}  Thr={m['Threshold']}\n")
    return m

# Run Sajid on both datasets
sajid_d1 = run_sajid("Dataset1_repo_text_benchmark", ta1, tb1, ty1, tea1, teb1, tey1)
sajid_d2 = run_sajid("Dataset2_quora_question_pairs", ta2, tb2, ty2, tea2, teb2, tey2)
```

---

### Cell 22 — Paper 2: Amirzhanov et al. (2025) — Standalone implementation

```python
class AmirzhanovStandalone:
    """
    Amirzhanov et al. (2025) — Frontiers in Computer Science baseline (self-contained).
    TF-IDF cosine similarity threshold classifier.
    No training required — uses a fixed threshold (optimized on validation).
    """
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=5000)

    def fit(self, ta, tb, y=None):
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)

        if y is not None:
            sims = cosine_batch(
                self.vec.transform([clean_text(t) for t in ta]),
                self.vec.transform([clean_text(t) for t in tb])
            )
            self.threshold, _ = optimize_threshold(sims, y, metric="mcc")

    def predict(self, ta, tb):
        va = self.vec.transform([clean_text(t) for t in ta])
        vb = self.vec.transform([clean_text(t) for t in tb])
        sims = cosine_batch(va, vb)
        return (sims >= self.threshold).astype(int), sims

def run_amirzhanov(dname, ta, tb, ty, tea, teb, tey):
    print(f"=== Amirzhanov et al. (2025) — {dname} ===")
    print(f"Train: {len(ta)} pairs | Test: {len(tea)} pairs\n")

    model = AmirzhanovStandalone()
    t0 = time.time()
    model.fit(ta, tb, ty)
    fit_t = time.time() - t0

    t0 = time.time()
    y_pred, y_score = model.predict(tea, teb)
    pred_t = time.time() - t0

    n = len(tey)
    latency = (pred_t / n) * 1000.0
    m = compute_metrics(tey, y_pred, y_score, latency)
    m.update({"FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4),
              "NPairs": int(n), "N_Features": 1})

    print(f"Accuracy={m['Accuracy']}  Precision={m['Precision']}  "
          f"Recall={m['Recall']}  F1={m['F1-Score']}  MCC={m['MCC']}  "
          f"ROC={m['ROC-AUC']}  N_Feat={m['N_Features']}  Thr={m['Threshold']}\n")
    return m

# Run Amirzhanov on both datasets
amir_d1 = run_amirzhanov("Dataset1_repo_text_benchmark", ta1, tb1, ty1, tea1, teb1, tey1)
amir_d2 = run_amirzhanov("Dataset2_quora_question_pairs", ta2, tb2, ty2, tea2, teb2, tey2)
```

---

### Cell 23 — Paper 3: Singh & Gupta (2026) — Standalone implementation

```python
class SinghGuptaStandalone:
    """
    Singh & Gupta (2026) — Knowledge and Information Systems baseline (self-contained).
    Hybrid: alpha * word_Jaccard + (1-alpha) * normalized_overlap.
    """
    def __init__(self, alpha=0.5, threshold=0.50):
        self.alpha = alpha
        self.threshold = threshold

    def fit(self, ta, tb, y=None):
        c1 = _compute_c1_features_standalone(ta, tb)
        jw = c1[:, 0]
        oc = c1[:, 1]
        hybrid = self.alpha * jw + (1 - self.alpha) * np.minimum(oc / 10.0, 1.0)
        if y is not None:
            self.threshold, _ = optimize_threshold(hybrid, y, metric="mcc")

    def predict(self, ta, tb):
        c1 = _compute_c1_features_standalone(ta, tb)
        jw = c1[:, 0]
        oc = c1[:, 1]
        hybrid = self.alpha * jw + (1 - self.alpha) * np.minimum(oc / 10.0, 1.0)
        return (hybrid >= self.threshold).astype(int), hybrid

def _compute_c1_features_standalone(ta, tb):
    """Compute C1 word-overlap count features (used by SinghGupta standalone)."""
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
    ba = [word_bigrams(t) for t in ca]
    bb = [word_bigrams(t) for t in cb]
    bo = np.array([len(ba[i] & bb[i]) / len(ba[i] | bb[i])
                    if ba[i] | bb[i] else 0.0 for i in range(n)])
    em = np.array([1.0 if ca[i] == cb[i] else 0.0 for i in range(n)])
    max_len = np.array([max(len(ws_a[i]), len(ws_b[i]), 1) for i in range(n)])
    oc_norm = oc / max_len
    bc = np.array([len(ba[i] & bb[i]) / max(len(ba[i]), 1) for i in range(n)])
    return np.column_stack([jw, oc, ldr, so, bo, em, oc_norm, bc])

def run_singhgupta(dname, ta, tb, ty, tea, teb, tey):
    print(f"=== Singh & Gupta (2026) — {dname} ===")
    print(f"Train: {len(ta)} pairs | Test: {len(tea)} pairs\n")

    model = SinghGuptaStandalone(alpha=0.5)
    t0 = time.time()
    model.fit(ta, tb, ty)
    fit_t = time.time() - t0

    t0 = time.time()
    y_pred, y_score = model.predict(tea, teb)
    pred_t = time.time() - t0

    n = len(tey)
    latency = (pred_t / n) * 1000.0
    m = compute_metrics(tey, y_pred, y_score, latency)
    m.update({"FitTime_sec": round(fit_t, 4), "PredTime_sec": round(pred_t, 4),
              "NPairs": int(n), "N_Features": 3})

    print(f"Accuracy={m['Accuracy']}  Precision={m['Precision']}  "
          f"Recall={m['Recall']}  F1={m['F1-Score']}  MCC={m['MCC']}  "
          f"ROC={m['ROC-AUC']}  N_Feat={m['N_Features']}  Thr={m['Threshold']}\n")
    return m

# Run SinghGupta on both datasets
sg_d1 = run_singhgupta("Dataset1_repo_text_benchmark", ta1, tb1, ty1, tea1, teb1, tey1)
sg_d2 = run_singhgupta("Dataset2_quora_question_pairs", ta2, tb2, ty2, tea2, teb2, tey2)
```

---

### Cell 24 — Display results table (single papers)

```python
# Collect all single-paper results
sp_results = []

for label, m in [("Sajid_2025_PLoS_D1", sajid_d1),
                 ("Sajid_2025_PLoS_D2", sajid_d2),
                 ("Amirzhanov_2025_D1", amir_d1),
                 ("Amirzhanov_2025_D2", amir_d2),
                 ("SinghGupta_2026_D1", sg_d1),
                 ("SinghGupta_2026_D2", sg_d2)]:
    dataset = "Dataset1_repo_text_benchmark" if "D1" in label else "Dataset2_quora_question_pairs"
    paper = label.replace("_D1", "").replace("_D2", "")
    row = {"Paper": paper, "Dataset": dataset, "Model": label}
    row.update(m)
    sp_results.append(row)

sp_results_df = pd.DataFrame(sp_results)
cols = ["Paper", "Dataset", "Model", "N_Test_Pairs",
        "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
        "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
        "Avg_Latency_ms", "FitTime_sec", "PredTime_sec",
        "Threshold", "N_Features"]
sp_results_df = sp_results_df[cols]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
display(sp_results_df)

# Save results
sp_results_df.to_csv("SINGLE_PAPERS_RESULTS.csv", index=False)
print("\nSingle-paper results saved to SINGLE_PAPERS_RESULTS.csv")
```

---

### Cell 25 — Confusion Matrix — Sajid et al. (2025) × Both Datasets

```python
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm_single(pname, dname, ta, tb, ty, tea, teb, tey, model_class):
    """Plot confusion matrix for a single paper on one dataset."""
    m = model_class()
    m.fit(ta, tb, ty)
    y_pred, y_score = m.predict(tea, teb)
    cm = confusion_matrix(tey, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Plagiarized', 'Plagiarized'],
                yticklabels=['Not Plagiarized', 'Plagiarized'])
    plt.title(f"{pname} — {dname}\nMCC={matthews_corrcoef(tey, y_pred):.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fname = f"confusion_matrix_{pname}_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")
    return y_pred, y_score

print("=" * 70)
print("CONFUSION MATRICES — Sajid et al. (2025) × Both Datasets")
print("=" * 70)

# Sajid on Dataset 1
print("\n--- Sajid — Dataset 1 ---")
sajid_y1, sajid_s1 = plot_cm_single("Sajid_2025_PLoS", "Dataset1_repo_text_benchmark",
                                     ta1, tb1, ty1, tea1, teb1, tey1, SajidStandalone)

# Sajid on Dataset 2
print("\n--- Sajid — Dataset 2 ---")
sajid_y2, sajid_s2 = plot_cm_single("Sajid_2025_PLoS", "Dataset2_quora_question_pairs",
                                     ta2, tb2, ty2, tea2, teb2, tey2, SajidStandalone)
```

---

### Cell 26 — Confusion Matrix — Amirzhanov et al. (2025) × Both Datasets

```python
print("=" * 70)
print("CONFUSION MATRICES — Amirzhanov et al. (2025) × Both Datasets")
print("=" * 70)

# Amirzhanov on Dataset 1
print("\n--- Amirzhanov — Dataset 1 ---")
amir_y1, amir_s1 = plot_cm_single("Amirzhanov_2025", "Dataset1_repo_text_benchmark",
                                   ta1, tb1, ty1, tea1, teb1, tey1, AmirzhanovStandalone)

# Amirzhanov on Dataset 2
print("\n--- Amirzhanov — Dataset 2 ---")
amir_y2, amir_s2 = plot_cm_single("Amirzhanov_2025", "Dataset2_quora_question_pairs",
                                   ta2, tb2, ty2, tea2, teb2, tey2, AmirzhanovStandalone)
```

---

### Cell 27 — Confusion Matrix — Singh & Gupta (2026) × Both Datasets

```python
print("=" * 70)
print("CONFUSION MATRICES — Singh & Gupta (2026) × Both Datasets")
print("=" * 70)

# SinghGupta on Dataset 1
print("\n--- SinghGupta — Dataset 1 ---")
sg_y1, sg_s1 = plot_cm_single("SinghGupta_2026", "Dataset1_repo_text_benchmark",
                               ta1, tb1, ty1, tea1, teb1, tey1, SinghGuptaStandalone)

# SinghGupta on Dataset 2
print("\n--- SinghGupta — Dataset 2 ---")
sg_y2, sg_s2 = plot_cm_single("SinghGupta_2026", "Dataset2_quora_question_pairs",
                               ta2, tb2, ty2, tea2, teb2, tey2, SinghGuptaStandalone)
```

---

### Cell 28 — ROC Curves — All Three Papers × Both Datasets

```python
print("=" * 70)
print("ROC CURVES — All Three Papers × Both Datasets")
print("=" * 70)

roc_results = []

# Sajid ROC curves
for dname, y_true, y_score, pname in [
    ("Dataset1_repo_text_benchmark", tey1, sajid_s1, "Sajid_2025_PLoS"),
    ("Dataset2_quora_question_pairs", tey2, sajid_s2, "Sajid_2025_PLoS"),
]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{pname} — {dname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curve — {pname} — {dname}")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_{pname}_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")
    roc_results.append({"Paper": pname, "Dataset": dname, "AUC": roc_auc})

# Amirzhanov ROC curves
for dname, y_true, y_score, pname in [
    ("Dataset1_repo_text_benchmark", tey1, amir_s1, "Amirzhanov_2025"),
    ("Dataset2_quora_question_pairs", tey2, amir_s2, "Amirzhanov_2025"),
]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{pname} — {dname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curve — {pname} — {dname}")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_{pname}_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")
    roc_results.append({"Paper": pname, "Dataset": dname, "AUC": roc_auc})

# SinghGupta ROC curves
for dname, y_true, y_score, pname in [
    ("Dataset1_repo_text_benchmark", tey1, sg_s1, "SinghGupta_2026"),
    ("Dataset2_quora_question_pairs", tey2, sg_s2, "SinghGupta_2026"),
]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{pname} — {dname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curve — {pname} — {dname}")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_{pname}_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")
    roc_results.append({"Paper": pname, "Dataset": dname, "AUC": roc_auc})

roc_df = pd.DataFrame(roc_results)
display(roc_df)
roc_df.to_csv("SINGLE_PAPERS_ROC.csv", index=False)
print("\nROC results saved to SINGLE_PAPERS_ROC.csv")
```

---

### Cell 29 — Summary comparison of all three papers

```python
print("=" * 70)
print("SUMMARY COMPARISON — All Three Papers × Both Datasets")
print("=" * 70)

summary_rows = []
for label, m in [("Sajid_2025_PLoS_D1", sajid_d1),
                 ("Sajid_2025_PLoS_D2", sajid_d2),
                 ("Amirzhanov_2025_D1", amir_d1),
                 ("Amirzhanov_2025_D2", amir_d2),
                 ("SinghGupta_2026_D1", sg_d1),
                 ("SinghGupta_2026_D2", sg_d2)]:
    dataset = "D1" if "D1" in label else "D2"
    paper = label.replace("_D1", "").replace("_D2", "")
    summary_rows.append({
        "Paper": paper,
        "Dataset": dataset,
        "Accuracy": m["Accuracy"],
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "F1-Score": m["F1-Score"],
        "MCC": m["MCC"],
        "ROC-AUC": m["ROC-AUC"],
        "N_Features": m["N_Features"],
        "Latency_ms": m["Avg_Latency_ms"],
    })
summary_df = pd.DataFrame(summary_rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
display(summary_df)

# Best performer per dataset
print("\nBest MCC per dataset:")
for ds in ["D1", "D2"]:
    sub = summary_df[summary_df["Dataset"] == ds]
    best = sub.loc[sub["MCC"].idxmax()]
    print(f"  {ds}: {best['Paper']} — MCC={best['MCC']} (N_Features={best['N_Features']})")

summary_df.to_csv("SINGLE_PAPERS_SUMMARY.csv", index=False)
print("\nSummary saved to SINGLE_PAPERS_SUMMARY.csv")
```

---

## Cell Index

| Cell | Section | Purpose |
|------|---------|---------|
| 1 | A | Install dependencies |
| 2 | A | Imports and global setup |
| 3 | A | Text preprocessing utilities |
| 4 | A | Levenshtein & Jaro-Winkler similarity (C3) |
| 5 | A | Cosine similarity helpers |
| 6 | A | C4 features: Question structure + Soundex |
| 7 | B | Progressive feature extractor class |
| 8 | B | Metrics, threshold optimization, evaluation |
| 9 | B | Three paper baselines (progressive versions) |
| 10 | B | Load datasets (Kaggle paths) |
| 11 | B | Run all 5 progressive models on a dataset |
| 12 | B | Run baselines (three papers) on a dataset |
| 13 | B | Execute all models on both datasets |
| 14 | B | Display and save quantitative results table |
| 15 | B | Ablation analysis (MCC drop per contribution) |
| 16 | B | Confusion Matrix plots (all models × both datasets) |
| 17 | B | ROC Curve plots (all models × both datasets) |
| 18 | B | Qualitative analysis output |
| 19 | B | Cost analysis (MCC per ms, pre-filtering efficiency) |
| 20 | B | Save all outputs |
| 20b | B | Comprehensive quantitative analysis |
| 21 | C | Paper 1: Sajid et al. (2025) — standalone |
| 22 | C | Paper 2: Amirzhanov et al. (2025) — standalone |
| 23 | C | Paper 3: Singh & Gupta (2026) — standalone |
| 24 | C | Results table (single papers) |
| 25 | C | Confusion Matrix — Sajid et al. (2025) × Both Datasets |
| 26 | C | Confusion Matrix — Amirzhanov et al. (2025) × Both Datasets |
| 27 | C | Confusion Matrix — Singh & Gupta (2026) × Both Datasets |
| 28 | C | ROC Curves — All Three Papers × Both Datasets |
| 29 | C | Summary comparison of all three papers |

---

## Expected results (from the completed study)

### Progressive models

| Dataset | Model | Config | MCC | Accuracy | N_Features |
|---------|-------|--------|-----|----------|------------|
| D1 | Model 1 (Sajid baseline) | BASELINE | 0.3573 | 0.7149 | 8 |
| D1 | Model 2 (+C1) | C1 | 0.3813 | 0.7218 | 16 |
| D1 | Model 3 (+C1+C2) | C1C2 | 0.3993 | 0.7392 | 21 |
| D1 | Model 4 (+C1+C2+C3) | C1C2C3 | 0.4035 | 0.7387 | 25 |
| D1 | Model 5 (+C1+C2+C3+C4) | C1C2C3C4 | 0.4705 | 0.7641 | 83 |
| D2 | Model 1 (Sajid baseline) | BASELINE | 0.1990 | 0.6009 | 8 |
| D2 | Model 2 (+C1) | C1 | 0.3476 | 0.6804 | 16 |
| D2 | Model 3 (+C1+C2) | C1C2 | 0.3516 | 0.6824 | 21 |
| D2 | Model 4 (+C1+C2+C3) | C1C2C3 | 0.3844 | 0.6976 | 25 |
| D2 | Model 5 (+C1+C2+C3+C4) | C1C2C3C4 | 0.3958 | 0.7046 | 83 |

### Single papers (standalone)

| Paper | Dataset | MCC | Accuracy | N_Features |
|-------|---------|-----|----------|------------|
| Sajid 2025 | D1 (9,824 test) | 0.3573 | 0.7149 | 8 |
| Sajid 2025 | D2 (8,000 test) | 0.1990 | 0.6009 | 8 |
| Amirzhanov 2025 | D1 | ~0.30-0.40 | ~0.65-0.75 | 1 |
| Amirzhanov 2025 | D2 | ~0.10-0.20 | ~0.55-0.65 | 1 |
| SinghGupta 2026 | D1 | ~0.35-0.45 | ~0.70-0.80 | 3 |
| SinghGupta 2026 | D2 | ~0.25-0.35 | ~0.60-0.70 | 3 |

**Notes for Kaggle:**
- Your files are at `/kaggle/input/the-ml-data-sets/` — Cell 10 uses these paths
- D2 uses 5,000-pair training subset (from 49,401 available) for speed
- Small differences in the last decimal may occur due to environment/version differences
- Run Cells 16-17 for progressive confusion matrices/ROC; Cell 20b for comprehensive quantitative analysis; Cells 25-27 for single-paper confusion matrices; Cell 28 for single-paper ROC curves
- Results save to CSV files in the working directory and `kaggle_outputs/`