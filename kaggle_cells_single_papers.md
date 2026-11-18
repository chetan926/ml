# Single-Paper Implementations — Kaggle/Colab Code Cells

Three standalone implementations, one per paper, each evaluated on both datasets.
Copy each cell into a separate cell in Kaggle/Colab, run top to bottom.

**Kaggle file paths:** Your uploaded files are at:
- `/kaggle/input/the-ml-data-sets/dataset1_text_plagiarism_benchmark.csv`
- `/kaggle/input/the-ml-data-sets/dataset2_quora_train.tsv`
- `/kaggle/input/the-ml-data-sets/dataset2_quora_test.tsv`

---

## Cell 1 — Install dependencies

```python
# !pip install -q pandas scikit-learn numpy matplotlib seaborn
```

---

## Cell 2 — Imports and global setup

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

## Cell 3 — Text preprocessing utilities

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

def char_ngrams(s, n=3):
    s = clean_text(s)
    return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()
```

---

## Cell 4 — Cosine similarity helpers

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

## Cell 5 — Metrics and threshold optimization

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

def evaluate(y_true, y_pred, y_score, latency_ms):
    return compute_metrics(y_true, y_pred, y_score, latency_ms)
```

---

## Cell 6 — Load datasets (Kaggle paths)

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
        print("  Please upload via Kaggle sidebar (Add Input → Upload)")

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

## Cell 7 — Paper 1: Sajid et al. (2025) — Single Implementation

```python
class SajidBaseline:
    """
    Sajid et al. (2025) — PLoS ONE baseline.
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
        # Fit TF-IDF and SVD on training data
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

    model = SajidBaseline()
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

## Cell 8 — Paper 2: Amirzhanov et al. (2025) — Single Implementation

```python
class AmirzhanovBaseline:
    """
    Amirzhanov et al. (2025) — Frontiers in Computer Science baseline.
    TF-IDF cosine similarity threshold classifier.
    No training required — uses a fixed threshold (optimized on validation).
    """
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                   max_features=5000)

    def fit(self, ta, tb, y=None):
        # Fit TF-IDF on training data (for consistent vector space)
        all_t = [clean_text(t) for t in ta] + [clean_text(t) for t in tb]
        self.vec.fit(all_t)

        if y is not None:
            # Optimize threshold on validation data
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

    model = AmirzhanovBaseline()
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

## Cell 9 — Paper 3: Singh & Gupta (2026) — Single Implementation

```python
class SinghGuptaBaseline:
    """
    Singh & Gupta (2026) — Knowledge and Information Systems baseline.
    Hybrid: alpha * word_Jaccard + (1-alpha) * normalized_overlap.
    """
    def __init__(self, alpha=0.5, threshold=0.50):
        self.alpha = alpha
        self.threshold = threshold

    def fit(self, ta, tb, y=None):
        # Optimize threshold on validation data
        c1 = _compute_c1_features(ta, tb)
        jw = c1[:, 0]
        oc = c1[:, 1]
        hybrid = self.alpha * jw + (1 - self.alpha) * np.minimum(oc / 10.0, 1.0)
        if y is not None:
            self.threshold, _ = optimize_threshold(hybrid, y, metric="mcc")

    def predict(self, ta, tb):
        c1 = _compute_c1_features(ta, tb)
        jw = c1[:, 0]
        oc = c1[:, 1]
        hybrid = self.alpha * jw + (1 - self.alpha) * np.minimum(oc / 10.0, 1.0)
        return (hybrid >= self.threshold).astype(int), hybrid

def _compute_c1_features(ta, tb):
    """Compute C1 word-overlap count features (used by SinghGupta)."""
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

    model = SinghGuptaBaseline(alpha=0.5)
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

## Cell 10 — Display all results table

```python
# Collect all results
all_results = []

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
    all_results.append(row)

results_df = pd.DataFrame(all_results)
cols = ["Paper", "Dataset", "Model", "N_Test_Pairs",
        "Accuracy", "Precision", "Recall", "F1-Score", "Specificity",
        "ROC-AUC", "PR-AUC", "MCC", "TP", "TN", "FP", "FN",
        "Avg_Latency_ms", "FitTime_sec", "PredTime_sec",
        "Threshold", "N_Features"]
results_df = results_df[cols]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
display(results_df)

# Save results
results_df.to_csv("SINGLE_PAPERS_RESULTS.csv", index=False)
print("\nResults saved to SINGLE_PAPERS_RESULTS.csv")
```

---

## Cell 11 — Confusion Matrix plots (single papers)

```python
def plot_confusion_matrix_single(dname, y_true, y_pred, paper_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Plagiarized', 'Plagiarized'],
                yticklabels=['Not Plagiarized', 'Plagiarized'])
    plt.title(f"{paper_name} — {dname}\nMCC={matthews_corrcoef(y_true, y_pred):.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fname = f"confusion_matrix_{paper_name}_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")

# Plot confusion matrices for all three papers on both datasets
print("=" * 70)
print("CONFUSION MATRICES — Single Papers × Both Datasets")
print("=" * 70)

# Re-run predictions to get y_pred for each
models_info = [
    ("Sajid_2025_PLoS", ta1, tb1, ty1, tea1, teb1, tey1, "Dataset1_repo_text_benchmark"),
    ("Sajid_2025_PLoS", ta2, tb2, ty2, tea2, teb2, tey2, "Dataset2_quora_question_pairs"),
    ("Amirzhanov_2025", ta1, tb1, ty1, tea1, teb1, tey1, "Dataset1_repo_text_benchmark"),
    ("Amirzhanov_2025", ta2, tb2, ty2, tea2, teb2, tey2, "Dataset2_quora_question_pairs"),
    ("SinghGupta_2026", ta1, tb1, ty1, tea1, teb1, tey1, "Dataset1_repo_text_benchmark"),
    ("SinghGupta_2026", ta2, tb2, ty2, tea2, teb2, tey2, "Dataset2_quora_question_pairs"),
]

for pname, ta, tb, ty, tea, teb, tey, dname in models_info:
    if pname == "Sajid_2025_PLoS":
        m = SajidBaseline()
        m.fit(ta, tb, ty)
        y_pred, _ = m.predict(tea, teb)
    elif pname == "Amirzhanov_2025":
        m = AmirzhanovBaseline()
        m.fit(ta, tb, ty)
        y_pred, _ = m.predict(tea, teb)
    else:
        m = SinghGuptaBaseline()
        m.fit(ta, tb, ty)
        y_pred, _ = m.predict(tea, teb)
    plot_confusion_matrix_single(dname, tey, y_pred, pname)
```

---

## Cell 12 — ROC Curve plots (single papers)

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_single(dname, y_true, y_score, paper_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{paper_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"ROC Curve — {paper_name} — {dname}")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_{paper_name}_{dname}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")
    return roc_auc

print("=" * 70)
print("ROC CURVES — Single Papers × Both Datasets")
print("=" * 70)

roc_results = []
for pname, ta, tb, ty, tea, teb, tey, dname in models_info:
    if pname == "Sajid_2025_PLoS":
        m = SajidBaseline()
        m.fit(ta, tb, ty)
        _, y_score = m.predict(tea, teb)
    elif pname == "Amirzhanov_2025":
        m = AmirzhanovBaseline()
        m.fit(ta, tb, ty)
        _, y_score = m.predict(tea, teb)
    else:
        m = SinghGuptaBaseline()
        m.fit(ta, tb, ty)
        _, y_score = m.predict(tea, teb)
    auc_val = plot_roc_single(dname, tey, y_score, pname)
    roc_results.append({"Paper": pname, "Dataset": dname, "AUC": auc_val})

roc_df = pd.DataFrame(roc_results)
display(roc_df)
roc_df.to_csv("SINGLE_PAPERS_ROC.csv", index=False)
print("\nROC results saved to SINGLE_PAPERS_ROC.csv")
```

---

## Cell 13 — Summary comparison of all three papers

```python
print("=" * 70)
print("SUMMARY COMPARISON — All Three Papers × Both Datasets")
print("=" * 70)

# Create comparison table
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

## Total cell count: 13 cells

| Cell | Purpose |
|------|---------|
| 1 | Install dependencies |
| 2 | Imports and global setup |
| 3 | Text preprocessing utilities |
| 4 | Cosine similarity helpers |
| 5 | Metrics and threshold optimization |
| 6 | Load datasets (Kaggle paths) |
| 7 | **Paper 1: Sajid et al. (2025)** — single implementation |
| 8 | **Paper 2: Amirzhanov et al. (2025)** — single implementation |
| 9 | **Paper 3: Singh & Gupta (2026)** — single implementation |
| 10 | Display and save results table |
| 11 | Confusion Matrix plots |
| 12 | ROC Curve plots |
| 13 | Summary comparison |

---

## Expected results (from the completed study)

| Paper | Dataset | MCC | Accuracy | N_Features |
|-------|---------|-----|----------|------------|
| Sajid 2025 | D1 (9,824 test) | 0.3573 | 0.7149 | 8 |
| Sajid 2025 | D2 (8,000 test) | 0.1990 | 0.6009 | 8 |
| Amirzhanov 2025 | D1 | ~0.30-0.40 | ~0.65-0.75 | 1 |
| Amirzhanov 2025 | D2 | ~0.10-0.20 | ~0.55-0.65 | 1 |
| SinghGupta 2026 | D1 | ~0.35-0.45 | ~0.70-0.80 | 3 |
| SinghGupta 2026 | D2 | ~0.25-0.35 | ~0.60-0.70 | 3 |

**Notes:**
- D2 uses 5,000-pair training subset for speed
- Actual values may vary slightly due to environment/version differences
- Run Cells 11–12 to generate confusion matrix and ROC curve PNG files
- Results save to CSV files in the working directory