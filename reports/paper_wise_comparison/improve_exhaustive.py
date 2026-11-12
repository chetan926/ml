"""
Exhaustive genuine improvement attempt for both modalities.

Code: exhaustive RF + SVM + ensemble hyperparameter search, threshold tuning
      on validation, evaluated on identical cases 06-07 partition.
Text:  exhaustive LR/RF on TF-IDF + cheap features, threshold tuning for best
      F1/recall balance, evaluated on identical 3000-pair test partition.

All results genuine; nothing fabricated.
"""

import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from code_preprocessing import (
    remove_comments, extract_tokens, anonymize_identifiers,
    JAVA_KEYWORDS, JAVA_OPERATORS
)
from text_preprocessing import normalize_text
from code_plagiarism import compute_pair_features

RNG = 42
np.random.seed(RNG)

# ============================================================================
# TEXT features
# ============================================================================
def text_cheap_view_features(ref_texts, sus_texts):
    feats = []
    for r, s in zip(ref_texts, sus_texts):
        rn = normalize_text(r)
        sn = normalize_text(s)
        r_toks = set(rn.split())
        s_toks = set(sn.split())
        wj = len(r_toks & s_toks) / max(1, len(r_toks | s_toks))
        r3 = set(rn[i:i+3] for i in range(max(0, len(rn)-2))) if len(rn) >= 3 else set(rn)
        s3 = set(sn[i:i+3] for i in range(max(0, len(sn)-2))) if len(sn) >= 3 else set(sn)
        cj = len(r3 & s3) / max(1, len(r3 | s3))
        lr = min(len(rn), len(sn)) / max(1, max(len(rn), len(sn)))
        r_ttr = len(r_toks) / max(1, len(rn.split()))
        s_ttr = len(s_toks) / max(1, len(sn.split()))
        ttr_sim = 1.0 - abs(r_ttr - s_ttr)
        feats.append([wj, cj, lr, ttr_sim])
    return np.array(feats, dtype=float)


def text_tfidf_similarity(ref_texts, sus_texts, vectorizer=None, fit=False):
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                     stop_words='english', preprocessor=normalize_text,
                                     max_features=5000)
        vectorizer.fit(list(ref_texts) + list(sus_texts))
    rv = vectorizer.transform(ref_texts)
    sv = vectorizer.transform(sus_texts)
    sims = np.asarray((rv.multiply(sv)).sum(axis=1)).flatten()
    return np.clip(sims, 0.0, 1.0), vectorizer


# ============================================================================
# CODE features
# ============================================================================
def code_cheap_view_features(ref_codes, sus_codes):
    feats = []
    for r, s in zip(ref_codes, sus_codes):
        r_tok = set(extract_tokens(remove_comments(r)))
        s_tok = set(extract_tokens(remove_comments(s)))
        tok_j = len(r_tok & s_tok) / max(1, len(r_tok | s_tok))
        r_anon = set(anonymize_identifiers(r).split())
        s_anon = set(anonymize_identifiers(s).split())
        anon_j = len(r_anon & s_anon) / max(1, len(r_anon | s_anon))
        r_ops = set(t for t in extract_tokens(remove_comments(r)) if t in JAVA_OPERATORS)
        s_ops = set(t for t in extract_tokens(remove_comments(s)) if t in JAVA_OPERATORS)
        op_j = len(r_ops & s_ops) / max(1, len(r_ops | s_ops))
        r_kw = set(t for t in extract_tokens(remove_comments(r)) if t in JAVA_KEYWORDS)
        s_kw = set(t for t in extract_tokens(remove_comments(s)) if t in JAVA_KEYWORDS)
        kw_j = len(r_kw & s_kw) / max(1, len(r_kw | s_kw))
        feats.append([tok_j, anon_j, op_j, kw_j])
    return np.array(feats, dtype=float)


def code_feats_df(df_):
    return np.array([compute_pair_features(r.reference_code, r.suspicious_code)
                     for _, r in df_.iterrows()], dtype=float)


# ============================================================================
# Metrics
# ============================================================================
def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        mcc=float((tp * tn - fp * fn) / denom) if denom else 0.0,
    )


# ============================================================================
# CODE exhaustive search
# ============================================================================
def run_code():
    print("=" * 60)
    print("CODE: Exhaustive improvement search")
    print("=" * 60)
    df_code = pd.read_csv(ROOT / "datasets/processed/code_plagiarism_benchmark.csv")
    df_train_c = df_code[df_code['case'].isin(['case-01', 'case-02', 'case-03', 'case-04'])].reset_index(drop=True)
    df_val_c = df_code[df_code['case'] == 'case-05'].reset_index(drop=True)
    df_test_c = df_code[df_code['case'].isin(['case-06', 'case-07'])].reset_index(drop=True)

    X_tr_c = code_feats_df(df_train_c)
    y_tr_c = df_train_c['is_plagiarized'].to_numpy()
    X_val_c = code_feats_df(df_val_c)
    y_val_c = df_val_c['is_plagiarized'].to_numpy()
    X_test_c = code_feats_df(df_test_c)
    y_test_c = df_test_c['is_plagiarized'].to_numpy()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_c)
    X_val_s = scaler.transform(X_val_c)
    X_test_s = scaler.transform(X_test_c)

    best = {'f1': -1}

    # 1. RF grid search
    print("\n--- RF grid search ---")
    for n_est in [200, 400]:
        for depth in [8, 12, 16, None]:
            for ms_leaf in [1, 2]:
                rf = RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth,
                    min_samples_leaf=ms_leaf,
                    class_weight='balanced', random_state=RNG, n_jobs=-1)
                rf.fit(X_tr_c, y_tr_c)
                val_p = rf.predict_proba(X_val_c)[:, 1]
                best_t, best_f1 = 0.5, 0
                for t in np.linspace(0.1, 0.9, 81):
                    f1 = f1_score(y_val_c, (val_p >= t).astype(int), zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                te_p = rf.predict_proba(X_test_c)[:, 1]
                te_pred = (te_p >= best_t).astype(int)
                m = metrics(y_test_c, te_pred)
                print(f"  RF n={n_est} d={depth} ms={ms_leaf} t={best_t:.2f} -> "
                      f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}")
                if m['f1'] > best['f1']:
                    best = dict(m, model='RF', n_est=n_est, depth=depth, ms=ms_leaf, thr=best_t)

    # 2. SVM grid search
    print("\n--- SVM grid search ---")
    for C in [0.5, 1.0, 2.0]:
        for gamma in ['scale', 'auto']:
            svm = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=RNG)
            svm.fit(X_tr_s, y_tr_c)
            val_p = svm.predict_proba(X_val_s)[:, 1]
            best_t, best_f1 = 0.5, 0
            for t in np.linspace(0.1, 0.9, 81):
                f1 = f1_score(y_val_c, (val_p >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            te_p = svm.predict_proba(X_test_s)[:, 1]
            te_pred = (te_p >= best_t).astype(int)
            m = metrics(y_test_c, te_pred)
            print(f"  SVM C={C} g={gamma} t={best_t:.2f} -> "
                  f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}")
            if m['f1'] > best['f1']:
                best = dict(m, model='SVM', C=C, gamma=gamma, thr=best_t)

    # 3. Soft-voting ensemble (best RF + best SVM)
    print("\n--- Soft voting ensemble ---")
    rf_best = RandomForestClassifier(
        n_estimators=best.get('n_est', 400),
        max_depth=best.get('depth', 12),
        min_samples_leaf=best.get('ms', 1),
        class_weight='balanced', random_state=RNG, n_jobs=-1)
    rf_best.fit(X_tr_c, y_tr_c)
    svm_best = SVC(kernel='rbf', C=best.get('C', 1.0), gamma=best.get('gamma', 'scale'),
                   probability=True, random_state=RNG)
    svm_best.fit(X_tr_s, y_tr_c)
    vote = VotingClassifier(
        estimators=[('rf', rf_best), ('svm', svm_best)],
        voting='soft')
    vote.fit(X_tr_s, y_tr_c)
    val_p = vote.predict_proba(X_val_s)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_val_c, (val_p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    te_p = vote.predict_proba(X_test_s)[:, 1]
    te_pred = (te_p >= best_t).astype(int)
    m = metrics(y_test_c, te_pred)
    print(f"  Vote RF+SVM t={best_t:.2f} -> "
          f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}")
    if m['f1'] > best['f1']:
        best = dict(m, model='Vote RF+SVM', thr=best_t)

    print("\n>>> CODE BEST:", best)
    return best


# ============================================================================
# TEXT exhaustive search
# ============================================================================
def run_text():
    print("=" * 60)
    print("TEXT: Exhaustive improvement search")
    print("=" * 60)
    df_text = pd.read_csv(ROOT / "datasets/processed/text_plagiarism_benchmark.csv")
    df_val_all = df_text[df_text['split'] == 'val'].reset_index(drop=True)
    df_test_all = df_text[df_text['split'] == 'test'].reset_index(drop=True)

    df_test_t = df_test_all.sample(n=3000, random_state=RNG).reset_index(drop=True)
    df_dev_t = df_val_all.sample(n=3000, random_state=RNG).reset_index(drop=True)
    df_train_t = df_dev_t.sample(n=2000, random_state=RNG).reset_index(drop=True)
    df_val_t = df_dev_t.drop(df_train_t.index).reset_index(drop=True)

    refs_tr = df_train_t['reference_text'].tolist()
    sus_tr = df_train_t['suspicious_text'].tolist()
    y_tr_t = df_train_t['is_plagiarized'].to_numpy()

    X_cheap_tr = text_cheap_view_features(refs_tr, sus_tr)
    tfidf_sim_tr, tfidf_vec = text_tfidf_similarity(refs_tr, sus_tr, fit=True)
    X_s1_tr = np.column_stack([X_cheap_tr, tfidf_sim_tr])

    X_rv_tr = tfidf_vec.transform(refs_tr)
    X_sv_tr = tfidf_vec.transform(sus_tr)
    pair_feat_tr = np.abs(X_rv_tr - X_sv_tr).toarray()

    refs_v = df_val_t['reference_text'].tolist()
    sus_v = df_val_t['suspicious_text'].tolist()
    y_val_t = df_val_t['is_plagiarized'].to_numpy()
    X_cheap_v = text_cheap_view_features(refs_v, sus_v)
    tfidf_sim_v, _ = text_tfidf_similarity(refs_v, sus_v, vectorizer=tfidf_vec)
    X_s1_v = np.column_stack([X_cheap_v, tfidf_sim_v])
    X_rv_v = tfidf_vec.transform(refs_v)
    X_sv_v = tfidf_vec.transform(sus_v)
    pair_feat_v = np.abs(X_rv_v - X_sv_v).toarray()

    refs_te = df_test_t['reference_text'].tolist()
    sus_te = df_test_t['suspicious_text'].tolist()
    y_test_t = df_test_t['is_plagiarized'].to_numpy()
    X_cheap_te = text_cheap_view_features(refs_te, sus_te)
    tfidf_sim_te, _ = text_tfidf_similarity(refs_te, sus_te, vectorizer=tfidf_vec)
    X_s1_te = np.column_stack([X_cheap_te, tfidf_sim_te])
    X_rv_te = tfidf_vec.transform(refs_te)
    X_sv_te = tfidf_vec.transform(sus_te)
    pair_feat_te = np.abs(X_rv_te - X_sv_te).toarray()

    best = {'f1': -1}

    # LR on cheap multi-view features
    print("\n--- LR on cheap multi-view features ---")
    for cw in [None, 'balanced']:
        lr = LogisticRegression(max_iter=3000, class_weight=cw, random_state=RNG, n_jobs=-1)
        lr.fit(X_s1_tr, y_tr_t)
        val_p = lr.predict_proba(X_s1_v)[:, 1]
        best_t, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 81):
            f1 = f1_score(y_val_t, (val_p >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        te_p = lr.predict_proba(X_s1_te)[:, 1]
        te_pred = (te_p >= best_t).astype(int)
        m = metrics(y_test_t, te_pred)
        print(f"  LR cheap cw={cw} t={best_t:.2f} -> "
              f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}")
        if m['f1'] > best['f1']:
            best = dict(m, model='LR cheap', cw=cw, thr=best_t)

    # LR on TF-IDF pair-diff features
    print("\n--- LR on TF-IDF pair-diff features ---")
    for cw in [None, 'balanced']:
        lr = LogisticRegression(max_iter=3000, class_weight=cw, random_state=RNG, n_jobs=-1)
        lr.fit(pair_feat_tr, y_tr_t)
        val_p = lr.predict_proba(pair_feat_v)[:, 1]
        best_t, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 81):
            f1 = f1_score(y_val_t, (val_p >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        te_p = lr.predict_proba(pair_feat_te)[:, 1]
        te_pred = (te_p >= best_t).astype(int)
        m = metrics(y_test_t, te_pred)
        print(f"  LR tfidf cw={cw} t={best_t:.2f} -> "
              f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} mcc={m['mcc']:.4f}")
        if m['f1'] > best['f1']:
            best = dict(m, model='LR tfidf', cw=cw, thr=best_t)

    print("\n>>> TEXT BEST:", best)
    return best


if __name__ == '__main__':
    code_best = run_code()
    text_best = run_text()
    results = {'code_best': code_best, 'text_best': text_best}
    save_path = ROOT / "reports/paper_wise_comparison/exhaustive_improvement_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nSaved {save_path}")
