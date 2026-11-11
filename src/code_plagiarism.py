import time
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from code_preprocessing import (
    remove_comments, extract_tokens, anonymize_identifiers,
    extract_code_features, JAVA_KEYWORDS, JAVA_OPERATORS
)

def compute_pair_features(ref_code: str, sus_code: str) -> list[float]:
    """
    Extracts multi-dimensional similarity features for a Java code pair,
    faithfully adapting Gandhi et al. (2024) to Java submissions:
    1. Raw token Jaccard similarity
    2. Anonymized token Jaccard similarity (resists variable/method renaming)
    3. Java keyword distribution cosine similarity
    4. Operator token Jaccard similarity
    5. Structural line-count ratio
    6. Cyclomatic complexity relative difference
    7. Clean character 3-gram overlap
    """
    f_ref = extract_code_features(ref_code)
    f_sus = extract_code_features(sus_code)

    # 1. Raw token Jaccard
    set_ref = set(f_ref['token_list'])
    set_sus = set(f_sus['token_list'])
    u_tokens = set_ref.union(set_sus)
    j_token = len(set_ref.intersection(set_sus)) / len(u_tokens) if u_tokens else 0.0

    # 2. Anonymized token Jaccard
    anon_ref = set(anonymize_identifiers(ref_code).split())
    anon_sus = set(anonymize_identifiers(sus_code).split())
    u_anon = anon_ref.union(anon_sus)
    j_anon = len(anon_ref.intersection(anon_sus)) / len(u_anon) if u_anon else 0.0

    # 3. Java keyword distribution cosine
    kw_ref = Counter([t for t in f_ref['token_list'] if t in JAVA_KEYWORDS])
    kw_sus = Counter([t for t in f_sus['token_list'] if t in JAVA_KEYWORDS])
    all_kws = sorted(list(JAVA_KEYWORDS))
    v_ref = np.array([kw_ref.get(k, 0) for k in all_kws], dtype=float)
    v_sus = np.array([kw_sus.get(k, 0) for k in all_kws], dtype=float)
    norm_r = np.linalg.norm(v_ref)
    norm_s = np.linalg.norm(v_sus)
    kw_cos = float(np.dot(v_ref, v_sus) / (norm_r * norm_s)) if (norm_r > 0 and norm_s > 0) else 0.0

    # 4. Operator Jaccard
    op_ref = set([t for t in f_ref['token_list'] if t in JAVA_OPERATORS])
    op_sus = set([t for t in f_sus['token_list'] if t in JAVA_OPERATORS])
    u_op = op_ref.union(op_sus)
    j_op = len(op_ref.intersection(op_sus)) / len(u_op) if u_op else 0.0

    # 5. Line ratio
    l_min = min(f_ref['clean_lines'], f_sus['clean_lines'])
    l_max = max(f_ref['clean_lines'], f_sus['clean_lines'])
    line_ratio = float(l_min / l_max) if l_max > 0 else 1.0

    # 6. Cyclomatic difference
    c_diff = abs(f_ref['cyclomatic_proxy'] - f_sus['cyclomatic_proxy'])
    c_max = max(f_ref['cyclomatic_proxy'], f_sus['cyclomatic_proxy'])
    cyclo_sim = float(1.0 - (c_diff / c_max)) if c_max > 0 else 1.0

    # 7. Character trigrams of clean code
    c_ref = re.sub(r'\s+', '', f_ref['clean_code'])
    c_sus = re.sub(r'\s+', '', f_sus['clean_code'])
    tri_r = set([c_ref[i:i+3] for i in range(len(c_ref)-2)]) if len(c_ref) >= 3 else set([c_ref])
    tri_s = set([c_sus[i:i+3] for i in range(len(c_sus)-2)]) if len(c_sus) >= 3 else set([c_sus])
    u_tri = tri_r.union(tri_s)
    tri_sim = len(tri_r.intersection(tri_s)) / len(u_tri) if u_tri else 0.0

    return [j_token, j_anon, kw_cos, j_op, line_ratio, cyclo_sim, tri_sim]

class CodeSVMDetector:
    """
    Method 3: SVM-Based Source Code Plagiarism Detector (Gandhi et al. 2024 adaptation to Java).
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.scaler = StandardScaler()
        self.clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        self.is_fitted = False
        self.threshold = 0.5

    def extract_dataset_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts feature matrix for all pairs in DataFrame."""
        feats = []
        for _, row in df.iterrows():
            feats.append(compute_pair_features(row['reference_code'], row['suspicious_code']))
        return np.array(feats, dtype=float)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits standard scaler and SVM classifier."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.clf.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs calibrated plagiarism probability scores."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, grid=np.linspace(0.1, 0.9, 81)):
        """Tunes threshold on validation set."""
        probs = self.predict_proba(X_val)
        best_thresh = 0.5
        best_f1 = -1.0
        for t in grid:
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)
        self.threshold = best_thresh
        print(f"[CodeSVM] Validation-tuned optimal threshold: {self.threshold:.4f} (Val F1: {best_f1:.4f})")
        return self.threshold

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, df_test: pd.DataFrame = None) -> dict:
        """Evaluates SVM detector, computing classification metrics and per-level obfuscation recall."""
        start_time = time.perf_counter()
        probs = self.predict_proba(X_test)
        total_time = time.perf_counter() - start_time

        preds = (probs >= self.threshold).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        roc_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5

        n_pairs = len(y_test)
        avg_latency_ms = (total_time / n_pairs) * 1000.0 if n_pairs > 0 else 0.0

        level_recall = {}
        if df_test is not None and 'obfuscation_level' in df_test.columns:
            for lvl in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
                mask = (df_test['obfuscation_level'] == lvl).values
                if mask.sum() > 0:
                    lvl_y = y_test[mask]
                    lvl_preds = preds[mask]
                    level_recall[lvl] = float(recall_score(lvl_y, lvl_preds, zero_division=0))

        return {
            'method': 'SVM Syntactic/Token Detector (Gandhi et al. 2024 adapted to Java)',
            'modality': 'Source Code',
            'threshold': self.threshold,
            'total_pairs': n_pairs,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'fpr': float(fpr),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'level_recall': level_recall,
            'total_runtime_sec': float(total_time),
            'avg_latency_ms': float(avg_latency_ms),
            'scores': probs,
            'predictions': preds
        }
