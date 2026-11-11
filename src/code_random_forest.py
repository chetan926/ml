import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from code_preprocessing import extract_tokens, remove_comments
from code_plagiarism import compute_pair_features

class CodeRandomForestDetector:
    """
    Method 3 (Recent 2024 Methodology):
    Source Code Plagiarism Detection via Machine Learning Random Forest Classifier.
    Based on Mehsen & Joshi (2024), International Journal of Computing, vol. 23(1), pp. 78-84.
    DOI: 10.47839/ijc.23.1.3438.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.threshold = 0.5
        self.is_fitted = False

    def extract_dataset_features(self, df_pairs: pd.DataFrame) -> np.ndarray:
        """Extracts the 7-dimensional syntactic/token features for code pairs."""
        features = []
        for _, row in df_pairs.iterrows():
            f = compute_pair_features(row['reference_code'], row['suspicious_code'])
            features.append(f)
        return np.array(features, dtype=float)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits the Random Forest ensemble on training data."""
        self.rf.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns positive class probability estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        return self.rf.predict_proba(X)[:, 1]

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray, grid=np.linspace(0.1, 0.9, 81)):
        """Finds the optimal classification threshold on validation data maximizing F1-score."""
        probs = self.predict_proba(X_val)
        best_th = 0.5
        best_f1 = -1.0

        for t in grid:
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(t)

        self.threshold = best_th
        print(f"[CodeRF] Validation-tuned optimal threshold: {self.threshold:.4f} (Val F1: {best_f1:.4f})")
        return self.threshold

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, df_test: pd.DataFrame = None) -> dict:
        """Evaluates detection performance on the test split."""
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
                mask = (df_test['obfuscation_level'] == lvl).to_numpy()
                if mask.sum() > 0:
                    lvl_preds = preds[mask]
                    level_recall[lvl] = float((lvl_preds == 1).sum() / mask.sum())

        return {
            'method': 'Method 3: Random Forest Ensemble (Mehsen & Joshi 2024)',
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
            'total_runtime_sec': float(total_time),
            'avg_latency_ms': float(avg_latency_ms),
            'level_recall': level_recall,
            'scores': probs,
            'predictions': preds
        }

if __name__ == '__main__':
    df = pd.read_csv('datasets/processed/code_plagiarism_benchmark.csv')
    rf_det = CodeRandomForestDetector()
    train_cases = ['case-01', 'case-02', 'case-03', 'case-04']
    val_cases = ['case-05']
    test_cases = ['case-06', 'case-07']

    df_train = df[df['case'].isin(train_cases)].reset_index(drop=True)
    df_val = df[df['case'].isin(val_cases)].reset_index(drop=True)
    df_test = df[df['case'].isin(test_cases)].reset_index(drop=True)

    X_train = rf_det.extract_dataset_features(df_train)
    y_train = df_train['is_plagiarized'].to_numpy()
    X_val = rf_det.extract_dataset_features(df_val)
    y_val = df_val['is_plagiarized'].to_numpy()
    X_test = rf_det.extract_dataset_features(df_test)
    y_test = df_test['is_plagiarized'].to_numpy()

    rf_det.fit(X_train, y_train)
    rf_det.tune_threshold(X_val, y_val)
    res = rf_det.evaluate(X_test, y_test, df_test)
    print(f"RF Test Acc: {res['accuracy']:.4f}, Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f}, F1: {res['f1']:.4f}")
    print("Level Recall:", res['level_recall'])

