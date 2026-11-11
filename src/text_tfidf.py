import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from text_preprocessing import normalize_text

class TextTFIDFDetector:
    """
    Method 1: Lexical Text Plagiarism Detection (TF-IDF + Cosine Similarity)
    Based on Sajid et al. (2025) comparative baseline.
    """
    def __init__(self, ngram_range=(1, 2), max_features=10000, sublinear_tf=True):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            preprocessor=normalize_text,
            stop_words='english'
        )
        self.threshold = 0.5
        self.is_fitted = False

    def fit(self, corpus: list[str]):
        """Fits TF-IDF vectorizer over the reference vocabulary."""
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        return self

    def compute_similarity(self, ref_texts: list[str], sus_texts: list[str]) -> np.ndarray:
        """Computes pairwise cosine similarity for corresponding text pairs."""
        if not self.is_fitted:
            # Fit on the combination of provided texts if not already fitted
            self.fit(list(ref_texts) + list(sus_texts))

        ref_vecs = self.vectorizer.transform(ref_texts)
        sus_vecs = self.vectorizer.transform(sus_texts)

        # Dot product of normalized rows gives cosine similarity
        sims = (ref_vecs.multiply(sus_vecs)).sum(axis=1)
        sims = np.asarray(sims).flatten()
        return np.clip(sims, 0.0, 1.0)

    def tune_threshold(self, ref_texts: list[str], sus_texts: list[str], y_true: np.ndarray, grid=np.linspace(0.1, 0.9, 81)):
        """Finds optimal decision threshold on validation data maximizing F1-score."""
        sims = self.compute_similarity(ref_texts, sus_texts)
        best_thresh = 0.5
        best_f1 = -1.0

        for t in grid:
            preds = (sims >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        self.threshold = best_thresh
        print(f"[TextTFIDF] Validation-tuned optimal threshold: {self.threshold:.4f} (Val F1: {best_f1:.4f})")
        return self.threshold

    def evaluate(self, ref_texts: list[str], sus_texts: list[str], y_true: np.ndarray) -> dict:
        """Evaluates detection performance on a test split, measuring execution latency."""
        start_time = time.perf_counter()
        sims = self.compute_similarity(ref_texts, sus_texts)
        total_time = time.perf_counter() - start_time

        preds = (sims >= self.threshold).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        roc_auc = roc_auc_score(y_true, sims) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = average_precision_score(y_true, sims) if len(np.unique(y_true)) > 1 else 0.5

        n_pairs = len(y_true)
        avg_latency_ms = (total_time / n_pairs) * 1000.0 if n_pairs > 0 else 0.0

        return {
            'method': 'TF-IDF + Cosine Similarity (Sajid et al. 2025)',
            'modality': 'Text',
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
            'scores': sims,
            'predictions': preds
        }

