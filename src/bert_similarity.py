import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

class ContextualSemanticDetector:
    """
    Method 2: Contextual Semantic Text Plagiarism Detection (BERT / Sentence-BERT)
    Based on Setu et al. (2025) (Springer, DOI: 10.1007/s43995-025-00108-1).
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        print(f"[SemanticDetector] Loading pretrained contextual model '{model_name}' on {device}...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = 0.5

    def compute_embeddings(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generates dense L2-normalized contextual sequence embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    def compute_similarity(self, ref_texts: list[str], sus_texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Computes pairwise contextual cosine similarities between reference and suspicious texts."""
        ref_vecs = self.compute_embeddings(ref_texts, batch_size=batch_size)
        sus_vecs = self.compute_embeddings(sus_texts, batch_size=batch_size)
        
        # Row-wise dot product of normalized vectors = cosine similarity
        sims = (ref_vecs * sus_vecs).sum(axis=1)
        return np.clip(sims, -1.0, 1.0)

    def tune_threshold(self, ref_texts: list[str], sus_texts: list[str], y_true: np.ndarray, grid=np.linspace(0.2, 0.9, 71)):
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
        print(f"[SemanticDetector] Validation-tuned optimal threshold: {self.threshold:.4f} (Val F1: {best_f1:.4f})")
        return self.threshold

    def evaluate(self, ref_texts: list[str], sus_texts: list[str], y_true: np.ndarray, batch_size: int = 64) -> dict:
        """Evaluates detection performance on test split and measures computational latency."""
        start_time = time.perf_counter()
        sims = self.compute_similarity(ref_texts, sus_texts, batch_size=batch_size)
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
            'method': 'Contextual SBERT Semantic Detector (Setu et al. 2025)',
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

