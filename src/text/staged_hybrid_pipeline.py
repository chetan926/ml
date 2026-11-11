"""
Setu-Inspired Text Detection Pipelines.
Based on Setu et al. (2025), DOI: 10.1007/s43995-025-00108-1.

Implements two conditions:
- E1: Paper-Faithful Full Hybrid:
      Computes both TF-IDF and SBERT for all candidate pairs.
      hybrid_score = max(tfidf_score, sbert_score)
      Evaluated at reference threshold (0.30) and validation-calibrated threshold.
- E2: Cost-Efficient Staged Hybrid (Proposed Contribution):
      Stage 1: Compute lightweight TF-IDF.
      Stage 2: Validation-calibrated routing (Prune, Escalate to SBERT, Direct Flag).
      Stage 3: Selectively invoke SBERT for escalated pairs.
      Stage 4: Combine scores and classify.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

try:
    from text.lexical_detector import LexicalTFIDFDetector
    from text.semantic_detector import SemanticSBERTDetector
    from text.hybrid_similarity import compute_hybrid_similarity
except ImportError:
    from src.text.lexical_detector import LexicalTFIDFDetector
    from src.text.semantic_detector import SemanticSBERTDetector
    from src.text.hybrid_similarity import compute_hybrid_similarity

class SetuTextPipeline:
    def __init__(self, tfidf_detector=None, sbert_detector=None):
        self.tfidf = tfidf_detector if tfidf_detector else LexicalTFIDFDetector()
        self.sbert = sbert_detector if sbert_detector else SemanticSBERTDetector()

    def run_full_hybrid_e1(
        self,
        ref_texts: list[str],
        sus_texts: list[str],
        y_true: np.ndarray,
        threshold: float = 0.30,
        ids: list[str] = None
    ) -> dict:
        """
        Executes Condition E1: Paper-Faithful Full Hybrid.
        Computes both TF-IDF and SBERT on 100% of candidate pairs.
        """
        y_true = np.asarray(y_true, dtype=int)
        n_pairs = len(y_true)
        start_total = time.perf_counter()

        # Step 1: TF-IDF
        t0_tfidf = time.perf_counter()
        tfidf_scores = self.tfidf.compute_similarity(ref_texts, sus_texts)
        time_tfidf = time.perf_counter() - t0_tfidf

        # Step 2: SBERT
        t0_sbert = time.perf_counter()
        sbert_scores = self.sbert.compute_similarity(ref_texts, sus_texts)
        time_sbert = time.perf_counter() - t0_sbert

        # Step 3: Hybrid fusion max(tfidf, sbert)
        hybrid_scores, categories = compute_hybrid_similarity(tfidf_scores, sbert_scores)
        total_time = time.perf_counter() - start_total

        # Predictions
        preds = (hybrid_scores >= threshold).astype(int)
        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        roc_auc = float(roc_auc_score(y_true, hybrid_scores)) if len(np.unique(y_true)) > 1 else 0.5

        # Pair records
        pair_records = []
        for i in range(n_pairs):
            pair_records.append({
                'id': ids[i] if ids else f"TXT_{i:05d}",
                'tfidf_score': float(tfidf_scores[i]),
                'sbert_score': float(sbert_scores[i]),
                'hybrid_score': float(hybrid_scores[i]),
                'dominant_category': categories[i],
                'prediction': int(preds[i]),
                'ground_truth': int(y_true[i]),
                'condition': 'E1_Full_Hybrid'
            })

        return {
            'condition': 'E1: Paper-Faithful Full Hybrid',
            'threshold': float(threshold),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'specificity': spec,
            'fpr': fpr,
            'roc_auc': roc_auc,
            'candidate_recall': 1.0,
            'pruning_ratio': 0.0,
            'expensive_call_rate': 1.0,
            'n_pairs': n_pairs,
            'sbert_calls': n_pairs,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'total_runtime_sec': float(total_time),
            'time_tfidf_sec': float(time_tfidf),
            'time_sbert_sec': float(time_sbert),
            'avg_latency_ms': float((total_time / n_pairs) * 1000.0) if n_pairs > 0 else 0.0,
            'pair_records': pair_records,
            'hybrid_scores': hybrid_scores,
            'tfidf_scores': tfidf_scores,
            'sbert_scores': sbert_scores
        }

    def run_staged_hybrid_e2(
        self,
        ref_texts: list[str],
        sus_texts: list[str],
        y_true: np.ndarray,
        theta_low: float = 0.05,
        theta_high: float = 0.80,
        decision_threshold: float = 0.50,
        ids: list[str] = None
    ) -> dict:
        """
        Executes Condition E2: Cost-Efficient Staged Hybrid Pipeline.
        Selectively invokes SBERT only for uncertain pairs in [theta_low, theta_high).
        """
        y_true = np.asarray(y_true, dtype=int)
        n_pairs = len(y_true)
        start_total = time.perf_counter()

        # Stage 1: TF-IDF
        t0_tfidf = time.perf_counter()
        tfidf_scores = self.tfidf.compute_similarity(ref_texts, sus_texts)
        time_tfidf = time.perf_counter() - t0_tfidf

        # Stage 2: Routing policy
        pruned_mask = (tfidf_scores < theta_low)
        flagged_mask = (tfidf_scores >= theta_high)
        escalate_mask = (~pruned_mask) & (~flagged_mask)

        escalate_indices = np.where(escalate_mask)[0]
        n_escalated = len(escalate_indices)
        n_pruned = int(pruned_mask.sum())
        n_flagged = int(flagged_mask.sum())

        # Stage 3: Selective SBERT invocation
        sbert_scores = np.zeros(n_pairs, dtype=float)
        time_sbert = 0.0

        if n_escalated > 0:
            esc_ref = [ref_texts[i] for i in escalate_indices]
            esc_sus = [sus_texts[i] for i in escalate_indices]

            t0_sbert = time.perf_counter()
            esc_sbert = self.sbert.compute_similarity(esc_ref, esc_sus)
            time_sbert = time.perf_counter() - t0_sbert

            for idx, orig_i in enumerate(escalate_indices):
                sbert_scores[orig_i] = esc_sbert[idx]

        total_time = time.perf_counter() - start_total

        # Stage 4: Combine scores and classify
        final_scores = np.zeros(n_pairs, dtype=float)
        predictions = np.zeros(n_pairs, dtype=int)
        categories = ['none'] * n_pairs

        for i in range(n_pairs):
            if pruned_mask[i]:
                final_scores[i] = tfidf_scores[i]
                predictions[i] = 0
                categories[i] = 'pruned_low_risk'
            elif flagged_mask[i]:
                final_scores[i] = max(tfidf_scores[i], 0.99)
                predictions[i] = 1
                categories[i] = 'direct_copy_flagged'
            else: # escalated
                final_scores[i] = max(tfidf_scores[i], sbert_scores[i])
                predictions[i] = int(final_scores[i] >= decision_threshold)
                categories[i] = 'paraphrase_like' if sbert_scores[i] > tfidf_scores[i] else 'direct_copy_like'

        # Metric evaluation
        tot_positives = int((y_true == 1).sum())
        retained_positives = int((~pruned_mask & (y_true == 1)).sum())
        candidate_recall = float(retained_positives / tot_positives) if tot_positives > 0 else 1.0
        pruning_ratio = float(n_pruned / n_pairs) if n_pairs > 0 else 0.0
        expensive_call_rate = float(n_escalated / n_pairs) if n_pairs > 0 else 0.0
        direct_flag_rate = float(n_flagged / n_pairs) if n_pairs > 0 else 0.0
        workload_reduction = float((n_pruned + n_flagged) / n_pairs) if n_pairs > 0 else 0.0

        acc = float(accuracy_score(y_true, predictions))
        prec = float(precision_score(y_true, predictions, zero_division=0))
        rec = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Pair records
        pair_records = []
        for i in range(n_pairs):
            pair_records.append({
                'id': ids[i] if ids else f"TXT_{i:05d}",
                'tfidf_score': float(tfidf_scores[i]),
                'sbert_score': float(sbert_scores[i]) if escalate_mask[i] else None,
                'final_score': float(final_scores[i]),
                'routing_status': 'PRUNED' if pruned_mask[i] else ('FLAGGED' if flagged_mask[i] else 'ESCALATED'),
                'dominant_category': categories[i],
                'prediction': int(predictions[i]),
                'ground_truth': int(y_true[i]),
                'condition': 'E2_Staged_Hybrid'
            })

        return {
            'condition': 'E2: Cost-Efficient Staged Hybrid',
            'theta_low': float(theta_low),
            'theta_high': float(theta_high),
            'decision_threshold': float(decision_threshold),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'specificity': spec,
            'fpr': fpr,
            'candidate_recall': candidate_recall,
            'pruning_ratio': pruning_ratio,
            'expensive_call_rate': expensive_call_rate,
            'direct_flag_rate': direct_flag_rate,
            'workload_reduction': workload_reduction,
            'n_pairs': n_pairs,
            'n_pruned': n_pruned,
            'n_escalated': n_escalated,
            'n_flagged': n_flagged,
            'total_positives': tot_positives,
            'retained_positives': retained_positives,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'total_runtime_sec': float(total_time),
            'time_tfidf_sec': float(time_tfidf),
            'time_sbert_sec': float(time_sbert),
            'avg_latency_ms': float((total_time / n_pairs) * 1000.0) if n_pairs > 0 else 0.0,
            'pair_records': pair_records
        }
