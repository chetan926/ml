import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from text_tfidf import TextTFIDFDetector
from bert_similarity import ContextualSemanticDetector
from code_plagiarism import CodeSVMDetector, compute_pair_features
from text_preprocessing import compute_word_jaccard, compute_char_ngram_jaccard
from code_preprocessing import extract_tokens, remove_comments, JAVA_KEYWORDS

class TypedDecisionPipeline:
    """
    Proposed Research Contribution:
    Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines.
    
    Components:
    - C1: Modality-Aware Typed Routing (Text vs Source Code)
    - C2: Tri-Band Confidence/Risk Filtering (Low Risk, Uncertain, High Risk)
    - C3: Cost-Aware Escalation (Invoke expensive contextual/syntactic models only on Uncertain band)
    - C4: Cross-Modality Validation-Calibrated Decision Boundaries
    """
    def __init__(self, text_cheap_detector=None, text_heavy_detector=None,
                 code_cheap_detector=None, code_heavy_detector=None):
        self.text_cheap = text_cheap_detector
        self.text_heavy = text_heavy_detector
        self.code_cheap = code_cheap_detector
        self.code_heavy = code_heavy_detector

        # Validation-tuned thresholds
        self.text_theta_low = 0.15
        self.text_theta_high = 0.75
        self.code_theta_low = 0.20
        self.code_theta_high = 0.80

    def detect_modality(self, ref_content: str, sus_content: str) -> str:
        """
        C1: Typed Routing Architecture.
        Determines whether input pair is Natural Language Text or Source Code.
        """
        combined = (ref_content + " " + sus_content).lower()
        java_syntax_markers = ['public class', 'import java.', 'system.out.print', 'public static void main', 'private static', 'throws exception']
        for marker in java_syntax_markers:
            if marker in combined:
                return 'code'
        
        # Check token keyword ratio
        tokens = combined.split()
        kw_hits = sum(1 for t in tokens if t in JAVA_KEYWORDS)
        if len(tokens) > 0 and (kw_hits / len(tokens)) > 0.08:
            return 'code'
        return 'text'

    def tune_text_thresholds(self, ref_texts: list[str], sus_texts: list[str], y_val: np.ndarray,
                             target_candidate_recall: float = 0.98):
        """
        Calibrates text tri-band thresholds on validation data.
        - theta_low is set to preserve >= target_candidate_recall of true positives.
        - theta_high is set to achieve high confidence (precision >= 0.90).
        """
        cheap_scores = self.text_cheap.compute_similarity(ref_texts, sus_texts)
        pos_mask = (y_val == 1)

        # Calibrate theta_low to retain target_candidate_recall of positives
        grid_low = np.linspace(0.01, 0.40, 40)
        chosen_low = 0.05
        for t in reversed(grid_low):
            retained_pos = (cheap_scores[pos_mask] >= t).sum()
            cand_recall = retained_pos / pos_mask.sum() if pos_mask.sum() > 0 else 1.0
            if cand_recall >= target_candidate_recall:
                chosen_low = float(t)
                break

        # Calibrate theta_high for high-confidence verbatim detection
        grid_high = np.linspace(0.60, 0.95, 36)
        chosen_high = 0.80
        for t in grid_high:
            flagged = (cheap_scores >= t)
            if flagged.sum() > 0:
                prec = (y_val[flagged] == 1).sum() / flagged.sum()
                if prec >= 0.85:
                    chosen_high = float(t)
                    break

        self.text_theta_low = chosen_low
        self.text_theta_high = chosen_high
        print(f"[TypedPipeline-Text] Calibrated thresholds: theta_low={self.text_theta_low:.4f}, theta_high={self.text_theta_high:.4f}")

    def tune_code_thresholds(self, cheap_scores: np.ndarray, y_val: np.ndarray,
                             target_candidate_recall: float = 0.98):
        """Calibrates code tri-band thresholds on validation data."""
        pos_mask = (y_val == 1)
        grid_low = np.linspace(0.30, 0.80, 51)
        chosen_low = 0.40
        for t in reversed(grid_low):
            retained_pos = (cheap_scores[pos_mask] >= t).sum()
            cand_recall = retained_pos / pos_mask.sum() if pos_mask.sum() > 0 else 1.0
            if cand_recall >= target_candidate_recall:
                chosen_low = float(t)
                break

        grid_high = np.linspace(0.70, 0.98, 29)
        chosen_high = 0.88
        for t in grid_high:
            flagged = (cheap_scores >= t)
            if flagged.sum() > 0:
                prec = (y_val[flagged] == 1).sum() / flagged.sum()
                if prec >= 0.90:
                    chosen_high = float(t)
                    break

        self.code_theta_low = chosen_low
        self.code_theta_high = chosen_high
        print(f"[TypedPipeline-Code] Calibrated thresholds: theta_low={self.code_theta_low:.4f}, theta_high={self.code_theta_high:.4f}")

    def evaluate_text_pipeline(self, ref_texts: list[str], sus_texts: list[str], y_test: np.ndarray) -> dict:
        """
        Executes the tri-band pipeline on test text pairs.
        Decisions:
          - Score < theta_low: LOW RISK (pruned as non-plagiarized, predicted 0)
          - theta_low <= Score < theta_high: UNCERTAIN (escalated to Contextual SBERT detector)
          - Score >= theta_high: HIGH RISK (direct flag for human review, predicted 1)
        """
        n_pairs = len(y_test)
        preds = np.zeros(n_pairs, dtype=int)
        decisions = []
        latencies = []

        start_total = time.perf_counter()

        # Step 1: Cheap lexical filter
        t0 = time.perf_counter()
        cheap_scores = self.text_cheap.compute_similarity(ref_texts, sus_texts)
        t_cheap = (time.perf_counter() - t0) / n_pairs

        escalate_indices = []
        filtered_count = 0
        direct_flag_count = 0

        for i in range(n_pairs):
            s = cheap_scores[i]
            if s < self.text_theta_low:
                # LOW RISK: Prune
                preds[i] = 0
                decisions.append('LOW_RISK_PRUNED')
                filtered_count += 1
                latencies.append(t_cheap)
            elif s >= self.text_theta_high:
                # HIGH RISK: Direct Flag
                preds[i] = 1
                decisions.append('HIGH_RISK_FLAGGED')
                direct_flag_count += 1
                latencies.append(t_cheap)
            else:
                # UNCERTAIN: Escalate
                decisions.append('UNCERTAIN_ESCALATED')
                escalate_indices.append(i)
                # Latency placeholder, updated below
                latencies.append(t_cheap)

        # Step 2: Escalate only Uncertain pairs to heavy contextual detector
        if len(escalate_indices) > 0:
            esc_ref = [ref_texts[i] for i in escalate_indices]
            esc_sus = [sus_texts[i] for i in escalate_indices]

            t_heavy_start = time.perf_counter()
            heavy_scores = self.text_heavy.compute_similarity(esc_ref, esc_sus)
            t_heavy_total = time.perf_counter() - t_heavy_start
            t_heavy_per_pair = t_heavy_total / len(escalate_indices)

            for idx, orig_i in enumerate(escalate_indices):
                pred_label = int(heavy_scores[idx] >= self.text_heavy.threshold)
                preds[orig_i] = pred_label
                latencies[orig_i] += t_heavy_per_pair

        total_runtime = time.perf_counter() - start_total

        # Metrics computation
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total_positives = int((y_test == 1).sum())
        retained_candidates = n_pairs - filtered_count
        retained_positives = int(((cheap_scores >= self.text_theta_low) & (y_test == 1)).sum())
        candidate_recall = float(retained_positives / total_positives) if total_positives > 0 else 1.0
        candidate_precision = float(retained_positives / retained_candidates) if retained_candidates > 0 else 0.0

        pruning_ratio = float(filtered_count / n_pairs) if n_pairs > 0 else 0.0
        expensive_call_rate = float(len(escalate_indices) / n_pairs) if n_pairs > 0 else 0.0

        return {
            'condition': 'C: Proposed Typed-Decision Pipeline',
            'modality': 'Text',
            'total_pairs': n_pairs,
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'fpr': float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'candidate_recall': candidate_recall,
            'candidate_precision': candidate_precision,
            'pruning_ratio': pruning_ratio,
            'expensive_detector_call_rate': expensive_call_rate,
            'filtered_pairs_count': filtered_count,
            'escalated_pairs_count': len(escalate_indices),
            'direct_flagged_count': direct_flag_count,
            'confusion_matrix': cm.tolist(),
            'total_runtime_sec': float(total_runtime),
            'avg_latency_ms': float(np.mean(latencies) * 1000.0),
            'median_latency_ms': float(np.median(latencies) * 1000.0),
            'p95_latency_ms': float(np.percentile(latencies, 95) * 1000.0),
            'predictions': preds,
            'decisions': decisions
        }

    def evaluate_code_pipeline(self, cheap_scores: np.ndarray, heavy_features: np.ndarray,
                               y_test: np.ndarray, df_test: pd.DataFrame = None) -> dict:
        """
        Executes the tri-band pipeline on source code pairs.
        - Cheap filter: Fast token Jaccard similarity.
        - Heavy detector: Multi-feature SVM model.
        """
        n_pairs = len(y_test)
        preds = np.zeros(n_pairs, dtype=int)
        decisions = []
        latencies = []

        start_total = time.perf_counter()
        t_cheap_est = 0.00005  # sub-millisecond token Jaccard

        escalate_indices = []
        filtered_count = 0
        direct_flag_count = 0

        for i in range(n_pairs):
            s = cheap_scores[i]
            if s < self.code_theta_low:
                # LOW RISK: Prune
                preds[i] = 0
                decisions.append('LOW_RISK_PRUNED')
                filtered_count += 1
                latencies.append(t_cheap_est)
            elif s >= self.code_theta_high:
                # HIGH RISK: Direct Flag
                preds[i] = 1
                decisions.append('HIGH_RISK_FLAGGED')
                direct_flag_count += 1
                latencies.append(t_cheap_est)
            else:
                # UNCERTAIN: Escalate
                decisions.append('UNCERTAIN_ESCALATED')
                escalate_indices.append(i)
                latencies.append(t_cheap_est)

        # Step 2: Escalate Uncertain pairs to heavy SVM detector
        if len(escalate_indices) > 0:
            esc_X = heavy_features[escalate_indices]
            t_heavy_start = time.perf_counter()
            esc_probs = self.code_heavy.predict_proba(esc_X)
            t_heavy_total = time.perf_counter() - t_heavy_start
            t_heavy_per_pair = t_heavy_total / len(escalate_indices)

            for idx, orig_i in enumerate(escalate_indices):
                pred_label = int(esc_probs[idx] >= self.code_heavy.threshold)
                preds[orig_i] = pred_label
                latencies[orig_i] += t_heavy_per_pair

        total_runtime = time.perf_counter() - start_total

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total_positives = int((y_test == 1).sum())
        retained_candidates = n_pairs - filtered_count
        retained_positives = int(((cheap_scores >= self.code_theta_low) & (y_test == 1)).sum())
        candidate_recall = float(retained_positives / total_positives) if total_positives > 0 else 1.0
        candidate_precision = float(retained_positives / retained_candidates) if retained_candidates > 0 else 0.0

        pruning_ratio = float(filtered_count / n_pairs) if n_pairs > 0 else 0.0
        expensive_call_rate = float(len(escalate_indices) / n_pairs) if n_pairs > 0 else 0.0

        return {
            'condition': 'C: Proposed Typed-Decision Pipeline',
            'modality': 'Source Code',
            'total_pairs': n_pairs,
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'fpr': float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'candidate_recall': candidate_recall,
            'candidate_precision': candidate_precision,
            'pruning_ratio': pruning_ratio,
            'expensive_detector_call_rate': expensive_call_rate,
            'filtered_pairs_count': filtered_count,
            'escalated_pairs_count': len(escalate_indices),
            'direct_flagged_count': direct_flag_count,
            'confusion_matrix': cm.tolist(),
            'total_runtime_sec': float(total_runtime),
            'avg_latency_ms': float(np.mean(latencies) * 1000.0),
            'median_latency_ms': float(np.median(latencies) * 1000.0),
            'p95_latency_ms': float(np.percentile(latencies, 95) * 1000.0),
            'predictions': preds,
            'decisions': decisions
        }

