"""
ES-Plag End-to-End Detector.
Based on Sulistiani & Karnalim (2019), "ES-Plag: Efficient and sensitive source code
plagiarism detection tool for academic environment", Comput. Appl. Eng. Educ.
DOI: 10.1002/cae.22066.

Workflow:
1. Tokenization & Token-Frequency Vectorization.
2. Fast Cosine Similarity Pre-Filtering with Dynamic Threshold:
       threshold = sim_min + initial_threshold * (sim_max - sim_min)
3. Running-Karp-Rabin Greedy-String-Tiling (RKRGST) on retained candidate pairs.
4. Penalty Normalization (Penalized Max and Penalized Average).
5. Final Evaluation & Metric Accounting.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

try:
    from code.esplag_tokenizer import tokenize_code
    from code.token_indexer import compute_cosine_similarity, compute_dynamic_esplag_threshold
    from code.rkr_gst import rkr_gst
    from code.penalty_similarity import compute_penalized_similarities
except ImportError:
    from src.code.esplag_tokenizer import tokenize_code
    from src.code.token_indexer import compute_cosine_similarity, compute_dynamic_esplag_threshold
    from src.code.rkr_gst import rkr_gst
    from src.code.penalty_similarity import compute_penalized_similarities

class ESPlagDetector:
    def __init__(self, mml: int = 2, initial_threshold: float = 0.50, decision_threshold: float = 0.50):
        """
        Args:
            mml: Minimum matching length for RKRGST (reference setting = 2).
            initial_threshold: Proportional cutoff for dynamic cosine filtering (reference setting = 0.50).
            decision_threshold: Final classification threshold on penalized similarity.
        """
        self.mml = mml
        self.initial_threshold = initial_threshold
        self.decision_threshold = decision_threshold
        self.dynamic_filter_threshold = 0.0

    def evaluate_pairs(
        self,
        df_pairs: pd.DataFrame,
        apply_filtering: bool = True,
        use_penalty: str = 'average', # 'average', 'max', or 'raw'
        custom_filter_threshold: float | None = None
    ) -> dict:
        """
        Executes ES-Plag evaluation across a DataFrame of code pairs.
        
        Expected df_pairs columns:
            ['id', 'reference_code', 'suspicious_code', 'is_plagiarized', 'obfuscation_level']
        """
        n_pairs = len(df_pairs)
        if n_pairs == 0:
            return {}

        start_total = time.perf_counter()

        # Step 1: Tokenize all pairs
        tokenized_pairs = []
        for _, row in df_pairs.iterrows():
            tok_ref = tokenize_code(str(row['reference_code']))
            tok_sus = tokenize_code(str(row['suspicious_code']))
            tokenized_pairs.append((tok_ref, tok_sus))

        # Step 2: Compute pairwise cosine similarities
        t0_cosine = time.perf_counter()
        cosine_sims = []
        for tok_ref, tok_sus in tokenized_pairs:
            sim = compute_cosine_similarity(tok_ref, tok_sus)
            cosine_sims.append(sim)
        cosine_sims = np.array(cosine_sims, dtype=float)
        time_cosine = time.perf_counter() - t0_cosine

        # Step 3: Determine dynamic filtering threshold
        if custom_filter_threshold is not None:
            filter_threshold = float(custom_filter_threshold)
        elif apply_filtering:
            filter_threshold = compute_dynamic_esplag_threshold(cosine_sims, self.initial_threshold)
        else:
            filter_threshold = -1.0  # Pass all pairs
        self.dynamic_filter_threshold = filter_threshold

        # Step 4: Apply candidate filtering
        retained_mask = (cosine_sims >= filter_threshold) if apply_filtering else np.ones(n_pairs, dtype=bool)
        n_retained = int(retained_mask.sum())
        n_pruned = n_pairs - n_retained

        # Step 5: Execute RKRGST on retained candidate pairs
        t0_rkr = time.perf_counter()
        final_scores = np.zeros(n_pairs, dtype=float)
        penalized_max_scores = np.zeros(n_pairs, dtype=float)
        penalized_avg_scores = np.zeros(n_pairs, dtype=float)
        raw_gst_scores = np.zeros(n_pairs, dtype=float)
        matched_tokens_list = np.zeros(n_pairs, dtype=int)
        num_subsequences_list = np.zeros(n_pairs, dtype=int)
        rkr_latencies = np.zeros(n_pairs, dtype=float)

        for i in range(n_pairs):
            if retained_mask[i]:
                t_pair_start = time.perf_counter()
                tok_ref, tok_sus = tokenized_pairs[i]
                gst_res = rkr_gst(tok_ref, tok_sus, mml=self.mml)
                pen = compute_penalized_similarities(
                    matched_tokens=gst_res.matched_tokens,
                    num_subsequences=gst_res.num_subsequences,
                    len_a=gst_res.len_a,
                    len_b=gst_res.len_b
                )
                t_pair_elapsed = time.perf_counter() - t_pair_start

                penalized_max_scores[i] = pen['penalized_max']
                penalized_avg_scores[i] = pen['penalized_average']
                raw_gst_scores[i] = pen['raw_gst_average']
                matched_tokens_list[i] = gst_res.matched_tokens
                num_subsequences_list[i] = gst_res.num_subsequences
                rkr_latencies[i] = t_pair_elapsed

                if use_penalty == 'max':
                    final_scores[i] = pen['penalized_max']
                elif use_penalty == 'raw':
                    final_scores[i] = pen['raw_gst_average']
                else:
                    final_scores[i] = pen['penalized_average']

        time_rkr = time.perf_counter() - t0_rkr
        total_runtime_sec = time.perf_counter() - start_total

        # Ground truth labels
        y_true = df_pairs['is_plagiarized'].to_numpy(dtype=int)
        tot_positives = int((y_true == 1).sum())

        # Filtering metrics
        retained_positives = int((retained_mask & (y_true == 1)).sum())
        candidate_recall = float(retained_positives / tot_positives) if tot_positives > 0 else 1.0
        pruning_ratio = float(n_pruned / n_pairs) if n_pairs > 0 else 0.0
        expensive_call_rate = float(n_retained / n_pairs) if n_pairs > 0 else 0.0

        # Classification metrics based on final_scores
        predictions = (final_scores >= self.decision_threshold).astype(int)
        acc = float(accuracy_score(y_true, predictions))
        prec = float(precision_score(y_true, predictions, zero_division=0))
        rec = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        roc_auc = float(roc_auc_score(y_true, final_scores)) if len(np.unique(y_true)) > 1 else 0.5

        # Obfuscation-level recall
        level_recalls = {}
        if 'obfuscation_level' in df_pairs.columns:
            for level in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
                mask = (df_pairs['obfuscation_level'] == level) & (y_true == 1)
                if mask.sum() > 0:
                    lvl_rec = float(predictions[mask].sum() / mask.sum())
                    level_recalls[level] = lvl_rec

        # Pair-by-pair result records
        pair_records = []
        for i, (_, row) in enumerate(df_pairs.iterrows()):
            pair_records.append({
                'id': row['id'],
                'cosine_similarity': float(cosine_sims[i]),
                'passed_filter': bool(retained_mask[i]),
                'matched_tokens': int(matched_tokens_list[i]),
                'num_subsequences': int(num_subsequences_list[i]),
                'penalized_max': float(penalized_max_scores[i]),
                'penalized_average': float(penalized_avg_scores[i]),
                'raw_gst_average': float(raw_gst_scores[i]),
                'final_score': float(final_scores[i]),
                'prediction': int(predictions[i]),
                'ground_truth': int(y_true[i]),
                'obfuscation_level': str(row.get('obfuscation_level', 'None')),
                'case': str(row.get('case', 'None'))
            })

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'specificity': spec,
            'fpr': fpr,
            'roc_auc': roc_auc,
            'candidate_recall': candidate_recall,
            'pruning_ratio': pruning_ratio,
            'expensive_call_rate': expensive_call_rate,
            'n_pairs': n_pairs,
            'n_retained': n_retained,
            'n_pruned': n_pruned,
            'total_positives': tot_positives,
            'retained_positives': retained_positives,
            'filter_threshold': float(filter_threshold),
            'decision_threshold': float(self.decision_threshold),
            'confusion_matrix': cm.tolist(),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'total_runtime_sec': float(total_runtime_sec),
            'time_cosine_sec': float(time_cosine),
            'time_rkr_sec': float(time_rkr),
            'avg_latency_ms': float((total_runtime_sec / n_pairs) * 1000.0) if n_pairs > 0 else 0.0,
            'level_recalls': level_recalls,
            'pair_records': pair_records
        }
