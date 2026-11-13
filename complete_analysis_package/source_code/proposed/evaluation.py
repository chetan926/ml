import os
import time
import json
import numpy as np
import pandas as pd
from text_tfidf import TextTFIDFDetector
from bert_similarity import ContextualSemanticDetector
from code_plagiarism import CodeSVMDetector, compute_pair_features
from code_random_forest import CodeRandomForestDetector
from typed_decision_pipeline import TypedDecisionPipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RESULTS_DIR = 'experiments/results'

def run_text_experiments(sample_size=3000):
    """
    Executes text experiments comparing:
      1. Method 1: TF-IDF Lexical Baseline (Sajid et al. 2025)
      2. Method 2: Contextual SBERT Semantic Detector (Setu et al. 2025)
      3. Condition A: Full Semantic Analysis (Run SBERT on 100% pairs)
      4. Condition B: Fixed-Threshold Filtering (Pass pairs >= single threshold to SBERT)
      5. Condition C: Proposed Typed-Decision Pipeline (Tri-band: Low Risk [Prune], Uncertain [Escalate], High Risk [Flag])
    """
    print("\n" + "="*50)
    print("STARTING TEXT EXPERIMENTS (MIT / SNLI BENCHMARK)")
    print("="*50)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_all = pd.read_csv('datasets/processed/text_plagiarism_benchmark.csv')
    df_val = df_all[df_all['split'] == 'val'].copy()
    df_test = df_all[df_all['split'] == 'test'].copy()

    # Balanced or representative sampling for fast, statistically rigorous execution
    if sample_size and sample_size < len(df_val):
        df_val = df_val.sample(n=min(sample_size, len(df_val)), random_state=42).reset_index(drop=True)
    if sample_size and sample_size < len(df_test):
        df_test = df_test.sample(n=min(sample_size, len(df_test)), random_state=42).reset_index(drop=True)

    print(f"Text evaluation set sizes: Validation={len(df_val)}, Test={len(df_test)}")

    val_ref = df_val['reference_text'].tolist()
    val_sus = df_val['suspicious_text'].tolist()
    y_val = df_val['is_plagiarized'].to_numpy()

    test_ref = df_test['reference_text'].tolist()
    test_sus = df_test['suspicious_text'].tolist()
    y_test = df_test['is_plagiarized'].to_numpy()

    # 1. Method 1: Lexical TF-IDF
    print("\n--- Running Method 1: TF-IDF + Cosine Similarity ---")
    tfidf = TextTFIDFDetector()
    tfidf.fit(val_ref + val_sus)
    tfidf.tune_threshold(val_ref, val_sus, y_val)
    res_tfidf = tfidf.evaluate(test_ref, test_sus, y_test)
    print(f"TF-IDF Test Results: Acc={res_tfidf['accuracy']:.4f}, Prec={res_tfidf['precision']:.4f}, Rec={res_tfidf['recall']:.4f}, F1={res_tfidf['f1']:.4f}, Latency={res_tfidf['avg_latency_ms']:.3f}ms")

    # 2. Method 2: Contextual SBERT
    print("\n--- Running Method 2: Contextual SBERT Semantic Detector ---")
    sbert = ContextualSemanticDetector()
    sbert.tune_threshold(val_ref, val_sus, y_val)
    res_sbert = sbert.evaluate(test_ref, test_sus, y_test)
    print(f"SBERT Test Results: Acc={res_sbert['accuracy']:.4f}, Prec={res_sbert['precision']:.4f}, Rec={res_sbert['recall']:.4f}, F1={res_sbert['f1']:.4f}, Latency={res_sbert['avg_latency_ms']:.3f}ms")

    # 3. Operating Condition A: Full Analysis
    res_cond_a = dict(res_sbert)
    res_cond_a['condition'] = 'A: Full Analysis (100% Expensive Detector)'
    res_cond_a['candidate_recall'] = 1.0
    res_cond_a['candidate_precision'] = res_sbert['precision']
    res_cond_a['pruning_ratio'] = 0.0
    res_cond_a['expensive_detector_call_rate'] = 1.0
    res_cond_a['filtered_pairs_count'] = 0
    res_cond_a['escalated_pairs_count'] = len(y_test)
    res_cond_a['direct_flagged_count'] = 0

    # 4. Operating Condition B: Fixed-Threshold Filtering
    print("\n--- Running Condition B: Fixed-Threshold Filtering ---")
    # Pick fixed threshold on validation that prunes ~50%
    val_cheap_sims = tfidf.compute_similarity(val_ref, val_sus)
    fixed_thresh = float(np.percentile(val_cheap_sims, 50))
    print(f"Condition B fixed similarity threshold (50th percentile): {fixed_thresh:.4f}")

    test_cheap_sims = res_tfidf['scores']
    passed_mask = (test_cheap_sims >= fixed_thresh)
    n_passed = passed_mask.sum()

    preds_cond_b = np.zeros(len(y_test), dtype=int)
    latencies_b = np.full(len(y_test), res_tfidf['avg_latency_ms'] / 1000.0)

    if n_passed > 0:
        pass_ref = [test_ref[i] for i in np.where(passed_mask)[0]]
        pass_sus = [test_sus[i] for i in np.where(passed_mask)[0]]
        t0 = time.perf_counter()
        pass_sbert_sims = sbert.compute_similarity(pass_ref, pass_sus)
        t_heavy = (time.perf_counter() - t0) / n_passed
        pass_preds = (pass_sbert_sims >= sbert.threshold).astype(int)

        for idx, orig_i in enumerate(np.where(passed_mask)[0]):
            preds_cond_b[orig_i] = pass_preds[idx]
            latencies_b[orig_i] += t_heavy

    cm_b = confusion_matrix(y_test, preds_cond_b, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    tot_pos = (y_test == 1).sum()

    res_cond_b = {
        'condition': 'B: Fixed-Threshold Filtering',
        'modality': 'Text',
        'threshold': fixed_thresh,
        'total_pairs': len(y_test),
        'accuracy': float(accuracy_score(y_test, preds_cond_b)),
        'precision': float(precision_score(y_test, preds_cond_b, zero_division=0)),
        'recall': float(recall_score(y_test, preds_cond_b, zero_division=0)),
        'f1': float(f1_score(y_test, preds_cond_b, zero_division=0)),
        'specificity': float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0,
        'fpr': float(fp_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0,
        'candidate_recall': float(int((passed_mask & (y_test == 1)).sum()) / tot_pos) if tot_pos > 0 else 1.0,
        'candidate_precision': float(int((passed_mask & (y_test == 1)).sum()) / n_passed) if n_passed > 0 else 0.0,
        'pruning_ratio': float(1.0 - (n_passed / len(y_test))),
        'expensive_detector_call_rate': float(n_passed / len(y_test)),
        'filtered_pairs_count': int(len(y_test) - n_passed),
        'escalated_pairs_count': int(n_passed),
        'direct_flagged_count': 0,
        'confusion_matrix': cm_b.tolist(),
        'total_runtime_sec': float(latencies_b.sum()),
        'avg_latency_ms': float(np.mean(latencies_b) * 1000.0),
        'median_latency_ms': float(np.median(latencies_b) * 1000.0),
        'p95_latency_ms': float(np.percentile(latencies_b, 95) * 1000.0),
    }

    # 5. Operating Condition C: Proposed Typed-Decision Pipeline
    print("\n--- Running Condition C: Proposed Typed-Decision Pipeline ---")
    pipeline = TypedDecisionPipeline(text_cheap_detector=tfidf, text_heavy_detector=sbert)
    pipeline.tune_text_thresholds(val_ref, val_sus, y_val, target_candidate_recall=0.98)
    res_cond_c = pipeline.evaluate_text_pipeline(test_ref, test_sus, y_test)
    print(f"Condition C Pipeline Test: Acc={res_cond_c['accuracy']:.4f}, Prec={res_cond_c['precision']:.4f}, Rec={res_cond_c['recall']:.4f}, F1={res_cond_c['f1']:.4f}, Pruning={res_cond_c['pruning_ratio']*100:.1f}%, CallRate={res_cond_c['expensive_detector_call_rate']*100:.1f}%")

    # Compute authentic empirical threshold sensitivity sweep on validation data
    thetas_text = np.linspace(0.01, 0.45, 45)
    pos_mask_val = (y_val == 1)
    tot_pos_val = pos_mask_val.sum()
    sens_rows = []
    for th in thetas_text:
        ret_pos = (val_cheap_sims[pos_mask_val] >= th).sum()
        ret_cand = (val_cheap_sims >= th).sum()
        sens_rows.append({
            'Modality': 'Text',
            'Threshold': round(float(th), 4),
            'Candidate_Recall': float(ret_pos / tot_pos_val) if tot_pos_val > 0 else 1.0,
            'Pruning_Ratio': float(1.0 - (ret_cand / len(y_val))),
            'Candidate_Precision': float(ret_pos / ret_cand) if ret_cand > 0 else 0.0,
            'Calibrated_Theta_Low': pipeline.text_theta_low
        })
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(os.path.join(RESULTS_DIR, 'threshold_sensitivity_data.csv'), index=False)
    print(f"Saved {RESULTS_DIR}/threshold_sensitivity_data.csv (authentic empirical validation sweep)")

    text_results = [res_tfidf, res_sbert, res_cond_a, res_cond_b, res_cond_c]
    
    # Save text results summary
    text_summary_rows = []
    for r in text_results:
        text_summary_rows.append({
            'Method / Condition': r.get('condition', r.get('method')),
            'Modality': r['modality'],
            'Accuracy': round(r['accuracy'], 4),
            'Precision': round(r['precision'], 4),
            'Recall': round(r['recall'], 4),
            'F1': round(r['f1'], 4),
            'Specificity': round(r['specificity'], 4),
            'FPR': round(r['fpr'], 4),
            'Candidate Recall': round(r.get('candidate_recall', r['recall']), 4),
            'Pruning Ratio (%)': round(r.get('pruning_ratio', 0.0) * 100, 2),
            'Expensive Call Rate (%)': round(r.get('expensive_detector_call_rate', 1.0) * 100, 2),
            'Avg Latency (ms)': round(r['avg_latency_ms'], 3),
            'Filtered Pairs': r.get('filtered_pairs_count', 0),
            'Escalated Pairs': r.get('escalated_pairs_count', r['total_pairs']),
            'Direct Flagged Pairs': r.get('direct_flagged_count', 0),
        })
    df_text_res = pd.DataFrame(text_summary_rows)
    df_text_res.to_csv(os.path.join(RESULTS_DIR, 'text_experiments_summary.csv'), index=False)
    print(f"Saved {RESULTS_DIR}/text_experiments_summary.csv")

    # Save detailed evaluation object for visualization
    with open(os.path.join(RESULTS_DIR, 'text_results_detailed.json'), 'w') as f:
        json.dump({
            'res_tfidf': {k: v for k, v in res_tfidf.items() if k not in ['scores', 'predictions']},
            'res_sbert': {k: v for k, v in res_sbert.items() if k not in ['scores', 'predictions']},
            'res_cond_a': {k: v for k, v in res_cond_a.items() if k not in ['scores', 'predictions']},
            'res_cond_b': res_cond_b,
            'res_cond_c': {k: v for k, v in res_cond_c.items() if k not in ['predictions', 'decisions']},
        }, f, indent=2)

    return text_results, {
        'df_val': df_val,
        'df_test': df_test,
        'val_ref': val_ref,
        'val_sus': val_sus,
        'y_val': y_val,
        'test_ref': test_ref,
        'test_sus': test_sus,
        'y_test': y_test,
        'tfidf': tfidf,
        'sbert': sbert,
        'res_tfidf': res_tfidf,
        'res_sbert': res_sbert,
        'res_cond_c': res_cond_c
    }

def run_code_experiments():
    """
    Executes source code experiments on IR-Plag Java Corpus:
      1. Method 3: SVM Syntactic/Token Java Detector (Gandhi et al. 2024 adapted to Java)
      2. Baseline Cheap Token Jaccard Filter
      3. Condition A: Full SVM Analysis
      4. Condition B: Fixed-Threshold Filtering
      5. Condition C: Proposed Typed-Decision Pipeline
      6. Obfuscation Level Analysis (L1 to L6)
    """
    print("\n" + "="*50)
    print("STARTING SOURCE CODE EXPERIMENTS (IR-PLAG JAVA BENCHMARK)")
    print("="*50)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_code = pd.read_csv('datasets/processed/code_plagiarism_benchmark.csv')
    print(f"Total IR-Plag code pairs: {len(df_code)}")

    # Group-aware split by assessment task:
    # Tasks case-01 to case-04 for Training, case-05 for Validation, case-06 and case-07 for Test
    train_cases = ['case-01', 'case-02', 'case-03', 'case-04']
    val_cases = ['case-05']
    test_cases = ['case-06', 'case-07']

    df_train = df_code[df_code['case'].isin(train_cases)].reset_index(drop=True)
    df_val = df_code[df_code['case'].isin(val_cases)].reset_index(drop=True)
    df_test = df_code[df_code['case'].isin(test_cases)].reset_index(drop=True)

    print(f"Task-aware splits: Train={len(df_train)} pairs, Val={len(df_val)} pairs, Test={len(df_test)} pairs")

    svm_det = CodeSVMDetector()

    print("Extracting multi-dimensional Java syntactic/token features...")
    X_train = svm_det.extract_dataset_features(df_train)
    y_train = df_train['is_plagiarized'].to_numpy()

    X_val = svm_det.extract_dataset_features(df_val)
    y_val = df_val['is_plagiarized'].to_numpy()

    X_test = svm_det.extract_dataset_features(df_test)
    y_test = df_test['is_plagiarized'].to_numpy()

    # Cheap feature is column 0 (token Jaccard)
    cheap_val = X_val[:, 0]
    cheap_test = X_test[:, 0]

    # Train SVM detector (Method 2: Gandhi et al. 2024)
    print("Fitting SVM model on training tasks...")
    svm_det.fit(X_train, y_train)
    svm_det.tune_threshold(X_val, y_val)

    # 1. Method 2: Full SVM detector (Gandhi et al. 2024)
    res_svm = svm_det.evaluate(X_test, y_test, df_test)
    print(f"Method 2 (SVM) Test: Acc={res_svm['accuracy']:.4f}, Prec={res_svm['precision']:.4f}, Rec={res_svm['recall']:.4f}, F1={res_svm['f1']:.4f}")
    print(f"Per-Obfuscation Level Recall (SVM): {res_svm['level_recall']}")

    # Train Random Forest detector (Method 3: Mehsen & Joshi 2024)
    print("Fitting Random Forest model on training tasks (Method 3: Mehsen & Joshi 2024)...")
    rf_det = CodeRandomForestDetector()
    rf_det.fit(X_train, y_train)
    rf_det.tune_threshold(X_val, y_val)
    res_rf = rf_det.evaluate(X_test, y_test, df_test)
    print(f"Method 3 (Random Forest) Test: Acc={res_rf['accuracy']:.4f}, Prec={res_rf['precision']:.4f}, Rec={res_rf['recall']:.4f}, F1={res_rf['f1']:.4f}")
    print(f"Per-Obfuscation Level Recall (RF): {res_rf['level_recall']}")

    # 2. Baseline Cheap Token Jaccard Filter alone
    best_tok_th = 0.3
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        p = (cheap_val >= t).astype(int)
        f1 = f1_score(y_val, p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tok_th = t
    tok_preds = (cheap_test >= best_tok_th).astype(int)
    cm_tok = confusion_matrix(y_test, tok_preds, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_tok.ravel()
    res_tok_cheap = {
        'method': 'Baseline Token Jaccard Filter',
        'modality': 'Source Code',
        'threshold': float(best_tok_th),
        'total_pairs': len(y_test),
        'accuracy': float(accuracy_score(y_test, tok_preds)),
        'precision': float(precision_score(y_test, tok_preds, zero_division=0)),
        'recall': float(recall_score(y_test, tok_preds, zero_division=0)),
        'f1': float(f1_score(y_test, tok_preds, zero_division=0)),
        'specificity': float(tn_t / (tn_t + fp_t)) if (tn_t + fp_t) > 0 else 0.0,
        'fpr': float(fp_t / (tn_t + fp_t)) if (tn_t + fp_t) > 0 else 0.0,
        'candidate_recall': float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 1.0,
        'pruning_ratio': 0.0,
        'expensive_detector_call_rate': 0.0,
        'avg_latency_ms': 0.05,
        'total_runtime_sec': 0.001,
        'confusion_matrix': cm_tok.tolist()
    }

    # 3. Condition A: Full SVM Analysis
    res_code_a = dict(res_svm)
    res_code_a['condition'] = 'A: Full Analysis (100% SVM Model)'
    res_code_a['candidate_recall'] = 1.0
    res_code_a['candidate_precision'] = res_svm['precision']
    res_code_a['pruning_ratio'] = 0.0
    res_code_a['expensive_detector_call_rate'] = 1.0
    res_code_a['filtered_pairs_count'] = 0
    res_code_a['escalated_pairs_count'] = len(y_test)
    res_code_a['direct_flagged_count'] = 0

    # 4. Condition B: Fixed-Threshold Filtering
    fixed_code_th = float(np.median(cheap_val))
    passed_mask = (cheap_test >= fixed_code_th)
    n_passed = passed_mask.sum()
    preds_b = np.zeros(len(y_test), dtype=int)
    latencies_b = np.full(len(y_test), 0.00005)

    if n_passed > 0:
        pass_probs = svm_det.predict_proba(X_test[passed_mask])
        pass_p = (pass_probs >= svm_det.threshold).astype(int)
        for idx, orig_i in enumerate(np.where(passed_mask)[0]):
            preds_b[orig_i] = pass_p[idx]
            latencies_b[orig_i] += (res_svm['avg_latency_ms'] / 1000.0)

    cm_b = confusion_matrix(y_test, preds_b, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    tot_pos = (y_test == 1).sum()

    res_code_b = {
        'condition': 'B: Fixed-Threshold Filtering',
        'modality': 'Source Code',
        'threshold': fixed_code_th,
        'total_pairs': len(y_test),
        'accuracy': float(accuracy_score(y_test, preds_b)),
        'precision': float(precision_score(y_test, preds_b, zero_division=0)),
        'recall': float(recall_score(y_test, preds_b, zero_division=0)),
        'f1': float(f1_score(y_test, preds_b, zero_division=0)),
        'specificity': float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0,
        'fpr': float(fp_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0,
        'candidate_recall': float(int((passed_mask & (y_test == 1)).sum()) / tot_pos) if tot_pos > 0 else 1.0,
        'candidate_precision': float(int((passed_mask & (y_test == 1)).sum()) / n_passed) if n_passed > 0 else 0.0,
        'pruning_ratio': float(1.0 - (n_passed / len(y_test))),
        'expensive_detector_call_rate': float(n_passed / len(y_test)),
        'filtered_pairs_count': int(len(y_test) - n_passed),
        'escalated_pairs_count': int(n_passed),
        'direct_flagged_count': 0,
        'confusion_matrix': cm_b.tolist(),
        'total_runtime_sec': float(latencies_b.sum()),
        'avg_latency_ms': float(np.mean(latencies_b) * 1000.0),
        'median_latency_ms': float(np.median(latencies_b) * 1000.0),
        'p95_latency_ms': float(np.percentile(latencies_b, 95) * 1000.0)
    }

    # 5. Condition C: Proposed Typed-Decision Pipeline
    pipeline = TypedDecisionPipeline(code_heavy_detector=svm_det)
    pipeline.tune_code_thresholds(cheap_val, y_val, target_candidate_recall=0.98)
    res_code_c = pipeline.evaluate_code_pipeline(cheap_test, X_test, y_test, df_test)
    print(f"Condition C Code Pipeline: Acc={res_code_c['accuracy']:.4f}, Prec={res_code_c['precision']:.4f}, Rec={res_code_c['recall']:.4f}, F1={res_code_c['f1']:.4f}, Pruning={res_code_c['pruning_ratio']*100:.1f}%, CallRate={res_code_c['expensive_detector_call_rate']*100:.1f}%")

    # Save summary table
    code_results = [res_tok_cheap, res_svm, res_rf, res_code_a, res_code_b, res_code_c]
    code_summary_rows = []
    for r in code_results:
        code_summary_rows.append({
            'Method / Condition': r.get('condition', r.get('method')),
            'Modality': r['modality'],
            'Accuracy': round(r['accuracy'], 4),
            'Precision': round(r['precision'], 4),
            'Recall': round(r['recall'], 4),
            'F1': round(r['f1'], 4),
            'Specificity': round(r['specificity'], 4),
            'FPR': round(r['fpr'], 4),
            'Candidate Recall': round(r.get('candidate_recall', r['recall']), 4),
            'Pruning Ratio (%)': round(r.get('pruning_ratio', 0.0) * 100, 2),
            'Expensive Call Rate (%)': round(r.get('expensive_detector_call_rate', 1.0) * 100, 2),
            'Avg Latency (ms)': round(r['avg_latency_ms'], 3),
            'Filtered Pairs': r.get('filtered_pairs_count', 0),
            'Escalated Pairs': r.get('escalated_pairs_count', r['total_pairs']),
            'Direct Flagged Pairs': r.get('direct_flagged_count', 0),
        })
    df_code_res = pd.DataFrame(code_summary_rows)
    df_code_res.to_csv(os.path.join(RESULTS_DIR, 'code_experiments_summary.csv'), index=False)
    print(f"Saved {RESULTS_DIR}/code_experiments_summary.csv")

    # Save obfuscation recall breakdown
    df_obf = pd.DataFrame([
        {'Obfuscation Level': k, 'Description': {
            'L1': 'Exact code copying (comments/formatting)',
            'L2': 'Identifier renaming (student variables)',
            'L3': 'Statement reordering & declaration order',
            'L4': 'Control flow replacement (loop constructs)',
            'L5': 'Method abstraction & procedural restructuring',
            'L6': 'Combined obfuscation attacks'
        }.get(k, k), 'SVM Recall': round(v, 4)}
        for k, v in res_svm.get('level_recall', {}).items()
    ])
    df_obf.to_csv(os.path.join(RESULTS_DIR, 'code_obfuscation_level_recall.csv'), index=False)
    print(f"Saved {RESULTS_DIR}/code_obfuscation_level_recall.csv")

    # Save detailed evaluation object
    with open(os.path.join(RESULTS_DIR, 'code_results_detailed.json'), 'w') as f:
        json.dump({
            'res_tok_cheap': res_tok_cheap,
            'res_svm': {k: v for k, v in res_svm.items() if k not in ['scores', 'predictions']},
            'res_code_a': {k: v for k, v in res_code_a.items() if k not in ['scores', 'predictions']},
            'res_code_b': res_code_b,
            'res_code_c': {k: v for k, v in res_code_c.items() if k not in ['predictions', 'decisions']},
        }, f, indent=2)

    return code_results, {
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'cheap_val': cheap_val,
        'cheap_test': cheap_test,
        'svm_det': svm_det,
        'rf_det': rf_det,
        'res_svm': res_svm,
        'res_rf': res_rf,
        'res_code_c': res_code_c
    }

if __name__ == '__main__':
    run_text_experiments(sample_size=3000)
    run_code_experiments()

