"""
Controlled Multi-View Feature Ablation Experiments (Phase 3.3, Workstream E)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from research_upgrade.src.candidate_generation.threshold_selection import ThresholdSelector
from research_upgrade.src.candidate_generation.candidate_evaluation import evaluate_candidate_threshold


TEXT_ABLATION_CONFIGS = {
    "word_only": ["text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio"],
    "character_only": ["text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio"],
    "semantic_only": ["text_semantic_minilm_sim"],
    "structural_only": ["text_length_ratio", "text_token_count_delta", "text_sentence_delta", "text_ttr_sim"],
    "word_plus_character": [
        "text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
        "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio"
    ],
    "word_plus_semantic": [
        "text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
        "text_semantic_minilm_sim"
    ],
    "character_plus_semantic": [
        "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio",
        "text_semantic_minilm_sim"
    ],
    "all_available_views": [
        "text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
        "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio",
        "text_semantic_minilm_sim",
        "text_length_ratio", "text_token_count_delta", "text_sentence_delta", "text_ttr_sim"
    ]
}

CODE_ABLATION_CONFIGS = {
    "raw_token_only": ["code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim"],
    "normalized_token_only": ["code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim"],
    "structural_only": ["code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim"],
    "complexity_only": ["code_operator_dist_sim", "code_control_flow_sim", "code_cyclomatic_sim", "code_sloc_ratio"],
    "raw_plus_normalized": [
        "code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
        "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim"
    ],
    "token_plus_structural": [
        "code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
        "code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim"
    ],
    "normalized_plus_structural": [
        "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim",
        "code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim"
    ],
    "all_available_views": [
        "code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
        "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim",
        "code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim",
        "code_operator_dist_sim", "code_control_flow_sim", "code_cyclomatic_sim", "code_sloc_ratio"
    ]
}


def run_text_ablation_experiments(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_recall: float = 0.98,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the 8 required text feature ablation configurations.
    Fits Logistic Regression on df_train features.
    Selects threshold tau* on df_val targeting recall >= target_recall.
    Evaluates out-of-sample performance on df_test.
    """
    selector = ThresholdSelector(num_grid_points=500)
    y_train = df_train["ground_truth_label"].to_numpy()
    y_val = df_val["ground_truth_label"].to_numpy()
    y_test = df_test["ground_truth_label"].to_numpy()

    ablation_results = {}

    for config_name, feature_list in TEXT_ABLATION_CONFIGS.items():
        X_tr = df_train[feature_list].to_numpy()
        X_va = df_val[feature_list].to_numpy()
        X_te = df_test[feature_list].to_numpy()

        # Fit logistic regression scorer
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        model.fit(X_tr, y_train)

        pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
        s_val = model.predict_proba(X_va)[:, pos_idx]
        s_test = model.predict_proba(X_te)[:, pos_idx]

        # Select threshold on validation set
        val_sel = selector.select_recall_constrained_threshold(y_val, s_val, target_recall=target_recall)
        tau_star = val_sel["selected_threshold"]

        # Evaluate on test set with selected threshold
        test_m = evaluate_candidate_threshold(y_test, s_test, tau_star)

        ablation_results[config_name] = {
            "configuration": config_name,
            "feature_count": len(feature_list),
            "features_used": feature_list,
            "selected_threshold": tau_star,
            "target_recall_constraint": target_recall,
            "validation_recall": val_sel["validation_metrics"]["candidate_recall"],
            "validation_workload_reduction": val_sel["validation_metrics"]["workload_reduction"],
            "validation_precision": val_sel["validation_metrics"]["candidate_precision"],
            "test_recall": test_m.candidate_recall,
            "test_workload_reduction": test_m.workload_reduction,
            "test_precision": test_m.candidate_precision,
            "test_false_negatives": test_m.false_negatives,
            "test_escalated_pairs": test_m.escalated_pairs
        }

    return ablation_results


def run_code_ablation_experiments(
    df_code: pd.DataFrame,
    target_recall: float = 0.98,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the 8 required code feature ablation configurations via 5-fold CV.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    selector = ThresholdSelector(num_grid_points=500)

    y = df_code["ground_truth_label"].to_numpy()
    ablation_results = {}

    for config_name, feature_list in CODE_ABLATION_CONFIGS.items():
        X = df_code[feature_list].to_numpy()

        fold_recalls = []
        fold_workloads = []
        fold_precisions = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
            model.fit(X_tr, y_tr)

            pos_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
            s_val = model.predict_proba(X_va)[:, pos_idx]

            val_sel = selector.select_recall_constrained_threshold(y_va, s_val, target_recall=target_recall)
            vm = val_sel["validation_metrics"]
            fold_recalls.append(vm["candidate_recall"])
            fold_workloads.append(vm["workload_reduction"])
            fold_precisions.append(vm["candidate_precision"])

        ablation_results[config_name] = {
            "configuration": config_name,
            "feature_count": len(feature_list),
            "features_used": feature_list,
            "target_recall_constraint": target_recall,
            "mean_validation_recall": round(float(np.mean(fold_recalls)), 4),
            "mean_validation_workload_reduction": round(float(np.mean(fold_workloads)), 4),
            "mean_validation_precision": round(float(np.mean(fold_precisions)), 4),
            "std_workload_reduction": round(float(np.std(fold_workloads)), 4)
        }

    return ablation_results

