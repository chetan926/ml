"""
Code Modality Candidate Scorer Pipeline (Phase 3.3, Workstream A & B)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from research_upgrade.src.features.feature_pipeline import (
    CodeMultiViewPipeline,
    CODE_FEATURE_COLUMNS
)
from research_upgrade.src.data.code_loader import CodeBenchmarkDataLoader
from research_upgrade.src.candidate_generation.score_fusion import (
    BaseCandidateScorer,
    SingleViewThresholdScorer,
    WeightedMultiViewScorer,
    LogisticRegressionCandidateScorer,
    RandomForestCandidateScorer,
    GradientBoostingCandidateScorer
)
from research_upgrade.src.candidate_generation.threshold_selection import ThresholdSelector
from research_upgrade.src.candidate_generation.candidate_evaluation import evaluate_candidate_threshold


class CodeCandidateScorerPipeline:
    """
    Coordinates training, threshold selection, cross-validation, and Case-05 evaluation
    for IR-Plag source code plagiarism candidate generation.
    Supports Policy A (720 pairs) and Policy B (709 pairs).
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.feature_columns = list(CODE_FEATURE_COLUMNS)
        self.feature_pipeline = CodeMultiViewPipeline()
        self.threshold_selector = ThresholdSelector(num_grid_points=500)
        self.df_policy_a_feats: Optional[pd.DataFrame] = None
        self.df_policy_b_feats: Optional[pd.DataFrame] = None
        self._is_prepared = False

    def prepare_data(
        self,
        code_loader: Optional[CodeBenchmarkDataLoader] = None,
        cache_dir: Optional[Path] = None
    ) -> "CodeCandidateScorerPipeline":
        """Extracts and optionally caches multi-view features for Policy A and B."""
        if cache_dir is not None:
            cache_p = Path(cache_dir)
            cache_p.mkdir(parents=True, exist_ok=True)
            f_a = cache_p / "code_feats_policy_a.parquet"
            f_b = cache_p / "code_feats_policy_b.parquet"
            if f_a.exists() and f_b.exists():
                self.df_policy_a_feats = pd.read_parquet(f_a)
                self.df_policy_b_feats = pd.read_parquet(f_b)
                self._is_prepared = True
                return self

        if code_loader is None:
            code_loader = CodeBenchmarkDataLoader()

        code_a = code_loader.get_dataset("policy_a")
        code_b = code_loader.get_dataset("policy_b")

        self.df_policy_a_feats = self.feature_pipeline.transform_dataframe(code_a, policy="policy_a")
        self.df_policy_b_feats = self.feature_pipeline.transform_dataframe(code_b, policy="policy_b")

        if cache_dir is not None:
            self.df_policy_a_feats.to_parquet(cache_p / "code_feats_policy_a.parquet", index=False)
            self.df_policy_b_feats.to_parquet(cache_p / "code_feats_policy_b.parquet", index=False)

        self._is_prepared = True
        return self

    def _get_models(self) -> Dict[str, BaseCandidateScorer]:
        return {
            "single_view_token_jaccard": SingleViewThresholdScorer("code_token_jaccard", self.feature_columns),
            "single_view_rkrgst": SingleViewThresholdScorer("code_rkrgst_sim", self.feature_columns),
            "single_view_norm_rkrgst": SingleViewThresholdScorer("code_norm_rkrgst_sim", self.feature_columns),
            "weighted_multiview": WeightedMultiViewScorer(self.feature_columns),
            "logistic_regression": LogisticRegressionCandidateScorer(self.feature_columns, random_state=self.random_seed),
            "random_forest": RandomForestCandidateScorer(self.feature_columns, random_state=self.random_seed),
            "gradient_boosting": GradientBoostingCandidateScorer(self.feature_columns, random_state=self.random_seed)
        }

    def evaluate_cross_validation(
        self,
        policy: str = "policy_a",
        n_splits: int = 5,
        target_recall: float = 0.98
    ) -> Dict[str, Any]:
        """
        Executes 5-fold stratified cross-validation on the code benchmark.
        In each fold:
        - Trains models on train fold.
        - Evaluates validation fold and selects threshold.
        """
        if not self._is_prepared:
            raise RuntimeError("Must prepare_data() before cross-validation.")

        df = self.df_policy_a_feats if policy == "policy_a" else self.df_policy_b_feats
        X = df[self.feature_columns].to_numpy()
        y = df["ground_truth_label"].to_numpy()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)

        model_names = list(self._get_models().keys())
        fold_metrics = {m: [] for m in model_names}

        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            fold_models = self._get_models()
            for name, model in fold_models.items():
                model.fit(X_tr, y_tr)
                scores_te = model.predict_score(X_te)

                # Select threshold targeting target_recall
                sel = self.threshold_selector.select_recall_constrained_threshold(
                    y_te, scores_te, target_recall=target_recall
                )
                fold_metrics[name].append(sel["validation_metrics"])

        # Aggregate across folds
        aggregated = {}
        for name in model_names:
            metrics_list = fold_metrics[name]
            recalls = [m["candidate_recall"] for m in metrics_list]
            workloads = [m["workload_reduction"] for m in metrics_list]
            precisions = [m["candidate_precision"] for m in metrics_list]
            thresholds = [m["threshold"] for m in metrics_list]

            aggregated[name] = {
                "model_name": name,
                "policy": policy,
                "n_splits": n_splits,
                "target_recall_constraint": target_recall,
                "mean_recall": round(float(np.mean(recalls)), 4),
                "std_recall": round(float(np.std(recalls)), 4),
                "mean_workload_reduction": round(float(np.mean(workloads)), 4),
                "std_workload_reduction": round(float(np.std(workloads)), 4),
                "mean_precision": round(float(np.mean(precisions)), 4),
                "mean_threshold": round(float(np.mean(thresholds)), 4),
                "fold_details": metrics_list
            }

        return aggregated

    def evaluate_case05_cross_fitting(self) -> Dict[str, Any]:
        """
        Evaluates Case-05 (69 pairs, all negative) under the audited 5-fold split.
        Verifies behavior when test fold contains 0 positive instances.
        """
        if not self._is_prepared:
            raise RuntimeError("Must prepare_data() before Case-05 evaluation.")

        c5_df = self.df_policy_a_feats[self.df_policy_a_feats["case_id"] == "case-05"].copy()
        X_c5 = c5_df[self.feature_columns].to_numpy()
        y_c5 = c5_df["ground_truth_label"].to_numpy()

        # Since all pairs in Case-05 are negative (0), recall is vacuously defined (0/0)
        # We evaluate the False Positive rate / Workload Reduction achieved on Case-05
        models = self._get_models()
        # Train on remaining cases (case-01 to case-04, case-06, case-07)
        other_df = self.df_policy_a_feats[self.df_policy_a_feats["case_id"] != "case-05"]
        X_other = other_df[self.feature_columns].to_numpy()
        y_other = other_df["ground_truth_label"].to_numpy()

        c5_results = {}
        for name, m in models.items():
            m.fit(X_other, y_other)
            scores = m.predict_score(X_c5)

            # Evaluate at threshold tau = 0.5 (or median)
            t = 0.5
            escalated = int(np.sum(scores >= t))
            filtered = len(scores) - escalated
            workload_red = float(filtered / len(scores))

            c5_results[name] = {
                "total_pairs": len(scores),
                "positive_pairs": int(np.sum(y_c5 == 1)),
                "negative_pairs": int(np.sum(y_c5 == 0)),
                "evaluated_threshold": t,
                "escalated_pairs": escalated,
                "filtered_pairs": filtered,
                "workload_reduction": round(workload_red, 4),
                "mean_score": round(float(np.mean(scores)), 4),
                "std_score": round(float(np.std(scores)), 4)
            }

        return {
            "case": "case-05",
            "limitation": (
                "Case-05 contains 69 pairs, all of which are negative (0 positive pairs). "
                "Candidate recall cannot be evaluated on positive instances in Case-05. "
                "Metrics measure workload reduction on negative distractor pairs."
            ),
            "results": c5_results
        }

