"""
Text Modality Candidate Scorer Pipeline (Phase 3.3, Workstream A & B)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import time

from research_upgrade.src.features.feature_pipeline import (
    TextMultiViewPipeline,
    TEXT_FEATURE_COLUMNS
)
from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
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


class TextCandidateScorerPipeline:
    """
    Coordinates training, validation threshold selection, and test evaluation
    for academic text plagiarism candidate generation.
    Strictly enforces training-only fitting and validation-only threshold tuning.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.feature_columns = list(TEXT_FEATURE_COLUMNS)
        self.feature_pipeline = TextMultiViewPipeline()
        self.threshold_selector = ThresholdSelector(num_grid_points=500)
        self.models: Dict[str, BaseCandidateScorer] = {}
        self.selected_thresholds: Dict[str, Dict[str, Any]] = {}
        self._is_prepared = False

        self.df_train_feats: Optional[pd.DataFrame] = None
        self.df_val_feats: Optional[pd.DataFrame] = None
        self.df_test_feats: Optional[pd.DataFrame] = None

    def prepare_data(
        self,
        text_loader: Optional[TextBenchmarkDataLoader] = None,
        cache_dir: Optional[Path] = None
    ) -> "TextCandidateScorerPipeline":
        """
        Loads partitions and extracts multi-view features.
        Caches feature matrices to disk if cache_dir is provided.
        """
        if cache_dir is not None:
            cache_p = Path(cache_dir)
            cache_p.mkdir(parents=True, exist_ok=True)
            f_train = cache_p / "text_feats_train.parquet"
            f_val = cache_p / "text_feats_val.parquet"
            f_test = cache_p / "text_feats_test.parquet"

            if f_train.exists() and f_val.exists() and f_test.exists():
                self.df_train_feats = pd.read_parquet(f_train)
                self.df_val_feats = pd.read_parquet(f_val)
                self.df_test_feats = pd.read_parquet(f_test)
                self._is_prepared = True
                return self

        if text_loader is None:
            text_loader = TextBenchmarkDataLoader()

        train_raw = text_loader.get_partition("train")
        val_raw = text_loader.get_partition("val")
        test_raw = text_loader.get_partition("test")

        # 1. Fit multi-view transformations strictly on D_train
        self.feature_pipeline.fit(train_raw)

        # 2. Transform partitions
        self.df_train_feats = self.feature_pipeline.transform_dataframe(train_raw, partition_name="train")
        self.df_val_feats = self.feature_pipeline.transform_dataframe(val_raw, partition_name="val")
        self.df_test_feats = self.feature_pipeline.transform_dataframe(test_raw, partition_name="test")

        if cache_dir is not None:
            self.df_train_feats.to_parquet(cache_p / "text_feats_train.parquet", index=False)
            self.df_val_feats.to_parquet(cache_p / "text_feats_val.parquet", index=False)
            self.df_test_feats.to_parquet(cache_p / "text_feats_test.parquet", index=False)

        self._is_prepared = True
        return self

    def train_models(self) -> Dict[str, BaseCandidateScorer]:
        """
        Initializes and fits all 7 candidate scoring models strictly on D_train features.
        """
        if not self._is_prepared:
            raise RuntimeError("Pipeline must prepare_data() before train_models().")

        X_train = self.df_train_feats[self.feature_columns].to_numpy()
        y_train = self.df_train_feats["ground_truth_label"].to_numpy()

        models = {
            "single_view_word_tfidf": SingleViewThresholdScorer("text_word_tfidf_sim", self.feature_columns),
            "single_view_word_jaccard": SingleViewThresholdScorer("text_word_jaccard", self.feature_columns),
            "single_view_dense_semantic": SingleViewThresholdScorer("text_semantic_minilm_sim", self.feature_columns),
            "weighted_multiview": WeightedMultiViewScorer(self.feature_columns),
            "logistic_regression": LogisticRegressionCandidateScorer(self.feature_columns, random_state=self.random_seed),
            "random_forest": RandomForestCandidateScorer(self.feature_columns, random_state=self.random_seed),
            "gradient_boosting": GradientBoostingCandidateScorer(self.feature_columns, random_state=self.random_seed)
        }

        for name, m in models.items():
            m.fit(X_train, y_train)

        self.models = models
        return self.models

    def evaluate_validation_and_select_thresholds(
        self,
        target_recall: float = 0.98
    ) -> Dict[str, Any]:
        """
        Evaluates candidate models on D_val and selects the optimal threshold
        maximizing workload reduction subject to validation recall >= target_recall.
        """
        if not self.models:
            raise RuntimeError("Must train_models() before threshold selection.")

        X_val = self.df_val_feats[self.feature_columns].to_numpy()
        y_val = self.df_val_feats["ground_truth_label"].to_numpy()

        results = {}
        for name, model in self.models.items():
            val_scores = model.predict_score(X_val)

            # 1. Primary recall-constrained selection
            selection_res = self.threshold_selector.select_recall_constrained_threshold(
                y_val, val_scores, target_recall=target_recall
            )
            self.selected_thresholds[name] = selection_res

            # 2. Multi-point operational profile
            profile = self.threshold_selector.generate_operating_points_profile(y_val, val_scores)

            results[name] = {
                "model_name": name,
                "selected_threshold": selection_res["selected_threshold"],
                "target_recall_constraint": target_recall,
                "target_achieved": selection_res["target_achieved"],
                "validation_metrics": selection_res["validation_metrics"],
                "operating_profile": profile
            }

        return results

    def evaluate_test_set(self) -> Dict[str, Any]:
        """
        Evaluates out-of-sample generalization on D_test using the thresholds
        strictly selected on D_val. D_test labels are NEVER used for threshold selection.
        """
        if not self.selected_thresholds:
            raise RuntimeError("Must select thresholds on validation set before test evaluation.")

        X_test = self.df_test_feats[self.feature_columns].to_numpy()
        y_test = self.df_test_feats["ground_truth_label"].to_numpy()

        test_results = {}
        for name, model in self.models.items():
            test_scores = model.predict_score(X_test)
            tau_star = self.selected_thresholds[name]["selected_threshold"]

            test_metrics = evaluate_candidate_threshold(y_test, test_scores, tau_star)
            val_metrics = self.selected_thresholds[name]["validation_metrics"]

            test_results[name] = {
                "model_name": name,
                "applied_threshold": tau_star,
                "validation_recall": val_metrics["candidate_recall"],
                "validation_workload_reduction": val_metrics["workload_reduction"],
                "test_recall": test_metrics.candidate_recall,
                "test_workload_reduction": test_metrics.workload_reduction,
                "test_precision": test_metrics.candidate_precision,
                "test_false_negatives": test_metrics.false_negatives,
                "test_false_positives": test_metrics.false_positives,
                "test_escalated_pairs": test_metrics.escalated_pairs,
                "test_total_pairs": test_metrics.total_pairs,
                "generalization_gap_recall": round(test_metrics.candidate_recall - val_metrics["candidate_recall"], 6),
                "full_test_metrics": test_metrics.to_dict()
            }

        return test_results

