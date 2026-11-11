"""
Multi-View Candidate Scoring and Score Fusion Architecture (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection

Methodological Standards:
- Outputs continuous candidate scores s(x).
- Uncalibrated candidate scores are strictly NOT described as calibrated probabilities.
- Fits supervised fusion models strictly on D_train feature representations.
- Zero label snooping: labels are NEVER used as feature inputs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


class BaseCandidateScorer(ABC):
    """Abstract base class for all candidate generation scoring models."""

    def __init__(self, model_name: str, feature_names: List[str]):
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseCandidateScorer":
        """Fits scoring model strictly using training features and training labels."""
        pass

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Outputs 1D array of continuous candidate scores s(x).
        Higher score indicates higher similarity / escalation priority.
        NOTE: Uncalibrated scores are NOT calibrated probabilities.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }


class SingleViewThresholdScorer(BaseCandidateScorer):
    """
    Baseline candidate scorer relying on a single feature view (e.g. Word TF-IDF or Token Jaccard).
    Non-parametric: does not require training.
    """

    def __init__(self, feature_name: str, all_feature_names: List[str]):
        super().__init__(f"single_view_{feature_name}", all_feature_names)
        assert feature_name in all_feature_names, (
            f"Feature '{feature_name}' not in provided feature names: {all_feature_names}"
        )
        self.target_feature = feature_name
        self.target_idx = all_feature_names.index(feature_name)
        self.is_fitted = True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SingleViewThresholdScorer":
        # Non-parametric baseline
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr[:, self.target_idx]


class WeightedMultiViewScorer(BaseCandidateScorer):
    """
    Combines feature views linearly: s(x) = sum(w_i * x_i) / sum(w_i).
    Weights can be pre-specified or learned via non-negative least squares on D_train.
    """

    def __init__(
        self,
        feature_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        model_name: str = "weighted_multiview_scorer"
    ):
        super().__init__(model_name, feature_names)
        self.weights = weights
        self._weight_vector: Optional[np.ndarray] = None
        if weights is not None:
            self._weight_vector = np.array([weights.get(fn, 1.0) for fn in feature_names], dtype=np.float64)
            sum_w = np.sum(self._weight_vector)
            if sum_w > 0:
                self._weight_vector /= sum_w
            self.is_fitted = True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "WeightedMultiViewScorer":
        """Fits non-negative weights proportional to Pearson correlation with y_train."""
        if self._weight_vector is None:
            X_arr = np.asarray(X_train, dtype=np.float64)
            y_arr = np.asarray(y_train, dtype=np.float64)
            
            # Simple univariate correlation weights bounded to non-negative
            corrs = []
            for j in range(X_arr.shape[1]):
                col = X_arr[:, j]
                std_c = np.std(col)
                std_y = np.std(y_arr)
                if std_c > 1e-9 and std_y > 1e-9:
                    r = float(np.corrcoef(col, y_arr)[0, 1])
                    corrs.append(max(0.0, r))
                else:
                    corrs.append(0.0)
            
            corrs_arr = np.array(corrs, dtype=np.float64)
            total = np.sum(corrs_arr)
            if total > 0:
                self._weight_vector = corrs_arr / total
            else:
                self._weight_vector = np.ones(X_arr.shape[1], dtype=np.float64) / X_arr.shape[1]
            
            self.weights = {fn: float(self._weight_vector[i]) for i, fn in enumerate(self.feature_names)}
            
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self._weight_vector is None:
            raise RuntimeError("WeightedMultiViewScorer must be fitted before predict_score.")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self._weight_vector)


class LogisticRegressionCandidateScorer(BaseCandidateScorer):
    """
    L2-regularized Logistic Regression candidate scorer fitted strictly on D_train.
    Outputs continuous sigmoid score in [0, 1].
    NOTE: Uncalibrated candidate escalation score, NOT a calibrated probability.
    """

    def __init__(
        self,
        feature_names: List[str],
        C: float = 1.0,
        random_state: int = 42
    ):
        super().__init__("logistic_regression_scorer", feature_names)
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionCandidateScorer":
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.int32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scorer must be fitted before predict_score.")
        X_arr = np.asarray(X, dtype=np.float64)
        # Use predict_proba for positive class continuous score in [0, 1]
        probs = self.model.predict_proba(X_arr)
        # Return probability column corresponding to class 1
        pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
        return probs[:, pos_idx]


class RandomForestCandidateScorer(BaseCandidateScorer):
    """
    Random Forest ensemble candidate scorer fitted strictly on D_train.
    Outputs mean tree agreement score in [0, 1].
    NOTE: Uncalibrated ensemble score, NOT a calibrated probability.
    """

    def __init__(
        self,
        feature_names: List[str],
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42
    ):
        super().__init__("random_forest_scorer", feature_names)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestCandidateScorer":
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.int32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scorer must be fitted before predict_score.")
        X_arr = np.asarray(X, dtype=np.float64)
        probs = self.model.predict_proba(X_arr)
        pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
        return probs[:, pos_idx]


class GradientBoostingCandidateScorer(BaseCandidateScorer):
    """
    Lightweight histogram-based gradient boosting candidate scorer.
    Fast inference and non-linear decision boundary.
    """

    def __init__(
        self,
        feature_names: List[str],
        max_iter: int = 100,
        max_depth: int = 5,
        random_state: int = 42
    ):
        super().__init__("gradient_boosting_scorer", feature_names)
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GradientBoostingCandidateScorer":
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.int32)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scorer must be fitted before predict_score.")
        X_arr = np.asarray(X, dtype=np.float64)
        probs = self.model.predict_proba(X_arr)
        pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
        return probs[:, pos_idx]

