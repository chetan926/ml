"""
Calibration Module for Academic Integrity Detection Framework (Phase 3.4).

Provides probability calibration algorithms:
- Uncalibrated baseline
- Regularized Platt Scaling (logistic sigmoid)
- Isotonic Regression (piecewise non-decreasing)
- Temperature Scaling
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar


class BaseCalibrator:
    """Abstract base class for calibration models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BaseCalibrator":
        raise NotImplementedError

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Returns array of shape (N, 2) where col 1 is P(Y=1|s)."""
        raise NotImplementedError

    def predict_positive_proba(self, scores: np.ndarray) -> np.ndarray:
        """Returns 1D array of shape (N,) of calibrated positive probabilities."""
        proba = self.predict_proba(scores)
        return proba[:, 1]


class UncalibratedBaseline(BaseCalibrator):
    """Pass-through calibrator treating raw continuous score as pseudo-probability."""

    def __init__(self):
        super().__init__(name="uncalibrated_baseline")

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "UncalibratedBaseline":
        self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - s, s])


class PlattScaler(BaseCalibrator):
    """
    Regularized Platt Scaling.
    Fits a 1D logistic regression model P(Y=1 | s) = 1 / (1 + exp(-(A*s + B))).
    Uses L2 regularization (C parameter) to prevent over-fitting on small calibration sets.
    """

    def __init__(self, C: float = 1.0, random_state: int = 42):
        super().__init__(name="platt_scaling")
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            penalty="l2",
            random_state=self.random_state,
            max_iter=1000
        )
        self.slope: float = 1.0
        self.intercept: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels, dtype=np.int32)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Handle degenerate case where calibration partition lacks both classes
            self.slope = 0.0
            self.intercept = 0.0 if unique_classes[0] == 0 else 5.0
            self.is_fitted = True
            return self

        self.model.fit(X, y)
        self.slope = float(self.model.coef_[0][0])
        self.intercept = float(self.model.intercept_[0])
        self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PlattScaler must be fitted before predict_proba.")
        X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        if len(self.model.classes_) < 2:
            # Degenerate case fallback
            p1 = 1.0 / (1.0 + np.exp(-(self.slope * X[:, 0] + self.intercept)))
            return np.column_stack([1.0 - p1, p1])
        return self.model.predict_proba(X)


class IsotonicCalibrator(BaseCalibrator):
    """
    Non-parametric Isotonic Regression Calibrator.
    Fits a monotonic non-decreasing step function mapping raw score to empirical probability.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        super().__init__(name="isotonic_regression")
        self.out_of_bounds = out_of_bounds
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=self.out_of_bounds)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        X = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before predict_proba.")
        X = np.asarray(scores, dtype=np.float64).ravel()
        p1 = self.model.predict(X)
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p1, p1])


class TemperatureScaler(BaseCalibrator):
    """
    Temperature Scaling Calibrator.
    Optimizes a single scalar T > 0 on logits z: P(Y=1) = sigma(z / T).
    Preserves model ranking (ROC-AUC invariant) while adjusting confidence dispersion.
    """

    def __init__(self):
        super().__init__(name="temperature_scaling")
        self.temperature: float = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        s = np.clip(np.asarray(scores, dtype=np.float64).ravel(), 1e-6, 1.0 - 1e-6)
        logits = np.log(s / (1.0 - s))
        y = np.asarray(labels, dtype=np.float64).ravel()

        def nll(T):
            T = max(T, 1e-4)
            scaled_logits = logits / T
            # Numerically stable log loss
            log_p = -np.logaddexp(0.0, -scaled_logits)
            log_1mp = -np.logaddexp(0.0, scaled_logits)
            loss = -(y * log_p + (1.0 - y) * log_1mp).mean()
            return loss

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=np.float64).ravel(), 1e-6, 1.0 - 1e-6)
        logits = np.log(s / (1.0 - s))
        scaled_logits = logits / self.temperature
        p1 = 1.0 / (1.0 + np.exp(-scaled_logits))
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p1, p1])

