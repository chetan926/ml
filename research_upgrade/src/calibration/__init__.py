"""
Calibration and Uncertainty Package Exports (Phase 3.4).
"""

from research_upgrade.src.calibration.calibrators import (
    BaseCalibrator,
    UncalibratedBaseline,
    PlattScaler,
    IsotonicCalibrator,
    TemperatureScaler,
)
from research_upgrade.src.calibration.calibration_metrics import (
    compute_calibration_metrics,
)
from research_upgrade.src.calibration.uncertainty import (
    compute_predictive_entropy,
    compute_confidence_margin,
    compute_maximum_class_probability,
    evaluate_selective_prediction,
)
from research_upgrade.src.calibration.group_conformal import (
    TextGroupConformalPredictor,
)

__all__ = [
    "BaseCalibrator",
    "UncalibratedBaseline",
    "PlattScaler",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "compute_calibration_metrics",
    "compute_predictive_entropy",
    "compute_confidence_margin",
    "compute_maximum_class_probability",
    "evaluate_selective_prediction",
    "TextGroupConformalPredictor",
]

