"""mltools: C's personal toolkit for comparing, tuning, and evaluating ML models."""
from .compare import ModelComparer
from .report import evaluate, plot_feature_importance
from .spaces import Categorical, Float, Int, get_preset
from .tune import Tuner

__version__ = "0.1.0"
__all__ = ["ModelComparer", "Tuner", "evaluate", "plot_feature_importance",
           "Int", "Float", "Categorical", "get_preset"]
