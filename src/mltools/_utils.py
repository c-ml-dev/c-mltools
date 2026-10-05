"""Internal helpers shared across the toolkit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

CLASSIFICATION = "classification"
REGRESSION = "regression"


def detect_task(y, max_classes: int = 20) -> str:
    """Guess whether y is a classification or regression target.

    Strings, booleans and categoricals are always classification. Numeric
    targets count as classification only if they are integer-valued with
    few unique values (default: 20 or fewer).
    """
    y = pd.Series(np.asarray(y).ravel())
    if y.dtype == bool or y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype):
        return CLASSIFICATION
    if pd.api.types.is_float_dtype(y) and not np.allclose(y.dropna() % 1, 0):
        return REGRESSION
    return CLASSIFICATION if y.nunique() <= max_classes else REGRESSION


def resolve_task(task: str | None, y) -> str:
    if task in (None, "auto"):
        return detect_task(y)
    if task not in (CLASSIFICATION, REGRESSION):
        raise ValueError(f"task must be 'classification', 'regression' or 'auto', got {task!r}")
    return task


def default_metrics(task: str, y) -> list[str]:
    """Sensible sklearn scorer names for comparing models."""
    if task == REGRESSION:
        return ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
    n_classes = pd.Series(np.asarray(y).ravel()).nunique()
    if n_classes == 2:
        return ["accuracy", "f1", "roc_auc"]
    return ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]


def default_primary_metric(task: str, y) -> str:
    """The single metric used to rank models / drive tuning by default."""
    if task == REGRESSION:
        return "neg_root_mean_squared_error"
    n_classes = pd.Series(np.asarray(y).ravel()).nunique()
    return "roc_auc" if n_classes == 2 else "f1_weighted"


def make_cv(task: str, cv, random_state: int | None):
    """Turn an int into a shuffled (Stratified)KFold; pass splitters through."""
    if not isinstance(cv, int):
        return cv
    if task == CLASSIFICATION:
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def pretty_metric(name: str) -> str:
    """'neg_root_mean_squared_error' -> 'root_mean_squared_error'."""
    return name[4:] if name.startswith("neg_") else name


def flip_if_neg(name: str, values):
    """sklearn reports error metrics as negatives; flip them back for display."""
    return -values if name.startswith("neg_") else values


def boxplot(ax, data, labels=None, horizontal=False, **kw):
    """ax.boxplot that works across matplotlib versions (orientation/tick_labels renames)."""
    import matplotlib
    major, minor = (int(v) for v in matplotlib.__version__.split(".")[:2])
    new = (major, minor) >= (3, 10)
    if labels is not None:
        kw["tick_labels" if (major, minor) >= (3, 9) else "labels"] = labels
    if horizontal:
        if new:
            kw["orientation"] = "horizontal"
        else:
            kw["vert"] = False
    return ax.boxplot(data, **kw)
