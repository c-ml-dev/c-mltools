"""Define a search space once, use it with grid, random, or Optuna search.

Example
-------
>>> from mltools.spaces import Int, Float, Categorical
>>> space = {
...     "n_estimators": Int(100, 800, step=100),
...     "max_depth": Int(3, 12),
...     "learning_rate": Float(1e-3, 0.3, log=True),
...     "subsample": Float(0.6, 1.0),
...     "booster": Categorical(["gbtree", "dart"]),
... }

Plain Python lists also work anywhere and are treated as Categorical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import stats


@dataclass
class Int:
    low: int
    high: int
    log: bool = False
    step: int = 1
    n_grid: int = 5  # points used for grid search when the range is large

    def to_random(self):
        if self.log:
            # sample in log space, round to int
            return _LogIntDist(self.low, self.high)
        if self.step == 1:
            return stats.randint(self.low, self.high + 1)
        return list(range(self.low, self.high + 1, self.step))

    def to_grid(self) -> list[int]:
        values = list(range(self.low, self.high + 1, self.step))
        if len(values) <= self.n_grid:
            return values
        if self.log:
            pts = np.geomspace(self.low, self.high, self.n_grid)
        else:
            pts = np.linspace(self.low, self.high, self.n_grid)
        return sorted({int(round(p)) for p in pts})

    def suggest(self, trial, name: str):
        step = 1 if self.log else self.step
        return trial.suggest_int(name, self.low, self.high, step=step, log=self.log)


@dataclass
class Float:
    low: float
    high: float
    log: bool = False
    n_grid: int = 5

    def to_random(self):
        if self.log:
            return stats.loguniform(self.low, self.high)
        return stats.uniform(self.low, self.high - self.low)

    def to_grid(self) -> list[float]:
        pts = np.geomspace(self.low, self.high, self.n_grid) if self.log \
            else np.linspace(self.low, self.high, self.n_grid)
        return [float(p) for p in pts]

    def suggest(self, trial, name: str):
        return trial.suggest_float(name, self.low, self.high, log=self.log)


@dataclass
class Categorical:
    choices: Sequence[Any]

    def to_random(self):
        return list(self.choices)

    def to_grid(self):
        return list(self.choices)

    def suggest(self, trial, name: str):
        # Optuna only stores None/bool/int/float/str; index anything else
        simple = all(c is None or isinstance(c, (bool, int, float, str)) for c in self.choices)
        if simple:
            return trial.suggest_categorical(name, list(self.choices))
        idx = trial.suggest_categorical(f"{name}__idx", list(range(len(self.choices))))
        return self.choices[idx]


class _LogIntDist:
    """Log-uniform integer distribution with an .rvs() for RandomizedSearchCV."""

    def __init__(self, low, high):
        self._dist = stats.loguniform(low, high + 1)

    def rvs(self, size=None, random_state=None):
        out = np.floor(self._dist.rvs(size=size, random_state=random_state)).astype(int)
        return out


def _as_param(p):
    if isinstance(p, (Int, Float, Categorical)):
        return p
    if isinstance(p, (list, tuple)):
        return Categorical(list(p))
    raise TypeError(f"Unsupported search-space entry: {p!r}. Use Int, Float, Categorical or a list.")


def normalize(space: dict) -> dict:
    return {k: _as_param(v) for k, v in space.items()}


def to_grid(space: dict) -> dict:
    return {k: p.to_grid() for k, p in normalize(space).items()}


def to_random(space: dict) -> dict:
    return {k: p.to_random() for k, p in normalize(space).items()}


# ---------------------------------------------------------------------------
# Presets: reasonable starting spaces keyed by estimator class name.
# ---------------------------------------------------------------------------
_TREE_ENSEMBLE = {
    "n_estimators": Int(100, 800, step=100),
    "max_depth": Categorical([None, 5, 10, 20, 30]),
    "min_samples_split": Int(2, 20),
    "min_samples_leaf": Int(1, 10),
    "max_features": Categorical(["sqrt", "log2", 0.5, 1.0]),
}

_XGB = {
    "n_estimators": Int(100, 1000, step=100),
    "max_depth": Int(3, 10),
    "learning_rate": Float(0.01, 0.3, log=True),
    "subsample": Float(0.6, 1.0),
    "colsample_bytree": Float(0.6, 1.0),
    "min_child_weight": Int(1, 10),
    "reg_lambda": Float(1e-3, 10, log=True),
}

_LGBM = {
    "n_estimators": Int(100, 1000, step=100),
    "num_leaves": Int(15, 255, log=True),
    "learning_rate": Float(0.01, 0.3, log=True),
    "subsample": Float(0.6, 1.0),
    "colsample_bytree": Float(0.6, 1.0),
    "min_child_samples": Int(5, 100, log=True),
    "reg_lambda": Float(1e-3, 10, log=True),
}

_HGB = {
    "learning_rate": Float(0.01, 0.3, log=True),
    "max_iter": Int(100, 1000, step=100),
    "max_leaf_nodes": Int(15, 127, log=True),
    "min_samples_leaf": Int(5, 100, log=True),
    "l2_regularization": Float(1e-4, 10, log=True),
}

_KNN = {
    "n_neighbors": Int(3, 50, log=True),
    "weights": Categorical(["uniform", "distance"]),
    "p": Categorical([1, 2]),
}

PRESETS: dict[str, dict] = {
    "RandomForestClassifier": _TREE_ENSEMBLE,
    "RandomForestRegressor": _TREE_ENSEMBLE,
    "ExtraTreesClassifier": _TREE_ENSEMBLE,
    "ExtraTreesRegressor": _TREE_ENSEMBLE,
    "DecisionTreeClassifier": {k: v for k, v in _TREE_ENSEMBLE.items() if k != "n_estimators"},
    "DecisionTreeRegressor": {k: v for k, v in _TREE_ENSEMBLE.items() if k != "n_estimators"},
    "GradientBoostingClassifier": {
        "n_estimators": Int(100, 600, step=100),
        "learning_rate": Float(0.01, 0.3, log=True),
        "max_depth": Int(2, 8),
        "subsample": Float(0.6, 1.0),
    },
    "GradientBoostingRegressor": {
        "n_estimators": Int(100, 600, step=100),
        "learning_rate": Float(0.01, 0.3, log=True),
        "max_depth": Int(2, 8),
        "subsample": Float(0.6, 1.0),
    },
    "HistGradientBoostingClassifier": _HGB,
    "HistGradientBoostingRegressor": _HGB,
    "XGBClassifier": _XGB,
    "XGBRegressor": _XGB,
    "LGBMClassifier": _LGBM,
    "LGBMRegressor": _LGBM,
    "LogisticRegression": {
        "C": Float(1e-3, 1e2, log=True),
        "class_weight": Categorical([None, "balanced"]),
    },
    "Ridge": {"alpha": Float(1e-3, 1e3, log=True)},
    "Lasso": {"alpha": Float(1e-4, 10, log=True)},
    "ElasticNet": {"alpha": Float(1e-4, 10, log=True), "l1_ratio": Float(0.05, 0.95)},
    "SVC": {"C": Float(1e-2, 1e2, log=True), "gamma": Categorical(["scale", "auto", 1e-3, 1e-2, 1e-1])},
    "SVR": {"C": Float(1e-2, 1e2, log=True), "epsilon": Float(1e-3, 1, log=True)},
    "KNeighborsClassifier": _KNN,
    "KNeighborsRegressor": _KNN,
}


def get_preset(model) -> dict:
    """Return the preset space for a model (or the final step of a Pipeline).

    Pipeline keys are automatically prefixed, e.g. ``model__max_depth``.
    """
    prefix = ""
    est = model
    if hasattr(model, "steps"):  # sklearn Pipeline
        step_name, est = model.steps[-1]
        prefix = f"{step_name}__"
    name = type(est).__name__
    if name not in PRESETS:
        raise KeyError(
            f"No preset search space for {name}. Pass param_space=... yourself. "
            f"Presets exist for: {', '.join(sorted(PRESETS))}"
        )
    return {prefix + k: v for k, v in PRESETS[name].items()}
