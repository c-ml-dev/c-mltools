"""Compare several models with cross-validation in one call."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from ._utils import (
    boxplot,
    default_metrics,
    default_primary_metric,
    flip_if_neg,
    make_cv,
    pretty_metric,
    resolve_task,
)


class ModelComparer:
    """Cross-validate a dict of models and return a ranked leaderboard.

    Parameters
    ----------
    models : dict[str, estimator]
        Anything with the sklearn fit/predict API (sklearn, XGBoost, LightGBM, Pipelines).
    task : {"auto", "classification", "regression"}
    cv : int or CV splitter
        An int becomes a shuffled StratifiedKFold (classification) or KFold (regression).
    metrics : list[str], optional
        sklearn scorer names. Defaults depend on the task.
    rank_by : str, optional
        Metric used to sort the leaderboard. Defaults to ROC AUC (binary),
        weighted F1 (multiclass) or RMSE (regression).
    n_jobs : int
        Parallel jobs for cross_validate (-1 = all cores).

    Example
    -------
    >>> cmp = ModelComparer({"rf": RandomForestClassifier(), "lr": LogisticRegression()})
    >>> cmp.fit(X_train, y_train)
    >>> cmp.results_          # leaderboard DataFrame
    >>> cmp.best_model_       # refit on all of X_train, ready to use
    >>> cmp.plot()
    """

    def __init__(self, models: dict, task: str = "auto", cv=5, metrics=None,
                 rank_by: str | None = None, refit_best: bool = True,
                 n_jobs: int = -1, random_state: int | None = 42, verbose: bool = True):
        if not isinstance(models, dict) or not models:
            raise ValueError("models must be a non-empty dict like {'name': estimator}")
        self.models = models
        self.task = task
        self.cv = cv
        self.metrics = metrics
        self.rank_by = rank_by
        self.refit_best = refit_best
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.task_ = resolve_task(self.task, y)
        metrics = list(self.metrics or default_metrics(self.task_, y))
        rank_by = self.rank_by or default_primary_metric(self.task_, y)
        if rank_by not in metrics:
            metrics.append(rank_by)
        self.metrics_, self.rank_by_ = metrics, rank_by
        cv = make_cv(self.task_, self.cv, self.random_state)

        rows, self.fold_scores_ = [], {}
        for name, model in self.models.items():
            if self.verbose:
                print(f"  cross-validating {name} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                out = cross_validate(clone(model), X, y, cv=cv, scoring=metrics,
                                     n_jobs=self.n_jobs, error_score="raise")
            except Exception as e:  # keep going if one model breaks
                if self.verbose:
                    print(f"FAILED ({type(e).__name__}: {e})")
                rows.append({"model": name, "error": f"{type(e).__name__}: {e}"})
                continue
            elapsed = time.perf_counter() - t0
            row = {"model": name}
            folds = {}
            for m in metrics:
                vals = flip_if_neg(m, out[f"test_{m}"])
                folds[pretty_metric(m)] = vals
                row[f"{pretty_metric(m)}_mean"] = vals.mean()
                row[f"{pretty_metric(m)}_std"] = vals.std()
            row["fit_time_s"] = out["fit_time"].mean()
            row["total_time_s"] = elapsed
            rows.append(row)
            self.fold_scores_[name] = pd.DataFrame(folds)
            if self.verbose:
                key = pretty_metric(rank_by)
                print(f"{key} = {row[f'{key}_mean']:.4f} ± {row[f'{key}_std']:.4f}  ({elapsed:.1f}s)")

        df = pd.DataFrame(rows)
        sort_col = f"{pretty_metric(rank_by)}_mean"
        ascending = rank_by.startswith("neg_")  # errors: lower is better
        if sort_col in df:
            df = df.sort_values(sort_col, ascending=ascending, na_position="last")
        self.results_ = df.reset_index(drop=True).set_index("model")

        ok = [n for n in self.results_.index if n in self.fold_scores_]
        if not ok:
            raise RuntimeError("Every model failed; see the 'error' column in results_.")
        self.best_name_ = ok[0]
        if self.refit_best:
            self.best_model_ = clone(self.models[self.best_name_]).fit(X, y)
        if self.verbose:
            print(f"Best: {self.best_name_}")
        return self

    def summary(self, decimals: int = 4) -> pd.DataFrame:
        """Compact leaderboard: 'mean ± std' strings, one column per metric."""
        self._check_fitted()
        out = pd.DataFrame(index=self.results_.index)
        for m in self.metrics_:
            k = pretty_metric(m)
            if f"{k}_mean" in self.results_:
                out[k] = (self.results_[f"{k}_mean"].round(decimals).astype(str) + " ± "
                          + self.results_[f"{k}_std"].round(decimals).astype(str))
        if "fit_time_s" in self.results_:
            out["fit_time_s"] = self.results_["fit_time_s"].round(2)
        return out

    def plot(self, metric: str | None = None, ax=None):
        """Box plot of per-fold scores for each model, best at the top."""
        self._check_fitted()
        key = pretty_metric(metric or self.rank_by_)
        names = [n for n in self.results_.index if n in self.fold_scores_][::-1]
        data = [self.fold_scores_[n][key].values for n in names]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.5))
        boxplot(ax, data, horizontal=True, widths=0.6, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.35),
                   medianprops=dict(color="#DD8452", linewidth=2))
        ax.set_yticks(range(1, len(names) + 1), names)
        for i, d in enumerate(data, start=1):
            ax.scatter(d, np.full_like(d, i, dtype=float), s=14, color="#4C72B0", zorder=3)
        ax.set_xlabel(key)
        ax.set_title(f"Cross-validated {key} by model ({len(data[0])} folds)", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return ax

    def _check_fitted(self):
        if not hasattr(self, "results_"):
            raise RuntimeError("Call .fit(X, y) first.")
