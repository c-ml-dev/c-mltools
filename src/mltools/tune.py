"""Hyperparameter tuning with one interface for grid, random, and Optuna search."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from . import spaces as sp
from ._utils import boxplot, default_primary_metric, flip_if_neg, make_cv, pretty_metric, resolve_task


class Tuner:
    """Tune one model's hyperparameters.

    Parameters
    ----------
    model : estimator
        sklearn-compatible model or Pipeline.
    param_space : dict, optional
        Values can be ``Int``, ``Float``, ``Categorical`` (from mltools.spaces)
        or plain lists. If omitted, a preset is used for common models.
    method : {"random", "grid", "optuna"}
    n_trials : int
        Number of candidates for random/optuna (ignored by grid).
    metric : str, optional
        sklearn scorer name. Defaults depend on the task.
    timeout : float, optional
        Optuna only: stop after this many seconds.

    Example
    -------
    >>> t = Tuner(XGBClassifier(), method="optuna", n_trials=50).fit(X_train, y_train)
    >>> t.best_params_, t.best_score_
    >>> t.best_model_.predict(X_test)
    """

    def __init__(self, model, param_space: dict | None = None, method: str = "random",
                 n_trials: int = 30, task: str = "auto", cv=5, metric: str | None = None,
                 n_jobs: int = -1, random_state: int | None = 42, timeout: float | None = None,
                 verbose: bool = True):
        if method not in ("random", "grid", "optuna"):
            raise ValueError("method must be 'random', 'grid' or 'optuna'")
        self.model = model
        self.param_space = param_space
        self.method = method
        self.n_trials = n_trials
        self.task = task
        self.cv = cv
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.timeout = timeout
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        self.task_ = resolve_task(self.task, y)
        self.metric_ = self.metric or default_primary_metric(self.task_, y)
        self.space_ = sp.normalize(self.param_space or sp.get_preset(self.model))
        cv = make_cv(self.task_, self.cv, self.random_state)

        t0 = time.perf_counter()
        if self.method == "optuna":
            self._fit_optuna(X, y, cv)
        else:
            self._fit_sklearn(X, y, cv)
        self.elapsed_s_ = time.perf_counter() - t0

        # store score in "human" orientation (RMSE positive, lower=better)
        self.best_score_ = float(flip_if_neg(self.metric_, self._raw_best_score))
        if self.verbose:
            print(f"Best {pretty_metric(self.metric_)}: {self.best_score_:.4f} "
                  f"({len(self.results_)} candidates, {self.elapsed_s_:.1f}s)")
            for k, v in self.best_params_.items():
                print(f"  {k}: {v}")
        return self

    def _fit_sklearn(self, X, y, cv):
        if self.method == "grid":
            grid = sp.to_grid(self.space_)
            n = int(np.prod([len(v) for v in grid.values()]))
            if self.verbose:
                print(f"Grid search over {n} combinations x {getattr(cv, 'n_splits', '?')} folds")
            search = GridSearchCV(clone(self.model), grid, scoring=self.metric_, cv=cv,
                                  n_jobs=self.n_jobs, refit=True)
        else:
            if self.verbose:
                print(f"Random search: {self.n_trials} candidates")
            search = RandomizedSearchCV(clone(self.model), sp.to_random(self.space_),
                                        n_iter=self.n_trials, scoring=self.metric_, cv=cv,
                                        n_jobs=self.n_jobs, refit=True,
                                        random_state=self.random_state)
        search.fit(X, y)
        self.search_ = search
        self.best_params_ = search.best_params_
        self.best_model_ = search.best_estimator_
        self._raw_best_score = search.best_score_

        res = pd.DataFrame(search.cv_results_)
        params = pd.json_normalize(res["params"])
        score = flip_if_neg(self.metric_, res["mean_test_score"])
        self.results_ = pd.concat(
            [params, pd.DataFrame({"score": score, "score_std": res["std_test_score"],
                                   "fit_time_s": res["mean_fit_time"]})], axis=1)
        self.results_["trial"] = np.arange(len(self.results_))
        self._sort_results()

    def _fit_optuna(self, X, y, cv):
        try:
            import optuna
        except ImportError as e:
            raise ImportError("method='optuna' needs optuna: pip install optuna "
                              "(or install this package with the [tuning] extra)") from e
        # per-trial INFO logs flood notebooks; the progress bar is enough
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {k: p.suggest(trial, k) for k, p in self.space_.items()}
            trial.set_user_attr("params", params)
            model = clone(self.model).set_params(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=self.metric_, n_jobs=self.n_jobs)
            trial.set_user_attr("std", float(scores.std()))
            return float(scores.mean())

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)  # sklearn scores: higher=better
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout,
                       show_progress_bar=self.verbose)
        self.study_ = study
        self.best_params_ = study.best_trial.user_attrs["params"]
        self._raw_best_score = study.best_value
        self.best_model_ = clone(self.model).set_params(**self.best_params_).fit(X, y)

        done = [t for t in study.trials if t.value is not None]
        params = pd.DataFrame([t.user_attrs["params"] for t in done])
        self.results_ = params.assign(
            score=flip_if_neg(self.metric_, np.array([t.value for t in done])),
            score_std=[t.user_attrs.get("std", np.nan) for t in done],
            fit_time_s=[t.duration.total_seconds() if t.duration else np.nan for t in done],
            trial=[t.number for t in done],
        )
        self._sort_results()

    def _sort_results(self):
        lower_is_better = self.metric_.startswith("neg_")
        self.results_ = (self.results_.sort_values("score", ascending=lower_is_better)
                         .reset_index(drop=True))

    # ---------------------------------------------------------------- plots
    def plot_history(self, ax=None):
        """Score per trial plus the running best. Useful for spotting when to stop."""
        self._check_fitted()
        df = self.results_.sort_values("trial")
        lower = self.metric_.startswith("neg_")
        best = df["score"].cummin() if lower else df["score"].cummax()
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(df["trial"], df["score"], s=18, color="#4C72B0", alpha=0.6, label="trial")
        ax.plot(df["trial"], best, color="#DD8452", linewidth=2, label="best so far")
        ax.set_xlabel("trial")
        ax.set_ylabel(pretty_metric(self.metric_))
        ax.set_title(f"{self.method.capitalize()} search: {pretty_metric(self.metric_)} per trial",
                     fontweight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return ax

    def plot_param(self, param: str, ax=None):
        """Score vs. one hyperparameter: shows whether your range is sensible."""
        self._check_fitted()
        if param not in self.results_:
            raise KeyError(f"{param!r} not in results. Options: {list(self.space_)}")
        df = self.results_[[param, "score"]].dropna(subset=["score"])
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        x = df[param]
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 6:
            ax.scatter(x, df["score"], s=20, color="#4C72B0", alpha=0.7)
            p = self.space_.get(param)
            if getattr(p, "log", False):
                ax.set_xscale("log")
        else:
            # None shows up as NaN in the results table; label it explicitly
            keys = x.astype(object).where(x.notna(), "None").map(str)
            labels, data = zip(*[(k, g.values) for k, g in df["score"].groupby(keys, sort=True)])
            boxplot(ax, list(data), labels=list(labels))
        ax.set_xlabel(param)
        ax.set_ylabel(pretty_metric(self.metric_))
        ax.set_title(f"{pretty_metric(self.metric_)} vs {param}", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return ax

    def _check_fitted(self):
        if not hasattr(self, "results_"):
            raise RuntimeError("Call .fit(X, y) first.")
