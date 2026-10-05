# mltools

My personal toolkit for the repetitive parts of ML projects: comparing models with
cross-validation, tuning hyperparameters (grid, random, or Optuna), and producing
test-set reports with diagnostic plots. Works for classification and regression,
and with anything that follows the scikit-learn API (sklearn, XGBoost, LightGBM, Pipelines).

## Install

```bash
# in a notebook / Colab
!pip install "mltools[all] @ git+https://github.com/c-ml-dev/c-mltools.git"

# pin a release so old projects don't break when the library changes
!pip install "mltools[all] @ git+https://github.com/c-ml-dev/c-mltools.git@v0.1.0"
```

Extras: `[tuning]` adds Optuna, `[boost]` adds XGBoost + LightGBM, `[all]` adds both.
Without extras you get the sklearn-only core.

## Quick start

```python
from mltools import ModelComparer, Tuner, evaluate, plot_feature_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1. Compare candidates with 5-fold CV (task is auto-detected from y)
cmp = ModelComparer({
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}).fit(X_train, y_train)
cmp.summary()        # leaderboard: mean ± std per metric
cmp.plot()           # per-fold score spread

# 2. Tune the winner (preset search space used automatically for common models)
tuner = Tuner(XGBClassifier(), method="optuna", n_trials=50).fit(X_train, y_train)
tuner.best_params_
tuner.plot_history()            # is it still improving?
tuner.plot_param("max_depth")   # is the range sensible?

# 3. Final report on the held-out test set
results = evaluate(tuner.best_model_, X_test, y_test)
plot_feature_importance(tuner.best_model_, top_n=15)
```

## Custom search spaces

Define a space once; it works with all three methods.

```python
from mltools import Int, Float, Categorical

space = {
    "n_estimators": Int(100, 800, step=100),
    "max_depth": Categorical([None, 5, 10, 20]),
    "learning_rate": Float(1e-3, 0.3, log=True),
}
Tuner(model, space, method="grid").fit(X, y)    # Int/Float become ~5 evenly spaced values
Tuner(model, space, method="random", n_trials=40).fit(X, y)
Tuner(model, space, method="optuna", n_trials=40).fit(X, y)
```

For a Pipeline, prefix keys with the step name (`"randomforestclassifier__max_depth"`);
presets do this for you. See `mltools.spaces.PRESETS` for the models with built-in spaces.

## Defaults

| | Classification | Regression |
|---|---|---|
| CV | shuffled StratifiedKFold(5) | shuffled KFold(5) |
| Ranking / tuning metric | ROC AUC (binary), weighted F1 (multiclass) | RMSE |
| Compare metrics | accuracy, F1, ROC AUC | R², RMSE, MAE |

Override with `metric=` / `metrics=` / `rank_by=` using any sklearn scorer name.
Error metrics are reported as positive numbers (RMSE 3.2, not -3.2).

## Development

```bash
git clone https://github.com/c-ml-dev/c-mltools.git
cd c-mltools
pip install -e ".[all,dev]"
pytest
```

Release a version: bump `version` in `pyproject.toml` and `__init__.py`, then
`git tag v0.1.1 && git push --tags`.
