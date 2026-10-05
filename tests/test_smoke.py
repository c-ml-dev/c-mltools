import matplotlib
matplotlib.use("Agg")
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mltools import ModelComparer, Tuner, evaluate, plot_feature_importance, Int, Float, Categorical


def split(loader):
    X, y = loader(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.mark.parametrize("loader", [load_breast_cancer, load_iris])
def test_classification_flow(loader):
    Xtr, Xte, ytr, yte = split(loader)
    cmp = ModelComparer({
        "rf": RandomForestClassifier(n_estimators=50, random_state=0),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }, cv=3, verbose=False).fit(Xtr, ytr)
    assert cmp.task_ == "classification"
    assert len(cmp.summary()) == 2
    cmp.plot()
    out = evaluate(cmp.best_model_, Xte, yte, show_report=False)
    assert 0 <= out["accuracy"] <= 1
    plot_feature_importance(cmp.best_model_)


@pytest.mark.parametrize("method", ["random", "grid", "optuna"])
def test_tuner_classification(method):
    Xtr, Xte, ytr, yte = split(load_breast_cancer)
    space = {"n_estimators": Int(20, 60, step=20), "max_depth": Categorical([None, 3, 6]),
             "max_features": Categorical(["sqrt", 0.5])}
    t = Tuner(RandomForestClassifier(random_state=0), space, method=method,
              n_trials=5, cv=3, verbose=False).fit(Xtr, ytr)
    assert set(space) <= set(t.best_params_)
    assert 0.5 < t.best_score_ <= 1
    t.plot_history(); t.plot_param("max_depth")
    t.best_model_.predict(Xte)


@pytest.mark.parametrize("method", ["random", "grid", "optuna"])
def test_tuner_regression_preset_pipeline(method):
    Xtr, Xte, ytr, yte = split(load_diabetes)
    pipe = make_pipeline(StandardScaler(), Ridge())
    t = Tuner(pipe, method=method, n_trials=8, cv=3, verbose=False).fit(Xtr, ytr)
    assert t.task_ == "regression"
    assert t.best_score_ > 0  # RMSE flipped to positive
    assert t.results_["score"].iloc[0] == t.results_["score"].min()  # lower is better
    t.plot_param("ridge__alpha")


def test_regression_compare_and_report():
    Xtr, Xte, ytr, yte = split(load_diabetes)
    cmp = ModelComparer({"rf": RandomForestRegressor(n_estimators=50, random_state=0),
                         "ridge": Ridge()}, cv=3, verbose=False).fit(Xtr, ytr)
    assert cmp.task_ == "regression"
    out = evaluate(cmp.best_model_, Xte, yte)
    assert out["rmse"] > 0


def test_xgboost_preset_optuna():
    xgb = pytest.importorskip("xgboost")
    Xtr, Xte, ytr, yte = split(load_breast_cancer)
    t = Tuner(xgb.XGBClassifier(n_jobs=1), method="optuna", n_trials=4, cv=3, verbose=False).fit(Xtr, ytr)
    assert "learning_rate" in t.best_params_


def test_bad_model_does_not_crash_comparer():
    Xtr, Xte, ytr, yte = split(load_breast_cancer)
    cmp = ModelComparer({"good": LogisticRegression(max_iter=5000),
                         "bad": LogisticRegression(C=-1)}, cv=3, verbose=False).fit(Xtr, ytr)
    assert cmp.best_name_ == "good"
    assert "error" in cmp.results_.columns
