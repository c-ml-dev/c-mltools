"""Metrics and diagnostic plots for a fitted model on held-out data."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize

from ._utils import CLASSIFICATION, resolve_task

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
_STYLES = ["-", "--", "-.", ":"]


def evaluate(model, X_test, y_test, task: str = "auto", plot: bool = True,
             normalize_cm: bool = False, show_report: bool = True) -> dict:
    """Score a fitted model on test data and (optionally) draw diagnostic plots.

    Classification: accuracy, balanced accuracy, precision/recall/F1, ROC AUC,
    log loss + confusion matrix, ROC curve and precision-recall curve.
    Regression: R², RMSE, MAE, MAPE + predicted-vs-actual, residuals, residual histogram.

    Returns a dict of metrics (the figure is under the "figure" key when plot=True).
    """
    task = resolve_task(task, y_test)
    if task == CLASSIFICATION:
        return _classification(model, X_test, y_test, plot, normalize_cm, show_report)
    return _regression(model, X_test, y_test, plot)


# ----------------------------------------------------------- classification
def _classification(model, X, y, plot, normalize_cm, show_report):
    y = np.asarray(y).ravel()
    pred = model.predict(X)
    classes = getattr(model, "classes_", np.unique(y))
    binary = len(classes) == 2
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    avg = "binary" if binary else "weighted"
    pos = classes[1] if binary else None

    out = {
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "precision": skm.precision_score(y, pred, average=avg, pos_label=pos, zero_division=0)
        if binary else skm.precision_score(y, pred, average=avg, zero_division=0),
        "recall": skm.recall_score(y, pred, average=avg, pos_label=pos, zero_division=0)
        if binary else skm.recall_score(y, pred, average=avg, zero_division=0),
        "f1": skm.f1_score(y, pred, average=avg, pos_label=pos, zero_division=0)
        if binary else skm.f1_score(y, pred, average=avg, zero_division=0),
    }
    if proba is not None:
        if binary:
            out["roc_auc"] = skm.roc_auc_score(y == pos, proba[:, 1])
            out["pr_auc"] = skm.average_precision_score(y == pos, proba[:, 1])
        else:
            out["roc_auc_ovr_weighted"] = skm.roc_auc_score(
                y, proba, multi_class="ovr", average="weighted", labels=classes)
        out["log_loss"] = skm.log_loss(y, proba, labels=classes)

    if show_report:
        print(skm.classification_report(y, pred, zero_division=0))
        print(pd.Series(out).round(4).to_string())

    if plot:
        ncols = 3 if proba is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6))
        axes = np.atleast_1d(axes)
        _confusion(axes[0], y, pred, classes, normalize_cm)
        if proba is not None:
            _curves(axes[1], axes[2], y, proba, classes)
        fig.suptitle(f"{type(_final(model)).__name__}: test-set diagnostics", fontweight="bold")
        plt.tight_layout()
        out["figure"] = fig
    return out


def _confusion(ax, y, pred, classes, normalize):
    cm = skm.confusion_matrix(y, pred, labels=classes,
                              normalize="true" if normalize else None)
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = range(len(classes))
    ax.set_xticks(ticks, [str(c) for c in classes], rotation=45 if len(classes) > 4 else 0)
    ax.set_yticks(ticks, [str(c) for c in classes])
    thresh = cm.max() / 2
    for i in ticks:
        for j in ticks:
            txt = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:,}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))


def _curves(ax_roc, ax_pr, y, proba, classes):
    if len(classes) == 2:
        series = [(str(classes[1]), (y == classes[1]).astype(int), proba[:, 1])]
    else:
        Y = label_binarize(y, classes=classes)
        series = [(str(c), Y[:, i], proba[:, i]) for i, c in enumerate(classes)][:8]

    for i, (name, yt, ps) in enumerate(series):
        c, ls = _COLORS[i % len(_COLORS)], _STYLES[i % len(_STYLES)]
        fpr, tpr, _ = skm.roc_curve(yt, ps)
        ax_roc.plot(fpr, tpr, color=c, linestyle=ls,
                    label=f"{name} (AUC {skm.auc(fpr, tpr):.3f})")
        prec, rec, _ = skm.precision_recall_curve(yt, ps)
        ax_pr.plot(rec, prec, color=c, linestyle=ls,
                   label=f"{name} (AP {skm.average_precision_score(yt, ps):.3f})")
        if len(series) == 1:
            ax_pr.axhline(yt.mean(), color="gray", linestyle=":", linewidth=1, label="baseline")

    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate",
               title="ROC curve" + (" (one-vs-rest)" if len(series) > 1 else ""))
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-recall curve")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=8, frameon=False, loc="lower right" if ax is ax_roc else "lower left")
        ax.spines[["top", "right"]].set_visible(False)


# --------------------------------------------------------------- regression
def _regression(model, X, y, plot):
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(model.predict(X), dtype=float).ravel()
    resid = y - pred
    out = {
        "r2": skm.r2_score(y, pred),
        "rmse": float(np.sqrt(skm.mean_squared_error(y, pred))),
        "mae": skm.mean_absolute_error(y, pred),
        "median_ae": skm.median_absolute_error(y, pred),
    }
    if np.all(y != 0):
        out["mape"] = skm.mean_absolute_percentage_error(y, pred)
    print(pd.Series(out).round(4).to_string())

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
        ax = axes[0]
        ax.scatter(y, pred, s=10, alpha=0.4, color=_COLORS[0])
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color=_COLORS[1], linestyle="--", label="perfect")
        ax.set(xlabel="actual", ylabel="predicted", title=f"Predicted vs actual (R² {out['r2']:.3f})")
        ax.legend(frameon=False)

        ax = axes[1]
        ax.scatter(pred, resid, s=10, alpha=0.4, color=_COLORS[0])
        ax.axhline(0, color=_COLORS[1], linestyle="--")
        ax.set(xlabel="predicted", ylabel="residual (actual − predicted)", title="Residuals vs predicted")

        ax = axes[2]
        ax.hist(resid, bins=40, color=_COLORS[0], edgecolor="white", alpha=0.85)
        ax.axvline(0, color=_COLORS[1], linestyle="--")
        ax.set(xlabel="residual", ylabel="count", title=f"Residual distribution (RMSE {out['rmse']:.3g})")
        for a in axes:
            a.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{type(_final(model)).__name__}: test-set diagnostics", fontweight="bold")
        plt.tight_layout()
        out["figure"] = fig
    return out


# ------------------------------------------------------------------ extras
def plot_feature_importance(model, feature_names=None, top_n: int = 20, ax=None):
    """Bar chart of feature_importances_ or |coef_| (works on Pipelines' final step)."""
    est = _final(model)
    if hasattr(est, "feature_importances_"):
        vals, label = np.asarray(est.feature_importances_), "importance"
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        label = "|coefficient|"
    else:
        raise AttributeError(f"{type(est).__name__} has no feature_importances_ or coef_. "
                             "Try sklearn.inspection.permutation_importance instead.")
    if feature_names is None:
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(vals))]
    s = pd.Series(vals, index=list(feature_names)).sort_values().tail(top_n)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=_COLORS[0])
    ax.set_xlabel(label)
    ax.set_title(f"Top {len(s)} features: {type(est).__name__}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


def _final(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model
