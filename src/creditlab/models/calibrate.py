"""Probability calibration fitted on training out-of-fold predictions only.

``CalibratedClassifierCV(cv=K, ensemble=False)`` does the following on the training split:
  1. cross_val_predict: for each of K folds, fit a clone of the pipeline on the other K-1 folds
     and predict the held-out fold, giving one out-of-fold score per training account;
  2. fit the calibrator (Platt sigmoid or isotonic regression) mapping those OOF scores to labels;
  3. refit the pipeline on the full training split.
At prediction time the refit pipeline's score is passed through the calibrator. No validation or
test account touches either the model or the calibrator.

This module also contains calibration diagnostics (reliability tables, ECE, calibration
intercept and slope).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from creditlab.models.tune import cv_splitter
from creditlab.utils.config import Config

EPS = 1e-12


def fit_calibrated(cfg: Config, estimator: Pipeline, method: str, X: pd.DataFrame,
                   y: pd.Series) -> CalibratedClassifierCV:
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(method)
    cal = CalibratedClassifierCV(
        estimator=clone(estimator), method=method,
        cv=cv_splitter(cfg, cfg["calibration"]["cv_folds"], name="calibration"), ensemble=False,
    )
    cal.fit(X, y)
    return cal


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10,
                      strategy: str = "quantile") -> pd.DataFrame:
    """Observed default rate vs mean predicted PD per score band, with binomial standard errors."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) == 1:  # constant predictions (e.g. the dummy model): one bin
            edges = np.array([edges[0], edges[0]])
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        obs, pred = y[m].mean(), p[m].mean()
        rows.append({
            "bin": b, "p_low": edges[b], "p_high": edges[b + 1], "n": n,
            "mean_predicted": pred, "observed_rate": obs,
            "observed_se": np.sqrt(max(pred * (1 - pred), EPS) / n),
            "expected_defaults": p[m].sum(), "observed_defaults": int(y[m].sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / n) |observed_b - predicted_b| over quantile bins."""
    t = reliability_table(y, p, n_bins)
    return float((t["n"] / t["n"].sum() * (t["observed_rate"] - t["mean_predicted"]).abs()).sum())


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Fit logit P(y=1) = a + b * logit(p). Perfect calibration: a = 0, b = 1.

    b < 1 indicates over-confident (too extreme) predictions; b > 1 under-confident.
    Unpenalised fit (C very large).
    """
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, max_iter=1000).fit(z, np.asarray(y))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def calibration_in_the_large(y, p) -> float:
    """Mean predicted PD minus observed default rate."""
    return float(np.mean(p) - np.mean(y))

