"""Milestone 3: calibration procedure (no leakage), diagnostics and selection rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from creditlab.models.calibrate import (
    calibration_in_the_large,
    calibration_intercept_slope,
    expected_calibration_error,
    fit_calibrated,
    reliability_table,
)
from creditlab.models.tune import cv_splitter, logistic_pipeline
from creditlab.utils.config import read_json


@pytest.fixture()
def calibrated_world():
    rng = np.random.default_rng(7)
    p = rng.beta(2, 6, 50_000)
    y = (rng.random(p.size) < p).astype(int)
    return y, p


def test_perfectly_calibrated_predictions_pass_diagnostics(calibrated_world):
    y, p = calibrated_world
    assert expected_calibration_error(y, p) < 0.01
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05
    assert abs(calibration_in_the_large(y, p)) < 0.005


def test_overconfident_predictions_have_slope_below_one(calibrated_world):
    y, p = calibrated_world
    logit = np.log(p / (1 - p))
    p_over = 1 / (1 + np.exp(-2.0 * logit))  # twice as extreme in log-odds
    _, b = calibration_intercept_slope(y, p_over)
    assert b == pytest.approx(0.5, abs=0.05)
    assert expected_calibration_error(y, p_over) > expected_calibration_error(y, p)


def test_reliability_table_counts_and_bins(calibrated_world):
    y, p = calibrated_world
    t = reliability_table(y, p, n_bins=10)
    assert t["n"].sum() == len(y)
    assert len(t) == 10
    assert (t["mean_predicted"].diff().dropna() > 0).all()
    assert t["observed_defaults"].sum() == y.sum()
    # constant predictions collapse to one bin instead of failing
    assert len(reliability_table(y, np.full(len(y), 0.2))) == 1


@pytest.mark.needs_data
def test_isotonic_calibrator_is_fit_on_training_out_of_fold_scores(cfg):
    """Reconstruct the calibrated model by hand: OOF decision scores on the training rows ->
    isotonic fit -> applied to a pipeline refit on all training rows. Must match exactly."""
    from creditlab.data.preprocess import TARGET, load_split, model_input_columns

    train = load_split(cfg, "train").sample(3000, random_state=1).reset_index(drop=True)
    X, y = train[model_input_columns(cfg)], train[TARGET].astype(int)
    base = logistic_pipeline(cfg, C=0.1)
    cal = fit_calibrated(cfg, base, "isotonic", X, y)
    assert len(cal.calibrated_classifiers_) == 1  # ensemble=False: one refit model + one calibrator

    cv = cv_splitter(cfg, cfg["calibration"]["cv_folds"], name="calibration")
    oof = cross_val_predict(clone(base), X, y, cv=cv, method="decision_function")
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    refit = clone(base).fit(X, y)
    X_new = load_split(cfg, "validation")[model_input_columns(cfg)].head(500)
    manual = iso.predict(refit.decision_function(X_new))
    np.testing.assert_allclose(cal.predict_proba(X_new)[:, 1], manual, atol=1e-12)


@pytest.mark.needs_data
def test_calibrated_probabilities_are_valid(cfg):
    from creditlab.data.preprocess import TARGET, load_split, model_input_columns

    train = load_split(cfg, "train").sample(3000, random_state=2).reset_index(drop=True)
    X, y = train[model_input_columns(cfg)], train[TARGET].astype(int)
    for method in ("sigmoid", "isotonic"):
        p = fit_calibrated(cfg, logistic_pipeline(cfg, C=1.0), method, X, y).predict_proba(X)[:, 1]
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1


@pytest.mark.needs_artifacts
def test_saved_selection_follows_documented_rule(cfg):
    path = cfg.artifacts / "metrics" / "selection.json"
    if not path.exists():
        pytest.skip("run `creditlab all` first")
    from creditlab.models.evaluate import log_loss_ci_high

    sel = read_json(path)
    vm = sel["validation_metrics"]
    boot = pd.DataFrame(sel["validation_bootstrap"])
    for fam, d in sel["families"].items():
        raw = vm[f"{fam}_raw"]
        qualifying = [m for m in cfg["calibration"]["methods"]
                      if vm[f"{fam}_{m}"]["log_loss"] < raw["log_loss"]
                      and vm[f"{fam}_{m}"]["brier"] <= raw["brier"]
                      and log_loss_ci_high(boot, f"{fam}_{m}", f"{fam}_raw") < 0]
        assert d["qualifying_calibrations"] == qualifying
        if qualifying:
            assert d["chosen_variant"] == min(qualifying, key=lambda m: vm[f"{fam}_{m}"]["log_loss"])
        else:
            assert d["chosen_variant"] == "raw"
    primary_ll = vm[sel["primary_model"]]["log_loss"]
    assert primary_ll == min(vm[d["model"]]["log_loss"] for d in sel["families"].values())
    # reliability tables exist for both splits and every non-dummy candidate
    rel = pd.read_csv(cfg.artifacts / "metrics" / "reliability_tables.csv")
    assert set(rel["split"]) == {"validation", "test"}
