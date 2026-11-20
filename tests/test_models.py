"""Milestone 2: pipelines, leakage guards, metric correctness, determinism and saved artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from creditlab.data.preprocess import TARGET, load_split, model_input_columns
from creditlab.models.baselines import MONEY_COLS, dummy_model, signed_log1p
from creditlab.models.evaluate import (
    bootstrap,
    probability_metrics,
    select_threshold_max_f1,
    threshold_metrics,
)
from creditlab.models.train import ids_fingerprint, train_models
from creditlab.models.tune import logistic_pipeline, xgboost_pipeline
from creditlab.utils.config import read_json

# ------------------------------------------------------------------ pure metric checks


def test_probability_metrics_hand_example():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = probability_metrics(y, p)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    expected_ll = -np.mean([np.log(0.9), np.log(0.6), np.log(0.35), np.log(0.8)])
    assert m["log_loss"] == pytest.approx(expected_ll)
    assert m["roc_auc"] == pytest.approx(0.75)  # 3 of 4 (pos, neg) pairs ordered correctly
    assert m["accuracy_at_0.5"] == pytest.approx(0.75)


def test_threshold_metrics_confusion_matrix():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.2, 0.6, 0.7, 0.4, 0.9])
    r = threshold_metrics(y, p, 0.5)
    assert r["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)


def test_max_f1_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    t = select_threshold_max_f1(y, p)
    best = max(f1_score(y, (p >= c).astype(int)) for c in np.unique(p))
    assert f1_score(y, (p >= t).astype(int)) == pytest.approx(best)


def test_accuracy_is_misleading_for_imbalanced_target():
    """A model that predicts 'no default' for everyone scores accuracy = 1 - base rate."""
    y = np.r_[np.zeros(78), np.ones(22)].astype(int)
    r = threshold_metrics(y, np.full(100, 0.22), 0.5)
    assert r["accuracy"] == pytest.approx(0.78) and r["recall"] == 0.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    p = rng.random(800)
    y = (rng.random(800) < p).astype(int)
    q = np.clip(p + rng.normal(0, 0.2, 800), 0.01, 0.99)
    s = bootstrap(y, {"a": p, "b": q}, reps=200, rng=np.random.default_rng(3), pairs=[("a", "b")])
    row = s[(s.model == "a") & (s.metric == "roc_auc")].iloc[0]
    assert row.ci_low < roc_auc_score(y, p) < row.ci_high
    diff = s[(s.model == "a - b") & (s.metric == "roc_auc")].iloc[0]
    assert diff.ci_low > 0  # the noisier model is genuinely worse here


# ------------------------------------------------------------------ pipelines / leakage


@pytest.fixture(scope="module")
def train_sample(cfg):
    try:
        train = load_split(cfg, "train")
    except FileNotFoundError:
        pytest.skip("run `creditlab prepare` first")
    return train.sample(4000, random_state=0).reset_index(drop=True)


@pytest.mark.needs_data
def test_preprocessing_statistics_learned_from_training_rows_only(cfg, train_sample):
    X, y = train_sample[model_input_columns(cfg)], train_sample[TARGET]
    pipe = logistic_pipeline(cfg, C=1.0).fit(X, y)
    scaler = pipe.named_steps["prep"].named_steps["columns"].named_transformers_["money"]["scale"]
    expected = signed_log1p(train_sample[MONEY_COLS].astype(float)).mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-10)
    # and NOT the statistics of the full dataset
    full = pd.concat([train_sample, load_split(cfg, "test")])
    full_mean = signed_log1p(full[MONEY_COLS].astype(float)).mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean)


@pytest.mark.needs_data
def test_model_inputs_never_include_sensitive_or_target(cfg, train_sample):
    X = train_sample[model_input_columns(cfg)]
    pipe = xgboost_pipeline(cfg, n_estimators=20, max_depth=2).fit(X, train_sample[TARGET])
    names = set(pipe.named_steps["prep"].get_feature_names_out())
    assert not names & {"SEX", "AGE", "MARRIAGE", "EDUCATION", "ID", TARGET, "split"}


def test_train_models_refuses_non_training_rows(cfg, synthetic_accounts):
    df = synthetic_accounts.assign(split=["train"] * 399 + ["test"])
    with pytest.raises(ValueError, match="only training-split rows"):
        train_models(cfg, df)


@pytest.mark.needs_data
def test_xgboost_fit_is_deterministic(cfg, train_sample):
    X, y = train_sample[model_input_columns(cfg)], train_sample[TARGET]
    p1 = xgboost_pipeline(cfg, n_estimators=50, max_depth=3, subsample=0.8,
                          colsample_bytree=0.8).fit(X, y).predict_proba(X)[:, 1]
    p2 = xgboost_pipeline(cfg, n_estimators=50, max_depth=3, subsample=0.8,
                          colsample_bytree=0.8).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(p1, p2)


def test_dummy_predicts_training_base_rate(synthetic_accounts):
    y = synthetic_accounts[TARGET]
    p = dummy_model().fit(synthetic_accounts[["LIMIT_BAL"]], y).predict_proba(
        synthetic_accounts[["LIMIT_BAL"]])[:, 1]
    assert np.allclose(p, y.mean())


# ------------------------------------------------------------------ saved research artifacts


@pytest.fixture(scope="module")
def artifacts(cfg):
    m = cfg.artifacts / "metrics"
    needed = [m / "selection.json", m / "test_evaluation.json",
              cfg.artifacts / "predictions" / "test_predictions.parquet"]
    if not all(p.exists() for p in needed):
        pytest.skip("research artifacts missing: run `creditlab all`")
    return {
        "selection": read_json(m / "selection.json"),
        "evaluation": read_json(m / "test_evaluation.json"),
        "meta": read_json(cfg.artifacts / "models" / "training_metadata.json"),
        "test_pred": pd.read_parquet(cfg.artifacts / "predictions" / "test_predictions.parquet"),
        "val_pred": pd.read_parquet(cfg.artifacts / "predictions" / "validation_predictions.parquet"),
    }


@pytest.mark.needs_artifacts
def test_saved_leakage_checks_passed(artifacts, cfg):
    assert all(artifacts["evaluation"]["leakage_checks"].values())
    train = load_split(cfg, "train")
    assert artifacts["meta"]["train_id_sha256"] == ids_fingerprint(train["ID"])
    assert artifacts["meta"]["n_train"] == 18000
    assert not set(artifacts["test_pred"]["ID"]) & set(train["ID"])
    assert artifacts["selection"]["selected_utc"] <= artifacts["evaluation"]["evaluated_utc"]


@pytest.mark.needs_artifacts
def test_reported_metrics_recompute_from_saved_predictions(artifacts):
    tp = artifacts["test_pred"]
    for name, m in artifacts["evaluation"]["test_metrics"].items():
        p = tp[f"p_{name}"].to_numpy()
        assert roc_auc_score(tp["y"], p) == pytest.approx(m["roc_auc"], abs=1e-12)
        assert log_loss(tp["y"], np.clip(p, 1e-15, 1 - 1e-15)) == pytest.approx(m["log_loss"])
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1


@pytest.mark.needs_artifacts
def test_saved_models_reproduce_saved_predictions(artifacts, cfg):
    from creditlab.models.train import load_model

    test = load_split(cfg, "test")
    X = test[model_input_columns(cfg)]
    for name in ("logistic_raw", "xgboost_raw", artifacts["selection"]["primary_model"]):
        p = load_model(cfg, name).predict_proba(X)[:, 1]
        np.testing.assert_allclose(p, artifacts["test_pred"][f"p_{name}"].to_numpy(), atol=1e-7)


@pytest.mark.needs_artifacts
def test_thresholds_come_from_validation(artifacts):
    vp = artifacts["val_pred"]
    for fam, d in artifacts["selection"]["families"].items():
        assert d["threshold_max_f1"] == pytest.approx(
            select_threshold_max_f1(vp["y"], vp[f"p_{d['model']}"])), fam
