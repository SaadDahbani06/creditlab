"""Research extension: PD estimation uncertainty propagated into VaR/ES.

Unit tests check the mathematics on synthetic portfolios; artifact tests check that the extension
reproduces the original engine exactly when PDs are held fixed, and that no test-split
information enters the refits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from creditlab.research.pd_uncertainty import (
    bootstrap_indices,
    mc_noise_diagnostic,
    pd_uncertainty_summary,
    risk_across_models,
    semi_analytic_mixture,
)
from creditlab.risk.simulation import (
    analytical_mean_and_variance_independent,
    one_factor_loss_quantiles,
    simulate_portfolio,
    simulate_portfolio_multi,
)
from creditlab.risk.var_es import expected_shortfall, value_at_risk
from creditlab.utils.config import read_json


@pytest.fixture(scope="module")
def portfolio():
    rng = np.random.default_rng(7)
    n = 1500
    p = np.clip(rng.beta(1.5, 6, n), 0.002, 0.95)
    w = rng.uniform(5_000, 80_000, n)
    return p, w


# ----------------------------------------------------------------------------- engine
@pytest.mark.parametrize("rho", [0.0, 0.04, 0.3])
@pytest.mark.parametrize("chunk", [97, 1000])
def test_multi_equals_single_runs_with_same_seed(portfolio, rho, chunk):
    """Common random numbers: row k must equal a separate run with the same seed, exactly."""
    p, w = portfolio
    rng = np.random.default_rng(1)
    P = np.vstack([p, np.clip(p * 1.3, 0, 1), np.clip(p + rng.normal(0, 0.01, p.size), 0, 1)])
    L = simulate_portfolio_multi(P, w, n_sims=600, rho=rho, seed=42, chunk_size=chunk)
    for k in range(P.shape[0]):
        single = simulate_portfolio(P[k], w, n_sims=600, rho=rho, seed=42, chunk_size=chunk).losses
        np.testing.assert_array_equal(L[k], single)


def test_multi_rejects_bad_input(portfolio):
    p, w = portfolio
    with pytest.raises(ValueError):
        simulate_portfolio_multi(np.vstack([p, p])[:, :-1], w, 10)
    with pytest.raises(ValueError):
        simulate_portfolio_multi(np.vstack([p, p + 2]), w, 10)
    with pytest.raises(ValueError):
        simulate_portfolio_multi(p, w, 10, rho=1.0)


def test_semi_analytic_independent_moments_exact(portfolio):
    p, w = portfolio
    sa = one_factor_loss_quantiles(p, w, 0.0)
    mean, var = analytical_mean_and_variance_independent(p, w)
    assert sa["mean"] == pytest.approx(mean, rel=1e-12)
    assert sa["std"] == pytest.approx(np.sqrt(var), rel=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.0, 0.04, 0.15])
def test_semi_analytic_matches_monte_carlo(portfolio, rho):
    """Noise-free reference agrees with a large MC run (normal approx of L|F: ~1% at 99%)."""
    p, w = portfolio
    sa = one_factor_loss_quantiles(p, w, rho)
    L = simulate_portfolio(p, w, n_sims=40_000, rho=rho, seed=3, chunk_size=2000).losses
    assert L.mean() == pytest.approx(sa["mean"], rel=0.01)
    assert L.std() == pytest.approx(sa["std"], rel=0.03)
    assert value_at_risk(L, 0.99) == pytest.approx(sa["var_99"], rel=0.02)
    assert expected_shortfall(L, 0.99) == pytest.approx(sa["es_99"], rel=0.025)
    assert sa["es_99"] >= sa["var_99"] >= sa["var_95"]


def test_semi_analytic_mean_invariant_to_rho(portfolio):
    p, w = portfolio
    means = [one_factor_loss_quantiles(p, w, r)["mean"] for r in (0.0, 0.05, 0.2)]
    assert np.allclose(means, float(p @ w), rtol=1e-6)


@pytest.mark.parametrize("rho", [0.0, 0.08])
def test_mixture_of_identical_models_equals_single_model(portfolio, rho):
    p, w = portfolio
    single = one_factor_loss_quantiles(p, w, rho, n_grid=2001)
    mix = semi_analytic_mixture(np.vstack([p, p, p]), w, rho)
    for k in ("mean", "var_95", "var_99", "es_99"):
        assert mix[k] == pytest.approx(single[k], rel=1e-9), k


def test_mixture_widens_independent_tail(portfolio):
    """Mixing location-shifted loss distributions raises the mixture's upper quantile above the
    quantile at the average PD (spread effect > 0 when defaults are independent)."""
    p, w = portfolio
    P = np.vstack([np.clip(p * m, 0, 1) for m in (0.9, 1.0, 1.1)])
    mix = semi_analytic_mixture(P, w, 0.0)
    mid = one_factor_loss_quantiles(P.mean(axis=0), w, 0.0)
    assert mix["mean"] == pytest.approx(mid["mean"], rel=1e-9)
    assert mix["var_99"] > mid["var_99"] and mix["es_99"] > mid["es_99"]


# ----------------------------------------------------------------------------- outer level
def test_bootstrap_indices_shape_range_determinism():
    a = bootstrap_indices(50, 8, np.random.default_rng(3))
    b = bootstrap_indices(50, 8, np.random.default_rng(3))
    assert a.shape == (8, 50) and a.min() >= 0 and a.max() < 50
    np.testing.assert_array_equal(a, b)
    # with replacement: typical bootstrap sample contains ~63% unique rows
    share_unique = np.mean([np.unique(r).size / 50 for r in bootstrap_indices(50, 400,
                                                                             np.random.default_rng(0))])
    assert 0.58 < share_unique < 0.68
    with pytest.raises(ValueError):
        bootstrap_indices(0, 3, np.random.default_rng(0))


def test_group_folds_never_split_duplicated_accounts():
    """Re-tuning inside a bootstrap sample must not score an account it was fitted on."""
    rng = np.random.default_rng(0)
    ids = np.arange(300)
    idx = bootstrap_indices(300, 1, rng)[0]
    groups, y = ids[idx], rng.integers(0, 2, 300)[idx]
    for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=1).split(idx, y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_pd_summary_identical_models_have_zero_spread(portfolio):
    p, _ = portfolio
    s = pd_uncertainty_summary(np.vstack([p] * 5), p)
    assert s["median_sd"] == 0 and s["mean_pd_bootstrap_sd"] == 0


def test_pd_summary_separates_common_and_account_specific_variation(portfolio):
    p, _ = portfolio
    rng = np.random.default_rng(5)
    shift = np.vstack([np.clip(p + c, 0, 1) for c in rng.normal(0, 0.01, 60)])
    noise = np.clip(p + rng.normal(0, 0.01, (60, p.size)), 0, 1)
    assert pd_uncertainty_summary(shift, p)["pd_variance_common_share"] > 0.95
    assert pd_uncertainty_summary(noise, p)["pd_variance_common_share"] < 0.05


# ----------------------------------------------------------------------------- inner level
def test_total_variance_decomposition_and_mixture(portfolio):
    p, w = portfolio
    rng = np.random.default_rng(2)
    P = np.vstack([np.clip(p * m, 0, 1) for m in rng.uniform(0.9, 1.1, 15)])
    r = risk_across_models(P, w, rho=0.0, n_sims=2000, seed=9, chunk=500, levels=(0.95, 0.99),
                           references={"original_fit": p}, semi_analytic=False)
    pooled = r["boot_losses"].ravel()
    d = r["decomposition"]
    # law of total variance: exact for the empirical equal-weight mixture (population moments)
    assert d["within_model_variance"] + d["between_model_variance"] == pytest.approx(
        pooled.var(), rel=1e-9)
    # between-model variance of simulated means ~ variance of analytic ELs
    assert d["between_model_variance"] == pytest.approx(d["between_model_variance_analytic_el"],
                                                        rel=0.15)
    assert r["mixture"]["var_99"] == value_at_risk(pooled, 0.99)
    # the reference row is simulated on the SAME draws as the bootstrap rows
    same = simulate_portfolio(p, w, n_sims=2000, rho=0.0, seed=9, chunk_size=500).losses
    assert r["references"]["original_fit"]["var_99"] == value_at_risk(same, 0.99)


def test_mc_noise_diagnostic_detects_pure_noise():
    rng = np.random.default_rng(0)
    sa = rng.normal(100, 2, 300)
    pm = pd.DataFrame({"var_99": sa + 5 + rng.normal(0, 0.1, 300), "sa_var_99": sa})
    d = mc_noise_diagnostic(pm, "var_99")
    assert d["common_offset_mc_minus_sa"] == pytest.approx(5, abs=0.05)
    assert d["sd_model_specific_mc_noise"] < 0.2 < d["sd_across_models_semi_analytic"]


# ----------------------------------------------------------------------------- research artifacts
@pytest.fixture(scope="module")
def ext(cfg):
    d = cfg.artifacts / "metrics" / "uncertainty"
    if not (d / "summary.json").exists():
        pytest.skip("run `creditlab uncertainty` first")
    return d, read_json(d / "summary.json")


@pytest.mark.needs_artifacts
def test_refit_on_original_data_reproduces_frozen_predictions(ext):
    _, s = ext
    for fam, v in s["families"].items():
        assert v["pd"]["refit_reproduces_saved_predictions_max_abs_diff"] < 1e-9, fam


@pytest.mark.needs_artifacts
def test_fixed_pd_rows_reproduce_original_scenarios_exactly(cfg, ext):
    """With fixed PDs, the extension must give the ORIGINAL engine's numbers (same seed/draws)."""
    d, s = ext
    if s["inner_sims"] != cfg["simulation"]["n_sims"]:
        pytest.skip("inner_sims differs from the original scenario path count")
    scen = pd.read_csv(cfg.artifacts / "metrics" / "risk" / "scenario_summary.csv").set_index("scenario")
    prim = read_json(cfg.artifacts / "metrics" / "selection.json")["primary_family"]
    for rho, name in ((0.0, "base"), (0.04, "base_correlated")):
        ref = s["families"][prim]["risk"][f"rho_{rho}"]["references"]["original_fit"]
        for m in ("var_95", "var_99", "es_95", "es_99"):
            assert ref[m] == pytest.approx(scen.loc[name, m], rel=1e-12), (name, m)


@pytest.mark.needs_artifacts
def test_bootstrap_uses_training_split_only(cfg, ext):
    """Bootstrap training default rates must be resamples of the training split's rate."""
    d, _ = ext
    perf = pd.read_csv(d / "bootstrap_model_validation_xgboost.csv")
    split = {r["split"]: r for r in read_json(cfg.artifacts / "metrics" / "split_summary.json")}
    n, rate = split["train"]["n"], split["train"]["default_rate"]
    se = np.sqrt(rate * (1 - rate) / n)
    assert abs(perf.train_default_rate.mean() - rate) < 4 * se / np.sqrt(len(perf)) + 1e-9
    assert perf.train_default_rate.std() == pytest.approx(se, rel=0.35)
