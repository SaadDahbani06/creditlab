"""Milestone 5: Monte Carlo engine invariants and statistical correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from creditlab.risk.simulation import (
    analytical_mean_and_variance_independent,
    conditional_pd,
    simulate_portfolio,
    vasicek_quantile,
)
from creditlab.risk.var_es import loss_summary


@pytest.fixture(scope="module")
def portfolio():
    rng = np.random.default_rng(11)
    n = 800
    p = np.clip(rng.beta(1.5, 6, n), 0.001, 0.95)
    w = rng.uniform(5_000, 80_000, n)  # LGD * EAD
    return p, w


def test_losses_finite_nonnegative_and_bounded(portfolio):
    p, w = portfolio
    for rho in (0.0, 0.2):
        r = simulate_portfolio(p, w, 2_000, rho=rho, seed=1, chunk_size=300)
        assert np.isfinite(r.losses).all() and (r.losses >= 0).all()
        assert r.losses.max() <= w.sum() + 1e-6
        assert ((r.default_counts >= 0) & (r.default_counts <= len(p))).all()


def test_fixed_seed_reproduces_and_chunking_is_irrelevant(portfolio):
    p, w = portfolio
    a = simulate_portfolio(p, w, 1_500, rho=0.1, seed=42, chunk_size=1_500)
    b = simulate_portfolio(p, w, 1_500, rho=0.1, seed=42, chunk_size=1_500)
    c = simulate_portfolio(p, w, 1_500, rho=0.1, seed=42, chunk_size=137)
    np.testing.assert_array_equal(a.losses, b.losses)
    np.testing.assert_allclose(a.losses, c.losses, rtol=1e-12)
    d = simulate_portfolio(p, w, 1_500, rho=0.1, seed=43, chunk_size=137)
    assert not np.array_equal(a.losses, d.losses)


def test_rho_zero_is_the_independent_case(portfolio):
    """With rho = 0 the latent variable is the idiosyncratic draw alone."""
    p, w = portfolio
    r = simulate_portfolio(p, w, 1_000, rho=0.0, seed=9, chunk_size=250)
    rng = np.random.default_rng(np.random.SeedSequence(9).spawn(2)[1])
    manual = (rng.standard_normal((1_000, len(p))) < norm.ppf(p)) @ w
    np.testing.assert_allclose(r.losses, manual)


def test_independent_mean_and_variance_match_analytical(portfolio):
    p, w = portfolio
    n_sims = 20_000
    r = simulate_portfolio(p, w, n_sims, rho=0.0, seed=2)
    mean, var = analytical_mean_and_variance_independent(p, w)
    se = np.sqrt(var / n_sims)
    assert abs(r.losses.mean() - mean) < 4 * se
    # sample variance of n iid draws: relative SE ~ sqrt(2/n) for near-normal losses
    assert r.losses.var(ddof=1) == pytest.approx(var, rel=4 * np.sqrt(2 / n_sims))


@pytest.mark.parametrize("rho", [0.0, 0.15, 0.4])
def test_marginal_default_probabilities_match_inputs(portfolio, rho):
    p, w = portfolio
    n_sims = 20_000
    r = simulate_portfolio(p, w, n_sims, rho=rho, seed=3)
    # Each account's default indicator over paths is iid Bernoulli(p_i) (paths are independent
    # even when accounts are correlated), so the binomial SE applies to each frequency.
    z = (r.account_default_freq - p) / np.sqrt(p * (1 - p) / n_sims)
    assert np.abs(z).max() < 5.0             # max of 800 approx-normal z-scores
    assert abs(z.mean()) < 4 / np.sqrt(len(p)) * (1 + 10 * rho)  # loose: z's correlated via F
    assert np.mean(np.abs(z) > 1.96) < 0.10   # ~5% expected


def test_correlated_mean_is_unbiased_across_seeds(portfolio):
    p, w = portfolio
    el = np.sum(p * w)
    z = []
    for s in range(40):
        r = simulate_portfolio(p, w, 1_000, rho=0.2, seed=100 + s)
        z.append((r.losses.mean() - el) / (r.losses.std(ddof=1) / np.sqrt(1_000)))
    z = np.array(z)
    assert abs(z.mean()) < 3 / np.sqrt(len(z))
    assert 0.6 < z.std(ddof=1) < 1.5


def test_pairwise_default_correlation_matches_bivariate_normal():
    """Homogeneous portfolio: Var(defaults) = n p(1-p) + n(n-1)(P11 - p^2), P11 = Phi2(t, t; rho)."""
    n, p, rho, n_sims = 200, 0.1, 0.25, 30_000
    r = simulate_portfolio(np.full(n, p), np.ones(n), n_sims, rho=rho, seed=5)
    t = norm.ppf(p)
    p11 = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([t, t])
    var_theory = n * p * (1 - p) + n * (n - 1) * (p11 - p * p)
    # relative SE of a sample variance for a skewed distribution: allow 6%
    assert r.default_counts.var(ddof=1) == pytest.approx(var_theory, rel=0.06)


def test_large_homogeneous_portfolio_quantile_matches_vasicek():
    n, p, rho = 5_000, 0.05, 0.1
    r = simulate_portfolio(np.full(n, p), np.ones(n), 20_000, rho=rho, seed=6, chunk_size=500)
    rate = r.default_counts / n
    for a in (0.95, 0.99):
        assert np.quantile(rate, a) == pytest.approx(vasicek_quantile(p, rho, a), rel=0.05)


def test_conditional_pd_averages_back_to_unconditional():
    p, rho = np.array([0.02, 0.2, 0.6]), 0.3
    f = norm.ppf((np.arange(100_000) + 0.5) / 100_000)  # quadrature over F
    avg = np.mean([conditional_pd(p, rho, x) for x in f[::50]], axis=0)
    np.testing.assert_allclose(avg, p, atol=2e-3)


def test_correlation_fattens_the_tail_but_not_every_quantile(portfolio):
    """Higher rho leaves the mean unchanged, raises upper-tail quantiles and variance, and
    LOWERS the median here: dependence moves mass to both 'quiet' and 'bad' states. So the
    effect of rho is not monotone across all quantiles, and is checked rather than assumed."""
    p, w = portfolio
    stats = {}
    for rho in (0.0, 0.1, 0.3):
        r = simulate_portfolio(p, w, 20_000, rho=rho, seed=8)
        stats[rho] = loss_summary(r.losses, (0.99,))
    el = np.sum(p * w)
    for rho, s in stats.items():
        assert abs(s["mean"] - el) < 4 * s["mc_se_mean"], rho
    assert stats[0.0]["std"] < stats[0.1]["std"] < stats[0.3]["std"]
    assert stats[0.0]["var_99"] < stats[0.1]["var_99"] < stats[0.3]["var_99"]
    assert stats[0.3]["median"] < stats[0.0]["median"]


def test_input_validation():
    with pytest.raises(ValueError):
        simulate_portfolio(np.array([0.5, 1.2]), np.ones(2), 10)
    with pytest.raises(ValueError):
        simulate_portfolio(np.array([0.5]), np.array([-1.0]), 10)
    with pytest.raises(ValueError):
        simulate_portfolio(np.array([0.5]), np.ones(1), 10, rho=1.0)


def test_degenerate_pds():
    r = simulate_portfolio(np.array([0.0, 1.0]), np.array([10.0, 7.0]), 500, rho=0.3, seed=0)
    assert (r.losses == 7.0).all()


def test_memory_is_bounded_by_chunk_size():
    """5,000 paths x 6,000 accounts runs in chunks; peak working array ~ chunk x accounts."""
    p, w = np.full(6_000, 0.2), np.ones(6_000)
    r = simulate_portfolio(p, w, 5_000, rho=0.05, seed=1, chunk_size=500)
    assert r.losses.shape == (5_000,)


@pytest.mark.needs_artifacts
def test_saved_scenarios_pass_sanity_checks(cfg):
    path = cfg.artifacts / "metrics" / "risk" / "scenario_summary.csv"
    if not path.exists():
        pytest.skip("run `creditlab simulate` first")
    s = pd.read_csv(path).set_index("scenario")
    assert {"base", "mild_stress", "severe_stress", "correlated_downturn"} <= set(s.index)
    assert (s["n_sims"] >= 5_000).all()
    assert (s["var_95"] <= s["var_99"]).all()
    assert (s["es_95"] >= s["var_95"]).all() and (s["es_99"] >= s["var_99"]).all()
    indep = s[s["rho"] == 0]
    assert (indep["mean_minus_el_in_se"].abs() < 4).all()
    corr = s[s["rho"] > 0]
    # correlated scenarios: any (mean - EL) gap must be explained by the factor sample
    for _, row in corr.iterrows():
        assert np.sign(row["mean_minus_el_in_se"]) == -np.sign(row["factor_sample_mean_z"]) \
            or abs(row["mean_minus_el_in_se"]) < 2
    assert s.loc["mild_stress", "analytical_el"] > s.loc["base", "analytical_el"]
    assert s.loc["severe_stress", "analytical_el"] > s.loc["mild_stress", "analytical_el"]
