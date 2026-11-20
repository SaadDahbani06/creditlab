"""Milestone 4: expected loss, EAD assumptions, stress transformation, VaR / ES."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from creditlab.risk.assumptions import RiskAssumptions, exposure_at_default
from creditlab.risk.expected_loss import (
    account_table,
    expected_loss,
    portfolio_summary,
    risk_band_table,
    sensitivity_grid,
)
from creditlab.risk.stress import stress_pd
from creditlab.risk.var_es import expected_shortfall, loss_summary, value_at_risk

# ------------------------------------------------------------------ EAD and EL by hand


def test_ead_balance_plus_ccf_hand_calculated():
    a = RiskAssumptions(ccf=0.5)
    limit = np.array([100_000, 100_000, 100_000, 50_000])
    balance = np.array([40_000, -5_000, 0, 60_000])  # normal, credit balance, zero, over-limit
    ead = exposure_at_default(limit, balance, a)
    # 40k + 0.5*60k = 70k; credit balance -> drawn 0 -> 0.5*100k; zero -> 50k; over-limit -> 60k
    np.testing.assert_allclose(ead, [70_000, 50_000, 50_000, 60_000])


def test_ead_limit_fraction_and_cap():
    a = RiskAssumptions(ead_method="limit_fraction", limit_fraction=0.4, ead_cap=30_000)
    ead = exposure_at_default([50_000, 100_000], [0, 0], a)
    np.testing.assert_allclose(ead, [20_000, 30_000])


@pytest.mark.parametrize("kwargs", [{"lgd": 1.2}, {"lgd": -0.1}, {"ccf": 1.5},
                                    {"ead_method": "credit_limit"}, {"limit_fraction": 0},
                                    {"ead_cap": -1}])
def test_invalid_assumptions_rejected(kwargs):
    with pytest.raises(ValueError):
        RiskAssumptions(**kwargs)


def test_expected_loss_hand_example():
    pd_ = np.array([0.1, 0.5, 0.02])
    ead = np.array([10_000, 2_000, 50_000])
    el = expected_loss(pd_, 0.6, ead)
    np.testing.assert_allclose(el, [600.0, 600.0, 600.0])


def test_aggregate_el_equals_sum_and_weighted_pd():
    a = RiskAssumptions(lgd=0.6, ccf=0.5)
    pd_ = np.array([0.1, 0.3, 0.05, 0.6])
    limit = np.array([100_000, 20_000, 300_000, 50_000])
    bal = np.array([50_000, 20_000, 0, 10_000])
    acc = account_table(pd_, limit, bal, a)
    s = portfolio_summary(acc)
    assert s["total_el"] == pytest.approx(acc["el"].sum())
    assert s["total_el"] == pytest.approx(np.sum(pd_ * 0.6 * acc["ead"]))
    assert s["ead_weighted_pd"] == pytest.approx(np.sum(pd_ * acc["ead"]) / acc["ead"].sum())
    assert s["el_rate_of_ead"] == pytest.approx(0.6 * s["ead_weighted_pd"])
    assert s["expected_defaults"] == pytest.approx(pd_.sum())


def test_expected_loss_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expected_loss([0.2, 1.1], 0.5, [1, 1])
    with pytest.raises(ValueError):
        expected_loss([0.2, np.nan], 0.5, [1, 1])
    with pytest.raises(ValueError):
        expected_loss([0.2], 0.5, [-1])


def test_risk_bands_partition_portfolio():
    rng = np.random.default_rng(0)
    acc = account_table(rng.random(500), rng.integers(10_000, 500_000, 500),
                        rng.integers(-1_000, 200_000, 500), RiskAssumptions())
    t = risk_band_table(acc, [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0], observed=rng.integers(0, 2, 500))
    assert t["n_accounts"].sum() == 500
    assert t["total_el"].sum() == pytest.approx(acc["el"].sum())
    assert t["share_of_el"].sum() == pytest.approx(1.0)


def test_el_is_linear_in_lgd_and_monotone_in_ccf():
    rng = np.random.default_rng(1)
    p, lim, bal = rng.random(200), rng.integers(10_000, 100_000, 200), rng.integers(0, 50_000, 200)
    g = sensitivity_grid(p, lim, bal, RiskAssumptions(), [0.3, 0.6, 0.9], [0.0, 0.5, 1.0])
    c = g[g.ead_method == "balance_plus_ccf"].pivot(index="ead_parameter", columns="lgd",
                                                    values="total_el")
    np.testing.assert_allclose(c[0.6], 2 * c[0.3])
    np.testing.assert_allclose(c[0.9], 3 * c[0.3])
    assert (c.diff().dropna() >= 0).all().all()


# ------------------------------------------------------------------ stress transformation


def test_stress_is_an_odds_multiplier():
    p = np.array([0.0, 0.01, 0.2, 0.5, 0.9, 1.0])
    s = stress_pd(p, 2.0)
    inner = (p > 0) & (p < 1)
    np.testing.assert_allclose(logit(s[inner]) - logit(p[inner]), np.log(2.0))
    assert s[0] == 0.0 and s[-1] == 1.0
    np.testing.assert_allclose(stress_pd(p, 1.0), p)
    assert np.all(np.diff(s) >= 0)  # ranking preserved
    assert np.all(s >= p)         # multiplier > 1 never lowers a PD
    with pytest.raises(ValueError):
        stress_pd(p, 0.0)


# ------------------------------------------------------------------ VaR / ES


def test_var_es_hand_example_no_ties():
    losses = np.arange(1, 101, dtype=float)  # 1..100, each prob 1/100
    assert value_at_risk(losses, 0.95) == 95.0
    assert expected_shortfall(losses, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert value_at_risk(losses, 0.99) == 99.0
    assert expected_shortfall(losses, 0.99) == pytest.approx(100.0)


def test_es_handles_ties_at_var():
    # 90 zeros and 10 tens: VaR_0.95 = 10, and ES must be 10 (all tail mass is at 10).
    losses = np.r_[np.zeros(90), np.full(10, 10.0)]
    assert value_at_risk(losses, 0.95) == 10.0
    assert expected_shortfall(losses, 0.95) == pytest.approx(10.0)
    # 96 zeros, 4 at 100: VaR_0.95 = 0 (P(L<=0)=0.96 >= 0.95).
    # ES_0.95 = (4*100 + 0*(96 - 95)) / 5 = 80, not the naive mean of the top 5% (= 80 too)
    # and not the mean of losses > VaR (= 100).
    losses = np.r_[np.zeros(96), np.full(4, 100.0)]
    assert value_at_risk(losses, 0.95) == 0.0
    assert expected_shortfall(losses, 0.95) == pytest.approx(80.0)


def test_es_matches_integral_definition_on_random_discrete_losses():
    rng = np.random.default_rng(5)
    losses = rng.integers(0, 15, 2_000).astype(float)  # heavy ties
    for a in (0.9, 0.95, 0.99):
        u = (np.arange(200_000) + 0.5) / 200_000
        u = u[u > a]
        integral = np.quantile(losses, u, method="inverted_cdf").mean()
        assert expected_shortfall(losses, a) == pytest.approx(integral, rel=1e-3)
        assert expected_shortfall(losses, a) >= value_at_risk(losses, a)


def test_loss_summary_orderings():
    rng = np.random.default_rng(3)
    losses = rng.lognormal(10, 0.5, 20_000)
    s = loss_summary(losses, (0.95, 0.99), {"t": float(np.median(losses))})
    assert s["var_95"] <= s["var_99"] <= s["max"]
    assert s["var_95"] <= s["es_95"] and s["var_99"] <= s["es_99"]
    assert s["es_95"] <= s["es_99"]
    assert s["p_exceed_t"] == pytest.approx(0.5, abs=1e-3)
    with pytest.raises(ValueError):
        loss_summary(np.array([1.0, np.inf]))


@pytest.mark.needs_artifacts
def test_saved_expected_loss_is_consistent(cfg):
    from creditlab.utils.config import read_json

    rdir = cfg.artifacts / "metrics" / "risk"
    if not (rdir / "expected_loss_summary.json").exists():
        pytest.skip("run `creditlab simulate` first")
    s = read_json(rdir / "expected_loss_summary.json")
    acc = pd.read_parquet(rdir / "account_expected_loss.parquet")
    assert s["total_el"] == pytest.approx(acc["el"].sum())
    np.testing.assert_allclose(acc["el"], acc["pd"] * acc["lgd"] * acc["ead"])
    assert s["assumptions"]["status"].startswith("HYPOTHETICAL")
    assert s["assumptions"]["horizon"] == "one month" and s["assumptions"]["currency"] == "NTD"
