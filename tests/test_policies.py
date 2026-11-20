"""Milestone 6: hypothetical policy economics, policy rules and validation-only cutoff choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditlab.decisions.policies import (
    PolicyEconomics,
    account_economics,
    break_even_revenue_rate,
    choose_pd_threshold,
    policy_expected_contribution,
    policy_pd_threshold,
    policy_random,
    policy_retain_all,
    summarize,
)
from creditlab.risk.assumptions import RiskAssumptions

ECON = PolicyEconomics(monthly_revenue_rate=0.02, monthly_cost_per_account=10.0,
                       risk=RiskAssumptions(lgd=0.5, ccf=0.5))


def test_account_economics_hand_example():
    # limit 100k, balance 40k: EAD = 40k + 0.5*60k = 70k; revenue = 0.02*40k = 800
    df = account_economics([0.01, 0.10], [100_000, 100_000], [40_000, 40_000], ECON, [0, 1])
    np.testing.assert_allclose(df["ead"], [70_000, 70_000])
    np.testing.assert_allclose(df["expected_loss"], [0.01 * 0.5 * 70_000, 0.10 * 0.5 * 70_000])
    np.testing.assert_allclose(df["expected_contribution"],
                               [0.99 * 800 - 350 - 10, 0.90 * 800 - 3_500 - 10])
    np.testing.assert_allclose(df["realized_hypothetical_contribution"], [800 - 10, -35_000 - 10])


def test_break_even_pd_and_revenue_rate_are_consistent():
    rng = np.random.default_rng(0)
    n = 2_000
    p = rng.uniform(0.001, 0.3, n)
    lim = rng.integers(10_000, 300_000, n)
    bal = rng.integers(-2_000, 200_000, n)
    df = account_economics(p, lim, bal, ECON)
    positive = df["expected_contribution"] > 0
    assert (positive == (df["pd"] < df["break_even_pd"])).all()
    r_star = break_even_revenue_rate(p, bal, df["ead"], ECON.risk.lgd, ECON.monthly_cost_per_account)
    ok = np.isfinite(r_star)
    at_break_even = (1 - p[ok]) * r_star[ok] * np.maximum(bal[ok], 0) - p[ok] * 0.5 * df["ead"][ok] - 10
    np.testing.assert_allclose(at_break_even, 0, atol=1e-6)


@pytest.fixture()
def toy():
    rng = np.random.default_rng(1)
    n = 1_000
    p = rng.beta(2, 7, n)
    y = (rng.random(n) < p).astype(int)
    return account_economics(p, rng.integers(10_000, 200_000, n), rng.integers(0, 100_000, n),
                             ECON, y)


def test_policy_rules(toy):
    assert policy_retain_all(toy).all()
    m = policy_pd_threshold(toy, 0.2)
    assert (m == (toy["pd"] < 0.2)).all()
    assert (policy_expected_contribution(toy) == (toy["expected_contribution"] > 0)).all()
    r1 = policy_random(len(toy), 300, np.random.default_rng(5))
    r2 = policy_random(len(toy), 300, np.random.default_rng(5))
    assert r1.sum() == 300 and (r1 == r2).all()


def test_summary_totals_and_empty_selection(toy):
    s = summarize(toy, policy_retain_all(toy))
    assert s["total_expected_loss"] == pytest.approx(toy["expected_loss"].sum())
    assert s["expected_net_contribution"] == pytest.approx(toy["expected_contribution"].sum())
    assert s["observed_default_rate"] == pytest.approx(toy["observed_default"].mean())
    e = summarize(toy, np.zeros(len(toy), bool))
    assert e["n_selected"] == 0 and e["expected_net_contribution"] == 0.0
    assert np.isnan(e["mean_pd"])


def test_cutoff_rule_max_bad_rate(toy):
    c = choose_pd_threshold(toy, 99, rule="max_bad_rate", target_bad_rate=0.12)
    kept = toy["pd"] < c["threshold"]
    assert toy.loc[kept, "observed_default"].mean() <= 0.12
    # the next candidate cutoff would breach the target (largest feasible cutoff)
    grid = np.asarray(c["grid"])
    nxt = grid[grid > c["threshold"]]
    if nxt.size:
        assert toy.loc[toy["pd"] < nxt[0], "observed_default"].mean() > 0.12


def test_cutoff_rule_max_realized_contribution(toy):
    c = choose_pd_threshold(toy, 99, rule="max_realized_contribution")
    assert c["validation_total"] == pytest.approx(max(c["grid_totals"]))


def test_cutoff_requires_observed_outcomes():
    df = account_economics([0.1], [10_000], [1_000], ECON)  # no labels
    with pytest.raises(ValueError):
        choose_pd_threshold(df, 9)


@pytest.mark.needs_artifacts
def test_policy_cutoff_does_not_depend_on_test_data(cfg):
    """Shuffle TEST labels and PDs: the validation-selected cutoff must not move."""
    from creditlab.decisions.compare import _cohort, evaluate_policies
    from creditlab.utils.config import read_json

    sel = cfg.artifacts / "metrics" / "selection.json"
    if not sel.exists():
        pytest.skip("run `creditlab evaluate` first")
    primary = read_json(sel)["primary_model"]
    val, test = _cohort(cfg, "validation", primary), _cohort(cfg, "test", primary)
    econ = PolicyEconomics.from_config(cfg)
    a = evaluate_policies(cfg, econ, val, test, with_tail=False)
    rng = np.random.default_rng(0)
    scrambled = test.assign(y=rng.permutation(test["y"].to_numpy()),
                            pd=rng.permutation(test["pd"].to_numpy()))
    b = evaluate_policies(cfg, econ, val, scrambled, with_tail=False)
    assert a["threshold"]["threshold"] == b["threshold"]["threshold"]


@pytest.mark.needs_artifacts
def test_saved_policy_table_is_consistent(cfg):
    path = cfg.artifacts / "metrics" / "policy" / "policy_comparison.csv"
    if not path.exists():
        pytest.skip("run `creditlab policy` first")
    t = pd.read_csv(path, index_col=0)
    assert t.loc["A_retain_all", "share_selected"] == 1.0
    assert t.loc["D_random_equal_size", "n_selected"] == t.loc["B_pd_threshold", "n_selected"]
    for pol in t.index:
        if t.loc[pol, "n_selected"] > 0:
            assert t.loc[pol, "expected_net_contribution"] == pytest.approx(
                t.loc[pol, "total_expected_revenue"] - t.loc[pol, "total_expected_loss"]
                - cfg["policy"]["monthly_cost_per_account"] * t.loc[pol, "n_selected"])
