"""Value-at-Risk and Expected Shortfall of a simulated loss distribution.

Conventions (losses are positive numbers; larger is worse):

VaR_a(L) = inf { l : P(L <= l) >= a }   (the lower a-quantile; ``inverted_cdf`` in NumPy)

ES_a(L)  = 1/(1-a) * integral_a^1 VaR_u(L) du

For an empirical distribution of n equally likely samples with ties, the integral form
(Acerbi & Tasche, 2002) evaluates to

ES_a = [ sum_{L_i > VaR_a} L_i  +  VaR_a * ( #{L_i <= VaR_a} - n*a ) ] / ( n*(1-a) ).

The second term assigns the probability mass of the atom at VaR_a that lies beyond the
a-level. The weights are non-negative and sum to one, so ES_a >= VaR_a always, and the
estimator stays correct when many simulated losses are tied (e.g. zero-default paths).

Substituting #{L_i <= VaR_a} = n - #{L_i > VaR_a} gives the algebraically identical
Rockafellar-Uryasev form used in the code,

ES_a = VaR_a + (1 / (n*(1-a))) * sum_i max(L_i - VaR_a, 0),

which avoids the floating-point error of n*a (e.g. 2000*0.95 != 1900 exactly) and makes
ES_a >= VaR_a hold numerically as well as mathematically.
"""

from __future__ import annotations

import numpy as np


def value_at_risk(losses, alpha: float) -> float:
    losses = np.asarray(losses, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return float(np.quantile(losses, alpha, method="inverted_cdf"))


def expected_shortfall(losses, alpha: float) -> float:
    losses = np.asarray(losses, dtype=float)
    n = losses.size
    v = value_at_risk(losses, alpha)
    excess = np.maximum(losses - v, 0.0).sum()
    return float(v + excess / (n * (1 - alpha)))


def batch_standard_error(losses, stat, n_batches: int = 20) -> float:
    """Monte Carlo standard error of a statistic via non-overlapping batches."""
    losses = np.asarray(losses, dtype=float)
    batches = np.array_split(losses, n_batches)
    vals = np.array([stat(b) for b in batches])
    return float(vals.std(ddof=1) / np.sqrt(n_batches))


def loss_summary(losses, levels=(0.95, 0.99), thresholds: dict[str, float] | None = None) -> dict:
    losses = np.asarray(losses, dtype=float)
    if not np.all(np.isfinite(losses)):
        raise ValueError("non-finite simulated loss")
    out = {
        "n_sims": int(losses.size),
        "mean": float(losses.mean()),
        "mc_se_mean": float(losses.std(ddof=1) / np.sqrt(losses.size)),
        "median": float(np.median(losses)),
        "std": float(losses.std(ddof=1)),
        "min": float(losses.min()),
        "max": float(losses.max()),
    }
    for a in levels:
        key = f"{round(a * 100)}"
        out[f"var_{key}"] = value_at_risk(losses, a)
        out[f"es_{key}"] = expected_shortfall(losses, a)
        out[f"var_{key}_mc_se"] = batch_standard_error(losses, lambda b, a=a: value_at_risk(b, a))
        out[f"es_{key}_mc_se"] = batch_standard_error(losses, lambda b, a=a: expected_shortfall(b, a))
    for name, t in (thresholds or {}).items():
        out[f"p_exceed_{name}"] = float(np.mean(losses > t))
    return out
