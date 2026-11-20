"""Monte Carlo portfolio default simulation: independent and one-factor Gaussian copula.

Latent variable for account i in simulation s:

    Z_is = sqrt(rho) * F_s + sqrt(1 - rho) * eps_is,    F_s, eps_is ~ iid N(0, 1)

Account i defaults when Z_is < Phi^{-1}(PD_i). Because Z_is ~ N(0, 1) for every rho,
P(default_i) = Phi(Phi^{-1}(PD_i)) = PD_i: correlation changes the joint behaviour, not the
marginal PDs. With rho = 0 the rule reduces to independent defaults, and with the same seed it
reproduces the independent simulation exactly (same eps draws).

Conditional on the factor, defaults are independent with
    PD_i(F) = Phi( (Phi^{-1}(PD_i) - sqrt(rho) F) / sqrt(1 - rho) ),
which gives the Vasicek large-portfolio quantile used as an analytical check.

Memory: simulations are generated in chunks of ``chunk_size`` rows, so peak memory is about
chunk_size * n_accounts * 8 bytes * 2 (latent draws and comparison), independent of n_sims.
The idiosyncratic and factor streams come from separate child seeds, so results do not depend
on the chunk size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class SimulationResult:
    losses: np.ndarray            # (n_sims,) hypothetical portfolio loss per path
    default_counts: np.ndarray    # (n_sims,) number of defaulted accounts per path
    factor: np.ndarray            # (n_sims,) common factor draws F_s
    account_default_freq: np.ndarray  # (n_accounts,) simulated default frequency per account
    rho: float
    seed: int


def simulate_portfolio(pd_, loss_given_default, n_sims: int, rho: float = 0.0,
                       seed: int = 0, chunk_size: int = 1000) -> SimulationResult:
    """Simulate portfolio losses.

    pd_: per-account default probability; loss_given_default: per-account loss if the account
    defaults (= LGD_i * EAD_i, NTD). Returns losses for each of ``n_sims`` paths.
    """
    pd_ = np.asarray(pd_, dtype=float)
    lgd_ead = np.asarray(loss_given_default, dtype=float)
    if pd_.shape != lgd_ead.shape or pd_.ndim != 1:
        raise ValueError("pd_ and loss_given_default must be 1-D arrays of equal length")
    if not np.all(np.isfinite(pd_)) or pd_.min() < 0 or pd_.max() > 1:
        raise ValueError("PD must be finite and in [0, 1]")
    if np.any(lgd_ead < 0) or not np.all(np.isfinite(lgd_ead)):
        raise ValueError("loss given default must be finite and non-negative")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if n_sims < 1 or chunk_size < 1:
        raise ValueError("n_sims and chunk_size must be positive")

    thresholds = norm.ppf(pd_)  # PD = 0 -> -inf (never defaults), PD = 1 -> +inf (always)
    factor_ss, idio_ss = np.random.SeedSequence(seed).spawn(2)
    factor = np.random.default_rng(factor_ss).standard_normal(n_sims)
    rng = np.random.default_rng(idio_ss)
    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)

    losses = np.empty(n_sims)
    counts = np.empty(n_sims, dtype=np.int64)
    freq = np.zeros(pd_.size)
    for start in range(0, n_sims, chunk_size):
        stop = min(start + chunk_size, n_sims)
        z = rng.standard_normal((stop - start, pd_.size))
        if rho > 0:
            z *= b
            z += a * factor[start:stop, None]
        defaults = z < thresholds
        losses[start:stop] = defaults @ lgd_ead
        counts[start:stop] = defaults.sum(axis=1)
        freq += defaults.sum(axis=0)
    return SimulationResult(losses, counts, factor, freq / n_sims, rho, seed)


def simulate_portfolio_multi(pd_sets, loss_given_default, n_sims: int, rho: float = 0.0,
                             seed: int = 0, chunk_size: int = 1000) -> np.ndarray:
    """Simulate losses for K alternative PD vectors with COMMON random numbers.

    ``pd_sets`` has shape (K, n_accounts): one PD vector per fitted model. Every PD vector is
    evaluated on exactly the same factor draws F_s and idiosyncratic draws eps_is, i.e. the same
    draws ``simulate_portfolio(pd_sets[k], ..., seed=seed)`` would use. Row k of the result is
    therefore identical to ``simulate_portfolio(pd_sets[k], loss_given_default, n_sims, rho,
    seed, chunk_size).losses``; generating the draws once makes the K runs ~K times cheaper and
    makes differences between models free of between-run sampling noise.

    Returns losses with shape (K, n_sims).
    """
    P = np.atleast_2d(np.asarray(pd_sets, dtype=float))
    w = np.asarray(loss_given_default, dtype=float)
    if P.ndim != 2 or P.shape[1] != w.size or w.ndim != 1:
        raise ValueError("pd_sets must be (K, n) and loss_given_default length n")
    if not np.all(np.isfinite(P)) or P.min() < 0 or P.max() > 1:
        raise ValueError("PD must be finite and in [0, 1]")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("loss given default must be finite and non-negative")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if n_sims < 1 or chunk_size < 1:
        raise ValueError("n_sims and chunk_size must be positive")

    thresholds = norm.ppf(P)  # (K, n)
    factor_ss, idio_ss = np.random.SeedSequence(seed).spawn(2)
    factor = np.random.default_rng(factor_ss).standard_normal(n_sims)
    rng = np.random.default_rng(idio_ss)
    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)
    losses = np.empty((P.shape[0], n_sims))
    for start in range(0, n_sims, chunk_size):
        stop = min(start + chunk_size, n_sims)
        z = rng.standard_normal((stop - start, w.size))
        if rho > 0:
            z *= b
            z += a * factor[start:stop, None]
        for k in range(P.shape[0]):
            losses[k, start:stop] = (z < thresholds[k]) @ w
    return losses


def one_factor_loss_quantiles(pd_, loss_given_default, rho: float,
                              levels=(0.95, 0.99), n_grid: int = 4001) -> dict[str, float]:
    """Semi-analytical mean, SD, VaR and ES of portfolio loss under the one-factor model.

    Conditional on the factor F, defaults are independent, so L | F has mean
    m(F) = sum_i w_i p_i(F) and variance v(F) = sum_i w_i^2 p_i(F)(1 - p_i(F)). For a portfolio
    of thousands of accounts L | F is close to normal (Lyapunov CLT), so

        P(L <= x) = E_F[ Phi((x - m(F)) / sqrt(v(F))) ],

    evaluated by quadrature over F on a fine grid. VaR is the root of P(L <= x) = alpha and
    ES_alpha = VaR + E[(L - VaR)^+] / (1 - alpha), using the closed form of E[(X - q)^+] for a
    normal X. No simulation noise, so it is a reference for the Monte Carlo engine. The normal
    approximation slightly understates right-tail skew when rho = 0 (small portfolios or
    very low PDs make it worse).
    """
    from scipy.optimize import brentq

    p = np.asarray(pd_, dtype=float)
    w = np.asarray(loss_given_default, dtype=float)
    if rho == 0:
        F, wF = np.array([0.0]), np.array([1.0])
        pf = p[None, :]
    else:
        F = np.linspace(-8.0, 8.0, n_grid)
        wF = norm.pdf(F)
        wF /= wF.sum()
        pf = norm.cdf((norm.ppf(p)[None, :] - np.sqrt(rho) * F[:, None]) / np.sqrt(1 - rho))
    m = pf @ w
    s = np.sqrt(np.maximum((pf * (1 - pf)) @ (w ** 2), 1e-12))
    mean = float(wF @ m)
    sd = float(np.sqrt(max(wF @ (s ** 2 + m ** 2) - mean ** 2, 0.0)))
    out = {"mean": mean, "std": sd}
    hi = m.max() + 12 * s.max()
    for alpha in levels:
        q = brentq(lambda x, a=alpha: float(wF @ norm.cdf((x - m) / s)) - a, 0.0, hi)
        z = (q - m) / s
        excess = float(wF @ (s * norm.pdf(z) + (m - q) * norm.sf(z)))
        key = f"{round(alpha * 100)}"
        out[f"var_{key}"] = float(q)
        out[f"es_{key}"] = float(q + excess / (1 - alpha))
    return out


def conditional_pd(pd_, rho: float, factor: float) -> np.ndarray:
    """PD_i(F) under the one-factor model."""
    if rho == 0:
        return np.asarray(pd_, dtype=float)
    return norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * factor) / np.sqrt(1 - rho))


def vasicek_quantile(pd_: float, rho: float, alpha: float) -> float:
    """alpha-quantile of the default RATE of an infinitely granular homogeneous portfolio.

    Bad outcomes correspond to low F, so the alpha-quantile of the loss rate uses
    F = -Phi^{-1}(alpha): Phi( (Phi^{-1}(PD) + sqrt(rho) Phi^{-1}(alpha)) / sqrt(1 - rho) ).
    """
    return float(norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho)))


def analytical_mean_and_variance_independent(pd_, loss_given_default) -> tuple[float, float]:
    """E[L] = sum p_i w_i;  Var[L] = sum p_i (1 - p_i) w_i^2 under independence."""
    p = np.asarray(pd_, dtype=float)
    w = np.asarray(loss_given_default, dtype=float)
    return float(np.sum(p * w)), float(np.sum(p * (1 - p) * w**2))
