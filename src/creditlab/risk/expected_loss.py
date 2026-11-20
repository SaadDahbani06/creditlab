"""Expected loss EL_i = PD_i * LGD_i * EAD_i and portfolio aggregates (all HYPOTHETICAL).

PD comes from the calibrated model; LGD and EAD come from ``RiskAssumptions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.risk.assumptions import RiskAssumptions, exposure_at_default


def _check_pd(pd_: np.ndarray) -> np.ndarray:
    pd_ = np.asarray(pd_, dtype=float)
    if not np.all(np.isfinite(pd_)) or pd_.min() < 0 or pd_.max() > 1:
        raise ValueError("PD must be finite and in [0, 1]")
    return pd_


def expected_loss(pd_, lgd, ead) -> np.ndarray:
    """Element-wise EL. ``lgd`` may be a scalar or per-account array."""
    pd_ = _check_pd(pd_)
    lgd = np.broadcast_to(np.asarray(lgd, dtype=float), pd_.shape)
    ead = np.asarray(ead, dtype=float)
    if ead.shape != pd_.shape:
        raise ValueError("PD and EAD must have the same shape")
    if np.any((lgd < 0) | (lgd > 1)) or np.any(ead < 0):
        raise ValueError("LGD must be in [0, 1] and EAD non-negative")
    return pd_ * lgd * ead


def account_table(pd_, limit, balance, a: RiskAssumptions) -> pd.DataFrame:
    ead = exposure_at_default(limit, balance, a)
    return pd.DataFrame({"pd": _check_pd(pd_), "ead": ead, "lgd": a.lgd,
                         "el": expected_loss(pd_, a.lgd, ead)})


def portfolio_summary(acc: pd.DataFrame) -> dict:
    """Aggregates for an ``account_table``. Total EL is the plain sum of account ELs."""
    total_ead = float(acc["ead"].sum())
    total_el = float(acc["el"].sum())
    by_el = np.sort(acc["el"].to_numpy())[::-1]
    top = max(1, int(np.ceil(0.10 * len(acc))))
    pd_top_decile = acc["pd"] >= acc["pd"].quantile(0.9)
    return {
        "n_accounts": len(acc),
        "total_ead": total_ead,
        "total_el": total_el,
        "el_rate_of_ead": total_el / total_ead if total_ead > 0 else np.nan,
        "mean_pd": float(acc["pd"].mean()),
        "ead_weighted_pd": float((acc["pd"] * acc["ead"]).sum() / total_ead),
        "expected_defaults": float(acc["pd"].sum()),
        "share_el_top10pct_accounts_by_el": float(by_el[:top].sum() / total_el),
        "share_el_top_pd_decile": float(acc.loc[pd_top_decile, "el"].sum() / total_el),
        "share_ead_top_pd_decile": float(acc.loc[pd_top_decile, "ead"].sum() / total_ead),
        "ead_herfindahl": float(((acc["ead"] / total_ead) ** 2).sum()),
    }


def risk_band_table(acc: pd.DataFrame, bands: list[float], observed=None) -> pd.DataFrame:
    """Concentration by PD band. ``observed`` (0/1 labels) adds the OBSERVED default rate."""
    cut = pd.cut(acc["pd"], bands, include_lowest=True, right=False)
    df = acc.assign(band=cut.astype(str))
    if observed is not None:
        df["observed_default"] = np.asarray(observed, int)
    g = df.groupby("band", observed=True, sort=False)
    out = pd.DataFrame({
        "n_accounts": g.size(),
        "mean_pd": g["pd"].mean(),
        "total_ead": g["ead"].sum(),
        "total_el": g["el"].sum(),
    })
    if observed is not None:
        out["observed_default_rate"] = g["observed_default"].mean()
        out["observed_defaults"] = g["observed_default"].sum()
    out["share_of_accounts"] = out["n_accounts"] / out["n_accounts"].sum()
    out["share_of_ead"] = out["total_ead"] / out["total_ead"].sum()
    out["share_of_el"] = out["total_el"] / out["total_el"].sum()
    order = sorted(out.index, key=lambda s: float(s.strip("[(").split(",")[0]))
    return out.loc[order].reset_index()


def sensitivity_grid(pd_, limit, balance, base: RiskAssumptions, lgds, ccfs,
                     limit_fractions=(0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Total EL across LGD x EAD assumptions (both EAD methods)."""
    rows = []
    for lgd in lgds:
        for ccf in ccfs:
            a = base.with_(lgd=lgd, ead_method="balance_plus_ccf", ccf=ccf)
            ead = exposure_at_default(limit, balance, a)
            rows.append({"lgd": lgd, "ead_method": "balance_plus_ccf", "ead_parameter": ccf,
                         "total_ead": ead.sum(), "total_el": expected_loss(pd_, lgd, ead).sum()})
        for f in limit_fractions:
            a = base.with_(lgd=lgd, ead_method="limit_fraction", limit_fraction=f)
            ead = exposure_at_default(limit, balance, a)
            rows.append({"lgd": lgd, "ead_method": "limit_fraction", "ead_parameter": f,
                         "total_ead": ead.sum(), "total_el": expected_loss(pd_, lgd, ead).sum()})
    return pd.DataFrame(rows)
