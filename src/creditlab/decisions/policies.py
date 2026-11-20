"""HYPOTHETICAL risk-selection policies for an existing-account research cohort.

This is NOT a lending or account-management engine. The data contain existing cardholders and
no historical approval/decline decisions, revenue, LGD or EAD. "Retain" below means "included in
a hypothetical retained cohort for research comparison". It makes no statement about any real
client and produces no decline reasons.

One-month economics per account i (all ASSUMED):
    B_i   = max(BILL_AMT1_i, 0)                        drawn balance (observed, Sep 2005)
    R_i   = r * B_i                                     revenue if the account performs
    EAD_i, LGD                                          from RiskAssumptions
    c                                                   fixed monthly cost per account
    E[contribution_i] = (1 - PD_i) * R_i - PD_i * LGD * EAD_i - c

E[contribution_i] > 0  <=>  PD_i < (R_i - c) / (R_i + LGD * EAD_i)   (account-specific break-even PD)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditlab.risk.assumptions import RiskAssumptions, exposure_at_default
from creditlab.utils.config import Config

POLICY_NAMES = {
    "A_retain_all": "A: retain entire cohort",
    "B_pd_threshold": "B: retain PD below validation-selected cutoff",
    "C_expected_contribution": "C: retain if expected contribution > 0",
    "D_random_equal_size": "D: random cohort, same size as B",
}


@dataclass(frozen=True)
class PolicyEconomics:
    monthly_revenue_rate: float = 0.015
    monthly_cost_per_account: float = 0.0
    risk: RiskAssumptions = RiskAssumptions()

    def __post_init__(self) -> None:
        if not 0 <= self.monthly_revenue_rate <= 1:
            raise ValueError("monthly_revenue_rate must be in [0, 1]")
        if self.monthly_cost_per_account < 0:
            raise ValueError("cost must be non-negative")

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> PolicyEconomics:
        pcfg = cfg["policy"]
        risk = RiskAssumptions.from_config(cfg, lgd=pcfg["lgd"])
        kwargs = {"monthly_revenue_rate": pcfg["monthly_revenue_rate"],
                  "monthly_cost_per_account": pcfg["monthly_cost_per_account"], "risk": risk}
        kwargs.update(overrides)
        return cls(**kwargs)


def account_economics(pd_, limit, balance, econ: PolicyEconomics, observed=None) -> pd.DataFrame:
    p = np.asarray(pd_, dtype=float)
    ead = exposure_at_default(limit, balance, econ.risk)
    revenue = econ.monthly_revenue_rate * np.maximum(np.asarray(balance, float), 0.0)
    lgd = econ.risk.lgd
    c = econ.monthly_cost_per_account
    df = pd.DataFrame({
        "pd": p, "ead": ead, "revenue_if_performing": revenue,
        "expected_loss": p * lgd * ead,
        "expected_revenue": (1 - p) * revenue,
    })
    df["expected_contribution"] = df["expected_revenue"] - df["expected_loss"] - c
    denom = revenue + lgd * ead
    df["break_even_pd"] = np.where(denom > 0, (revenue - c) / np.where(denom > 0, denom, 1), 0.0)
    if observed is not None:
        y = np.asarray(observed, int)
        df["observed_default"] = y
        # Observed outcome x ASSUMED economics: still hypothetical money.
        df["realized_hypothetical_contribution"] = (1 - y) * revenue - y * lgd * ead - c
    return df


# ----------------------------------------------------------------------------- policies
def policy_retain_all(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype=bool)


def policy_pd_threshold(df: pd.DataFrame, threshold: float) -> np.ndarray:
    return df["pd"].to_numpy() < threshold


def policy_expected_contribution(df: pd.DataFrame) -> np.ndarray:
    return df["expected_contribution"].to_numpy() > 0


def policy_random(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask


def choose_pd_threshold(val_df: pd.DataFrame, grid_size: int, rule: str = "max_bad_rate",
                        target_bad_rate: float = 0.10) -> dict:
    """Choose policy B's PD cutoff on the VALIDATION split (observed outcomes required).

    Candidate cutoffs are quantiles of the validation PD distribution plus "retain all".
    max_bad_rate: the largest cutoff whose retained validation accounts have an OBSERVED default
        rate <= target_bad_rate (a hypothetical risk-appetite constraint).
    max_realized_contribution: the cutoff maximising observed outcomes x assumed economics.
    """
    if "realized_hypothetical_contribution" not in val_df:
        raise ValueError("validation frame needs observed outcomes")
    qs = np.linspace(0.01, 0.99, grid_size)
    grid = np.unique(np.r_[np.quantile(val_df["pd"], qs), 1.0 + 1e-12])
    p = val_df["pd"].to_numpy()
    y = val_df["observed_default"].to_numpy()
    v = val_df["realized_hypothetical_contribution"].to_numpy()
    totals = np.array([v[p < t].sum() for t in grid])
    bad = np.array([y[p < t].mean() if np.any(p < t) else np.nan for t in grid])
    if rule == "max_realized_contribution":
        best = int(np.argmax(totals))
    elif rule == "max_bad_rate":
        ok = np.where(bad <= target_bad_rate)[0]
        if ok.size == 0:
            raise ValueError(f"no cutoff achieves observed bad rate <= {target_bad_rate}")
        best = int(ok.max())
    else:
        raise ValueError(f"unknown rule {rule}")
    return {"rule": rule, "target_bad_rate": target_bad_rate if rule == "max_bad_rate" else None,
            "threshold": float(grid[best]), "validation_total": float(totals[best]),
            "validation_bad_rate": float(bad[best]),
            "validation_share_retained": float(np.mean(p < grid[best])),
            "grid": grid.tolist(), "grid_totals": totals.tolist(),
            "grid_bad_rates": bad.tolist()}


def break_even_revenue_rate(pd_, balance, ead, lgd: float, cost: float = 0.0) -> np.ndarray:
    """Monthly revenue rate r_i at which E[contribution_i] = 0 (inf if no drawn balance).

    (1 - PD) r B = PD LGD EAD + c  =>  r* = (PD LGD EAD + c) / ((1 - PD) B).
    """
    p = np.asarray(pd_, float)
    b = np.maximum(np.asarray(balance, float), 0.0)
    num = p * lgd * np.asarray(ead, float) + cost
    den = (1 - p) * b
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.inf)


def summarize(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sel = df[mask]
    total_el = float(sel["expected_loss"].sum())
    by_el = np.sort(sel["expected_loss"].to_numpy())[::-1]
    top = max(1, int(np.ceil(0.10 * len(sel)))) if len(sel) else 0
    out = {
        "n_selected": int(mask.sum()),
        "share_selected": float(mask.mean()),
        "mean_pd": float(sel["pd"].mean()) if len(sel) else np.nan,
        "total_ead": float(sel["ead"].sum()),
        "total_expected_loss": total_el,
        "total_expected_revenue": float(sel["expected_revenue"].sum()),
        "expected_net_contribution": float(sel["expected_contribution"].sum()),
        "share_el_top10pct_accounts": float(by_el[:top].sum() / total_el) if total_el > 0 else np.nan,
        "ead_herfindahl": float(((sel["ead"] / sel["ead"].sum()) ** 2).sum()) if len(sel) else np.nan,
    }
    if "observed_default" in df:
        out["observed_default_rate"] = float(sel["observed_default"].mean()) if len(sel) else np.nan
        out["realized_hypothetical_contribution"] = float(
            sel["realized_hypothetical_contribution"].sum())
    return out
