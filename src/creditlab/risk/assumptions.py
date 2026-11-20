"""HYPOTHETICAL loss-given-default and exposure-at-default assumptions.

Only PD is estimated from data. The dataset contains no LGD, EAD, recovery or write-off
observations, so every value here is an assumption chosen for illustration. Outputs computed
from these assumptions are hypothetical and are not observed bank losses.

Units: New Taiwan dollars (NTD). Horizon: one month (the target is default on the next
month's payment).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from creditlab.utils.config import Config

EAD_METHODS = ("balance_plus_ccf", "limit_fraction")


@dataclass(frozen=True)
class RiskAssumptions:
    lgd: float = 0.60
    ead_method: str = "balance_plus_ccf"
    ccf: float = 0.50
    limit_fraction: float = 0.50
    ead_cap: float | None = None
    currency: str = "NTD"
    horizon: str = "one month"

    def __post_init__(self) -> None:
        if not 0.0 <= self.lgd <= 1.0:
            raise ValueError(f"LGD must be in [0, 1], got {self.lgd}")
        if self.ead_method not in EAD_METHODS:
            raise ValueError(f"ead_method must be one of {EAD_METHODS}")
        if not 0.0 <= self.ccf <= 1.0:
            raise ValueError(f"CCF must be in [0, 1], got {self.ccf}")
        if not 0.0 < self.limit_fraction <= 1.0:
            raise ValueError(f"limit_fraction must be in (0, 1], got {self.limit_fraction}")
        if self.ead_cap is not None and self.ead_cap <= 0:
            raise ValueError("ead_cap must be positive or None")

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> RiskAssumptions:
        r = cfg["risk"]
        base = cls(
            lgd=r["lgd_base"], ead_method=r["ead"]["method"], ccf=r["ead"]["ccf"],
            limit_fraction=r["ead"]["limit_fraction"], ead_cap=r["ead"]["cap"],
            currency=r["currency"], horizon=r["horizon"],
        )
        return replace(base, **overrides)

    def with_(self, **overrides) -> RiskAssumptions:
        return replace(self, **overrides)

    def describe(self) -> dict:
        d = asdict(self)
        d["status"] = "HYPOTHETICAL: LGD and EAD are not observed in the dataset"
        if self.ead_method == "balance_plus_ccf":
            d["ead_formula"] = "max(BILL_AMT1, 0) + CCF * max(LIMIT_BAL - max(BILL_AMT1, 0), 0)"
        else:
            d["ead_formula"] = "limit_fraction * LIMIT_BAL"
        if self.ead_cap is not None:
            d["ead_formula"] = f"min({d['ead_formula']}, {self.ead_cap})"
        return d


def exposure_at_default(limit, balance, a: RiskAssumptions) -> np.ndarray:
    """Hypothetical EAD per account (NTD).

    balance_plus_ccf: EAD = drawn + CCF * undrawn, with drawn = max(latest statement, 0) and
    undrawn = max(limit - drawn, 0). Over-limit balances are kept in full (undrawn = 0).
    limit_fraction:   EAD = f * limit.
    Credit limit is never silently used as EAD: it enters only through these explicit formulas.
    """
    limit = np.asarray(limit, dtype=float)
    balance = np.asarray(balance, dtype=float)
    if np.any(limit <= 0):
        raise ValueError("credit limits must be positive")
    if a.ead_method == "limit_fraction":
        ead = a.limit_fraction * limit
    else:
        drawn = np.maximum(balance, 0.0)
        undrawn = np.maximum(limit - drawn, 0.0)
        ead = drawn + a.ccf * undrawn
    if a.ead_cap is not None:
        ead = np.minimum(ead, a.ead_cap)
    return ead
