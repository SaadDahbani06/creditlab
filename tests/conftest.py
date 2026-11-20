"""Shared pytest fixtures.

Tests that need the real dataset are marked ``needs_data`` and skip with an explicit
reason when data/raw/uci_350_data.csv is missing (run ``creditlab fetch`` first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditlab.utils.config import PROJECT_ROOT, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config(PROJECT_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def fast_cfg():
    return load_config(PROJECT_ROOT / "configs" / "fast.yaml")


@pytest.fixture(scope="session")
def raw_df(cfg):
    from creditlab.data.fetch import load_raw

    if not cfg.raw_data_file.exists():
        pytest.skip(f"{cfg.raw_data_file} missing: run `creditlab fetch` first")
    return load_raw(cfg)


@pytest.fixture(scope="session")
def accounts(raw_df):
    from creditlab.data.preprocess import to_semantic

    return to_semantic(raw_df)


@pytest.fixture()
def synthetic_accounts():
    """Small synthetic frame with the semantic schema, for unit tests of pure logic."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"ID": np.arange(1, n + 1)})
    df["LIMIT_BAL"] = rng.choice([10_000, 50_000, 100_000, 200_000], n)
    df["SEX"] = rng.choice([1, 2], n)
    df["EDUCATION"] = rng.choice([1, 2, 3, 4], n)
    df["MARRIAGE"] = rng.choice([1, 2, 3], n)
    df["AGE"] = rng.integers(21, 70, n)
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        df[c] = rng.choice([-2, -1, 0, 1, 2, 3], n, p=[0.1, 0.2, 0.5, 0.1, 0.08, 0.02])
    for k in range(1, 7):
        df[f"BILL_AMT{k}"] = rng.integers(-1_000, 80_000, n)
        df[f"PAY_AMT{k}"] = rng.integers(0, 10_000, n)
    logit = -1.5 + 0.8 * (df["PAY_0"] >= 1) - 0.000004 * df["LIMIT_BAL"]
    df["DEFAULT_NEXT_MONTH"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
