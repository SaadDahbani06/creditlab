"""Milestone 1-2: semantic mapping, engineered features, split integrity and leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditlab.data.preprocess import (
    BILL_COLS,
    ENGINEERED,
    PAY_COLS,
    RENAME,
    SOURCE_FEATURES,
    engineer_features,
    make_split,
    model_input_columns,
)


def test_rename_covers_every_source_variable():
    assert len(SOURCE_FEATURES) == 23
    assert len(set(RENAME.values())) == 24  # 23 features + target
    assert RENAME["X6"] == "PAY_0" and RENAME["X7"] == "PAY_2"  # official (gap in numbering)
    assert RENAME["X12"] == "BILL_AMT1" and RENAME["X23"] == "PAY_AMT6"


def test_model_inputs_exclude_sensitive_proxy_and_id(cfg):
    cols = model_input_columns(cfg)
    assert not {"SEX", "AGE", "MARRIAGE", "EDUCATION", "ID", "DEFAULT_NEXT_MONTH"} & set(cols)
    assert len(cols) == 19
    assert set(PAY_COLS) <= set(cols) and "LIMIT_BAL" in cols


def test_engineered_features_hand_calculated():
    row = {"LIMIT_BAL": 100_000, **{c: v for c, v in zip(PAY_COLS, [2, 1, 0, -1, -2, -2], strict=True)}}
    row |= {f"BILL_AMT{k}": v for k, v in zip(range(1, 7), [50_000, 40_000, -1_000, 0, 10_000, 20_000], strict=True)}
    row |= {f"PAY_AMT{k}": v for k, v in zip(range(1, 7), [1_000, 2_000, 0, 0, 3_000, 4_000], strict=True)}
    out = engineer_features(pd.DataFrame([row])).iloc[0]
    assert out["UTIL_1"] == pytest.approx(0.5)
    assert out["UTIL_MEAN_6"] == pytest.approx((50_000 + 40_000 - 1_000 + 0 + 10_000 + 20_000) / 6 / 100_000)
    assert out["BILL_TREND_6"] == pytest.approx((50_000 - 20_000) / 100_000)
    assert out["PAY_TO_BILL_6M"] == pytest.approx(10_000 / 120_000)  # negative bill excluded
    assert out["NO_POSITIVE_BILL_6M"] == 0
    assert out["N_MONTHS_DELAYED"] == 2
    assert out["MAX_DELAY"] == 2
    assert out["N_MONTHS_CODE_M2"] == 2 and out["N_MONTHS_CODE_M1"] == 1


def test_engineered_ratio_edge_cases():
    base = {"LIMIT_BAL": 10_000, **dict.fromkeys(PAY_COLS, 0)}
    no_bill_paid = base | dict.fromkeys(BILL_COLS, 0) | {f"PAY_AMT{k}": 500 for k in range(1, 7)}
    inactive = base | dict.fromkeys(BILL_COLS, 0) | {f"PAY_AMT{k}": 0 for k in range(1, 7)}
    out = engineer_features(pd.DataFrame([no_bill_paid, inactive]), ratio_clip=5.0)
    assert out.loc[0, "PAY_TO_BILL_6M"] == 5.0          # capped, not infinite
    assert out.loc[1, "PAY_TO_BILL_6M"] == 0.0
    assert out["NO_POSITIVE_BILL_6M"].tolist() == [1, 1]
    assert np.isfinite(out[ENGINEERED].to_numpy(dtype=float)).all()


def test_engineering_is_row_wise(synthetic_accounts):
    """Row-wise => applying before or after a split gives identical values (no leakage)."""
    full = engineer_features(synthetic_accounts)
    part = engineer_features(synthetic_accounts.iloc[:57])
    pd.testing.assert_frame_equal(full.iloc[:57], part)


@pytest.mark.needs_data
def test_split_is_disjoint_complete_stratified_and_reproducible(cfg, accounts):
    s1 = make_split(accounts, cfg)
    s2 = make_split(accounts, cfg)
    pd.testing.assert_series_equal(s1, s2)
    counts = s1.value_counts()
    assert counts.to_dict() == {"train": 18000, "validation": 6000, "test": 6000}
    rates = accounts.groupby(s1)["DEFAULT_NEXT_MONTH"].mean()
    overall = accounts["DEFAULT_NEXT_MONTH"].mean()
    assert (rates - overall).abs().max() < 0.002
    ids = {name: set(accounts.loc[s1 == name, "ID"]) for name in counts.index}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]


@pytest.mark.needs_data
def test_persisted_split_matches_recomputed(cfg, accounts):
    path = cfg.processed_dir / "splits.csv"
    if not path.exists():
        pytest.skip("run `creditlab prepare` first")
    saved = pd.read_csv(path).set_index("ID")["split"]
    recomputed = make_split(accounts, cfg)
    recomputed.index = accounts["ID"]
    assert (saved.sort_index() == recomputed.sort_index()).all()
