"""Milestone 1: raw data integrity, validation rules and SQL analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from creditlab.data.fetch import RAW_COLUMNS, content_sha256, provenance_path
from creditlab.data.sql_analysis import connect, parse_sql_file, run_queries
from creditlab.data.validate import DataValidationError, validate_raw
from creditlab.utils.config import read_json, sha256_file

pytestmark = pytest.mark.needs_data


def test_raw_shape_and_columns(raw_df):
    assert raw_df.shape == (30000, 25)
    assert list(raw_df.columns) == RAW_COLUMNS
    assert raw_df["ID"].is_unique
    assert set(raw_df["Y"].unique()) == {0, 1}
    assert not raw_df.isna().any().any()


def test_raw_file_matches_provenance(cfg):
    prov = read_json(provenance_path(cfg))
    assert sha256_file(cfg.raw_data_file) == prov["sha256"], "raw file modified after fetch"
    assert prov["rows"] == 30000
    assert prov["license"].startswith("CC BY 4.0")


def test_raw_content_identical_to_official_csv(raw_df, cfg):
    prov = read_json(provenance_path(cfg))
    assert prov["content_identical_to_official_csv"] is True
    assert content_sha256(raw_df) == prov["content_sha256"]


def test_validation_report_on_real_data(raw_df):
    rep = validate_raw(raw_df)
    assert rep["rows"] == 30000 and rep["n_features"] == 23
    assert rep["target_counts"] == {0: 23364, 1: 6636}
    # Known, documented anomalies are reported rather than silently fixed.
    assert set(rep["undocumented_pay_codes"]["X6"]) == {-2, 0}
    assert set(rep["undocumented_education"]) == {0, 5, 6}
    assert set(rep["undocumented_marriage"]) == {0}
    assert rep["accounts_with_any_negative_bill"] > 0


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d.assign(Y=d["Y"].replace({1: 2})), "target"),
        (lambda d: d.iloc[:-1], "row count"),
        (lambda d: d.assign(ID=1), "ID is not unique"),
        (lambda d: d.assign(X18=-d["X18"] - 1), "PAY_AMT"),
        (lambda d: d.assign(X6=d["X6"].replace({8: 12})), "status codes"),
        (lambda d: d.drop(columns=["X5"]), "columns"),
    ],
)
def test_validation_rejects_corrupted_data(raw_df, mutation, message):
    with pytest.raises(DataValidationError, match=message):
        validate_raw(mutation(raw_df.copy()))


def test_sql_file_has_required_queries(cfg):
    names = {q.name for q in parse_sql_file(cfg.path("sql_file"))}
    required = {
        "overall_default_rate", "default_rate_by_pay0", "repayment_delay_patterns",
        "bill_amount_distribution", "payment_to_bill_ratio", "credit_limit_deciles",
        "default_by_risk_bands", "data_quality_checks",
    }
    assert required <= names
    assert len(names) >= 8


@pytest.fixture(scope="module")
def sql_results(cfg, tmp_path_factory):
    from creditlab.data.preprocess import processed_path

    if not pd.io.common.file_exists(processed_path(cfg)):
        pytest.skip("processed data missing: run `creditlab prepare` first")
    db = tmp_path_factory.mktemp("duck") / "test.duckdb"
    con = connect(cfg, database=str(db))
    try:
        yield run_queries(con, parse_sql_file(cfg.path("sql_file")))
    finally:
        con.close()


def test_sql_queries_execute_and_agree_with_pandas(sql_results, accounts):
    overall = sql_results["overall_default_rate"].set_index("split")
    assert overall.loc["ALL", "n_accounts"] == 30000
    assert overall.loc["ALL", "n_defaults"] == accounts["DEFAULT_NEXT_MONTH"].sum()
    assert overall.loc[["train", "validation", "test"], "n_accounts"].tolist() == [18000, 6000, 6000]

    dq = sql_results["data_quality_checks"].set_index("check_name")["n"]
    assert dq["NULL cells (all columns)"] == 0
    assert dq["PAY_0 = -2 (undocumented)"] == (accounts["PAY_0"] == -2).sum()
    assert dq["negative payment amount"] == 0


def test_sql_eda_restricted_to_training_split(sql_results):
    for name in ["default_rate_by_pay0", "repayment_delay_patterns", "credit_limit_deciles"]:
        total = sql_results[name]["n_accounts"].sum()
        assert total == 18000, f"{name} should only see the training split"
