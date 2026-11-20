"""Schema and content validation of the raw UCI table.

Hard checks raise ``DataValidationError``. Soft findings (undocumented codes, negative bills,
duplicate feature vectors, ...) are reported, not "fixed": they are real properties of the
source data and are documented in docs/DATA_DICTIONARY.md.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from creditlab.data.fetch import RAW_COLUMNS

# Codes documented in the official UCI variable description.
DOCUMENTED_PAY_CODES = {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9}
DOCUMENTED_SEX = {1, 2}
DOCUMENTED_EDUCATION = {1, 2, 3, 4}
DOCUMENTED_MARRIAGE = {1, 2, 3}
PAY_RAW = [f"X{i}" for i in range(6, 12)]
BILL_RAW = [f"X{i}" for i in range(12, 18)]
PAYAMT_RAW = [f"X{i}" for i in range(18, 24)]
FEATURE_RAW = [f"X{i}" for i in range(1, 24)]


class DataValidationError(ValueError):
    pass


def _counts(series: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in sorted(series.value_counts().items())}


def validate_raw(df: pd.DataFrame, expected_rows: int = 30000) -> dict[str, Any]:
    errors: list[str] = []

    if list(df.columns) != RAW_COLUMNS:
        # Every later check depends on the schema, so stop here.
        raise DataValidationError(f"columns {list(df.columns)} != {RAW_COLUMNS}")
    if len(df) != expected_rows:
        errors.append(f"row count {len(df)} != expected {expected_rows}")
    if df.isna().any().any():
        errors.append(f"missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    non_int = [c for c in df.columns if not pd.api.types.is_integer_dtype(df[c])]
    if non_int:
        errors.append(f"non-integer columns: {non_int}")
    if not df["ID"].is_unique:
        errors.append("ID is not unique")
    target_values = set(df["Y"].unique().tolist())
    if not target_values <= {0, 1}:
        errors.append(f"target has values outside {{0,1}}: {sorted(target_values)}")
    if (df["X1"] <= 0).any():
        errors.append("LIMIT_BAL (X1) has non-positive values")
    if (df[PAYAMT_RAW] < 0).any().any():
        errors.append("PAY_AMT (X18-X23) has negative values")
    if not df["X5"].between(18, 100).all():
        errors.append("AGE (X5) outside [18, 100]")
    pay_values = set(pd.unique(df[PAY_RAW].to_numpy().ravel()).tolist())
    if not pay_values <= set(range(-2, 10)):
        errors.append(f"repayment status codes outside [-2, 9]: {sorted(pay_values)}")
    if errors:
        raise DataValidationError("; ".join(errors))

    pay_codes = {c: _counts(df[c]) for c in PAY_RAW}
    undocumented_pay = {
        c: {k: v for k, v in counts.items() if k not in DOCUMENTED_PAY_CODES}
        for c, counts in pay_codes.items()
    }
    features_dup = df.duplicated(subset=FEATURE_RAW, keep=False)
    with_target_dup = df.duplicated(subset=[*FEATURE_RAW, "Y"], keep=False)
    # Groups of identical feature vectors that carry different labels.
    conflicting = (
        df[features_dup].groupby(FEATURE_RAW)["Y"].nunique().gt(1).sum()
    )
    bill_over_limit = df[BILL_RAW].gt(df["X1"], axis=0)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "n_features": len(FEATURE_RAW),
        "missing_values": 0,
        "id_unique": True,
        "target_counts": _counts(df["Y"]),
        "default_rate": float(df["Y"].mean()),
        "pay_status_codes": pay_codes,
        "undocumented_pay_codes": undocumented_pay,
        "sex_codes": _counts(df["X2"]),
        "education_codes": _counts(df["X3"]),
        "undocumented_education": {k: v for k, v in _counts(df["X3"]).items()
                                   if k not in DOCUMENTED_EDUCATION},
        "marriage_codes": _counts(df["X4"]),
        "undocumented_marriage": {k: v for k, v in _counts(df["X4"]).items()
                                  if k not in DOCUMENTED_MARRIAGE},
        "negative_bill_counts": {c: int((df[c] < 0).sum()) for c in BILL_RAW},
        "accounts_with_any_negative_bill": int((df[BILL_RAW] < 0).any(axis=1).sum()),
        "accounts_with_any_bill_over_limit": int(bill_over_limit.any(axis=1).sum()),
        "rows_in_duplicate_feature_groups": int(features_dup.sum()),
        "rows_in_duplicate_feature_and_label_groups": int(with_target_dup.sum()),
        "duplicate_feature_groups_with_conflicting_labels": int(conflicting),
        "limit_bal_range": [int(df["X1"].min()), int(df["X1"].max())],
        "age_range": [int(df["X5"].min()), int(df["X5"].max())],
    }
