"""Semantic renaming, deterministic feature engineering and the persisted data split.

Leakage note: ``engineer_features`` is a pure row-wise function (each output row depends only
on the same input row and on constants from the config). It learns nothing from other rows,
so applying it before or after splitting gives identical results. All *fitted* preprocessing
(scaling, encoding) lives inside the scikit-learn pipelines in ``creditlab.models`` and is fit
on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from creditlab.data.fetch import load_raw
from creditlab.data.validate import validate_raw
from creditlab.utils.config import Config, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]   # Sep .. Apr 2005
BILL_COLS = [f"BILL_AMT{k}" for k in range(1, 7)]                   # Sep .. Apr 2005
PAYAMT_COLS = [f"PAY_AMT{k}" for k in range(1, 7)]                  # Sep .. Apr 2005
DEMOGRAPHIC_COLS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
TARGET = "DEFAULT_NEXT_MONTH"

# Official variable names from the UCI metadata (variables table, "description" column).
RENAME = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    **{f"X{6 + i}": c for i, c in enumerate(PAY_COLS)},
    **{f"X{12 + i}": c for i, c in enumerate(BILL_COLS)},
    **{f"X{18 + i}": c for i, c in enumerate(PAYAMT_COLS)},
    "Y": TARGET,
}
SOURCE_FEATURES = [RENAME[f"X{i}"] for i in range(1, 24)]

ENGINEERED = [
    "UTIL_1", "UTIL_MEAN_6", "BILL_TREND_6", "PAY_TO_BILL_6M", "NO_POSITIVE_BILL_6M",
    "N_MONTHS_DELAYED", "MAX_DELAY", "N_MONTHS_CODE_M2", "N_MONTHS_CODE_M1",
]


def to_semantic(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.rename(columns=RENAME)


def engineer_features(df: pd.DataFrame, ratio_clip: float = 5.0) -> pd.DataFrame:
    """Return a copy of ``df`` with row-wise engineered features appended.

    Requires LIMIT_BAL, PAY_*, BILL_AMT*, PAY_AMT* columns. Every feature uses only
    information recorded up to September 2005, i.e. before the target month.
    """
    out = df.copy()
    limit = out["LIMIT_BAL"].astype(float)
    bills = out[BILL_COLS].astype(float)
    pays = out[PAYAMT_COLS].astype(float)
    status = out[PAY_COLS]

    # Utilisation: statement balance relative to the credit limit. Negative balances
    # (credit in the customer's favour) give negative utilisation; clipped to [-1, clip].
    out["UTIL_1"] = (bills["BILL_AMT1"] / limit).clip(-1.0, ratio_clip)
    out["UTIL_MEAN_6"] = (bills.mean(axis=1) / limit).clip(-1.0, ratio_clip)
    # Change in balance from April to September, relative to the limit.
    out["BILL_TREND_6"] = ((bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit).clip(
        -ratio_clip, ratio_clip
    )
    # Six-month payments relative to six-month positive billed amounts. Aggregating over
    # six months avoids assuming which month's bill a given payment settles (the source
    # does not say). Denominator floored at 1 NTD; result capped at ratio_clip.
    pos_bills = bills.clip(lower=0).sum(axis=1)
    out["PAY_TO_BILL_6M"] = (pays.sum(axis=1) / np.maximum(pos_bills, 1.0)).clip(0, ratio_clip)
    out["NO_POSITIVE_BILL_6M"] = (pos_bills == 0).astype(int)
    # Delinquency summaries. Only codes >= 1 are documented as payment delays.
    out["N_MONTHS_DELAYED"] = (status >= 1).sum(axis=1)
    out["MAX_DELAY"] = status.clip(lower=0).max(axis=1)
    # Counts of the two most common status codes. -1 is documented ("pay duly");
    # -2 is NOT documented by the source, so this is a count of an observed code,
    # not an interpretation of it.
    out["N_MONTHS_CODE_M2"] = (status == -2).sum(axis=1)
    out["N_MONTHS_CODE_M1"] = (status == -1).sum(axis=1)
    return out


def model_input_columns(cfg: Config) -> list[str]:
    """Raw (pre-engineering) columns given to the model pipelines."""
    excluded = set(cfg["features"]["exclude_sensitive"]) | set(cfg["features"]["exclude_proxy"])
    return [c for c in SOURCE_FEATURES if c not in excluded]


def make_split(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Stratified 60/20/20 client-level split. Returns a Series of split labels indexed like df."""
    s = cfg["split"]
    seed = cfg.sub_seed("split")
    y = df[TARGET] if s.get("stratify", True) else None
    idx_rest, idx_test = train_test_split(
        df.index, test_size=s["test"], stratify=y, random_state=seed
    )
    y_rest = df.loc[idx_rest, TARGET] if y is not None else None
    val_share_of_rest = s["validation"] / (s["train"] + s["validation"])
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=val_share_of_rest, stratify=y_rest, random_state=seed + 1
    )
    split = pd.Series("train", index=df.index, name="split")
    split.loc[idx_val] = "validation"
    split.loc[idx_test] = "test"
    return split


def processed_path(cfg: Config) -> str:
    return str(cfg.processed_dir / "accounts.parquet")


def run_prepare(cfg: Config) -> pd.DataFrame:
    raw = load_raw(cfg)
    report = validate_raw(raw, expected_rows=cfg["data"]["expected_rows"])
    write_json(cfg.artifact_dir("metrics") / "data_validation.json", report)
    log.info("Validation passed: %d rows, default rate %.4f", report["rows"],
             report["default_rate"])

    df = engineer_features(to_semantic(raw), cfg["features"]["ratio_clip"])
    df["split"] = make_split(df, cfg)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path(cfg), index=False)
    df[["ID", "split"]].to_csv(cfg.processed_dir / "splits.csv", index=False)

    summary = (
        df.groupby("split")[TARGET].agg(n="size", defaults="sum", default_rate="mean")
        .reindex(["train", "validation", "test"])
    )
    write_json(cfg.artifact_dir("metrics") / "split_summary.json",
               summary.reset_index().to_dict(orient="records"))
    update_manifest(cfg, "prepare", {
        "processed_file": processed_path(cfg),
        "split_counts": summary["n"].to_dict(),
        "model_input_columns": model_input_columns(cfg),
        "engineered_features": ENGINEERED,
    })
    log.info("Split sizes: %s", summary["n"].to_dict())
    return df


def load_processed(cfg: Config) -> pd.DataFrame:
    path = processed_path(cfg)
    try:
        return pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{path} not found. Run `creditlab prepare` first.") from exc


def load_split(cfg: Config, name: str) -> pd.DataFrame:
    if name not in {"train", "validation", "test"}:
        raise ValueError(name)
    df = load_processed(cfg)
    return df[df["split"] == name].reset_index(drop=True)
