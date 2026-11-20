"""Baseline model and the preprocessing pipelines shared by every model.

All fitted preprocessing lives inside a scikit-learn ``Pipeline`` so that ``fit`` on the
training split is the only place statistics (scaler means, encoder categories) are learned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from creditlab.data.preprocess import (
    BILL_COLS,
    ENGINEERED,
    PAY_COLS,
    PAYAMT_COLS,
    engineer_features,
)

MONEY_COLS = ["LIMIT_BAL", *BILL_COLS, *PAYAMT_COLS]


def signed_log1p(x):
    """sign(x) * log(1 + |x|): compresses heavy tails, keeps sign of negative balances."""
    return np.sign(x) * np.log1p(np.abs(x))


def group_pay_codes(x):
    """Group status codes >= 3 (sparse, all documented delays) into one category '3+'."""
    return np.minimum(np.asarray(x), 3)


def engineered_feature_names(transformer, input_features):
    """Output column names of ``engineer_features``: inputs followed by engineered columns."""
    return np.asarray([*input_features, *ENGINEERED], dtype=object)


def _engineer_step(ratio_clip: float) -> FunctionTransformer:
    return FunctionTransformer(
        engineer_features, kw_args={"ratio_clip": ratio_clip},
        feature_names_out=engineered_feature_names, validate=False,
    ).set_output(transform="pandas")


def logistic_preprocessor(ratio_clip: float) -> Pipeline:
    """Engineer features, one-hot status codes, log+scale money, scale engineered features."""
    columns = ColumnTransformer(
        [
            ("status", Pipeline([
                ("group", FunctionTransformer(group_pay_codes, feature_names_out="one-to-one")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                         sparse_output=False)),
            ]), PAY_COLS),
            ("money", Pipeline([
                ("slog", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]), MONEY_COLS),
            ("engineered", StandardScaler(), ENGINEERED),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("engineer", _engineer_step(ratio_clip)), ("columns", columns)]).set_output(
        transform="pandas"
    )


def tree_preprocessor(ratio_clip: float) -> Pipeline:
    """Trees need no scaling; status codes stay ordinal so splits can isolate any code."""
    columns = ColumnTransformer(
        [("keep", "passthrough", [*PAY_COLS, *MONEY_COLS, *ENGINEERED])],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("engineer", _engineer_step(ratio_clip)), ("columns", columns)]).set_output(
        transform="pandas"
    )


def dummy_model() -> DummyClassifier:
    """Predicts the training base rate for every account (AUC = 0.5 by construction)."""
    return DummyClassifier(strategy="prior")


def as_float_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].astype("float64")
