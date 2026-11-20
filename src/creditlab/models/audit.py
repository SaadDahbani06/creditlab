"""Aggregate research audit of the primary model across sensitive attributes (TEST split).

Sensitive attributes are not model inputs. This audit only *describes* how the model's
predictions and calibration differ across groups, with sample sizes. It is not used for any
decision rule, and group-level differences do not establish their cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditlab.data.preprocess import TARGET
from creditlab.utils.config import Config

MIN_CLASS_COUNT = 30  # minimum defaults AND non-defaults in a group to report AUC


def group_labels(df: pd.DataFrame) -> dict[str, pd.Series]:
    age_band = pd.cut(df["AGE"], [0, 29, 39, 49, 200], labels=["21-29", "30-39", "40-49", "50+"])
    edu = df["EDUCATION"].map({1: "1 graduate", 2: "2 university", 3: "3 high school"}
                              ).fillna("other/undocumented (0,4,5,6)")
    mar = df["MARRIAGE"].map({1: "1 married", 2: "2 single"}).fillna("other/undocumented (0,3)")
    sex = df["SEX"].map({1: "1 male", 2: "2 female"})
    return {"SEX": sex, "AGE_BAND": age_band.astype(str), "EDUCATION": edu, "MARRIAGE": mar}


def audit_table(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    y = df[TARGET].to_numpy(int)
    rows = []
    for attr, labels in group_labels(df).items():
        for g in sorted(labels.unique()):
            m = (labels == g).to_numpy()
            n, d = int(m.sum()), int(y[m].sum())
            ok = d >= MIN_CLASS_COUNT and (n - d) >= MIN_CLASS_COUNT
            rows.append({
                "attribute": attr, "group": g, "n": n, "observed_defaults": d,
                "observed_default_rate": y[m].mean(), "mean_predicted_pd": p[m].mean(),
                "predicted_minus_observed": p[m].mean() - y[m].mean(),
                "roc_auc": roc_auc_score(y[m], p[m]) if ok else np.nan,
                "auc_reported": ok,
            })
    return pd.DataFrame(rows)


def run_audit(cfg: Config, test: pd.DataFrame, p_primary: np.ndarray) -> pd.DataFrame:
    table = audit_table(test, p_primary)
    table.to_csv(cfg.artifact_dir("metrics") / "audit_test_primary_model.csv", index=False)
    return table
