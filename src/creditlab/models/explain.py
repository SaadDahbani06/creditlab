"""Model explanation: logistic coefficients, XGBoost gain importance, permutation importance.

Permutation importance is computed on the VALIDATION split (never the test split), permuting
the raw input columns given to each pipeline and measuring the drop in ROC-AUC. Importances
describe what the model uses for prediction, not causal effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from creditlab.data.preprocess import TARGET, model_input_columns
from creditlab.models.train import load_model
from creditlab.utils.config import Config


def logistic_coefficients(cfg: Config) -> pd.DataFrame:
    pipe = load_model(cfg, "logistic_raw")
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["odds_ratio_per_sd"] = np.exp(coef)  # inputs are standardised (or one-hot)
    df["abs"] = df["coefficient"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def xgboost_gain(cfg: Config) -> pd.DataFrame:
    pipe = load_model(cfg, "xgboost_raw")
    booster = pipe.named_steps["clf"].get_booster()
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    gain = booster.get_score(importance_type="total_gain")
    # Booster keys are the training column names (pandas input) or f0.. otherwise.
    rows = [{"feature": k if k in names else names[int(k[1:])], "total_gain": v}
            for k, v in gain.items()]
    df = pd.DataFrame(rows)
    df["share_of_total_gain"] = df["total_gain"] / df["total_gain"].sum()
    return df.sort_values("total_gain", ascending=False).reset_index(drop=True)


def permutation_table(cfg: Config, val: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    X, y = val[model_input_columns(cfg)], val[TARGET].astype(int)
    frames = []
    for name in ("logistic_raw", "xgboost_raw"):
        r = permutation_importance(load_model(cfg, name), X, y, scoring="roc_auc",
                                   n_repeats=n_repeats, random_state=cfg.sub_seed("permutation"),
                                   n_jobs=1)
        frames.append(pd.DataFrame({"model": name, "feature": X.columns,
                                    "auc_drop_mean": r.importances_mean,
                                    "auc_drop_std": r.importances_std}))
    return pd.concat(frames).sort_values(["model", "auc_drop_mean"], ascending=[True, False])


def run_explain(cfg: Config, val: pd.DataFrame) -> None:
    out = cfg.artifact_dir("metrics")
    logistic_coefficients(cfg).to_csv(out / "logistic_coefficients.csv", index=False)
    xgboost_gain(cfg).to_csv(out / "xgboost_gain_importance.csv", index=False)
    permutation_table(cfg, val).to_csv(out / "permutation_importance_validation.csv", index=False)
