"""Hyperparameter search using cross-validation on the TRAINING split only."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from creditlab.models.baselines import logistic_preprocessor, tree_preprocessor
from creditlab.utils.config import Config

SCORING = {
    "neg_log_loss": "neg_log_loss",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "neg_brier_score": "neg_brier_score",
}


def cv_splitter(cfg: Config, n_splits: int | None = None, name: str = "cv") -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=n_splits or cfg["models"]["cv_folds"], shuffle=True,
        random_state=cfg.sub_seed(name),
    )


def logistic_pipeline(cfg: Config, **params: Any) -> Pipeline:
    return Pipeline([
        ("prep", logistic_preprocessor(cfg["features"]["ratio_clip"])),
        ("clf", LogisticRegression(max_iter=cfg["models"]["logistic"]["max_iter"], **params)),
    ])


def xgboost_pipeline(cfg: Config, **params: Any) -> Pipeline:
    # n_jobs=1 inside the model: parallelism is at the CV level, and single-threaded
    # tree building keeps results bit-for-bit reproducible.
    return Pipeline([
        ("prep", tree_preprocessor(cfg["features"]["ratio_clip"])),
        ("clf", XGBClassifier(
            tree_method="hist", objective="binary:logistic", eval_metric="logloss",
            random_state=cfg.sub_seed("xgboost"), n_jobs=1, **params,
        )),
    ])


def _cv_table(search) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    keep = [c for c in res.columns if c.startswith("param_")] + [
        f"{s}_test_{m}" for m in SCORING for s in ("mean", "std")
    ] + ["rank_test_neg_log_loss"]
    return res[keep].sort_values("rank_test_neg_log_loss").reset_index(drop=True)


def tune_logistic(cfg: Config, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict, pd.DataFrame]:
    grid = {
        "clf__C": cfg["models"]["logistic"]["C"],
        "clf__class_weight": cfg["models"]["logistic"]["class_weight"],
    }
    search = GridSearchCV(
        logistic_pipeline(cfg), grid, scoring=SCORING, refit=cfg["models"]["selection_metric"],
        cv=cv_splitter(cfg), n_jobs=cfg["models"]["xgboost"]["n_jobs"], error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, _strip(search.best_params_), _cv_table(search)


def tune_xgboost(cfg: Config, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict, pd.DataFrame]:
    xcfg = cfg["models"]["xgboost"]
    dist = {f"clf__{k}": list(v) for k, v in xcfg["param_distributions"].items()}
    ratio = float((y == 0).sum() / (y == 1).sum())
    dist["clf__scale_pos_weight"] = [ratio if v == "balanced" else float(v)
                                     for v in dist["clf__scale_pos_weight"]]
    search = RandomizedSearchCV(
        xgboost_pipeline(cfg), dist, n_iter=xcfg["n_iter"], scoring=SCORING,
        refit=cfg["models"]["selection_metric"], cv=cv_splitter(cfg),
        random_state=cfg.sub_seed("xgb_search"), n_jobs=xcfg["n_jobs"], error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, _strip(search.best_params_), _cv_table(search)


def _strip(params: dict) -> dict:
    return {k.removeprefix("clf__"): v for k, v in params.items()}
