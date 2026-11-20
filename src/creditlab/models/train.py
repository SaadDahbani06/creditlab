"""Training stage: tune, fit and calibrate every model on the TRAINING split only.

Outputs (artifacts/models/):
  dummy_raw.joblib, logistic_{raw,sigmoid,isotonic}.joblib, xgboost_{raw,sigmoid,isotonic}.joblib
  training_metadata.json  (best params, train-ID hash, feature lists, sizes)
  cv_results_{logistic,xgboost}.csv
"""

from __future__ import annotations

import hashlib
import time

import joblib
import numpy as np
import pandas as pd

from creditlab.data.preprocess import TARGET, load_split, model_input_columns
from creditlab.models.baselines import dummy_model
from creditlab.models.calibrate import fit_calibrated
from creditlab.models.tune import tune_logistic, tune_xgboost
from creditlab.utils.config import Config, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)

FAMILIES = ["logistic", "xgboost"]
VARIANTS = ["raw", "sigmoid", "isotonic"]


def ids_fingerprint(ids) -> str:
    arr = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def candidate_names(cfg: Config) -> list[str]:
    methods = cfg["calibration"]["methods"]
    return ["dummy_raw"] + [f"{f}_{v}" for f in FAMILIES for v in ["raw", *methods]]


def model_path(cfg: Config, name: str):
    return cfg.artifact_dir("models") / f"{name}.joblib"


def load_model(cfg: Config, name: str):
    path = model_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `creditlab train` first.")
    return joblib.load(path)


def train_models(cfg: Config, train: pd.DataFrame) -> dict:
    """Fit all candidates on ``train`` (the training split) and return metadata.

    This function receives only the training split; it has no access to validation or test
    rows by construction.
    """
    if set(train["split"].unique()) != {"train"}:
        raise ValueError("train_models must receive only training-split rows")
    features = model_input_columns(cfg)
    X, y = train[features], train[TARGET].astype(int)
    meta: dict = {
        "n_train": len(train),
        "n_train_defaults": int(y.sum()),
        "train_id_sha256": ids_fingerprint(train["ID"]),
        "input_features": features,
        "selection_metric": cfg["models"]["selection_metric"],
        "models": {},
    }
    out = cfg.artifact_dir("models")

    t0 = time.perf_counter()
    dummy = dummy_model().fit(X, y)
    joblib.dump(dummy, model_path(cfg, "dummy_raw"))
    meta["models"]["dummy_raw"] = {"params": {"strategy": "prior"},
                                   "fit_seconds": time.perf_counter() - t0}

    tuners = {"logistic": tune_logistic, "xgboost": tune_xgboost}
    for family in FAMILIES:
        t0 = time.perf_counter()
        best, params, cv_table = tuners[family](cfg, X, y)
        cv_table.to_csv(out / f"cv_results_{family}.csv", index=False)
        joblib.dump(best, model_path(cfg, f"{family}_raw"))
        top = cv_table.iloc[0]
        meta["models"][f"{family}_raw"] = {
            "params": params,
            "cv_mean_log_loss": -float(top["mean_test_neg_log_loss"]),
            "cv_mean_roc_auc": float(top["mean_test_roc_auc"]),
            "cv_candidates": len(cv_table),
            "fit_seconds": time.perf_counter() - t0,
        }
        log.info("%s tuned in %.1fs: %s | CV log loss %.4f, AUC %.4f", family,
                 time.perf_counter() - t0, params, -top["mean_test_neg_log_loss"],
                 top["mean_test_roc_auc"])
        for method in cfg["calibration"]["methods"]:
            t0 = time.perf_counter()
            cal = fit_calibrated(cfg, best, method, X, y)
            joblib.dump(cal, model_path(cfg, f"{family}_{method}"))
            meta["models"][f"{family}_{method}"] = {
                "base_params": params, "method": method,
                "calibration": f"{cfg['calibration']['cv_folds']}-fold training OOF, ensemble=False",
                "fit_seconds": time.perf_counter() - t0,
            }
            log.info("%s %s calibration fitted in %.1fs", family, method, time.perf_counter() - t0)

    write_json(out / "training_metadata.json", meta)
    return meta


def run_train(cfg: Config) -> dict:
    train = load_split(cfg, "train")
    meta = train_models(cfg, train)
    update_manifest(cfg, "train", {
        "n_train": meta["n_train"],
        "best_params": {k: v.get("params") for k, v in meta["models"].items() if "params" in v},
    })
    return meta
