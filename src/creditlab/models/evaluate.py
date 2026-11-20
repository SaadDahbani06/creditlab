"""Evaluation stage.

Phase 1 (VALIDATION only): score every candidate, choose each family's calibration variant,
choose the primary model, choose classification thresholds. Write selection.json.
Phase 2 (TEST, once): load the frozen selection, score the test split, bootstrap confidence
intervals, save predictions and figures. Nothing computed in phase 2 feeds back into phase 1.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from creditlab.data.preprocess import TARGET, load_split, model_input_columns
from creditlab.models.calibrate import (
    calibration_in_the_large,
    calibration_intercept_slope,
    expected_calibration_error,
    reliability_table,
)
from creditlab.models.train import FAMILIES, candidate_names, ids_fingerprint, load_model
from creditlab.utils.config import Config, read_json, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)
CLIP = 1e-15


# ----------------------------------------------------------------------------- metrics
def probability_metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    intercept, slope = calibration_intercept_slope(y, p) if np.ptp(p) > 0 else (np.nan, np.nan)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_pd_minus_default_rate": calibration_in_the_large(y, p),
        "accuracy_at_0.5": float(np.mean((p >= 0.5) == y)),
    }


def threshold_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y, int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y)),
        "flagged_share": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold_max_f1(y, p) -> float:
    """Threshold maximising F1 on the data given (call with VALIDATION data only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)[:-1]
    return float(thr[int(np.argmax(f1))])


def _vector_log_loss(y, p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap(y, preds: dict[str, np.ndarray], reps: int, rng: np.random.Generator,
              pairs: list[tuple[str, str]] = ()) -> pd.DataFrame:
    """Paired nonparametric bootstrap of metric values and pairwise differences.

    The same resampled indices are used for every model in a replicate, so differences
    reflect model differences rather than sampling noise between models.
    """
    y = np.asarray(y, int)
    n = len(y)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        rec = {}
        for name, p in preds.items():
            pb = p[idx]
            rec[(name, "roc_auc")] = roc_auc_score(yb, pb)
            rec[(name, "average_precision")] = average_precision_score(yb, pb)
            rec[(name, "brier")] = np.mean((pb - yb) ** 2)
            rec[(name, "log_loss")] = np.mean(_vector_log_loss(yb, pb))
        for a, b in pairs:
            for m in ("roc_auc", "average_precision", "brier", "log_loss"):
                rec[(f"{a} - {b}", m)] = rec[(a, m)] - rec[(b, m)]
        rows.append(rec)
    boot = pd.DataFrame(rows)
    summary = boot.quantile([0.025, 0.975]).T
    summary.columns = ["ci_low", "ci_high"]
    summary["boot_mean"] = boot.mean()
    summary.index = pd.MultiIndex.from_tuples(summary.index, names=["model", "metric"])
    return summary.reset_index()


def log_loss_ci_high(boot: pd.DataFrame, a: str, b: str) -> float:
    """Upper 95% bootstrap bound of log_loss(a) - log_loss(b)."""
    row = boot[(boot["model"] == f"{a} - {b}") & (boot["metric"] == "log_loss")]
    return float(row["ci_high"].iloc[0])


# ----------------------------------------------------------------------------- phase 1
def predict_all(cfg: Config, df: pd.DataFrame) -> dict[str, np.ndarray]:
    X = df[model_input_columns(cfg)]
    out = {}
    for name in candidate_names(cfg):
        p = load_model(cfg, name).predict_proba(X)[:, 1]
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            raise ValueError(f"{name} produced probabilities outside [0, 1]")
        out[name] = p
    return out


def select_on_validation(cfg: Config, y_val, preds_val: dict[str, np.ndarray]) -> dict:
    """All model, calibration and threshold choices. Receives validation data only.

    Calibration rule (documented in docs/DECISIONS.md): a calibrated variant replaces the raw
    model only if, on validation, (1) it lowers log loss, (2) it does not raise Brier score, and
    (3) the paired-bootstrap 95% CI of its log-loss difference versus raw lies entirely below 0.
    Among qualifying variants the lowest log loss wins; otherwise the simpler raw model is kept.
    A point improvement in one metric is not enough.
    """
    y_val = np.asarray(y_val, int)
    val_metrics = {name: probability_metrics(y_val, p) for name, p in preds_val.items()}
    rng = cfg.rng("validation_bootstrap")
    pairs = [(f"{f}_{m}", f"{f}_raw") for f in FAMILIES for m in cfg["calibration"]["methods"]]
    boot_val = bootstrap(y_val, {k: v for k, v in preds_val.items() if k != "dummy_raw"},
                         reps=max(200, cfg["evaluation"]["bootstrap_reps"] // 2), rng=rng,
                         pairs=pairs)
    families = {}
    for fam in FAMILIES:
        raw = val_metrics[f"{fam}_raw"]
        qualifying = [
            m for m in cfg["calibration"]["methods"]
            if val_metrics[f"{fam}_{m}"]["log_loss"] < raw["log_loss"]
            and val_metrics[f"{fam}_{m}"]["brier"] <= raw["brier"]
            and log_loss_ci_high(boot_val, f"{fam}_{m}", f"{fam}_raw") < 0
        ]
        chosen = min(qualifying, key=lambda m: val_metrics[f"{fam}_{m}"]["log_loss"]) \
            if qualifying else "raw"
        name = f"{fam}_{chosen}"
        families[fam] = {
            "chosen_variant": chosen,
            "model": name,
            "qualifying_calibrations": qualifying,
            "threshold_max_f1": select_threshold_max_f1(y_val, preds_val[name]),
        }
    primary = min(FAMILIES, key=lambda f: val_metrics[families[f]["model"]]["log_loss"])
    dummy_thr = select_threshold_max_f1(y_val, preds_val["dummy_raw"])
    return {
        "selected_using": "validation split only",
        "selected_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule_calibration": ("calibrated variant must lower validation log loss, not raise Brier, "
                             "and have a paired-bootstrap 95% CI of the log-loss change below 0"),
        "rule_primary": "lowest validation log loss among each family's chosen variant",
        "rule_threshold": cfg["evaluation"]["threshold_rule"],
        "families": families,
        "primary_family": primary,
        "primary_model": families[primary]["model"],
        "dummy_threshold_max_f1": dummy_thr,
        "validation_metrics": val_metrics,
        "validation_bootstrap": boot_val.to_dict(orient="records"),
    }


# ----------------------------------------------------------------------------- run
def _check_no_leakage(cfg: Config, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    meta = read_json(cfg.artifact_dir("models") / "training_metadata.json")
    train = load_split(cfg, "train")
    checks = {
        "train_ids_match_model_metadata": ids_fingerprint(train["ID"]) == meta["train_id_sha256"],
        "n_train_matches": len(train) == meta["n_train"],
        "validation_disjoint_from_train": not set(val["ID"]) & set(train["ID"]),
        "test_disjoint_from_train": not set(test["ID"]) & set(train["ID"]),
        "test_disjoint_from_validation": not set(test["ID"]) & set(val["ID"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Leakage check failed: {checks}")
    return checks


def run_evaluate(cfg: Config) -> dict:
    metrics_dir = cfg.artifact_dir("metrics")
    pred_dir = cfg.artifact_dir("predictions")

    # ---- phase 1: validation only
    val = load_split(cfg, "validation")
    preds_val = predict_all(cfg, val)
    selection = select_on_validation(cfg, val[TARGET], preds_val)
    write_json(metrics_dir / "selection.json", selection)
    pd.DataFrame({"ID": val["ID"], "y": val[TARGET], **{f"p_{k}": v for k, v in preds_val.items()}}
                 ).to_parquet(pred_dir / "validation_predictions.parquet", index=False)
    log.info("Selection frozen (validation): primary=%s; %s", selection["primary_model"],
             {f: d["model"] for f, d in selection["families"].items()})

    # ---- phase 2: test, scored once with frozen choices
    test = load_split(cfg, "test")
    leakage = _check_no_leakage(cfg, val, test)
    preds_test = predict_all(cfg, test)
    y_test = test[TARGET].to_numpy(int)
    test_metrics = {name: probability_metrics(y_test, p) for name, p in preds_test.items()}
    thresholds = {f: d["threshold_max_f1"] for f, d in selection["families"].items()}
    thr_results = {
        selection["families"][f]["model"]: threshold_metrics(
            y_test, preds_test[selection["families"][f]["model"]], thresholds[f])
        for f in FAMILIES
    }
    thr_results["dummy_raw"] = threshold_metrics(y_test, preds_test["dummy_raw"],
                                                 selection["dummy_threshold_max_f1"])
    chosen = {f: selection["families"][f]["model"] for f in FAMILIES}
    boot = bootstrap(
        y_test, {**{n: preds_test[n] for n in chosen.values()}, "dummy_raw": preds_test["dummy_raw"]},
        reps=cfg["evaluation"]["bootstrap_reps"], rng=cfg.rng("test_bootstrap"),
        pairs=[(chosen["xgboost"], chosen["logistic"])],
    )
    boot.to_csv(metrics_dir / "test_bootstrap_ci.csv", index=False)

    pd.DataFrame({"ID": test["ID"], "y": y_test, **{f"p_{k}": v for k, v in preds_test.items()}}
                 ).to_parquet(pred_dir / "test_predictions.parquet", index=False)

    n_bins = cfg["calibration"]["n_bins"]
    rel = []
    for split_name, y, preds in (("validation", val[TARGET].to_numpy(int), preds_val),
                                 ("test", y_test, preds_test)):
        for name, p in preds.items():
            if name == "dummy_raw":
                continue
            t = reliability_table(y, p, n_bins)
            t.insert(0, "model", name)
            t.insert(0, "split", split_name)
            rel.append(t)
    pd.concat(rel).to_csv(metrics_dir / "reliability_tables.csv", index=False)

    table = pd.DataFrame(test_metrics).T
    table.index.name = "model"
    table.to_csv(metrics_dir / "test_metrics.csv")
    pd.DataFrame(selection["validation_metrics"]).T.rename_axis("model").to_csv(
        metrics_dir / "validation_metrics.csv")
    result = {
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split_label": "out-of-sample client split (NOT a chronological backtest)",
        "n_test": len(test), "n_test_defaults": int(y_test.sum()),
        "leakage_checks": leakage,
        "primary_model": selection["primary_model"],
        "test_metrics": test_metrics,
        "test_threshold_metrics": thr_results,
    }
    write_json(metrics_dir / "test_evaluation.json", result)

    from creditlab.models.audit import run_audit
    from creditlab.models.explain import run_explain
    from creditlab.reporting.model_figures import make_model_figures

    run_explain(cfg, val)
    run_audit(cfg, test, preds_test[selection["primary_model"]])
    make_model_figures(cfg)
    update_manifest(cfg, "evaluate", {
        "primary_model": selection["primary_model"],
        "test_auc_primary": test_metrics[selection["primary_model"]]["roc_auc"],
        "leakage_checks": leakage,
    })
    log.info("Test (primary %s): AUC %.4f, Brier %.4f, log loss %.4f",
             selection["primary_model"], test_metrics[selection["primary_model"]]["roc_auc"],
             test_metrics[selection["primary_model"]]["brier"],
             test_metrics[selection["primary_model"]]["log_loss"])
    return result
