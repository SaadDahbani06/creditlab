"""Figures for model comparison, calibration and importance (from saved artifacts only)."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from creditlab.reporting.style import (
    ENTITY_COLORS,
    NEUTRAL,
    SERIES,
    TEXT_SECONDARY,
    apply_style,
    caption,
    plt,
    save,
)
from creditlab.utils.config import Config, read_json

LABELS = {"dummy_raw": "Dummy (base rate)", "logistic": "Logistic regression", "xgboost": "XGBoost"}


def _family(name: str) -> str:
    return name.split("_")[0]


def make_model_figures(cfg: Config) -> None:
    apply_style()
    figs = cfg.artifact_dir("figures")
    metrics = cfg.artifact_dir("metrics")
    sel = read_json(metrics / "selection.json")
    test = pd.read_parquet(cfg.artifact_dir("predictions") / "test_predictions.parquet")
    test_m = pd.read_csv(metrics / "test_metrics.csv", index_col=0)
    chosen = [sel["families"][f]["model"] for f in ("logistic", "xgboost")]
    y = test["y"].to_numpy()

    # ROC + PR on test
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name in ["dummy_raw", *chosen]:
        fam = _family(name)
        color = ENTITY_COLORS.get(fam, NEUTRAL)
        label_base = LABELS.get(name, LABELS.get(fam, name))
        variant = "" if fam == "dummy" else f" ({name.split('_', 1)[1]})"
        fpr, tpr, _ = roc_curve(y, test[f"p_{name}"])
        a1.plot(fpr, tpr, color=color, ls="--" if fam == "dummy" else "-",
                label=f"{label_base}{variant}: AUC {test_m.loc[name, 'roc_auc']:.3f}")
        ap_label = f"{label_base}{variant}: AP {test_m.loc[name, 'average_precision']:.3f}"
        if fam == "dummy":
            # A constant score has precision = base rate at every recall (no-skill line);
            # precision_recall_curve returns only two endpoints, which would draw a diagonal.
            a2.axhline(y.mean(), color=color, ls="--", lw=1.5, label=ap_label)
            continue
        prec, rec, _ = precision_recall_curve(y, test[f"p_{name}"])
        a2.plot(rec, prec, color=color, label=ap_label)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test split)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve (test split)",
           ylim=(0, 1))
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    caption(a1, f"Out-of-sample client split, n = {len(y):,}; not a chronological backtest.")
    save(fig, figs / "roc_pr_test.png")

    # Reliability: validation, each family raw vs calibrated
    rel = pd.read_csv(metrics / "reliability_tables.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, fam in zip(axes, ("logistic", "xgboost"), strict=True):
        for variant in ["raw", *cfg["calibration"]["methods"]]:
            t = rel[(rel["split"] == "validation") & (rel["model"] == f"{fam}_{variant}")]
            ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", markersize=4,
                    color=ENTITY_COLORS[variant], label=variant)
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls=":", label="perfect calibration")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted PD (decile bin)",
               title=f"{LABELS[fam]}: reliability on validation")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Observed default rate")
    caption(axes[0], "Calibrators fitted on training out-of-fold predictions; shown on validation.")
    save(fig, figs / "reliability_validation.png")

    # Reliability: test, chosen models, with binomial error bars
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name in chosen:
        t = rel[(rel["split"] == "test") & (rel["model"] == name)]
        ax.errorbar(t["mean_predicted"], t["observed_rate"], yerr=1.96 * t["observed_se"],
                    marker="o", markersize=4, capsize=2, color=ENTITY_COLORS[_family(name)],
                    label=f"{LABELS[_family(name)]} ({name.split('_', 1)[1]})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted PD (decile bin)",
           ylabel="Observed default rate", title="Reliability on test (selected variants)")
    ax.legend(loc="upper left")
    caption(ax, "Bars: ±1.96 binomial SE under the predicted rate.")
    save(fig, figs / "reliability_test.png")

    # Importance
    perm = pd.read_csv(metrics / "permutation_importance_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, name, color in zip(axes, ("logistic_raw", "xgboost_raw"), (SERIES[0], SERIES[1]),
                               strict=True):
        t = perm[perm["model"] == name].nlargest(10, "auc_drop_mean").iloc[::-1]
        ax.barh(t["feature"], t["auc_drop_mean"], xerr=t["auc_drop_std"], color=color,
                height=0.65, error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
        ax.set(xlabel="Mean drop in validation ROC-AUC when permuted",
               title=f"{LABELS[_family(name)]}: permutation importance")
        ax.grid(axis="y", visible=False)
    caption(axes[0], "Validation split, 5 repeats. Predictive association, not causal effect.")
    save(fig, figs / "permutation_importance.png")

    coefs = pd.read_csv(metrics / "logistic_coefficients.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(coefs["feature"], coefs["coefficient"],
            color=[SERIES[7] if c > 0 else SERIES[0] for c in coefs["coefficient"]], height=0.65)
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set(xlabel="Coefficient (log-odds per SD of a numeric input, or per one-hot category)",
           title="Logistic regression: 15 largest |coefficients|")
    ax.grid(axis="y", visible=False)
    caption(ax, "Red raises predicted default odds, blue lowers them. No reference category is dropped "
                "(L2 penalty identifies them):\ncompare categories within one status column. Correlated inputs share weight.")
    save(fig, figs / "logistic_coefficients.png")
