"""Predictive modeling: logistic regression vs XGBoost, discrimination, calibration, drivers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from creditlab.reporting import dashboard as d

cfg = d.config()
need = ["metrics/selection.json", "metrics/test_evaluation.json", "metrics/test_bootstrap_ci.csv",
        "predictions/test_predictions.parquet", "metrics/reliability_tables.csv",
        "metrics/permutation_importance_validation.csv"]
if d.missing(cfg, need):
    st.title("Predictive Modeling")
    st.error(d.SETUP)
    st.stop()

sel = d.read(cfg, "metrics/selection.json")
ev = d.read(cfg, "metrics/test_evaluation.json")
boot = d.read_csv(cfg, "metrics/test_bootstrap_ci.csv")
pred = pd.read_parquet(cfg.artifacts / "predictions" / "test_predictions.parquet")
logit, xgb = sel["families"]["logistic"]["model"], sel["families"]["xgboost"]["model"]
NAMES = {"dummy_raw": "Base-rate baseline", logit: "Logistic regression", xgb: "XGBoost"}
COLORS = {"dummy_raw": d.NEUTRAL, logit: d.BLUE, xgb: d.RED}

d.page_header(
    "Predictive modeling",
    "Logistic regression vs gradient-boosted trees",
    "Both models predict the probability that an account misses its next monthly payment, using "
    "six months of repayment status, balances and payments. Hyperparameters were tuned by "
    "cross-validation on the training split, every choice was frozen on the validation split, and "
    "the 6,000-account test split was scored afterwards.")


def ci(model: str, metric: str) -> str:
    r = boot[(boot.model == model) & (boot.metric == metric)].iloc[0]
    return f'<span class="ci">[{r.ci_low:.3f}, {r.ci_high:.3f}]</span>'


rows = []
for name in ("dummy_raw", logit, xgb):
    t = ev["test_metrics"][name]
    rows.append([NAMES[name], f"{t['roc_auc']:.3f}{ci(name, 'roc_auc')}",
                 f"{t['average_precision']:.3f}{ci(name, 'average_precision')}",
                 f"{t['brier']:.3f}", f"{t['log_loss']:.3f}"])
d.table(["Test split (n = 6,000)", "ROC-AUC [95% CI]", "PR-AUC [95% CI]", "Brier", "Log loss"],
        rows, highlight={2})
diff = boot[(boot.model == f"{xgb} - {logit}")].set_index("metric")
st.caption(
    f"Bracketed: 1,000-replicate bootstrap 95% intervals on the test split. XGBoost − logistic "
    f"(paired bootstrap): ΔAUC {diff.loc['roc_auc', 'boot_mean']:+.3f} "
    f"[{diff.loc['roc_auc', 'ci_low']:+.3f}, {diff.loc['roc_auc', 'ci_high']:+.3f}], "
    f"Δlog loss {diff.loc['log_loss', 'boot_mean']:+.4f} "
    f"[{diff.loc['log_loss', 'ci_low']:+.4f}, {diff.loc['log_loss', 'ci_high']:+.4f}]. "
    "Lower Brier and log loss are better. The baseline predicts the training default rate for "
    "everyone.")

st.header("Discrimination", anchor=False)
roc, pr = go.Figure(), go.Figure()
for name in ("dummy_raw", logit, xgb):
    p = pred[f"p_{name}"].to_numpy()
    dash = "dot" if name == "dummy_raw" else "solid"
    fpr, tpr, _ = roc_curve(pred.y, p)
    roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=NAMES[name],
                             line={"color": COLORS[name], "width": 2, "dash": dash},
                             hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>" + NAMES[name] + "</extra>"))
    if name == "dummy_raw":
        pr.add_trace(go.Scatter(x=[0, 1], y=[pred.y.mean()] * 2, mode="lines", name=NAMES[name],
                                line={"color": COLORS[name], "width": 2, "dash": dash}))
    else:
        prec, rec, _ = precision_recall_curve(pred.y, p)
        pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=NAMES[name],
                                line={"color": COLORS[name], "width": 2},
                                hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<extra>"
                                + NAMES[name] + "</extra>"))
roc.update_xaxes(title="False positive rate", range=[0, 1])
roc.update_yaxes(title="True positive rate", range=[0, 1.02])
pr.update_xaxes(title="Recall (share of defaulters caught)", range=[0, 1])
pr.update_yaxes(title="Precision", range=[0, 1.02])
c1, c2 = st.columns(2, gap="large")
d.chart(roc, "ROC curve", 380, container=c1)
d.chart(pr, "Precision–recall curve", 380, container=c2)
st.markdown(
    "The two models rank accounts almost identically across most of the ROC curve. XGBoost's "
    "advantage is concentrated among the highest-risk accounts (low recall, high precision). The "
    "difference is statistically supported but small, both in ranking (AUC) and in probability "
    "quality (log loss and Brier score).")

st.header("Calibration", anchor=False)
rel = d.read_csv(cfg, "metrics/reliability_tables.csv")
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                         line={"color": d.NEUTRAL, "dash": "dot", "width": 1}, hoverinfo="skip"))
for name in (logit, xgb):
    t = rel[(rel.split == "test") & (rel.model == name)]
    fig.add_trace(go.Scatter(
        x=t.mean_predicted, y=t.observed_rate, mode="lines+markers", name=NAMES[name],
        line={"color": COLORS[name], "width": 2}, marker={"size": 8},
        error_y={"type": "data", "array": 1.96 * t.observed_se, "thickness": 1, "width": 3},
        customdata=np.c_[t.n], hovertemplate="mean PD %{x:.3f}<br>observed %{y:.3f}<br>n = "
        "%{customdata[0]}<extra>" + NAMES[name] + "</extra>"))
fig.update_xaxes(title="Mean predicted PD (decile of accounts)", range=[0, 0.85])
fig.update_yaxes(title="Observed default rate", range=[0, 0.85])
c1, c2 = st.columns([3, 2], gap="large")
d.chart(fig, "Predicted vs observed default rate, test split", 400, container=c1)
with c2:
    tx, tl = ev["test_metrics"][xgb], ev["test_metrics"][logit]
    d.table(["Calibration (test)", "XGBoost", "Logistic"], [
        ["Calibration slope (ideal 1)", f"{tx['calibration_slope']:.2f}", f"{tl['calibration_slope']:.2f}"],
        ["Intercept (ideal 0)", f"{tx['calibration_intercept']:+.2f}", f"{tl['calibration_intercept']:+.2f}"],
        ["Expected calibration error", f"{tx['ece']:.3f}", f"{tl['ece']:.3f}"],
        ["Mean PD − default rate", f"{tx['mean_pd_minus_default_rate']:+.3f}",
         f"{tl['mean_pd_minus_default_rate']:+.3f}"],
    ])
    st.markdown(
        f"Each point groups about {int(t.n.median()):,} test accounts by predicted PD. Points on "
        "the diagonal mean a "
        "predicted 30% corresponds to about 30% observed defaults. Error bars are 95% binomial "
        "intervals. Sigmoid and isotonic recalibration (fitted on training out-of-fold "
        "predictions) did not improve validation log loss by a bootstrap-supported margin, so the "
        "raw models were kept.")

st.header("What drives the predictions", anchor=False)
perm = d.read_csv(cfg, "metrics/permutation_importance_validation.csv")
t = perm[perm.model == xgb].nlargest(8, "auc_drop_mean").iloc[::-1]
LABELS = {"PAY_0": "Repayment status, Sep", "PAY_2": "Repayment status, Aug",
          "PAY_3": "Repayment status, Jul", "PAY_4": "Repayment status, Jun",
          "PAY_5": "Repayment status, May", "PAY_6": "Repayment status, Apr",
          "LIMIT_BAL": "Credit limit", "BILL_AMT1": "Statement balance, Sep"}
names = [LABELS.get(f, f.replace("_", " ").title()) for f in t.feature]
fig = go.Figure(go.Bar(x=t.auc_drop_mean, y=names, orientation="h", marker_color=d.RED,
                       error_x={"type": "data", "array": t.auc_drop_std, "thickness": 1,
                                "color": d.INK_2},
                       hovertemplate="%{y}: AUC drop %{x:.3f}<extra></extra>"))
fig.update_xaxes(title="Drop in validation ROC-AUC when the feature is shuffled")
c1, c2 = st.columns([3, 2], gap="large")
d.chart(fig, "XGBoost permutation importance (validation split)", 340, container=c1)
c2.markdown(
    "The most recent repayment status dominates: shuffling it costs about "
    f"{t.auc_drop_mean.iloc[-1]:.2f} AUC, {t.auc_drop_mean.iloc[-1] / t.auc_drop_mean.iloc[-2]:.1f}× "
    "the next most important input. Credit limit and "
    "the latest balance come next. Importance is predictive association, not causation: higher "
    "limits partly reflect the bank's own earlier judgement of who was safe.")

with st.expander("Validation protocol and disclosed deviation"):
    st.markdown(
        "1. Tuning by 5-fold cross-validation on the training split (18,000 accounts), selecting "
        "on log loss because the probabilities feed the loss model.\n"
        "2. Calibrators fitted on training out-of-fold predictions only.\n"
        "3. Calibration variant, primary model and classification thresholds chosen on the "
        "validation split and written to disk before the test split is loaded.\n"
        "4. The test split is scored with the frozen choices, with automated leakage checks.\n\n"
        "**Deviation (D9).** The first calibration rule selected logistic + sigmoid on a "
        "negligible validation gain. Test metrics had already been displayed when the rule was "
        "tightened to require a bootstrap-supported improvement. The change did not alter the "
        "primary model or any reported test number, but the test split cannot be described as "
        "untouched.")
d.footer(cfg)
