"""Calibration: reliability curves and raw vs calibrated probability comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from creditlab.reporting import dashboard as d

cfg = d.config()
st.title("Calibration")
st.caption("Can the predicted default probabilities be read as probabilities? Calibrators were "
           "fitted on training out-of-fold predictions; the choice was made on validation.")

need = ["metrics/reliability_tables.csv", "metrics/selection.json",
        "predictions/validation_predictions.parquet"]
if d.missing(cfg, need):
    st.error(d.SETUP)
    st.stop()

rel = d.read_csv(cfg, "metrics/reliability_tables.csv")
sel = d.read(cfg, "metrics/selection.json")

c1, c2 = st.columns([1, 3])
split = c1.radio("Split", ["validation", "test"], horizontal=False)
family = c1.radio("Model family", ["logistic", "xgboost"])
c1.markdown(f"Chosen variant: **{sel['families'][family]['chosen_variant']}**")
fig = go.Figure()
for variant in ["raw", *cfg["calibration"]["methods"]]:
    t = rel[(rel.split == split) & (rel.model == f"{family}_{variant}")]
    fig.add_trace(go.Scatter(
        x=t.mean_predicted, y=t.observed_rate, mode="lines+markers", name=variant,
        line={"color": d.ENTITY_COLORS[variant], "width": 2},
        error_y={"type": "data", "array": 1.96 * t.observed_se, "visible": variant == "raw"},
        customdata=t[["n", "observed_defaults", "expected_defaults"]],
        hovertemplate="mean PD %{x:.3f}<br>observed %{y:.3f}<br>n %{customdata[0]}<br>"
                      "observed defaults %{customdata[1]}<br>expected %{customdata[2]:.1f}"
                      "<extra>%{fullData.name}</extra>"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                         line={"color": d.NEUTRAL, "dash": "dot", "width": 1}))
fig.update_xaxes(title="Mean predicted PD (decile bin)", range=[0, 1])
fig.update_yaxes(title="Observed default rate", range=[0, 1])
c2.plotly_chart(d.style(fig, 460, f"Reliability: {family}, {split} split"),
                use_container_width=True)

st.subheader("Raw vs calibrated (validation)")
vm = pd.DataFrame(sel["validation_metrics"]).T
st.dataframe(vm[["roc_auc", "brier", "log_loss", "ece", "calibration_intercept",
                 "calibration_slope", "mean_pd_minus_default_rate"]].style.format(precision=4),
             use_container_width=True)
boot = pd.DataFrame(sel["validation_bootstrap"])
boot = boot[boot.model.str.contains(" - ") & boot.metric.isin(["log_loss", "brier"])]
st.markdown("**Paired bootstrap differences (calibrated − raw), validation split**")
st.dataframe(boot[["model", "metric", "boot_mean", "ci_low", "ci_high"]].style.format(
    precision=5, subset=["boot_mean", "ci_low", "ci_high"]), hide_index=True,
    use_container_width=True)
st.markdown(f"Selection rule: *{sel['rule_calibration']}*. "
            "A calibrated variant is not declared better because one number moved.")

st.subheader("Distribution of predicted PDs (validation)")
vp = pd.read_parquet(cfg.artifacts / "predictions" / "validation_predictions.parquet")
fig = go.Figure()
for variant in ["raw", *cfg["calibration"]["methods"]]:
    fig.add_trace(go.Histogram(x=vp[f"p_{family}_{variant}"], name=variant, nbinsx=60,
                               opacity=0.55, marker_color=d.ENTITY_COLORS[variant]))
fig.update_layout(barmode="overlay")
fig.update_xaxes(title="Predicted PD")
fig.update_yaxes(title="Accounts")
st.plotly_chart(d.style(fig, 340), use_container_width=True)
st.caption("Isotonic regression produces a step function, so many accounts share identical PDs. "
           "This creates ties that lower ranking metrics such as average precision.")
