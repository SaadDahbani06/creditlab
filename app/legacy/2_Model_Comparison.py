"""Model Comparison: dummy vs logistic regression vs XGBoost on the test split."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from creditlab.reporting import dashboard as d

cfg = d.config()
st.title("Model Comparison")
st.caption("Test split (n = 6,000), scored once after all choices were made on the validation "
           "split. Out-of-sample client split, not a chronological backtest.")

need = ["metrics/selection.json", "metrics/test_evaluation.json", "metrics/test_bootstrap_ci.csv",
        "predictions/test_predictions.parquet"]
if d.missing(cfg, need):
    st.error(d.SETUP)
    st.stop()

sel = d.read(cfg, "metrics/selection.json")
ev = d.read(cfg, "metrics/test_evaluation.json")
pred = pd.read_parquet(cfg.artifacts / "predictions" / "test_predictions.parquet")
chosen = [sel["families"][f]["model"] for f in ("logistic", "xgboost")]
models = ["dummy_raw", *chosen]
color = {"dummy_raw": d.NEUTRAL, chosen[0]: d.ENTITY_COLORS["logistic"],
         chosen[1]: d.ENTITY_COLORS["xgboost"]}

c1, c2 = st.columns(2)
roc, pr = go.Figure(), go.Figure()
for m in models:
    auc = ev["test_metrics"][m]["roc_auc"]
    ap = ev["test_metrics"][m]["average_precision"]
    fpr, tpr, _ = roc_curve(pred["y"], pred[f"p_{m}"])
    roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{m} (AUC {auc:.3f})", mode="lines",
                             line={"color": color[m], "width": 2,
                                   "dash": "dash" if m == "dummy_raw" else "solid"}))
    if m == "dummy_raw":
        pr.add_trace(go.Scatter(x=[0, 1], y=[pred["y"].mean()] * 2, name=f"{m} (AP {ap:.3f})",
                                mode="lines", line={"color": color[m], "dash": "dash"}))
    else:
        p, r, _ = precision_recall_curve(pred["y"], pred[f"p_{m}"])
        pr.add_trace(go.Scatter(x=r, y=p, name=f"{m} (AP {ap:.3f})", mode="lines",
                                line={"color": color[m], "width": 2}))
roc.update_xaxes(title="False positive rate")
roc.update_yaxes(title="True positive rate")
pr.update_xaxes(title="Recall")
pr.update_yaxes(title="Precision", range=[0, 1])
c1.plotly_chart(d.style(roc, 420, "ROC curve"), use_container_width=True)
c2.plotly_chart(d.style(pr, 420, "Precision–recall curve"), use_container_width=True)

st.subheader("Metrics with 95% bootstrap confidence intervals")
boot = d.read_csv(cfg, "metrics/test_bootstrap_ci.csv")
rows = []
for m in models:
    tm = ev["test_metrics"][m]
    b = boot[boot.model == m].set_index("metric")
    rows.append({"model": m, **{
        k: f"{tm[k]:.4f}  [{b.loc[k, 'ci_low']:.4f}, {b.loc[k, 'ci_high']:.4f}]"
        for k in ("roc_auc", "average_precision", "brier", "log_loss")}})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
diff = boot[boot.model == f"{chosen[1]} - {chosen[0]}"]
st.markdown("**Paired bootstrap difference (XGBoost − logistic regression)**")
st.dataframe(diff[["metric", "boot_mean", "ci_low", "ci_high"]].style.format(precision=4),
             hide_index=True, use_container_width=True)

st.subheader("Threshold-dependent metrics")
st.caption("Thresholds maximise F1 on the validation split and were frozen before the test split "
           "was scored. F1 is a convention for this comparison, not a business optimum.")
thr = ev["test_threshold_metrics"]
st.dataframe(pd.DataFrame([{"model": k, "threshold": v["threshold"], "precision": v["precision"],
                            "recall": v["recall"], "F1": v["f1"], "accuracy": v["accuracy"],
                            **v["confusion_matrix"]} for k, v in thr.items()]
                          ).style.format(precision=4, subset=["threshold", "precision", "recall",
                                                              "F1", "accuracy"]),
             hide_index=True, use_container_width=True)

st.subheader("What the models use")
perm = d.read_csv(cfg, "metrics/permutation_importance_validation.csv")
c1, c2 = st.columns(2)
for col, m, colr in ((c1, "logistic_raw", d.ENTITY_COLORS["logistic"]),
                      (c2, "xgboost_raw", d.ENTITY_COLORS["xgboost"])):
    t = perm[perm.model == m].nlargest(10, "auc_drop_mean").iloc[::-1]
    fig = go.Figure(go.Bar(x=t.auc_drop_mean, y=t.feature, orientation="h", marker_color=colr,
                           error_x={"type": "data", "array": t.auc_drop_std}))
    fig.update_xaxes(title="Drop in validation ROC-AUC when permuted")
    col.plotly_chart(d.style(fig, 380, f"{m}: permutation importance"), use_container_width=True)
st.caption("Computed on the validation split. Importance is predictive association, not a causal "
           "effect, and correlated inputs share importance.")
with st.expander("Logistic regression coefficients"):
    st.dataframe(d.read_csv(cfg, "metrics/logistic_coefficients.csv").style.format(precision=4),
                 hide_index=True, use_container_width=True)
with st.expander("Cross-validation results (training split)"):
    for fam in ("logistic", "xgboost"):
        st.markdown(f"**{fam}**")
        st.dataframe(pd.read_csv(cfg.artifacts / "models" / f"cv_results_{fam}.csv").head(10),
                     hide_index=True, use_container_width=True)
