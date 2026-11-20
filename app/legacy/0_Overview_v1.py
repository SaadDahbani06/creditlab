"""Overview page of the CREDITLAB research dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from creditlab.reporting import dashboard as d

cfg = d.config()

st.title("CREDITLAB")
st.caption("Credit default modeling, calibration, portfolio stress testing and decision research "
           "on the UCI *Default of Credit Card Clients* dataset (Taiwan, 2005).")
st.info("Research project. Not an underwriting or account-management system. "
        "Predictions concern existing cardholders in 2005 and are not decisions about any person.")

need = ["metrics/data_validation.json", "metrics/selection.json", "metrics/test_evaluation.json",
        "metrics/split_summary.json"]
if d.missing(cfg, need):
    st.error(d.SETUP)
    st.stop()

val = d.read(cfg, "metrics/data_validation.json")
sel = d.read(cfg, "metrics/selection.json")
ev = d.read(cfg, "metrics/test_evaluation.json")
primary = sel["primary_model"]
pm = ev["test_metrics"][primary]

c = st.columns(5)
c[0].metric("Accounts", f"{val['rows']:,}")
c[1].metric("Observed default rate", f"{val['default_rate']:.1%}")
c[2].metric("Selected model", primary.replace("_", " "))
c[3].metric("Test ROC-AUC", f"{pm['roc_auc']:.3f}")
c[4].metric("Test Brier score", f"{pm['brier']:.4f}")

st.subheader("Dataset")
split = pd.DataFrame(d.read(cfg, "metrics/split_summary.json"))
left, right = st.columns([1, 1])
with left:
    st.dataframe(split.style.format({"default_rate": "{:.4f}"}), hide_index=True,
                 use_container_width=True)
with right:
    st.markdown(
        "- 30,000 existing credit-card accounts, April–September 2005 history.\n"
        "- Target: default on the **next month's** payment (observed label).\n"
        "- Stratified 60/20/20 **client-level** split (fixed seed). This is an out-of-sample client "
        "split, **not** a chronological backtest.\n"
        "- SEX, AGE, MARRIAGE and EDUCATION are excluded from the model and used only for an "
        "aggregate audit."
    )

st.subheader("Model performance (test split, scored once after validation-based selection)")
rows = []
for name in ["dummy_raw", *[sel["families"][f]["model"] for f in ("logistic", "xgboost")]]:
    m = ev["test_metrics"][name]
    rows.append({"model": name, "ROC-AUC": m["roc_auc"], "PR-AUC": m["average_precision"],
                 "Brier": m["brier"], "log loss": m["log_loss"], "ECE": m["ece"],
                 "accuracy @0.5": m["accuracy_at_0.5"]})
st.dataframe(pd.DataFrame(rows).style.format(precision=4), hide_index=True,
             use_container_width=True)
st.caption("Accuracy is shown to illustrate why it is insufficient: the dummy model that predicts "
           "the base rate for everyone reaches 77.9% accuracy at a 0.5 threshold while having no "
           "ability to rank risk (AUC 0.5).")

manifest_path = cfg.artifacts / "metrics" / "run_manifest.json"
if manifest_path.exists():
    with st.expander("Run provenance"):
        man = d.read(cfg, "metrics/run_manifest.json")
        st.write({"seed": man["seed"], "config": man["config_source"],
                  "config_fingerprint": man["config_fingerprint"],
                  "python": man["environment"]["python"],
                  "git_commit": man["environment"]["git_commit"]})
        st.dataframe(pd.DataFrame([{"stage": k, "completed (UTC)": v["completed_utc"]}
                                   for k, v in man["stages"].items()]), hide_index=True)
