"""Overview: what CreditLab did and what it found, in plain English."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from creditlab.reporting import dashboard as d

cfg = d.config()
need = ["metrics/data_validation.json", "metrics/selection.json", "metrics/test_evaluation.json",
        "metrics/test_bootstrap_ci.csv", "metrics/risk/scenario_summary.csv",
        "metrics/risk/expected_loss_summary.json"]
if d.missing(cfg, need):
    st.title("CreditLab")
    st.error(d.SETUP)
    st.stop()

val = d.read(cfg, "metrics/data_validation.json")
sel = d.read(cfg, "metrics/selection.json")
ev = d.read(cfg, "metrics/test_evaluation.json")
boot = d.read_csv(cfg, "metrics/test_bootstrap_ci.csv")
scen = d.read_csv(cfg, "metrics/risk/scenario_summary.csv").set_index("scenario")
el = d.read(cfg, "metrics/risk/expected_loss_summary.json")
primary = sel["primary_model"]
pm = ev["test_metrics"][primary]
ci = boot[(boot.model == primary) & (boot.metric == "roc_auc")].iloc[0]
logit_auc = ev["test_metrics"][sel["families"]["logistic"]["model"]]["roc_auc"]

d.page_header(
    "Research project · credit risk",
    "CreditLab",
    "How well can six months of repayment history predict next-month credit-card default, and "
    "how much should we trust the portfolio tail-risk numbers built on those predictions? "
    f"An end-to-end study of {val['rows']:,} Taiwanese cardholders (UCI, 2005).")

ind, cor = scen.loc["base"], scen.loc["base_correlated"]
stats = [
    (f"{pm['roc_auc']:.3f}", "Test ROC-AUC, XGBoost",
     f"95% CI {ci.ci_low:.3f}–{ci.ci_high:.3f}; logistic regression {logit_auc:.3f}"),
    (f"{el['expected_defaults']:,.0f}", "Predicted vs observed defaults",
     f"{el['observed_defaults_in_portfolio']:,} observed in the 6,000-account test portfolio"),
    (d.m(ind["var_99"]), "99% VaR, independent defaults",
     f"rises to {d.m(cor['var_99'])} with asset correlation ρ = {cor['rho']:g} (hypothetical)"),
]
ext_path = cfg.artifacts / "metrics" / "uncertainty" / "summary.json"
ext = d.read(cfg, "metrics/uncertainty/summary.json") if ext_path.exists() else None
if ext:
    fam = sel["primary_family"]
    r0 = ext["families"][fam]["risk"]
    rho_c = max(float(k.removeprefix("rho_")) for k in r0)
    rc = r0[f"rho_{rho_c}"]
    a = rc["across_models"]["var_99"]
    stats.append((f"±{(a['p97_5'] - a['p2_5']) / 2 / 1e6:.1f}M", "VaR uncertainty from PD estimation",
                  f"half-width of the 95% range of 99% VaR across {ext['n_bootstrap']} refitted "
                  f"models at ρ = {rho_c:g}"))
d.stats(stats)
st.caption("All money figures are hypothetical one-month losses in NTD under assumed loss-given-"
           "default and exposure; they are not observed bank losses.")

st.header("What the project does", anchor=False)
d.steps([
    ("Predict default", "Logistic regression and gradient-boosted trees (XGBoost) estimate each "
     "account's probability of default (PD). Every modelling choice was made on a validation "
     "split; the test split was scored afterwards."),
    ("Check the probabilities", "Calibration analysis tests whether a predicted 20% really means "
     "about 20% of such accounts default, because the PDs feed a loss model, not just a ranking."),
    ("Simulate portfolio losses", "A one-factor Gaussian default model turns PDs into a "
     "distribution of hypothetical losses, with Value-at-Risk and Expected Shortfall."),
    ("Stress the estimate itself", "The research extension refits the model on bootstrap "
     "resamples to measure how much the tail-risk numbers move because the PDs are estimated "
     "from finite data."),
])

st.header("Main findings", anchor=False)
findings = (
    f"- **Moderate, well-calibrated prediction.** XGBoost reached test ROC-AUC {pm['roc_auc']:.3f}, "
    f"a small but statistically supported gain over logistic regression "
    f"({pm['roc_auc'] - logit_auc:+.3f}). Its probabilities were already calibrated, so neither "
    "Platt nor isotonic recalibration was adopted.\n"
    f"- **Correlation, not the PD model, drives the tail.** With independent defaults the 99% VaR "
    f"is only {100 * (ind['var_99'] / ind['analytical_el'] - 1):.0f}% above expected loss; a small "
    f"common-factor correlation (ρ = {cor['rho']:g}) raises it by "
    f"{100 * (cor['var_99'] / ind['var_99'] - 1):.0f}% while leaving expected loss unchanged.\n")
if ext:
    fam = sel["primary_family"]
    r_ind = ext["families"][fam]["risk"]["rho_0.0"]
    findings += (
        f"- **PD estimation uncertainty matters most where diversification is strongest.** "
        f"Across {ext['n_bootstrap']} refitted models, estimation uncertainty accounts for "
        f"{d.pct(r_ind['decomposition']['share_between'], 0)} of total loss variance when defaults "
        f"are independent, but only {d.pct(rc['decomposition']['share_between'], 1)} at "
        f"ρ = {rho_c:g}, where the common factor dominates. Details on the "
        "*Portfolio Risk & Findings* page.")
st.markdown(findings)
d.note("CreditLab is a research project on public data from 2005. It is not an underwriting "
       "system and makes no decision about any person. The test split is not completely untouched: "
       "test metrics were displayed once before a disclosed, non-material rule change (see "
       "Methodology). No genuinely independent validation data exists.")

st.header("Data", anchor=False)
split = pd.DataFrame(d.read(cfg, "metrics/split_summary.json"))
d.table(["Split", "Accounts", "Observed defaults", "Default rate"],
        [[r.split.capitalize(), f"{r.n:,}", f"{r.defaults:,}", d.pct(r.default_rate)]
         for r in split.itertuples()])
st.caption("UCI Machine Learning Repository dataset 350 (Yeh, 2009; CC BY 4.0). Stratified 60/20/20 "
           "client-level split. This is a split of clients from one period, not a chronological "
           "backtest. Sex, age, marital status and education are excluded from the models.")
d.footer(cfg)
