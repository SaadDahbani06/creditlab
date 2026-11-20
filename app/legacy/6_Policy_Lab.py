"""Policy Lab: hypothetical selection strategies for an existing-account research cohort."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from creditlab.decisions.compare import _cohort, evaluate_policies
from creditlab.decisions.policies import PolicyEconomics
from creditlab.reporting import dashboard as d
from creditlab.utils.config import Config

cfg = d.config()
st.title("Policy Lab")
st.error("**Not a lending engine.** These are hypothetical selection rules applied to *existing* "
         "2005 cardholders. The data contain no approval decisions, rejected applicants, revenue, "
         "LGD or EAD. No result here is a proven profit improvement or a causal effect, and "
         "nothing here is a decision or reason about any person.")

if d.missing(cfg, ["predictions/validation_predictions.parquet",
                   "predictions/test_predictions.parquet", "metrics/selection.json"]):
    st.error(d.SETUP)
    st.stop()

primary = d.read(cfg, "metrics/selection.json")["primary_model"]


@st.cache_data(show_spinner=False)
def cohorts():
    return _cohort(cfg, "validation", primary), _cohort(cfg, "test", primary)


with st.sidebar:
    st.header("Assumed one-month economics")
    r = st.slider("Monthly net revenue rate on drawn balance", 0.0, 0.30,
                  float(cfg["policy"]["monthly_revenue_rate"]), 0.005, format="%.3f")
    cost = st.number_input("Fixed monthly cost per account (NTD)", 0.0, 2_000.0,
                           float(cfg["policy"]["monthly_cost_per_account"]), 10.0)
    lgd = st.slider("LGD", 0.05, 1.0, float(cfg["policy"]["lgd"]), 0.05)
    ccf = st.slider("CCF on undrawn limit", 0.0, 1.0, float(cfg["risk"]["ead"]["ccf"]), 0.05)
    st.header("Policy B (chosen on validation)")
    target = st.slider("Risk appetite: max observed bad rate of retained (validation)",
                       0.04, 0.22, float(cfg["policy"]["b_target_bad_rate"]), 0.01)


@st.cache_data(show_spinner=False, max_entries=16)
def compute(r, cost, lgd, ccf, target):
    raw = {**cfg.raw, "policy": {**cfg.raw["policy"], "b_target_bad_rate": target,
                                 "random_baseline_draws": 200}}
    c2 = Config(raw=raw, source=cfg.source)
    econ = PolicyEconomics.from_config(c2, monthly_revenue_rate=r, monthly_cost_per_account=cost)
    econ = PolicyEconomics(r, cost, econ.risk.with_(lgd=lgd, ccf=ccf))
    val, test = cohorts()
    res = evaluate_policies(c2, econ, val, test, with_tail=False)
    return res["table"], res["threshold"]


table, thr = compute(r, cost, lgd, ccf, target)
st.markdown(f"Policy B cutoff **PD < {thr['threshold']:.3f}** was selected on the validation split "
            f"(observed bad rate there {thr['validation_bad_rate']:.3f}, retaining "
            f"{thr['validation_share_retained']:.1%}). It is then applied unchanged to the test cohort.")

labels = {"A_retain_all": "A all", "B_pd_threshold": "B cutoff",
          "C_expected_contribution": "C E[c]>0", "D_random_equal_size": "D random"}
cols = st.columns(3)
for col, key, title, fmt in (
    (cols[0], "share_selected", "Share retained", ".0%"),
    (cols[1], "observed_default_rate", "OBSERVED default rate of retained", ".1%"),
    (cols[2], "expected_net_contribution", "Exp. net contribution, NTD M (hyp.)", ",.1f"),
):
    y = table[key] / (1e6 if key == "expected_net_contribution" else 1)
    fig = go.Figure(go.Bar(x=[labels[i] for i in table.index], y=y,
                           marker_color=[d.ENTITY_COLORS[i] for i in table.index],
                           hovertemplate="%{x}<br>%{y:" + fmt + "}<extra></extra>"))
    fig.update_yaxes(tickformat=fmt if "%" in fmt else None)
    col.plotly_chart(d.style(fig, 330, title), use_container_width=True)

show = table[["description", "share_selected", "mean_pd", "observed_default_rate", "total_ead",
              "total_expected_loss", "total_expected_revenue", "expected_net_contribution",
              "realized_hypothetical_contribution", "share_el_top10pct_accounts"]].copy()
money = ["total_ead", "total_expected_loss", "total_expected_revenue", "expected_net_contribution",
         "realized_hypothetical_contribution"]
show[money] = show[money].astype(float) / 1e6
st.dataframe(show.style.format(precision=3, na_rep="—"), use_container_width=True)
st.caption("A: retain all · B: PD below validation-selected cutoff · C: expected one-month "
           "contribution > 0 · D: random cohort of B's size.")
st.caption("Money columns: NTD millions, one month, hypothetical. `mean_pd` = model prediction. "
           "`observed_default_rate` = observed test labels. `realized_hypothetical_contribution` "
           "applies the ASSUMED economics to OBSERVED outcomes. Policy D is the mean of 200 random draws.")

st.markdown(
    "**Why policy C often retains nobody.** Over one month an account earns at most "
    "r × balance, but a default costs LGD × EAD at once. With a 22% one-month default rate in "
    "this sample, the break-even revenue rate for most accounts is far above plausible card "
    "margins. Raise the revenue slider to see where C starts retaining accounts. The point is "
    "the mechanics, not a recommended price.")

with st.expander("Ethics and limitations"):
    st.markdown(d.doc("LIMITATIONS.md"))
