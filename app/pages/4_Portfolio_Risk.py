"""Portfolio Risk: hypothetical expected loss, VaR and Expected Shortfall."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from creditlab.reporting import dashboard as d

cfg = d.config()
st.title("Portfolio Risk")
st.warning(d.HYPOTHETICAL_BANNER)

need = ["metrics/risk/expected_loss_summary.json", "metrics/risk/scenario_summary.csv",
        "metrics/risk/scenario_losses.npz", "metrics/risk/risk_bands.csv"]
if d.missing(cfg, need):
    st.error(d.SETUP)
    st.stop()

el = d.read(cfg, "metrics/risk/expected_loss_summary.json")
scen = d.read_csv(cfg, "metrics/risk/scenario_summary.csv").set_index("scenario")
base = scen.loc["base"]


def m(v: float) -> str:
    return f"NTD {v / 1e6:,.1f}M"


st.markdown(f"Portfolio: **test split, {el['n_accounts']:,} accounts**, PDs from "
            f"**{el['model']}** (out-of-sample). EAD: `{el['assumptions']['ead_formula']}` with "
            f"CCF {el['assumptions']['ccf']}; LGD {el['assumptions']['lgd']:.0%}.")
c = st.columns(5)
c[0].metric("Hypothetical EAD", m(el["total_ead"]))
c[1].metric("Expected loss (Σ PD·LGD·EAD)", m(el["total_el"]))
c[2].metric("EAD-weighted PD", f"{el['ead_weighted_pd']:.3f}")
c[3].metric("99% VaR (independent)", m(base["var_99"]))
c[4].metric("99% ES (independent)", m(base["es_99"]))

losses = np.load(cfg.artifacts / "metrics" / "risk" / "scenario_losses.npz")
name = st.selectbox("Scenario", list(scen.index), index=0)
x = losses[name]
s = scen.loc[name]
fig = go.Figure(go.Histogram(x=x / 1e6, nbinsx=90, marker_color=d.ENTITY_COLORS.get(name, d.SERIES[0]),
                             hovertemplate="loss %{x:.1f}M<br>paths %{y}<extra></extra>"))
# VaR and ES are often close together: label VaR to the left of its line, ES to the right.
for val, label, dash, pos in ((s["analytical_el"], "EL", "solid", "top"),
                              (s["var_99"], "VaR 99%", "dash", "top left"),
                              (s["es_99"], "ES 99%", "dot", "top right")):
    fig.add_vline(x=val / 1e6, line_dash=dash, line_color="#52514e",
                  annotation_text=f"{label} {val / 1e6:.1f}M", annotation_position=pos)
fig.update_xaxes(title="Simulated one-month loss (NTD millions)")
fig.update_yaxes(title="Paths")
st.plotly_chart(d.style(fig, 400, f"Loss distribution: {name} ({int(s['n_sims']):,} paths)"),
                use_container_width=True)
st.caption(s["description"])

st.subheader("Scenario summary")
show = scen[["odds_multiplier", "lgd", "rho", "mean_stressed_pd", "analytical_el", "mean",
             "median", "std", "var_95", "var_99", "es_95", "es_99"]].copy()
money = ["analytical_el", "mean", "median", "std", "var_95", "var_99", "es_95", "es_99"]
show[money] = show[money] / 1e6
st.dataframe(show.style.format(precision=2).format("{:.4f}", subset=["mean_stressed_pd"]),
             use_container_width=True)
st.caption("Money columns in NTD millions. All scenarios use the same random numbers.")

st.subheader("Concentration by PD band")
bands = d.read_csv(cfg, "metrics/risk/risk_bands.csv")
fig = go.Figure()
fig.add_trace(go.Bar(x=bands.band, y=bands.share_of_accounts, name="share of accounts",
                     marker_color=d.SERIES[1]))
fig.add_trace(go.Bar(x=bands.band, y=bands.share_of_el, name="share of expected loss",
                     marker_color=d.SERIES[0]))
fig.update_yaxes(tickformat=".0%")
fig.update_xaxes(title="Predicted PD band")
st.plotly_chart(d.style(fig, 340), use_container_width=True)
st.dataframe(bands.style.format(precision=3), hide_index=True, use_container_width=True)
st.caption("`observed_default_rate` is OBSERVED (test labels); `mean_pd` is model-predicted; "
           "EAD and EL are hypothetical.")
