"""Stress Testing: interactive, reproducible Monte Carlo with user-chosen assumptions."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from creditlab.reporting import dashboard as d
from creditlab.risk.assumptions import RiskAssumptions, exposure_at_default
from creditlab.risk.simulation import simulate_portfolio
from creditlab.risk.stress import load_portfolio, stress_pd
from creditlab.risk.var_es import loss_summary

cfg = d.config()
st.title("Stress Testing")
st.warning(d.HYPOTHETICAL_BANNER + " Stress multipliers are assumed scenarios, not estimated "
           "macroeconomic effects.")

if d.missing(cfg, ["predictions/test_predictions.parquet", "metrics/selection.json"]):
    st.error(d.SETUP)
    st.stop()


@st.cache_data(show_spinner=False)
def portfolio() -> pd.DataFrame:
    p = load_portfolio(cfg)
    return p[["ID", "LIMIT_BAL", "BILL_AMT1", "pd", "observed_default"]]


@st.cache_data(show_spinner=False, max_entries=16)
def run(lgd, method, ccf, frac, cap, m, rho, n_sims, seed):
    port = portfolio()
    a = RiskAssumptions(lgd=lgd, ead_method=method, ccf=ccf, limit_fraction=frac,
                        ead_cap=cap or None)
    ead = exposure_at_default(port.LIMIT_BAL, port.BILL_AMT1, a)
    p = stress_pd(port["pd"].to_numpy(), m)
    t0 = time.perf_counter()
    res = simulate_portfolio(p, lgd * ead, n_sims=n_sims, rho=rho, seed=seed,
                             chunk_size=cfg["simulation"]["chunk_size"])
    el = float(np.sum(p * lgd * ead))
    s = loss_summary(res.losses, (0.95, 0.99), {"1.25x_EL": 1.25 * el})
    return res.losses, s, el, float(ead.sum()), float(p.mean()), time.perf_counter() - t0


with st.sidebar:
    st.header("Assumptions")
    lgd = st.slider("LGD (loss given default)", 0.05, 1.0, float(cfg["risk"]["lgd_base"]), 0.05)
    method = st.radio("EAD method", ["balance_plus_ccf", "limit_fraction"],
                      format_func=lambda s: {"balance_plus_ccf": "drawn + CCF × undrawn",
                                             "limit_fraction": "fraction of limit"}[s])
    ccf = st.slider("CCF on undrawn limit", 0.0, 1.0, float(cfg["risk"]["ead"]["ccf"]), 0.05,
                    disabled=method != "balance_plus_ccf")
    frac = st.slider("EAD as fraction of limit", 0.05, 1.0,
                     float(cfg["risk"]["ead"]["limit_fraction"]), 0.05,
                     disabled=method != "limit_fraction")
    cap = st.number_input("EAD cap per account (NTD, 0 = none)", 0, 2_000_000, 0, 10_000)
    st.header("Scenario")
    m = st.slider("Default-odds multiplier", 0.5, 4.0, 1.0, 0.1,
                  help="PD' = m·PD / (1 − PD + m·PD), i.e. logit shift of log(m)")
    rho = st.slider("Common-factor correlation ρ", 0.0, 0.5, 0.0, 0.01)
    n_sims = st.select_slider("Simulation paths", [1_000, 2_000, 5_000, 10_000, 20_000],
                              value=5_000)
    n_sims = min(n_sims, cfg["dashboard"]["max_interactive_sims"])
    seed = st.number_input("Random seed", 0, 2**31 - 1, cfg.sub_seed("simulation"))

losses, s, el, ead, mean_pd, secs = run(lgd, method, ccf, frac, cap, m, rho, n_sims, int(seed))
base_losses, base_s, base_el, *_ = run(float(cfg["risk"]["lgd_base"]), "balance_plus_ccf",
                                       float(cfg["risk"]["ead"]["ccf"]), 0.5, 0, 1.0, 0.0,
                                       n_sims, int(seed))

c = st.columns(6)
c[0].metric("Mean stressed PD", f"{mean_pd:.3f}")
c[1].metric("Hypothetical EAD", f"{ead / 1e6:,.0f}M")
c[2].metric("Analytical EL", f"{el / 1e6:,.1f}M", f"{(el / base_el - 1):+.0%} vs base")
c[3].metric("99% VaR", f"{s['var_99'] / 1e6:,.1f}M", f"{(s['var_99'] / base_s['var_99'] - 1):+.0%}")
c[4].metric("99% ES", f"{s['es_99'] / 1e6:,.1f}M", f"{(s['es_99'] / base_s['es_99'] - 1):+.0%}")
c[5].metric("P(loss > 1.25 × EL)", f"{s['p_exceed_1.25x_EL']:.3f}")
st.caption(f"{n_sims:,} paths, seed {int(seed)}, simulated in {secs:.2f}s (cached). Same seed and "
           "settings reproduce identical results. Deltas compare with the configured base case "
           "at the same seed and path count.")

fig = go.Figure()
fig.add_trace(go.Histogram(x=base_losses / 1e6, name="configured base case", nbinsx=80,
                           marker_color=d.NEUTRAL, opacity=0.5))
fig.add_trace(go.Histogram(x=losses / 1e6, name="your scenario", nbinsx=80,
                           marker_color=d.SERIES[0], opacity=0.65))
fig.update_layout(barmode="overlay")
fig.add_vline(x=s["var_99"] / 1e6, line_dash="dash", line_color="#52514e",
              annotation_text="your 99% VaR")
fig.update_xaxes(title="Simulated one-month loss (NTD millions)")
fig.update_yaxes(title="Paths")
st.plotly_chart(d.style(fig, 420), use_container_width=True)

st.dataframe(pd.DataFrame([{"statistic": k, "value": v} for k, v in s.items()]),
             hide_index=True, use_container_width=True)
st.markdown(
    "**Reading the result.** With ρ = 0, defaults are independent and a 6,000-account portfolio "
    "diversifies most idiosyncratic risk, so VaR sits close to EL. A common factor (ρ > 0) leaves "
    "the mean unchanged but makes many accounts default together, which fattens the right tail. "
    "It also *lowers* the median, because most paths are now quieter than average.")
