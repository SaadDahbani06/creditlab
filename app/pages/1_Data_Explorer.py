"""Data Explorer: historical account characteristics from the DuckDB SQL analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from creditlab.data.sql_analysis import parse_sql_file
from creditlab.reporting import dashboard as d

cfg = d.config()
st.title("Data Explorer")
st.caption("Observed historical data. Exploratory views use the development training split "
           "(18,000 accounts); data-quality checks use all 30,000 rows.")

if d.missing(cfg, ["metrics/sql/index.json"]):
    st.error(d.SETUP)
    st.stop()


def q(name: str) -> pd.DataFrame:
    return d.read_csv(cfg, f"metrics/sql/{name}.csv")


tab1, tab2, tab3, tab4 = st.tabs(["Repayment status", "Balances & payments", "Data quality",
                                  "SQL queries"])
with tab1:
    pay0 = q("default_rate_by_pay0")
    fig = px.bar(pay0, x=pay0["status_code"].astype(str), y="default_rate",
                 hover_data=["official_meaning", "n_accounts", "n_defaults"],
                 color_discrete_sequence=[d.SERIES[0]])
    fig.update_yaxes(tickformat=".0%", title="Observed default rate")
    fig.update_xaxes(title="September 2005 status code (PAY_0)")
    st.plotly_chart(d.style(fig, title="Next-month default rate by latest repayment status"),
                    use_container_width=True)
    st.caption("Codes -2 and 0 are not documented by the source and are not interpreted. "
               "Codes 5–8 have small samples (see hover).")
    c1, c2 = st.columns(2)
    with c1:
        rd = q("repayment_delay_patterns")
        fig = px.bar(rd, x="months_delayed_of_6", y="default_rate", hover_data=["n_accounts"],
                     color_discrete_sequence=[d.SERIES[0]])
        fig.update_yaxes(tickformat=".0%", title="Observed default rate")
        fig.update_xaxes(title="Months with documented delay (of 6)")
        st.plotly_chart(d.style(fig, title="Persistence of delays"), use_container_width=True)
    with c2:
        sm = q("status_code_by_month")
        fig = go.Figure()
        for col, label, color in (("share_code_m2", "-2 (undoc.)", d.SERIES[2]),
                                  ("share_code_m1", "-1 pay duly", d.SERIES[0]),
                                  ("share_code_0", "0 (undoc.)", d.SERIES[1]),
                                  ("share_code_1", "1 one-month delay", d.SERIES[3]),
                                  ("share_code_2", "2 two-month delay", d.SERIES[4])):
            fig.add_trace(go.Scatter(x=sm["month"], y=sm[col], name=label, mode="lines+markers",
                                     line={"color": color, "width": 2}))
        fig.update_yaxes(tickformat=".0%", title="Share of accounts")
        st.plotly_chart(d.style(fig, title="Status codes by month"), use_container_width=True)
        st.caption("Code 1 appears almost only in September: the coding may differ across months.")
    tr = q("status_transition_aug_to_sep")
    st.markdown("**August → September status transitions** (groups with n ≥ 30)")
    st.dataframe(tr.style.format({"default_rate": "{:.3f}"}), hide_index=True,
                 use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        lim = q("credit_limit_deciles")
        lim["range"] = [f"{a / 1000:.0f}–{b / 1000:.0f}k" for a, b in zip(lim.min_limit, lim.max_limit, strict=True)]
        fig = px.bar(lim, x="range", y="default_rate", hover_data=["n_accounts"],
                     color_discrete_sequence=[d.SERIES[0]])
        fig.update_yaxes(tickformat=".0%", title="Observed default rate")
        fig.update_xaxes(title="Credit-limit decile (NTD)")
        st.plotly_chart(d.style(fig, title="Default rate by credit-limit decile"),
                        use_container_width=True)
    with c2:
        pb = q("payment_to_bill_ratio")
        pb["band"] = pb["pay_to_bill_band"].str.split(": ").str[1]
        fig = px.bar(pb, x="band", y="default_rate", hover_data=["n_accounts", "median_ratio"],
                     color_discrete_sequence=[d.SERIES[0]])
        fig.update_yaxes(tickformat=".0%", title="Observed default rate")
        fig.update_xaxes(title="Six-month payments / positive bills")
        st.plotly_chart(d.style(fig, title="Default rate by payment-to-bill ratio"),
                        use_container_width=True)
    rb = q("default_by_risk_bands")
    piv = rb.pivot(index="utilisation_band", columns="worst_delay_6m", values="default_rate")
    piv = piv[["no delay", "1 month", "2 months", "3+ months"]]
    cnt = rb.pivot(index="utilisation_band", columns="worst_delay_6m", values="n_accounts")[piv.columns]
    fig = go.Figure(go.Heatmap(z=piv.values, x=list(piv.columns), y=[s.split(": ")[1] for s in piv.index],
                               colorscale="Blues", zmin=0, zmax=1, customdata=cnt.values,
                               hovertemplate="utilisation %{y}<br>worst delay %{x}<br>"
                                             "default rate %{z:.1%}<br>n = %{customdata}<extra></extra>",
                               colorbar={"tickformat": ".0%"}))
    fig.update_xaxes(title="Worst delay in six months")
    fig.update_yaxes(title="September utilisation")
    st.plotly_chart(d.style(fig, 360, "Default rate by utilisation and worst delay"),
                    use_container_width=True)
    st.markdown("**Bill amount distribution (NTD)**")
    st.dataframe(q("bill_amount_distribution"), hide_index=True, use_container_width=True)

with tab3:
    st.dataframe(q("data_quality_checks"), hide_index=True, use_container_width=True)
    st.markdown(
        "**Limitations visible in the data**\n"
        "- Existing cardholders only, not applicants; credit limits reflect past lending decisions.\n"
        "- One observation window and one outcome month: no chronological out-of-time test.\n"
        "- Undocumented status codes (-2, 0) and education/marriage codes; kept as recorded.\n"
        "- No LGD, EAD, revenue, recovery or macroeconomic data.")
    st.markdown("**Audit-only aggregate view of sensitive attributes** (training split; never "
                "model inputs or policy rules)")
    st.dataframe(q("demographic_audit").style.format({"default_rate": "{:.3f}"}),
                 hide_index=True, use_container_width=True)

with tab4:
    queries = parse_sql_file(cfg.path("sql_file"))
    name = st.selectbox("Query", [x.name for x in queries])
    qq = next(x for x in queries if x.name == name)
    st.caption(qq.description)
    st.code(qq.sql, language="sql")
    st.dataframe(q(name), hide_index=True, use_container_width=True)
    st.caption("Results were produced by `creditlab prepare` (DuckDB) and are read from "
               "artifacts/metrics/sql/.")
