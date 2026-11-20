"""Portfolio risk & research findings: read-only summary of the simulation and the extension."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from creditlab.reporting import dashboard as d

cfg = d.config()
need = ["metrics/risk/scenario_summary.csv", "metrics/risk/scenario_losses.npz",
        "metrics/risk/expected_loss_summary.json", "metrics/selection.json"]
if d.missing(cfg, need):
    st.title("Portfolio Risk & Findings")
    st.error(d.SETUP)
    st.stop()

scen = d.read_csv(cfg, "metrics/risk/scenario_summary.csv").set_index("scenario")
el = d.read(cfg, "metrics/risk/expected_loss_summary.json")
sel = d.read(cfg, "metrics/selection.json")
ind, cor = scen.loc["base"], scen.loc["base_correlated"]
rho_c = float(cor["rho"])

d.page_header(
    "Portfolio risk & research findings",
    "From default probabilities to tail risk",
    f"The model's PDs for the {el['n_accounts']:,} test accounts are combined with assumed "
    "loss-given-default and exposure to simulate a distribution of one-month portfolio losses. "
    "The research extension then asks how much those tail-risk numbers move because the PDs are "
    "themselves estimated from finite data.")
d.note("<b>Hypothetical money.</b> Only the probabilities of default are estimated from data. "
       f"Loss-given-default ({el['assumptions']['lgd']:.0%}), exposure (drawn balance + "
       f"{el['assumptions']['ccf']:.0%} of the undrawn limit) and correlation are assumptions. "
       "The dataset's default label is a missed payment, not a write-off, so these figures are "
       "upper-bound illustrations, not observed bank losses.")

# ----------------------------------------------------------------------------- loss distribution
st.header("Correlation fattens the tail, not the mean", anchor=False)
losses = np.load(cfg.artifacts / "metrics" / "risk" / "scenario_losses.npz")
fig = go.Figure()
for name, row, color, label in (("base", ind, d.BLUE, "Independent defaults"),
                                ("base_correlated", cor, d.RED, f"Correlated, ρ = {rho_c:g}")):
    x = losses[name] / 1e6
    fig.add_trace(go.Histogram(x=x, name=label, marker_color=color, opacity=0.75,
                               xbins={"size": 1.0}, histnorm="probability",
                               hovertemplate="loss %{x:.0f}M<br>share of paths %{y:.3f}<extra>"
                               + label + "</extra>"))
    fig.add_vline(x=row["var_99"] / 1e6, line_color=color, line_dash="dash", line_width=1.5)
fig.add_vline(x=ind["analytical_el"] / 1e6, line_color=d.INK, line_width=1)
fig.add_annotation(x=ind["analytical_el"] / 1e6, y=1, yref="paper", text="Expected loss",
                   showarrow=False, xanchor="right", xshift=-4, font={"size": 11, "color": d.INK})
fig.add_annotation(x=cor["var_99"] / 1e6, y=0.92, yref="paper", text="99% VaR, correlated",
                   showarrow=False, xanchor="left", xshift=4, font={"size": 11, "color": d.RED})
fig.update_layout(barmode="overlay")
fig.update_xaxes(title="One-month loss, NTD millions (hypothetical)")
fig.update_yaxes(title="Share of simulated paths", tickformat=".0%")
d.chart(fig, f"Loss distribution, {int(ind['n_sims']):,} simulated months", 400)

d.table(["Fixed PDs (original model)", "Expected loss", "95% VaR", "99% VaR", "99% ES"], [
    ["Independent defaults", d.m(ind["analytical_el"]), d.m(ind["var_95"]), d.m(ind["var_99"]),
     d.m(ind["es_99"])],
    [f"Correlated, ρ = {rho_c:g}", d.m(cor["analytical_el"]), d.m(cor["var_95"]),
     d.m(cor["var_99"]), d.m(cor["es_99"])],
], highlight={1})
st.markdown(
    f"With independent defaults, {el['n_accounts']:,} accounts diversify each other: the 99% VaR "
    f"sits only {d.m(ind['var_99'] - ind['analytical_el'])} above expected loss. A common factor "
    "leaves every account's default probability unchanged, so expected loss stays the same, but it "
    "makes defaults arrive together. In a bad month many accounts default at once, so the right "
    f"tail stretches: the 99% VaR rises by {d.m(cor['var_99'] - ind['var_99'])} "
    f"({100 * (cor['var_99'] / ind['var_99'] - 1):.0f}%) and Expected Shortfall, the average loss "
    f"in the worst 1% of months, reaches {d.m(cor['es_99'])}. The median falls slightly, because "
    "most months are quieter than average.")

# ----------------------------------------------------------------------------- extension
ext_dir = cfg.artifacts / "metrics" / "uncertainty"
if not (ext_dir / "summary.json").exists():
    st.info("Run `creditlab uncertainty` to add the PD-estimation-uncertainty findings.")
    d.footer(cfg)
    st.stop()

s = d.read(cfg, "metrics/uncertainty/summary.json")
fam = sel["primary_family"]
F = s["families"][fam]
p = F["pd"]
r0, rc = F["risk"]["rho_0.0"], F["risk"][f"rho_{rho_c}"]
reg = d.read_csv(cfg, "metrics/uncertainty/regime_comparison.csv")
reg = reg[reg.family == fam]

st.header("Research question: how reliable is a reported VaR?", anchor=False)
st.markdown(
    "The numbers above treat each account's PD as exact. In reality the PDs come from a model "
    "fitted to 18,000 historical accounts, and a different sample of 18,000 would have produced a "
    f"different model. To measure this, the frozen model specification was refitted on "
    f"**{s['n_bootstrap']} bootstrap resamples** of the training data (hyperparameters, "
    "preprocessing and features unchanged; no validation or test rows used). Each refit gives a "
    "new set of PDs for the same test portfolio, and every set is run through the same Monte Carlo "
    "engine with identical random draws, so the differences come from the model alone.")

d.stats([
    (f"{p['median_sd']:.3f}", "Median SD of an account's PD",
     f"{100 * p['median_relative_sd']:.0f}% of the PD itself; 90% of accounts below "
     f"{p['p90_sd']:.3f}"),
    (f"{p['mean_pd_bootstrap_sd']:.4f}", "SD of the portfolio-average PD",
     f"around a mean PD of {p['mean_pd_original']:.3f}; most account-level noise cancels"),
    (d.m(r0["across_models"]["var_99"]["p97_5"] - r0["across_models"]["var_99"]["p2_5"]),
     "Width of 95% range of 99% VaR",
     f"independent defaults; {d.m(rc['across_models']['var_99']['p97_5'] - rc['across_models']['var_99']['p2_5'])}"
     f" at ρ = {rho_c:g}"),
])

fig = go.Figure()
labels, y = [], 0
for rho in (0.0, rho_c):
    for kind, color in (("fixed", d.INK_2), ("uncertain", d.RED)):
        row = reg[(reg.rho == rho) & (reg.pd == kind)].iloc[0]
        lab = (("Fixed PDs" if kind == "fixed" else "With PD uncertainty")
               + (", independent" if rho == 0 else f", ρ = {rho:g}"))
        labels.append(lab)
        if kind == "uncertain":
            fig.add_trace(go.Scatter(x=[row.var_99_p2_5 / 1e6, row.var_99_p97_5 / 1e6], y=[lab, lab],
                                     mode="lines", line={"color": color, "width": 6},
                                     opacity=0.35, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[row.var_99 / 1e6], y=[lab], mode="markers", showlegend=False,
            marker={"color": color, "size": 11, "line": {"color": "white", "width": 2}},
            hovertemplate=f"{lab}<br>99% VaR %{{x:.1f}}M"
            + (f"<br>95% range across models {row.var_99_p2_5 / 1e6:.1f}–{row.var_99_p97_5 / 1e6:.1f}M"
               if kind == "uncertain" else "") + "<extra></extra>"))
        y += 1
fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], showgrid=False)
fig.update_xaxes(title="99% Value-at-Risk, NTD millions (hypothetical)")
c1, c2 = st.columns([3, 2], gap="large")
d.chart(fig, "99% VaR under four modelling assumptions", 300, container=c1)
c2.markdown(
    "Gray: the original fixed-PD estimate. Red dot: the VaR of the mixture distribution, which "
    "averages over all refitted models. Red band: the 95% range of the VaR reported by the "
    "individual refitted models, i.e. how far a single analyst's number could plausibly sit "
    "from another's with a different training sample.")

dc0, dcc = r0["decomposition"], rc["decomposition"]
mech_path = ext_dir / "mechanism.json"
mech = d.read(cfg, "metrics/uncertainty/mechanism.json") if mech_path.exists() else None
nf_path = ext_dir / "noise_free_comparison.json"
nf = d.read(cfg, "metrics/uncertainty/noise_free_comparison.json") if nf_path.exists() else None
a0, ac = r0["across_models"]["var_99"], rc["across_models"]["var_99"]
st.subheader("What the experiment shows", anchor=False)
st.markdown(
    f"1. **Individual PDs are noisy, the portfolio average much less so.** A typical account's PD "
    f"moves by about {p['median_sd']:.3f} between refits, but only "
    f"{100 * p['pd_variance_common_share']:.0f}% of that variation is a shift shared by all "
    "accounts. Account-specific noise largely cancels in a 6,000-account sum; the shared shift "
    "does not, and it moves expected loss and VaR together."
    + (f" About {100 * mech[fam]['r2_mean_pd_on_train_default_rate']:.0f}% of that shared shift is "
       "explained simply by how many defaults landed in each resampled training set." if mech else "")
    + "\n"
    f"2. **Under independence, estimation uncertainty is of the same order as the tail itself.** "
    f"The 99% VaR from different refits ranges over {d.m(a0['p2_5'])}–{d.m(a0['p97_5'])}, while "
    f"the gap between VaR and expected loss is only {d.m(ind['var_99'] - ind['analytical_el'])}. "
    f"PD estimation explains {100 * dc0['share_between']:.0f}% of total loss variance in this "
    "regime, so a fixed-PD VaR understates the real uncertainty.\n"
    f"3. **With correlation, the common factor dominates.** At ρ = {rho_c:g} the spread across "
    f"refits is similar in money ({d.m(ac['p2_5'])}–{d.m(ac['p97_5'])}), but it is small "
    f"next to the tail created by correlation: PD estimation explains "
    f"{100 * dcc['share_between']:.1f}% of loss variance.\n"
    + (f"4. **Folding PD uncertainty into one number barely changes it.** Computed without "
       f"simulation noise, the uncertainty-adjusted 99% VaR differs from the fixed-PD value by "
       f"{100 * nf['rho_0.0']['total_change_var_99'] / nf['rho_0.0']['fixed']['var_99']:+.1f}% "
       f"(independent) and "
       f"{100 * nf[f'rho_{rho_c}']['total_change_var_99'] / nf[f'rho_{rho_c}']['fixed']['var_99']:+.1f}% "
       f"(ρ = {rho_c:g}). The far larger uncertainty is the correlation assumption itself, which "
       "the data cannot estimate." if nf else ""))

g = d.read_csv(cfg, "metrics/uncertainty/rho_grid.csv")
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.r_[g.rho, g.rho[::-1]], y=np.r_[g.var_99_p97_5, g.var_99_p2_5[::-1]] / 1e6,
                         mode="lines", fill="toself", fillcolor="rgba(200,16,46,0.14)",
                         line={"width": 0},
                         name="95% range across refitted models", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=g.rho, y=g.fixed_var_99 / 1e6, mode="lines+markers", name="Fixed PDs",
                         line={"color": d.INK_2, "width": 2}, marker={"size": 6},
                         hovertemplate="ρ %{x}<br>VaR99 %{y:.1f}M<extra>fixed PDs</extra>"))
fig.add_trace(go.Scatter(x=g.rho, y=g.mixture_var_99 / 1e6, mode="lines+markers",
                         name="With PD uncertainty", line={"color": d.RED, "width": 2},
                         marker={"size": 6},
                         hovertemplate="ρ %{x}<br>VaR99 %{y:.1f}M<extra>mixture</extra>"))
fig.update_xaxes(title="Assumed asset correlation ρ")
fig.update_yaxes(title="99% VaR, NTD millions")
c1, c2 = st.columns([3, 2], gap="large")
d.chart(fig, "The correlation assumption outweighs estimation error", 360, container=c1)
c2.markdown(
    f"Moving ρ from 0 to {g.rho.max():g} raises the 99% VaR from "
    f"{d.m(g.fixed_var_99.iloc[0])} to {d.m(g.fixed_var_99.iloc[-1])}. The band from PD "
    f"estimation stays roughly {d.m((g.var_99_p97_5 - g.var_99_p2_5).mean())} wide throughout. A "
    "single outcome month cannot pin down ρ, so for this portfolio the correlation assumption, "
    "not the PD model, is the dominant source of tail-risk uncertainty.")

with st.expander("Diagnostics and robustness checks"):
    noise = {n["metric"]: n for n in rc["mc_noise"]}["var_99"]
    seed = F.get("seed_only_pd", {})
    rt = F.get("retuned_pd", {})
    o = s["observed_vs_expected_defaults"]
    rows = [
        ["Refit on original data reproduces the frozen PDs",
         f"max difference {p['refit_reproduces_saved_predictions_max_abs_diff']:.0e}"],
        [f"Monte Carlo noise in the across-model VaR spread (ρ = {rho_c:g})",
         f"{d.m(noise['sd_model_specific_mc_noise'], 2)} vs spread {d.m(noise['sd_across_models_semi_analytic'], 2)}"],
        ["XGBoost seed-only variation (median PD SD)",
         f"{seed['median_sd']:.4f}" if seed else "—"],
        ["Re-tuning hyperparameters in each resample (median PD SD)",
         f"{rt['median_sd']:.4f} vs {rt['paired_fixed_hyperparameters']['median_sd']:.4f} fixed"
         if rt else "—"],
        ["Expected test defaults, 95% range across refits",
         f"{o['expected_bootstrap_p2_5']:,.0f}–{o['expected_bootstrap_p97_5']:,.0f} "
         f"(observed {o['observed_test_defaults']:,})"],
    ]
    d.table(["Check", "Result"], rows)
    st.caption("Monte Carlo noise is measured against a noise-free semi-analytical calculation of "
               "the same one-factor model. All refits use the training split only. This is a "
               "post-hoc analysis on the same data as the original study, not an independent "
               "validation.")
d.footer(cfg)
