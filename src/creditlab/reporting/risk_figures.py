"""Figures for the hypothetical risk engine and Monte Carlo results (from saved artifacts)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.reporting.style import (
    ENTITY_COLORS,
    NEUTRAL,
    SERIES,
    TEXT_SECONDARY,
    apply_style,
    caption,
    plt,
    save,
)
from creditlab.utils.config import Config, read_json

HYP = "HYPOTHETICAL: assumed LGD/EAD; one-month horizon; NTD; test-split portfolio (n = 6,000)."


def _m(x, _=None):
    return f"{x / 1e6:.0f}M" if abs(x) >= 1e8 else f"{x / 1e6:.1f}M"


def make_risk_figures(cfg: Config) -> None:
    apply_style()
    figs = cfg.artifact_dir("figures")
    rdir = cfg.artifact_dir("metrics") / "risk"
    scen = pd.read_csv(rdir / "scenario_summary.csv").set_index("scenario")
    losses = dict(np.load(rdir / "scenario_losses.npz"))
    el = read_json(rdir / "expected_loss_summary.json")

    # Base loss distribution with EL, VaR, ES
    b = losses["base"]
    s = scen.loc["base"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(b, bins=80, color=SERIES[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    top = ax.get_ylim()[1]
    for (val, label, ls), h in zip(((s["analytical_el"], "analytical EL", "-"),
                                    (s["var_99"], "99% VaR", "--"), (s["es_99"], "99% ES", ":")),
                                   (0.95, 0.75, 0.55), strict=True):
        ax.axvline(val, color=TEXT_SECONDARY, ls=ls, lw=1.3)
        ax.annotate(f"{label}\n{_m(val)}", (val, top * h), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY, va="top",
                    bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1})
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_m))
    ax.set(xlabel="Simulated one-month portfolio loss (NTD)", ylabel="Paths",
           title=f"Base scenario: simulated loss distribution ({int(s['n_sims']):,} paths, independent defaults)")
    ax.grid(axis="x", visible=False)
    caption(ax, HYP)
    save(fig, figs / "loss_distribution_base.png")

    # Scenario overlay (density) + tail comparison bars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.4, 1]})
    hi = max(np.quantile(v, 0.999) for v in losses.values())
    bins = np.linspace(0, hi, 120)
    for name in scen.index:
        a1.hist(losses[name], bins=bins, density=True, histtype="step", lw=1.8,
                color=ENTITY_COLORS.get(name, NEUTRAL), label=name.replace("_", " "))
    a1.xaxis.set_major_formatter(plt.FuncFormatter(_m))
    a1.set(xlabel="Simulated loss (NTD)", ylabel="Density", title="Loss distributions by scenario")
    a1.set_yticks([])
    a1.legend(loc="upper right")
    x = np.arange(len(scen))
    width = 0.2
    for k, (col, label) in enumerate((("mean", "mean"), ("var_99", "99% VaR"), ("es_99", "99% ES"))):
        a2.bar(x + (k - 1) * width, scen[col], width=width * 0.92, color=SERIES[[0, 1, 2][k]],
               label=label)
    a2.set_xticks(x, [n.replace("_", "\n") for n in scen.index], fontsize=7.5)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(_m))
    a2.set(title="Mean and tail loss by scenario", ylabel="NTD")
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper left")
    caption(a1, HYP + " Common random numbers across scenarios.")
    save(fig, figs / "scenario_comparison.png")

    # Convergence
    conv = pd.read_csv(rdir / "convergence_base.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(conv["n_sims"], conv["running_mean"], color=SERIES[0], label="running mean")
    ax.plot(conv["n_sims"], conv["running_var_99"], color=SERIES[1], label="running 99% VaR")
    ax.plot(conv["n_sims"], conv["running_es_99"], color=SERIES[2], label="running 99% ES")
    ax.axhline(conv["analytical_el"].iloc[0], color=NEUTRAL, ls=":", lw=1.2, label="analytical EL")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_m))
    ax.set(xlabel="Number of simulated paths (log scale)", ylabel="NTD",
           title="Monte Carlo convergence, base scenario")
    ax.legend(loc="center right")
    caption(ax, HYP)
    save(fig, figs / "mc_convergence.png")

    # Correlation sensitivity
    rho = pd.read_csv(rdir / "sensitivity_rho.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for col, label, c in (("mean", "mean", SERIES[0]), ("median", "median", NEUTRAL),
                          ("var_99", "99% VaR", SERIES[1]), ("es_99", "99% ES", SERIES[2])):
        ax.plot(rho["rho"], rho[col], marker="o", markersize=4, color=c, label=label)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_m))
    ax.set(xlabel="Common-factor correlation ρ", ylabel="NTD",
           title="Sensitivity of the loss distribution to default correlation")
    ax.legend(loc="upper left")
    caption(ax, HYP + f" {int(rho['n_sims'].iloc[0]):,} paths per point.")
    save(fig, figs / "sensitivity_rho.png")

    # EL sensitivity grid (LGD x CCF)
    grid = pd.read_csv(rdir / "el_sensitivity_grid.csv")
    g = grid[grid["ead_method"] == "balance_plus_ccf"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, lgd in enumerate(sorted(g["lgd"].unique())):
        t = g[g["lgd"] == lgd]
        ax.plot(t["ead_parameter"], t["total_el"], marker="o", markersize=4, color=SERIES[k],
                label=f"LGD {lgd:.0%}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_m))
    ax.set(xlabel="Credit conversion factor on undrawn limit (CCF)", ylabel="Total EL (NTD)",
           title="Expected loss sensitivity to LGD and EAD assumptions")
    ax.legend(loc="upper left")
    caption(ax, HYP + " EAD = drawn + CCF × undrawn.")
    save(fig, figs / "el_sensitivity.png")

    # Risk band concentration
    bands = pd.read_csv(rdir / "risk_bands.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(bands))
    ax.bar(x - 0.2, bands["share_of_accounts"], width=0.38, color=SERIES[0], label="share of accounts")
    ax.bar(x + 0.2, bands["share_of_el"], width=0.38, color=SERIES[1], label="share of hypothetical EL")
    ax.set_xticks(x, bands["band"], fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(xlabel="Predicted PD band", title="Concentration of expected loss by PD band")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    caption(ax, HYP + f" Total EL {_m(el['total_el'])} NTD.")
    save(fig, figs / "risk_band_concentration.png")
