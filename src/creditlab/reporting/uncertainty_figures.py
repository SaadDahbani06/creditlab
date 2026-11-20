"""Figures for the PD-estimation-uncertainty extension (report style, matplotlib)."""

from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from creditlab.reporting.style import GRID, SERIES, TEXT_SECONDARY, apply_style, caption, save
from creditlab.utils.config import PROJECT_ROOT, Config, read_json

M = 1e6


def make_uncertainty_figures(cfg: Config) -> list[str]:
    apply_style()
    d = cfg.artifacts / "metrics" / "uncertainty"
    fig_dir = cfg.artifact_dir("figures")
    s = read_json(d / "summary.json")
    fam = "xgboost" if "xgboost" in s["families"] else next(iter(s["families"]))
    rhos = sorted(float(k.removeprefix("rho_")) for k in s["families"][fam]["risk"])
    made = []

    # 1. How much do individual PDs move?
    acc = pd.read_parquet(d / f"pd_per_account_{fam}.parquet")
    pdb = np.load(d / f"pd_bootstrap_{fam}.npz")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.42})
    hb = ax[0].hexbin(acc.pd_original, acc.pd_bootstrap_sd, gridsize=45, mincnt=1, cmap="Blues",
                      bins="log", linewidths=0)
    ax[0].set(xlabel="PD from the original fit", ylabel="Bootstrap SD of the account's PD",
              title=f"Account-level PD noise ({int(s['n_bootstrap'])} refits)")
    fig.colorbar(hb, ax=ax[0], label="accounts (log scale)")
    mean_pd = pdb["test"].astype(float).mean(axis=1)
    ax[1].hist(mean_pd, bins=30, color=SERIES[0], alpha=0.85)
    ax[1].axvline(acc.pd_original.mean(), color=TEXT_SECONDARY, ls="--", lw=1.4,
                  label=f"original fit {acc.pd_original.mean():.4f}")
    ax[1].set(xlabel="Portfolio mean PD", ylabel="Bootstrap models",
              title="Portfolio-average PD across refits")
    ax[1].legend()
    caption(ax[0], "Test-split portfolio (6,000 accounts). Refits use the training split only.")
    save(fig, fig_dir / "ext_pd_uncertainty.png")
    made.append("ext_pd_uncertainty.png")

    # 2. VaR99 across fitted models, independent vs correlated
    fig, ax = plt.subplots(1, len(rhos), figsize=(5 * len(rhos), 3.6))
    ax = np.atleast_1d(ax)
    for i, rho in enumerate(rhos):
        pm = pd.read_csv(d / f"risk_per_model_{fam}_rho{rho}.csv")
        r = s["families"][fam]["risk"][f"rho_{rho}"]
        ax[i].hist(pm.var_99 / M, bins=30, color=SERIES[0], alpha=0.85, label="per fitted model")
        ax[i].axvline(r["references"]["original_fit"]["var_99"] / M, color=TEXT_SECONDARY, ls="--",
                      lw=1.4, label="fixed PDs (original fit)")
        ax[i].axvline(r["mixture"]["var_99"] / M, color=SERIES[1], lw=1.8,
                      label="PD uncertainty included (mixture)")
        ax[i].set(xlabel="99% VaR, NTD millions (hypothetical)", ylabel="Bootstrap models",
                  title=f"99% VaR across refitted models, ρ = {rho:g}")
        ax[i].set_ylim(0, ax[i].get_ylim()[1] * 1.45)
        ax[i].legend(fontsize=7.5, loc="upper left", frameon=False)
    caption(ax[0], "Same simulated default draws for every model (common random numbers).")
    save(fig, fig_dir / "ext_var_across_models.png")
    made.append("ext_var_across_models.png")

    # 3. Four regimes: one panel per correlation setting (own x scale), rows = measure x PD treatment
    reg = pd.read_csv(d / "regime_comparison.csv")
    reg = reg[reg.family == fam]
    fig, ax = plt.subplots(1, len(rhos), figsize=(11, 3.4), sharey=True,
                           gridspec_kw={"wspace": 0.08})
    ax = np.atleast_1d(ax)
    rows = [("var_99", "fixed"), ("var_99", "uncertain"), ("es_99", "fixed"), ("es_99", "uncertain")]
    labels = [f"{'99% VaR' if m_ == 'var_99' else '99% ES'}, "
              f"{'fixed PDs' if k == 'fixed' else 'with PD uncertainty'}" for m_, k in rows]
    for j, rho in enumerate(rhos):
        for y, (metric, pdk) in enumerate(rows):
            row = reg[(reg.rho == rho) & (reg.pd == pdk)].iloc[0]
            col = TEXT_SECONDARY if pdk == "fixed" else SERIES[0]
            if pdk == "uncertain":
                ax[j].hlines(y, row[f"{metric}_p2_5"] / M, row[f"{metric}_p97_5"] / M, color=col,
                             lw=3, alpha=0.45)
            ax[j].plot(row[metric] / M, y, "o", color=col, ms=7, zorder=3)
        ax[j].set_yticks(range(len(rows)), labels)
        ax[j].set_ylim(len(rows) - 0.5, -0.5)
        ax[j].set(xlabel="NTD millions (hypothetical)",
                  title="Independent defaults" if rho == 0 else f"Correlated, ρ = {rho:g}")
    caption(ax[0], "Dot: point estimate (gray fixed PDs; blue mixture over refitted models). "
                   "Bar: 95% range of the per-model estimate.")
    save(fig, fig_dir / "ext_regime_comparison.png")
    made.append("ext_regime_comparison.png")

    # 4. Correlation dominates: VaR99 vs rho with the estimation band
    g = pd.read_csv(d / "rho_grid.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(g.rho, g.var_99_p2_5 / M, g.var_99_p97_5 / M, color=SERIES[0], alpha=0.2,
                    lw=0, label="95% range across refitted models")
    ax.plot(g.rho, g.fixed_var_99 / M, "-o", color=TEXT_SECONDARY, ms=4, label="fixed PDs")
    ax.plot(g.rho, g.mixture_var_99 / M, "-o", color=SERIES[0], ms=4, label="PD uncertainty included")
    ax.set(xlabel="Asset correlation ρ (assumed)", ylabel="99% VaR, NTD millions",
           title="Correlation assumption vs PD estimation uncertainty")
    ax.legend()
    caption(ax, "Hypothetical one-month losses; LGD 60%, EAD = drawn + 50% of undrawn limit.")
    save(fig, fig_dir / "ext_rho_grid.png")
    made.append("ext_rho_grid.png")

    # 5. Convergence diagnostics
    c = pd.read_csv(d / "convergence.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, rho in enumerate(rhos):
        cb = c[(c.rho == rho) & c.n_models.notna()]
        ax[0].plot(cb.n_models, cb.var_99_sd / M, "-o", ms=3, color=SERIES[i], label=f"ρ = {rho:g}")
        ci = c[(c.rho == rho) & c.inner_paths.notna()]
        ax[1].plot(ci.inner_paths, ci.mixture_var_99 / ci.mixture_var_99.iloc[-1], "-o", ms=3,
                   color=SERIES[i], label=f"ρ = {rho:g}")
    ax[0].set(xscale="log", xlabel="Number of bootstrap models B",
              ylabel="SD of 99% VaR across models (M)", title="Stability in B (noise-free values)")
    ax[1].axhline(1, color=GRID, lw=1)
    ax[1].set(xscale="log", xlabel="Inner simulated paths per model",
              ylabel="Mixture VaR99 / final value", title="Stability in inner paths")
    for a in ax:
        a.legend()
    save(fig, fig_dir / "ext_convergence.png")
    made.append("ext_convergence.png")

    if cfg.artifacts.resolve() == (PROJECT_ROOT / "artifacts").resolve():
        dst = PROJECT_ROOT / "docs" / "figures"
        dst.mkdir(parents=True, exist_ok=True)
        for f in made:
            shutil.copy2(fig_dir / f, dst / f)
    plt.close("all")
    return made
