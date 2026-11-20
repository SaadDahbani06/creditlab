"""Figures for the hypothetical policy lab."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.reporting.style import ENTITY_COLORS, NEUTRAL, apply_style, caption, plt, save
from creditlab.utils.config import Config, read_json

NOTE = ("HYPOTHETICAL: assumed revenue rate, LGD and EAD; one-month horizon; NTD; "
        "test-split cohort of existing accounts. Not lending decisions.")
SHORT = {"A_retain_all": "A all", "B_pd_threshold": "B PD cutoff",
         "C_expected_contribution": "C E[contrib]>0", "D_random_equal_size": "D random"}


def _m(x, _=None):
    return f"{x / 1e6:.1f}M"


def make_policy_figures(cfg: Config) -> None:
    apply_style()
    pdir = cfg.artifact_dir("metrics") / "policy"
    figs = cfg.artifact_dir("figures")
    tab = pd.read_csv(pdir / "policy_comparison.csv", index_col=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(tab))
    colors = [ENTITY_COLORS.get(p, NEUTRAL) for p in tab.index]
    for ax, col, title, fmt in (
        (axes[0], "share_selected", "Share of cohort retained", lambda v, _: f"{v:.0%}"),
        (axes[1], "observed_default_rate", "OBSERVED default rate of retained", lambda v, _: f"{v:.0%}"),
        (axes[2], "expected_net_contribution", "Expected net contribution (model PDs)", _m),
    ):
        ax.bar(x, tab[col], color=colors, width=0.65)
        ax.set_xticks(x, [SHORT[p] for p in tab.index], fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    d = tab.loc["D_random_equal_size"]
    axes[2].errorbar(3, d["expected_net_contribution"],
                     yerr=[[d["expected_net_contribution"] - d["expected_net_contribution_p05"]],
                           [d["expected_net_contribution_p95"] - d["expected_net_contribution"]]],
                     color="#0b0b0b", capsize=3, lw=1)
    caption(axes[0], NOTE + " D: mean of random draws (bar: 5–95%).")
    save(fig, figs / "policy_comparison.png")

    thr = read_json(pdir / "policy_threshold_selection.json")
    grid = np.asarray(thr["grid"][:-1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(grid, np.asarray(thr["grid_bad_rates"][:-1]), color=ENTITY_COLORS["B_pd_threshold"])
    if thr.get("target_bad_rate") is not None:
        a1.axhline(thr["target_bad_rate"], color=NEUTRAL, ls=":", lw=1.2)
        a1.annotate(f"risk-appetite target {thr['target_bad_rate']:.0%}",
                    (grid.max(), thr["target_bad_rate"]), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=8)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set(xlabel="PD cutoff (retain if PD < cutoff)",
           ylabel="Observed default rate of retained",
           title="Validation: observed bad rate vs cutoff")
    a2.plot(grid, np.asarray(thr["grid_totals"][:-1]), color=ENTITY_COLORS["C_expected_contribution"])
    a2.yaxis.set_major_formatter(plt.FuncFormatter(_m))
    a2.set(xlabel="PD cutoff (retain if PD < cutoff)",
           ylabel="Realized hypothetical contribution",
           title="Validation: observed outcomes x assumed economics")
    for ax in (a1, a2):
        ax.axvline(thr["threshold"], color=NEUTRAL, ls="--", lw=1.2)
    a1.annotate(f"selected cutoff {thr['threshold']:.3f}", (thr["threshold"], a1.get_ylim()[0]),
                xytext=(5, 8), textcoords="offset points", fontsize=8)
    caption(a1, f"Policy B cutoff chosen on the validation split, rule: {thr['rule']}. " + NOTE[:58] + "…")
    save(fig, figs / "policy_threshold_selection.png")
