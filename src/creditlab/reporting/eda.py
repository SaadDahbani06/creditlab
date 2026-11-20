"""Exploratory figures built from the SQL results (training split only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.reporting.style import SERIES, TEXT_SECONDARY, apply_style, caption, plt, save
from creditlab.utils.config import Config

NOTE = "Source: UCI dataset 350, development training split (n = 18,000)."


def _rate_bars(ax, labels, rates, ns, color=SERIES[0]):
    x = np.arange(len(labels))
    ax.bar(x, rates, color=color, width=0.7, edgecolor="white", linewidth=1)
    ax.set_xticks(x, labels)
    for xi, r, n in zip(x, rates, ns, strict=True):
        ax.text(xi, r + 0.01, f"{r:.0%}\nn={n:,}", ha="center", va="bottom", fontsize=7,
                color=TEXT_SECONDARY)
    ax.set_ylim(0, min(1.0, max(rates) * 1.3 + 0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)


def make_eda_figures(cfg: Config, results: dict[str, pd.DataFrame]) -> None:
    apply_style()
    out = cfg.artifact_dir("figures")

    d = results["default_rate_by_pay0"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _rate_bars(ax, d["status_code"].astype(str), d["default_rate"], d["n_accounts"])
    ax.set_title("Next-month default rate by September 2005 repayment status code")
    ax.set_xlabel("PAY_0 code (-2 and 0 are undocumented by the source; -1 = pay duly; k >= 1 = k-month delay)")
    ax.set_ylabel("Observed default rate")
    caption(ax, NOTE)
    save(fig, out / "eda_default_by_pay0.png")

    d = results["repayment_delay_patterns"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _rate_bars(ax, d["months_delayed_of_6"].astype(str), d["default_rate"], d["n_accounts"])
    ax.set_title("Default rate by number of delayed months (Apr–Sep 2005)")
    ax.set_xlabel("Months with documented delay (status >= 1), of 6")
    ax.set_ylabel("Observed default rate")
    caption(ax, NOTE)
    save(fig, out / "eda_default_by_months_delayed.png")

    d = results["credit_limit_deciles"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = [f"{lo / 1000:.0f}–{hi / 1000:.0f}k" for lo, hi in zip(d["min_limit"], d["max_limit"], strict=True)]
    _rate_bars(ax, labels, d["default_rate"], d["n_accounts"])
    ax.set_title("Default rate by credit-limit decile")
    ax.set_xlabel("LIMIT_BAL decile range (NTD thousands)")
    ax.set_ylabel("Observed default rate")
    ax.tick_params(axis="x", labelsize=7.5)
    caption(ax, NOTE)
    save(fig, out / "eda_default_by_limit_decile.png")

    d = results["payment_to_bill_ratio"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = [s.split(": ", 1)[1] for s in d["pay_to_bill_band"]]
    _rate_bars(ax, labels, d["default_rate"], d["n_accounts"])
    ax.set_title("Default rate by six-month payment-to-bill ratio")
    ax.set_xlabel("Sum of payments / sum of positive bills (Apr–Sep 2005)")
    ax.set_ylabel("Observed default rate")
    ax.tick_params(axis="x", labelsize=7.5)
    caption(ax, NOTE)
    save(fig, out / "eda_default_by_pay_to_bill.png")

    d = results["default_by_risk_bands"]
    pivot = d.pivot(index="utilisation_band", columns="worst_delay_6m", values="default_rate")
    counts = d.pivot(index="utilisation_band", columns="worst_delay_6m", values="n_accounts")
    order = ["no delay", "1 month", "2 months", "3+ months"]
    pivot, counts = pivot[order], counts[order]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(pivot.index)), [s.split(": ")[1] for s in pivot.index])
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v, n = pivot.values[i, j], counts.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0%}\nn={int(n):,}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.5 else "#0b0b0b")
    ax.set_xlabel("Worst repayment delay in six months")
    ax.set_ylabel("Sept. utilisation (bill / limit)")
    ax.set_title("Observed default rate by utilisation and worst delay")
    fig.colorbar(im, ax=ax, fraction=0.04, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    caption(ax, NOTE)
    save(fig, out / "eda_risk_band_heatmap.png")

    d = results["status_code_by_month"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    cols = [("share_code_m2", "code -2 (undoc.)", SERIES[2]), ("share_code_m1", "code -1 (pay duly)", SERIES[0]),
            ("share_code_0", "code 0 (undoc.)", SERIES[1]), ("share_code_2", "code 2 (2-month delay)", SERIES[7])]
    for col, label, color in cols:
        ax.plot(d["month"], d[col], marker="o", markersize=4, color=color, label=label)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Share of accounts by repayment status code, per month")
    ax.set_ylabel("Share of accounts")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0))
    caption(ax, NOTE + " Code 1 is almost absent after September.")
    save(fig, out / "eda_status_codes_by_month.png")
