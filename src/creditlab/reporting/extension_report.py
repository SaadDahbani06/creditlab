"""Generated tables for docs/RESEARCH_EXTENSION.md (injected between BEGIN/END markers)."""

from __future__ import annotations

import pandas as pd

from creditlab.reporting.generate import _md, inject
from creditlab.utils.config import PROJECT_ROOT, Config, read_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)
DOC = PROJECT_ROOT / "docs" / "RESEARCH_EXTENSION.md"


def _m(v: float, d: int = 2) -> str:
    return f"{v / 1e6:,.{d}f}"


def _rng(lo: float, hi: float, d: int = 2) -> str:
    return f"[{lo / 1e6:,.{d}f}, {hi / 1e6:,.{d}f}]"


def build_extension_tables(cfg: Config) -> dict[str, str]:
    d = cfg.artifacts / "metrics" / "uncertainty"
    s = read_json(d / "summary.json")
    sel = read_json(cfg.artifacts / "metrics" / "selection.json")
    prim = sel["primary_family"]
    t: dict[str, str] = {}

    # PD movement
    rows = []
    sets = [(f"{f} bootstrap (B = {s['families'][f]['pd']['n_models']})", s["families"][f]["pd"])
            for f in s["families"]]
    x = s["families"].get("xgboost", {})
    if "seed_only_pd" in x:
        sets.append((f"xgboost seed only (R = {x['seed_only_pd']['n_models']})", x["seed_only_pd"]))
    if "retuned_pd" in x:
        k = x["retuned_pd"]["n_models"]
        sets.append((f"xgboost bootstrap, fixed hyperparameters (first {k})",
                     x["retuned_pd"]["paired_fixed_hyperparameters"]))
        sets.append((f"xgboost bootstrap, re-tuned in each sample ({k})", x["retuned_pd"]))
    for name, v in sets:
        rows.append({"model set": name, "median SD of account PD": f"{v['median_sd']:.4f}",
                     "90th pct SD": f"{v['p90_sd']:.4f}",
                     "median relative SD": f"{100 * v['median_relative_sd']:.1f}%",
                     "accounts with 90% range > 0.05": f"{100 * v['share_accounts_90pct_width_gt_0.05']:.1f}%",
                     "SD of portfolio mean PD": f"{v['mean_pd_bootstrap_sd']:.4f}",
                     "common share of PD variance": f"{100 * v['pd_variance_common_share']:.1f}%"})
    t["ext_pd"] = _md(pd.DataFrame(rows))

    rows = []
    for f, v in s["families"].items():
        p = v["pd"]
        rows.append({"family": f, "validation AUC, original fit": f"{p['validation_auc_original']:.4f}",
                     "validation AUC, 95% range across refits":
                         f"[{p['validation_auc_bootstrap_p2_5']:.4f}, {p['validation_auc_bootstrap_p97_5']:.4f}]",
                     "validation log loss, original": f"{p['validation_log_loss_original']:.4f}",
                     "validation log loss, 95% range":
                         f"[{p['validation_log_loss_bootstrap_p2_5']:.4f}, {p['validation_log_loss_bootstrap_p97_5']:.4f}]",
                     "refit reproduces frozen PDs (max abs diff)":
                         f"{p['refit_reproduces_saved_predictions_max_abs_diff']:.1e}"})
    t["ext_models"] = _md(pd.DataFrame(rows))

    # Main results: four regimes for each family
    reg = pd.read_csv(d / "regime_comparison.csv")
    for f in s["families"]:
        rows = []
        for r in reg[reg.family == f].itertuples():
            unc = r.pd == "uncertain"
            rows.append({
                "PDs": "estimation uncertainty (mixture)" if unc else "fixed (original fit)",
                "ρ": f"{r.rho:g}",
                "EL": _m(r.el),
                "VaR 95": _m(r.var_95), "VaR 99": _m(r.var_99), "ES 95": _m(r.es_95),
                "ES 99": _m(r.es_99),
                "95% range of VaR 99 across models": _rng(r.var_99_p2_5, r.var_99_p97_5) if unc else "—",
                "95% range of ES 99 across models": _rng(r.es_99_p2_5, r.es_99_p97_5) if unc else "—",
            })
        t[f"ext_regimes_{f}"] = _md(pd.DataFrame(rows)) + (
            "\n\nNTD millions, hypothetical one-month losses (LGD 60%, EAD = drawn + 50% of undrawn). "
            f"{s['inner_sims']:,} common-random-number paths per fitted model.")

    # Separate bagging shift from spread
    rows = []
    for f in s["families"]:
        for key, r in s["families"][f]["risk"].items():
            ref, bag, mix = (r["references"]["original_fit"], r["references"]["bagged_mean"],
                             r["mixture"])
            rows.append({"family": f, "ρ": key.removeprefix("rho_"),
                         "VaR 99 fixed (original fit)": _m(ref["var_99"]),
                         "VaR 99 fixed (bagged mean PD)": _m(bag["var_99"]),
                         "VaR 99 mixture": _m(mix["var_99"]),
                         "bagging shift": f"{(bag['var_99'] - ref['var_99']) / 1e6:+.2f}",
                         "spread effect": f"{(mix['var_99'] - bag['var_99']) / 1e6:+.2f}",
                         "total change": f"{(mix['var_99'] - ref['var_99']) / 1e6:+.2f}",
                         "MC SE of fixed VaR 99": f"{ref['var_99_mc_se'] / 1e6:.2f}"})
    t["ext_shift"] = _md(pd.DataFrame(rows)) + (
        "\n\nNTD millions. Bagging shift = VaR at the average bootstrap PD minus VaR at the original "
        "PDs; spread effect = mixture VaR minus VaR at the average PD. MC SE: batch-means standard "
        "error of a single fixed-PD VaR estimate.")

    nf_path = d / "noise_free_comparison.json"
    if nf_path.exists():
        nf = read_json(nf_path)
        rows = []
        for key, r in nf.items():
            rows.append({"ρ": key.removeprefix("rho_"),
                         "VaR 99 fixed": _m(r["fixed"]["var_99"]),
                         "VaR 99 at bagged-mean PD": _m(r["bagged_mean"]["var_99"]),
                         "VaR 99 mixture": _m(r["mixture"]["var_99"]),
                         "bagging shift": f"{r['bagging_shift_var_99'] / 1e6:+.2f}",
                         "spread effect": f"{r['spread_effect_var_99'] / 1e6:+.2f}",
                         "total change": f"{r['total_change_var_99'] / 1e6:+.2f}",
                         "total change, %": f"{100 * r['total_change_var_99'] / r['fixed']['var_99']:+.2f}%",
                         "ES 99 spread effect": f"{r['spread_effect_es_99'] / 1e6:+.2f}"})
        t["ext_shift_noise_free"] = _md(pd.DataFrame(rows)) + (
            "\n\nNTD millions. Semi-analytic (no Monte Carlo noise): each model's loss CDF by "
            "quadrature over the common factor; the mixture CDF is their average.")

    # Variance decomposition
    rows = []
    for f in s["families"]:
        for key, r in s["families"][f]["risk"].items():
            dc = r["decomposition"]
            tot = dc["within_model_variance"] + dc["between_model_variance"]
            rows.append({"family": f, "ρ": key.removeprefix("rho_"),
                         "SD of loss, total": _m(tot ** 0.5),
                         "SD from default randomness": _m(dc["within_model_variance"] ** 0.5),
                         "SD from PD estimation": _m(dc["between_model_variance"] ** 0.5),
                         "PD-estimation share of variance": f"{100 * dc['share_between']:.1f}%",
                         "check: Var of analytic EL across models":
                             f"{100 * dc['between_model_variance_analytic_el'] / tot:.1f}%"})
    t["ext_decomposition"] = _md(pd.DataFrame(rows)) + "\n\nNTD millions."

    # MC noise
    rows = []
    for key, r in s["families"][prim]["risk"].items():
        for n in r["mc_noise"]:
            rows.append({"ρ": key.removeprefix("rho_"), "metric": n["metric"],
                         "SD across models (Monte Carlo)": _m(n["sd_across_models_mc"], 3),
                         "SD across models (semi-analytic, noise-free)":
                             _m(n["sd_across_models_semi_analytic"], 3),
                         "model-specific MC noise SD": _m(n["sd_model_specific_mc_noise"], 3),
                         "common MC offset": f"{n['common_offset_mc_minus_sa'] / 1e6:+.3f}",
                         "corr(MC, semi-analytic)": f"{n['corr_mc_sa']:.3f}"})
    t["ext_mc_noise"] = _md(pd.DataFrame(rows)) + "\n\nNTD millions."

    g = pd.read_csv(d / "rho_grid.csv")
    t["ext_rho"] = _md(pd.DataFrame({
        "ρ": g.rho.map("{:g}".format), "VaR 99 fixed": g.fixed_var_99.map(_m),
        "VaR 99, 95% range across models": [_rng(a, b) for a, b in zip(g.var_99_p2_5, g.var_99_p97_5, strict=True)],
        "VaR 99 mixture": g.mixture_var_99.map(_m), "ES 99 fixed": g.fixed_es_99.map(_m),
        "ES 99 mixture": g.mixture_es_99.map(_m),
        "PD-estimation share of variance": g.share_between.map(lambda v: f"{100 * v:.1f}%")})) + (
        f"\n\nNTD millions; {cfg['uncertainty']['grid_sims']:,} paths per model.")

    sens = pd.read_csv(d / "sensitivity_model_sets.csv")
    t["ext_sensitivity"] = _md(pd.DataFrame({
        "model set": sens["set"], "ρ": sens.rho.map("{:g}".format), "models": sens.n_models,
        "SD of EL": sens.el_sd.map(lambda v: _m(v, 3)),
        "SD of VaR 99 (semi-analytic)": sens.sa_var_99_sd.map(lambda v: _m(v, 3)),
        "SD of ES 99 (semi-analytic)": sens.sa_es_99_sd.map(lambda v: _m(v, 3)),
        "SD of portfolio mean PD": sens.mean_pd_sd.map("{:.4f}".format)})) + "\n\nNTD millions."

    c = pd.read_csv(d / "convergence.csv")
    cb = c[c.n_models.notna()]
    rows = []
    for rho in sorted(cb.rho.unique()):
        x_ = cb[cb.rho == rho].set_index("n_models")
        avail = x_.index.to_numpy()
        picks = sorted({int(avail[abs(avail - k).argmin()]) for k in (25, 50, 100, avail.max())})
        rows.append({"ρ": f"{rho:g}", **{f"SD of VaR 99, B = {k}": _m(x_.loc[k, "var_99_sd"], 3)
                                       for k in picks}})
    ci = c[c.inner_paths.notna()]
    conv_rows = []
    for rho in sorted(ci.rho.unique()):
        x_ = ci[ci.rho == rho]
        last = x_.mixture_var_99.iloc[-1]
        half = x_[x_.inner_paths >= x_.inner_paths.max() / 2].mixture_var_99
        conv_rows.append({"ρ": f"{rho:g}", "mixture VaR 99 (all paths)": _m(last),
                          "max relative change over the last half of the paths":
                              f"{100 * (half.max() - half.min()) / last:.2f}%"})
    t["ext_convergence"] = _md(pd.DataFrame(rows)) + "\n\n" + _md(pd.DataFrame(conv_rows)) + (
        "\n\nNTD millions. Top: stability of the across-model spread as B grows (noise-free "
        "semi-analytic values). Bottom: stability of the mixture VaR as inner paths grow.")

    # Comparison with the original CreditLab findings
    scen = pd.read_csv(cfg.artifacts / "metrics" / "risk" / "scenario_summary.csv").set_index("scenario")
    rows = []
    for rho, name in ((0.0, "base"), (0.04, "base_correlated")):
        key = f"rho_{rho}"
        if key not in s["families"][prim]["risk"]:
            continue
        r = s["families"][prim]["risk"][key]
        a = r["across_models"]
        rows.append({"scenario": name, "original VaR 99": _m(scen.loc[name, "var_99"]),
                     "extension, fixed PDs (same draws)": _m(r["references"]["original_fit"]["var_99"]),
                     "noise-free semi-analytic, fixed PDs": _m(r["references"]["original_fit"]["sa_var_99"]),
                     "95% range across refitted models": _rng(a["var_99"]["p2_5"], a["var_99"]["p97_5"]),
                     "mixture": _m(r["mixture"]["var_99"])})
    t["ext_vs_original"] = _md(pd.DataFrame(rows)) + "\n\nNTD millions."

    mech = read_json(d / "mechanism.json") if (d / "mechanism.json").exists() else {}
    rows = []
    for f, v in mech.items():
        for key, w in v["var99"].items():
            rows.append({"family": f, "ρ": key.removeprefix("rho_"),
                         "corr(training default rate, portfolio mean PD)":
                             f"{v['corr_train_default_rate_vs_mean_pd']:.3f}",
                         "R² of mean PD on training default rate":
                             f"{v['r2_mean_pd_on_train_default_rate']:.3f}",
                         "R² of VaR 99 on EL": f"{w['r2_on_el']:.3f}",
                         "SD(VaR 99) / SD(EL)": f"{w['sd_var99_over_sd_el']:.2f}",
                         "residual SD of VaR 99": _m(w["residual_sd"], 3),
                         "corr(residual, PD dispersion)":
                             f"{w['corr_residual_with_pd_dispersion']:+.2f}"})
    if rows:
        t["ext_mechanism"] = _md(pd.DataFrame(rows)) + (
            "\n\nAcross the 200 bootstrap models. VaR values are noise-free semi-analytic; NTD "
            "millions. PD dispersion = cross-sectional SD of a model's 6,000 PDs.")

    o = s["observed_vs_expected_defaults"]
    t["ext_defaults"] = (f"Observed test defaults: {o['observed_test_defaults']:,}. Expected under the "
                         f"original fit: {o['expected_original']:,.1f}; 95% range across refitted "
                         f"models: [{o['expected_bootstrap_p2_5']:,.1f}, "
                         f"{o['expected_bootstrap_p97_5']:,.1f}]. Descriptive only.")
    return t


def run_extension_report(cfg: Config) -> list[str]:
    if not (cfg.artifacts / "metrics" / "uncertainty" / "summary.json").exists():
        return []
    tables = build_extension_tables(cfg)
    (cfg.artifact_dir("metrics") / "uncertainty" / "extension_tables.md").write_text(
        "# Extension tables\n\n" + "\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items()) + "\n")
    if cfg.artifacts.resolve() == (PROJECT_ROOT / "artifacts").resolve() and DOC.exists():
        used = inject(DOC, tables)
        log.info("Injected %d extension tables into %s", len(used), DOC.name)
        return used
    return []
