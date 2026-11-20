"""`report` stage: regenerate figures and inject measured-result tables into the docs.

Tables are written between markers ``<!-- BEGIN:name -->`` and ``<!-- END:name -->`` in
docs/RESEARCH_REPORT.md and docs/MODEL_CARD.md, so every number in those tables comes from the
saved artifacts of the last pipeline run. Narrative text outside the markers is hand-written.
All tables are also written to artifacts/metrics/report_tables.md.
"""

from __future__ import annotations

import re
import shutil

import pandas as pd

from creditlab.utils.config import PROJECT_ROOT, Config, read_json, update_manifest
from creditlab.utils.logging import get_logger

log = get_logger(__name__)
DOCS = ["docs/RESEARCH_REPORT.md", "docs/MODEL_CARD.md"]


def _md(df: pd.DataFrame, floatfmt: dict[str, str] | None = None, index: bool = False) -> str:
    df = df.copy()
    for col, fmt in (floatfmt or {}).items():
        if col in df:
            df[col] = df[col].map(lambda v, f=fmt: "—" if pd.isna(v) else format(v, f))
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("—" if (not isinstance(v, str) and pd.isna(v)) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines)


def _ntd(v: float) -> str:
    return f"{v / 1e6:,.2f}M"


def build_tables(cfg: Config) -> dict[str, str]:
    m = cfg.artifacts / "metrics"
    t: dict[str, str] = {}

    val = read_json(m / "data_validation.json")
    split = pd.DataFrame(read_json(m / "split_summary.json"))
    split["default_rate"] = split["default_rate"].map(lambda v: f"{v:.4f}")
    t["dataset"] = _md(split.rename(columns={"n": "accounts", "defaults": "observed defaults"})) + (
        f"\n\nTotal: {val['rows']:,} accounts, {val['n_features']} source features, observed "
        f"default rate {val['default_rate']:.4f} ({val['target_counts']['1']:,} defaults).")

    sel = read_json(m / "selection.json")
    ev = read_json(m / "test_evaluation.json")
    boot = pd.read_csv(m / "test_bootstrap_ci.csv")
    chosen = [sel["families"][f]["model"] for f in ("logistic", "xgboost")]
    rows = []
    for name in ["dummy_raw", *chosen]:
        tm = ev["test_metrics"][name]
        b = boot[boot["model"] == name].set_index("metric")

        def ci(k, b=b):
            return f"[{b.loc[k, 'ci_low']:.4f}, {b.loc[k, 'ci_high']:.4f}]" if k in b.index else ""

        rows.append({"model": name, "ROC-AUC": f"{tm['roc_auc']:.4f}", "AUC 95% CI": ci("roc_auc"),
                     "PR-AUC (AP)": f"{tm['average_precision']:.4f}", "AP 95% CI": ci("average_precision"),
                     "Brier": f"{tm['brier']:.4f}", "log loss": f"{tm['log_loss']:.4f}",
                     "accuracy @0.5": f"{tm['accuracy_at_0.5']:.4f}"})
    t["test_metrics"] = _md(pd.DataFrame(rows))
    diff = boot[boot["model"] == f"{chosen[1]} - {chosen[0]}"].set_index("metric")
    t["test_difference"] = _md(pd.DataFrame([
        {"metric": k, "difference (XGBoost − logistic)": f"{diff.loc[k, 'boot_mean']:+.4f}",
         "paired bootstrap 95% CI": f"[{diff.loc[k, 'ci_low']:+.4f}, {diff.loc[k, 'ci_high']:+.4f}]"}
        for k in ("roc_auc", "average_precision", "brier", "log_loss")]))

    thr = ev["test_threshold_metrics"]
    t["threshold_metrics"] = _md(pd.DataFrame([
        {"model": k, "threshold (from validation)": f"{v['threshold']:.4f}",
         "precision": f"{v['precision']:.4f}", "recall": f"{v['recall']:.4f}", "F1": f"{v['f1']:.4f}",
         "accuracy": f"{v['accuracy']:.4f}", "flagged": f"{v['flagged_share']:.3f}",
         "TN": v["confusion_matrix"]["tn"], "FP": v["confusion_matrix"]["fp"],
         "FN": v["confusion_matrix"]["fn"], "TP": v["confusion_matrix"]["tp"]}
        for k, v in thr.items()]))

    vm = pd.DataFrame(sel["validation_metrics"]).T
    vm = vm[["roc_auc", "average_precision", "brier", "log_loss", "ece", "calibration_intercept",
             "calibration_slope"]]
    t["validation_all"] = _md(vm.rename_axis("model"), {c: ".4f" for c in vm.columns}, index=True)
    tm_all = pd.read_csv(m / "test_metrics.csv", index_col=0)[vm.columns]
    t["test_all"] = _md(tm_all.rename_axis("model"), {c: ".4f" for c in vm.columns}, index=True)

    vb = pd.DataFrame(sel["validation_bootstrap"])
    vb = vb[vb["model"].str.contains(" - ") & (vb["metric"].isin(["log_loss", "brier"]))]
    vb = vb.assign(**{"95% CI": vb.apply(lambda r: f"[{r.ci_low:+.5f}, {r.ci_high:+.5f}]", axis=1),
                      "mean difference": vb["boot_mean"].map(lambda v: f"{v:+.5f}")})
    t["calibration_bootstrap"] = _md(vb[["model", "metric", "mean difference", "95% CI"]]
                                     .rename(columns={"model": "comparison (validation)"}))
    t["selection"] = _md(pd.DataFrame([
        {"family": f, "chosen variant": d["chosen_variant"],
         "qualifying calibrations": ", ".join(d["qualifying_calibrations"]) or "none",
         "F1 threshold": f"{d['threshold_max_f1']:.4f}"} for f, d in sel["families"].items()]
    )) + f"\n\nPrimary model (lowest validation log loss): **{sel['primary_model']}**."

    meta = read_json(cfg.artifacts / "models" / "training_metadata.json")
    t["hyperparameters"] = _md(pd.DataFrame([
        {"model": k, "selected hyperparameters": ", ".join(f"{p}={v}" for p, v in d["params"].items()),
         "CV log loss": f"{d.get('cv_mean_log_loss', float('nan')):.4f}",
         "CV ROC-AUC": f"{d.get('cv_mean_roc_auc', float('nan')):.4f}",
         "candidates": d.get("cv_candidates", 1)}
        for k, d in meta["models"].items() if "params" in d and k != "dummy_raw"]))

    perm = pd.read_csv(m / "permutation_importance_validation.csv")
    top = perm.groupby("model").head(6)
    t["importance"] = _md(top[["model", "feature", "auc_drop_mean", "auc_drop_std"]],
                          {"auc_drop_mean": ".4f", "auc_drop_std": ".4f"})

    audit = pd.read_csv(m / "audit_test_primary_model.csv")
    t["audit"] = _md(audit[["attribute", "group", "n", "observed_default_rate", "mean_predicted_pd",
                            "predicted_minus_observed", "roc_auc"]],
                     {"observed_default_rate": ".3f", "mean_predicted_pd": ".3f",
                      "predicted_minus_observed": "+.3f", "roc_auc": ".3f"})

    r = m / "risk"
    el = read_json(r / "expected_loss_summary.json")
    t["expected_loss"] = _md(pd.DataFrame([
        ("Accounts (test split)", f"{el['n_accounts']:,}"),
        ("Model", el["model"]),
        ("Assumed LGD", f"{el['assumptions']['lgd']:.0%}"),
        ("Assumed EAD", el["assumptions"]["ead_formula"] + f" (CCF = {el['assumptions']['ccf']})"),
        ("Hypothetical total EAD (NTD)", _ntd(el["total_ead"])),
        ("Hypothetical total EL (NTD)", _ntd(el["total_el"])),
        ("EL / EAD", f"{el['el_rate_of_ead']:.4f}"),
        ("Mean PD", f"{el['mean_pd']:.4f}"),
        ("EAD-weighted PD", f"{el['ead_weighted_pd']:.4f}"),
        ("Expected defaults (Σ PD)", f"{el['expected_defaults']:.1f}"),
        ("Observed defaults in test split", f"{el['observed_defaults_in_portfolio']:,}"),
        ("Assumed LGD/EAD applied to observed defaults (NTD)", _ntd(el["hypothetical_loss_at_observed_defaults"])),
        ("Share of EL in top 10% of accounts by EL", f"{el['share_el_top10pct_accounts_by_el']:.3f}"),
        ("Share of EL / EAD in top PD decile", f"{el['share_el_top_pd_decile']:.3f} / {el['share_ead_top_pd_decile']:.3f}"),
    ], columns=["quantity", "value"]))
    bands = pd.read_csv(r / "risk_bands.csv")
    t["risk_bands"] = _md(bands[["band", "n_accounts", "mean_pd", "observed_default_rate",
                                 "share_of_accounts", "share_of_ead", "share_of_el"]],
                          {"mean_pd": ".3f", "observed_default_rate": ".3f",
                           "share_of_accounts": ".3f", "share_of_ead": ".3f", "share_of_el": ".3f"})
    grid = pd.read_csv(r / "el_sensitivity_grid.csv")
    g = grid.pivot_table(index=["ead_method", "ead_parameter"], columns="lgd", values="total_el")
    g.columns = [f"LGD {c:.0%}" for c in g.columns]
    t["el_sensitivity"] = _md((g / 1e6).round(2).reset_index().rename(
        columns={"ead_parameter": "CCF or limit fraction"})) + "\n\nValues in NTD millions."

    sc = pd.read_csv(r / "scenario_summary.csv")
    cols = {"scenario": "scenario", "odds_multiplier": "odds ×", "lgd": "LGD", "rho": "ρ",
            "mean_stressed_pd": "mean PD", "analytical_el": "analytical EL", "mean": "MC mean",
            "median": "median", "std": "std", "var_95": "VaR 95", "var_99": "VaR 99",
            "es_95": "ES 95", "es_99": "ES 99"}
    s2 = sc[list(cols)].rename(columns=cols)
    for c in ["analytical EL", "MC mean", "median", "std", "VaR 95", "VaR 99", "ES 95", "ES 99"]:
        s2[c] = s2[c].map(_ntd)
    s2["mean PD"] = s2["mean PD"].map(lambda v: f"{v:.4f}")
    t["scenarios"] = _md(s2) + f"\n\n{int(sc['n_sims'].iloc[0]):,} paths per scenario; NTD."
    ex = [c for c in sc.columns if c.startswith("p_exceed_")]
    s3 = sc[["scenario", "mean_minus_el_in_se", "factor_sample_mean_z", "var_99_mc_se",
             "es_99_mc_se", "max_abs_marginal_pd_error", *ex]].copy()
    s3["var_99_mc_se"] = s3["var_99_mc_se"].map(_ntd)
    s3["es_99_mc_se"] = s3["es_99_mc_se"].map(_ntd)
    s3 = s3.rename(columns={c: "P(L > " + c.removeprefix("p_exceed_").replace("x_base_el", "× base EL)")
                            for c in ex})
    t["scenario_checks"] = _md(s3, {"mean_minus_el_in_se": "+.2f", "factor_sample_mean_z": "+.2f",
                                    "max_abs_marginal_pd_error": ".4f",
                                    **{c: ".4f" for c in s3.columns if c.startswith("P(")}})
    ub = read_json(r / "mc_unbiasedness_check.json")
    t["mc_unbiasedness"] = (
        f"Unbiasedness check ({ub['scenario']}, {ub['n_seeds']} independent seeds × "
        f"{ub['paths_per_seed']:,} paths): standardised error (MC mean − analytical EL)/SE has "
        f"mean {ub['z_mean']:+.2f} and SD {ub['z_sd']:.2f}; share with |z| > 2: "
        f"{ub['share_abs_z_gt_2']:.2f} (≈0.05 expected for an unbiased estimator).")
    rho = pd.read_csv(r / "sensitivity_rho.csv")[["rho", "mean", "median", "std", "var_99", "es_99"]]
    for c in ["mean", "median", "std", "var_99", "es_99"]:
        rho[c] = rho[c].map(_ntd)
    t["rho_sensitivity"] = _md(rho)
    lgd = pd.read_csv(r / "sensitivity_lgd.csv")[["lgd", "rho", "mean", "var_99", "es_99"]]
    for c in ["mean", "var_99", "es_99"]:
        lgd[c] = lgd[c].map(_ntd)
    t["lgd_sensitivity"] = _md(lgd)
    ovs = read_json(r / "observed_vs_simulated_defaults.json")
    t["observed_vs_simulated"] = _md(pd.DataFrame([
        {"scenario": k, "observed defaults": v["observed_defaults"],
         "simulated mean": f"{v['simulated_mean']:.1f}",
         "simulated 95% interval": f"[{v['simulated_p2_5']:.0f}, {v['simulated_p97_5']:.0f}]",
         "share of paths below observed": f"{v['share_of_paths_below_observed']:.3f}"}
        for k, v in ovs.items()]))

    p = m / "policy"
    pol = pd.read_csv(p / "policy_comparison.csv", index_col=0)
    cols = ["share_selected", "mean_pd", "observed_default_rate", "total_ead", "total_expected_loss",
            "total_expected_revenue", "expected_net_contribution",
            "realized_hypothetical_contribution", "share_el_top10pct_accounts",
            "loss_var_99_rho_0.04"]
    pt = pol[cols].copy()
    for c in ["total_ead", "total_expected_loss", "total_expected_revenue",
              "expected_net_contribution", "realized_hypothetical_contribution",
              "loss_var_99_rho_0.04"]:
        pt[c] = pt[c].map(lambda v: "—" if pd.isna(v) else _ntd(v))
    t["policy"] = _md(pt.rename_axis("policy"), {"share_selected": ".3f", "mean_pd": ".3f",
                                                 "observed_default_rate": ".3f",
                                                 "share_el_top10pct_accounts": ".3f"}, index=True)
    d = pol.loc["D_random_equal_size"]
    t["policy"] += (f"\n\nPolicy D: mean of {cfg['policy']['random_baseline_draws']} random draws. "
                    f"5–95% range of observed default rate: [{d['observed_default_rate_p05']:.3f}, "
                    f"{d['observed_default_rate_p95']:.3f}]; of expected net contribution: "
                    f"[{_ntd(d['expected_net_contribution_p05'])}, {_ntd(d['expected_net_contribution_p95'])}].")
    thr = read_json(p / "policy_threshold_selection.json")
    be = read_json(p / "break_even_revenue_rate.json")
    t["policy_threshold"] = (
        f"Rule `{thr['rule']}` (target observed bad rate {thr['target_bad_rate']}). "
        f"Selected PD cutoff **{thr['threshold']:.4f}** on validation; validation observed bad rate "
        f"{thr['validation_bad_rate']:.4f}; validation share retained {thr['validation_share_retained']:.3f}. "
        f"Break-even analysis (test): share of accounts with positive expected one-month contribution at "
        f"r = {be['assumed_rate']}: {be['share_positive_expected_contribution']:.3f}; highest account "
        f"break-even PD {be['max_break_even_pd']:.4f} vs lowest model PD {be['min_model_pd']:.4f}; median "
        f"break-even monthly revenue rate {be['quantiles']['0.5']:.3f} "
        f"({be['n_without_drawn_balance']} accounts have no drawn balance).")
    sens = pd.read_csv(p / "policy_revenue_sensitivity.csv")
    sp = sens.pivot_table(index="monthly_revenue_rate", columns="policy", values="share_selected")
    t["policy_revenue"] = _md(sp.round(3).reset_index())
    aud = pd.read_csv(p / "policy_selection_by_group_audit.csv")
    t["policy_audit"] = _md(aud, {"share_selected_B_pd_threshold": ".3f",
                                  "share_selected_C_expected_contribution": ".3f"})
    return t


def inject(path, tables: dict[str, str]) -> list[str]:
    text = path.read_text()
    used = []
    for name, body in tables.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
        if pattern.search(text):
            text = pattern.sub(lambda mt, b=body: f"{mt.group(1)}\n{b}\n{mt.group(3)}", text)
            used.append(name)
    path.write_text(text)
    return used


def run_report(cfg: Config) -> dict[str, str]:
    from creditlab.data.sql_analysis import parse_sql_file
    from creditlab.reporting.eda import make_eda_figures
    from creditlab.reporting.model_figures import make_model_figures
    from creditlab.reporting.policy_figures import make_policy_figures
    from creditlab.reporting.risk_figures import make_risk_figures

    sql_dir = cfg.artifacts / "metrics" / "sql"
    results = {q.name: pd.read_csv(sql_dir / f"{q.name}.csv")
               for q in parse_sql_file(cfg.path("sql_file"))}
    make_eda_figures(cfg, results)
    make_model_figures(cfg)
    make_risk_figures(cfg)
    make_policy_figures(cfg)

    tables = build_tables(cfg)
    out = cfg.artifact_dir("metrics") / "report_tables.md"
    out.write_text("# Generated result tables\n\n" + "\n\n".join(
        f"## {k}\n\n{v}" for k, v in tables.items()) + "\n")
    injected = {}
    # Only the research configuration writes into docs/; fast runs stay in their artifact dir.
    if cfg.artifacts.resolve() == (PROJECT_ROOT / "artifacts").resolve():
        # Copy figures next to the report so it renders without the (git-ignored) artifacts.
        fig_dst = PROJECT_ROOT / "docs" / "figures"
        fig_dst.mkdir(parents=True, exist_ok=True)
        for png in (cfg.artifacts / "figures").glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
        for doc in DOCS:
            path = PROJECT_ROOT / doc
            if path.exists():
                injected[doc] = inject(path, tables)
                log.info("Injected %d tables into %s", len(injected[doc]), doc)
    update_manifest(cfg, "report", {"tables": list(tables), "injected": injected})
    return tables
