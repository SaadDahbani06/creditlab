"""HYPOTHETICAL stress scenarios and the `simulate` stage.

Stress transformation (odds multiplier m > 0):

    odds' = m * odds,   i.e.   logit(PD') = logit(PD) + log(m),
    PD'   = m * PD / (1 - PD + m * PD).

This keeps PD' in [0, 1], preserves the ranking of accounts, maps 0 -> 0 and 1 -> 1, and is the
shift a logistic model would produce if its intercept moved by log(m). The multipliers are
ASSUMED scenario inputs, not estimated macroeconomic effects.

Portfolio: the TEST split (6,000 accounts) with out-of-sample PDs from the primary model chosen
on validation. All scenarios use the same random seed (common random numbers), so differences
between scenarios reflect the assumptions rather than Monte Carlo noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditlab.data.preprocess import load_split
from creditlab.risk.assumptions import RiskAssumptions, exposure_at_default
from creditlab.risk.expected_loss import (
    account_table,
    portfolio_summary,
    risk_band_table,
    sensitivity_grid,
)
from creditlab.risk.simulation import simulate_portfolio, vasicek_quantile
from creditlab.risk.var_es import loss_summary, value_at_risk
from creditlab.utils.config import Config, read_json, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)


def stress_pd(pd_, odds_multiplier: float) -> np.ndarray:
    if odds_multiplier <= 0:
        raise ValueError("odds multiplier must be positive")
    p = np.asarray(pd_, dtype=float)
    return odds_multiplier * p / (1.0 - p + odds_multiplier * p)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    odds_multiplier: float
    lgd: float
    rho: float


def scenarios_from_config(cfg: Config) -> list[Scenario]:
    return [Scenario(name=k, **v) for k, v in cfg["simulation"]["scenarios"].items()]


def load_portfolio(cfg: Config) -> pd.DataFrame:
    """Test-split accounts with the primary model's out-of-sample PD and observed outcome."""
    pred_file = cfg.artifacts / "predictions" / "test_predictions.parquet"
    sel_file = cfg.artifacts / "metrics" / "selection.json"
    if not pred_file.exists() or not sel_file.exists():
        raise FileNotFoundError("Predictions missing. Run `creditlab train` and `creditlab evaluate`.")
    primary = read_json(sel_file)["primary_model"]
    preds = pd.read_parquet(pred_file)[["ID", "y", f"p_{primary}"]].rename(
        columns={f"p_{primary}": "pd", "y": "observed_default"})
    test = load_split(cfg, "test")[["ID", "LIMIT_BAL", "BILL_AMT1"]]
    port = test.merge(preds, on="ID", how="inner", validate="one_to_one")
    if len(port) != len(test):
        raise RuntimeError("prediction / test split mismatch")
    port.attrs["model"] = primary
    return port


def run_scenario(port: pd.DataFrame, sc: Scenario, base: RiskAssumptions, n_sims: int,
                 seed: int, chunk: int, levels, thresholds: dict[str, float]):
    a = base.with_(lgd=sc.lgd)
    p = stress_pd(port["pd"].to_numpy(), sc.odds_multiplier)
    ead = exposure_at_default(port["LIMIT_BAL"], port["BILL_AMT1"], a)
    w = a.lgd * ead
    res = simulate_portfolio(p, w, n_sims=n_sims, rho=sc.rho, seed=seed, chunk_size=chunk)
    s = loss_summary(res.losses, levels, thresholds)
    analytic_el = float(np.sum(p * w))
    s.update({
        "scenario": sc.name, "description": sc.description,
        "odds_multiplier": sc.odds_multiplier, "lgd": sc.lgd, "rho": sc.rho,
        "mean_stressed_pd": float(p.mean()),
        "analytical_el": analytic_el,
        "mean_minus_el_in_se": (s["mean"] - analytic_el) / s["mc_se_mean"],
        "mean_default_count": float(res.default_counts.mean()),
        "default_count_p99": float(value_at_risk(res.default_counts, 0.99)),
        "max_abs_marginal_pd_error": float(np.max(np.abs(res.account_default_freq - p))),
        # For rho > 0 most loss variance comes from the common factor; a factor sample mean
        # away from 0 shifts the simulated mean (positive F => fewer defaults). Reported so a
        # (mean - EL) gap can be attributed to factor sampling rather than bias.
        "factor_sample_mean_z": float(res.factor.mean() * np.sqrt(res.factor.size))
        if sc.rho > 0 else np.nan,
    })
    return s, res


def run_simulate(cfg: Config) -> dict:
    sim_cfg, risk_cfg = cfg["simulation"], cfg["risk"]
    out = cfg.artifact_dir("metrics") / "risk"
    out.mkdir(parents=True, exist_ok=True)
    port = load_portfolio(cfg)
    base = RiskAssumptions.from_config(cfg)
    seed = cfg.sub_seed("simulation")
    n_sims, chunk, levels = sim_cfg["n_sims"], sim_cfg["chunk_size"], sim_cfg["var_levels"]

    # ---- Milestone 4: expected loss (analytical)
    acc = account_table(port["pd"], port["LIMIT_BAL"], port["BILL_AMT1"], base)
    acc.insert(0, "ID", port["ID"].to_numpy())
    acc.to_parquet(out / "account_expected_loss.parquet", index=False)
    summary = portfolio_summary(acc)
    realized = float((port["observed_default"] * acc["lgd"] * acc["ead"]).sum())
    summary.update({
        "model": port.attrs["model"], "assumptions": base.describe(),
        "observed_defaults_in_portfolio": int(port["observed_default"].sum()),
        "hypothetical_loss_at_observed_defaults": realized,
        "note": "EL uses model PDs; 'hypothetical_loss_at_observed_defaults' applies the same "
                "ASSUMED LGD/EAD to the OBSERVED test-split defaults. Neither is an observed loss.",
    })
    write_json(out / "expected_loss_summary.json", summary)
    risk_band_table(acc, risk_cfg["risk_bands"], port["observed_default"]).to_csv(
        out / "risk_bands.csv", index=False)
    sensitivity_grid(acc["pd"], port["LIMIT_BAL"], port["BILL_AMT1"], base,
                     risk_cfg["lgd_scenarios"], risk_cfg["ccf_scenarios"]).to_csv(
        out / "el_sensitivity_grid.csv", index=False)
    log.info("Hypothetical EL %.0f NTD on EAD %.0f NTD (EAD-weighted PD %.4f)",
             summary["total_el"], summary["total_ead"], summary["ead_weighted_pd"])

    # ---- Milestone 5: Monte Carlo scenarios
    base_el = summary["total_el"]
    thresholds = {f"{m}x_base_el": m * base_el for m in sim_cfg["exceedance_multiples"]}
    rows, losses, counts = [], {}, {}
    for sc in scenarios_from_config(cfg):
        s, res = run_scenario(port, sc, base, n_sims, seed, chunk, levels, thresholds)
        rows.append(s)
        losses[sc.name] = res.losses
        counts[sc.name] = res.default_counts
        log.info("%-20s mean %.0f | VaR99 %.0f | ES99 %.0f | (mean-EL)/SE %+.2f", sc.name,
                 s["mean"], s["var_99"], s["es_99"], s["mean_minus_el_in_se"])
    scen = pd.DataFrame(rows)
    scen.to_csv(out / "scenario_summary.csv", index=False)
    np.savez_compressed(out / "scenario_losses.npz", **losses)
    np.savez_compressed(out / "scenario_default_counts.npz", **counts)

    # Observed default count vs simulated distribution (calibration-in-the-large check)
    observed = int(port["observed_default"].sum())
    count_check = {
        name: {"observed_defaults": observed,
               "simulated_mean": float(c.mean()),
               "simulated_p2_5": float(np.quantile(c, 0.025)),
               "simulated_p97_5": float(np.quantile(c, 0.975)),
               "share_of_paths_below_observed": float(np.mean(c < observed))}
        for name, c in counts.items() if name in ("base", "base_correlated")
    }
    write_json(out / "observed_vs_simulated_defaults.json", count_check)

    # Convergence of the base scenario
    base_losses = losses["base"]
    pts = np.unique(np.geomspace(100, n_sims, sim_cfg["convergence_points"]).astype(int))
    conv = pd.DataFrame({
        "n_sims": pts,
        "running_mean": [base_losses[:k].mean() for k in pts],
        "running_var_99": [value_at_risk(base_losses[:k], 0.99) for k in pts],
        "running_es_99": [loss_summary(base_losses[:k], (0.99,))["es_99"] for k in pts],
    })
    conv["analytical_el"] = base_el
    conv.to_csv(out / "convergence_base.csv", index=False)

    # Sensitivity to correlation (base PDs, base LGD) and to LGD (independent)
    sens_n = sim_cfg["sensitivity_n_sims"]
    rho_rows = []
    for rho in sim_cfg["rho_grid"]:
        sc = Scenario(f"rho_{rho}", "", 1.0, base.lgd, rho)
        s, _ = run_scenario(port, sc, base, sens_n, seed, chunk, levels, thresholds)
        rho_rows.append(s)
    pd.DataFrame(rho_rows).to_csv(out / "sensitivity_rho.csv", index=False)
    lgd_rows = []
    for lgd in risk_cfg["lgd_scenarios"]:
        for rho in (0.0, 0.04):
            sc = Scenario(f"lgd_{lgd}_rho_{rho}", "", 1.0, lgd, rho)
            s, _ = run_scenario(port, sc, base, sens_n, seed, chunk, levels, thresholds)
            lgd_rows.append(s)
    pd.DataFrame(lgd_rows).to_csv(out / "sensitivity_lgd.csv", index=False)

    # Unbiasedness check for the correlated simulator: repeat the correlated-downturn setting
    # with independent seeds and standardise each mean against the analytical EL. An unbiased
    # estimator gives z-scores with mean ~0 and SD ~1.
    sc = next(x for x in scenarios_from_config(cfg) if x.rho > 0 and x.odds_multiplier > 1)
    n_seeds, n_each = (30, 2_000) if n_sims >= 5_000 else (8, 500)
    zs = []
    for k in range(n_seeds):
        s, _ = run_scenario(port, sc, base, n_each, cfg.sub_seed(f"unbiasedness_{k}"), chunk,
                            levels, thresholds)
        zs.append(s["mean_minus_el_in_se"])
    zs = np.asarray(zs)
    write_json(out / "mc_unbiasedness_check.json", {
        "scenario": sc.name, "n_seeds": n_seeds, "paths_per_seed": n_each,
        "z_mean": float(zs.mean()), "z_sd": float(zs.std(ddof=1)),
        "share_abs_z_gt_2": float(np.mean(np.abs(zs) > 2)), "z": zs.tolist(),
    })

    # Analytical cross-check: Vasicek large-portfolio quantile at the portfolio's mean PD.
    pbar = float(port["pd"].mean())
    vas = [{"rho": r, "alpha": a, "default_rate_quantile": vasicek_quantile(pbar, r, a)}
           for r in sim_cfg["rho_grid"] if r > 0 for a in levels]
    write_json(out / "vasicek_reference.json", {"mean_pd": pbar, "quantiles": vas,
                                                "note": "homogeneous, infinitely granular "
                                                        "approximation; reference only"})

    from creditlab.reporting.risk_figures import make_risk_figures

    make_risk_figures(cfg)
    update_manifest(cfg, "simulate", {
        "n_sims": n_sims, "seed": seed, "portfolio": "test split",
        "model": port.attrs["model"], "scenarios": scen["scenario"].tolist(),
    })
    return {"expected_loss": summary, "scenarios": scen}

