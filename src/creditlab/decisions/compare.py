"""`policy` stage: compare hypothetical selection policies A-D on the TEST cohort.

Order of operations:
  1. VALIDATION: choose policy B's PD cutoff (observed validation outcomes x assumed economics).
  2. TEST: apply the frozen cutoff and policies A, C, D; report model-based expected values,
     observed default rates, and realized hypothetical contribution separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditlab.data.preprocess import load_split
from creditlab.decisions.policies import (
    POLICY_NAMES,
    PolicyEconomics,
    account_economics,
    break_even_revenue_rate,
    choose_pd_threshold,
    policy_expected_contribution,
    policy_pd_threshold,
    policy_random,
    policy_retain_all,
    summarize,
)
from creditlab.models.audit import group_labels
from creditlab.risk.simulation import simulate_portfolio
from creditlab.risk.var_es import expected_shortfall, value_at_risk
from creditlab.utils.config import Config, read_json, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)
TAIL_RHO = 0.04  # same common-factor assumption as the base_correlated scenario


def _cohort(cfg: Config, split: str, primary: str) -> pd.DataFrame:
    preds = pd.read_parquet(cfg.artifacts / "predictions" / f"{split}_predictions.parquet")
    preds = preds[["ID", "y", f"p_{primary}"]].rename(columns={f"p_{primary}": "pd"})
    df = load_split(cfg, split)
    return df.merge(preds, on="ID", how="inner", validate="one_to_one")


def evaluate_policies(cfg: Config, econ: PolicyEconomics, val: pd.DataFrame, test: pd.DataFrame,
                      with_tail: bool = True) -> dict:
    pcfg = cfg["policy"]
    v = account_economics(val["pd"], val["LIMIT_BAL"], val["BILL_AMT1"], econ, val["y"])
    choice = choose_pd_threshold(v, pcfg["pd_threshold_grid_size"], rule=pcfg["b_rule"],
                                 target_bad_rate=pcfg["b_target_bad_rate"])
    t = account_economics(test["pd"], test["LIMIT_BAL"], test["BILL_AMT1"], econ, test["y"])

    masks = {
        "A_retain_all": policy_retain_all(t),
        "B_pd_threshold": policy_pd_threshold(t, choice["threshold"]),
        "C_expected_contribution": policy_expected_contribution(t),
    }
    rows = {k: summarize(t, m) for k, m in masks.items()}

    rng = cfg.rng("policy_random")
    k = int(masks["B_pd_threshold"].sum())
    draws = [summarize(t, policy_random(len(t), k, rng)) for _ in range(pcfg["random_baseline_draws"])]
    dd = pd.DataFrame(draws)
    rows["D_random_equal_size"] = {
        **dd.mean().to_dict(),
        **{f"{c}_p05": float(dd[c].quantile(0.05)) for c in
           ("expected_net_contribution", "realized_hypothetical_contribution", "observed_default_rate")},
        **{f"{c}_p95": float(dd[c].quantile(0.95)) for c in
           ("expected_net_contribution", "realized_hypothetical_contribution", "observed_default_rate")},
    }
    masks["D_random_equal_size"] = policy_random(len(t), k, cfg.rng("policy_random_tail"))

    if with_tail:
        for name, m in masks.items():
            sel = t[m]
            if sel.empty:  # retaining nobody is a valid (degenerate) hypothetical outcome
                rows[name]["loss_var_99_rho_0.04"] = 0.0
                rows[name]["loss_es_99_rho_0.04"] = 0.0
                continue
            res = simulate_portfolio(sel["pd"].to_numpy(),
                                     econ.risk.lgd * sel["ead"].to_numpy(),
                                     n_sims=pcfg["policy_tail_sims"], rho=TAIL_RHO,
                                     seed=cfg.sub_seed("policy_tail"),
                                     chunk_size=cfg["simulation"]["chunk_size"])
            rows[name]["loss_var_99_rho_0.04"] = value_at_risk(res.losses, 0.99)
            rows[name]["loss_es_99_rho_0.04"] = expected_shortfall(res.losses, 0.99)
            if name == "D_random_equal_size":
                rows[name]["tail_note"] = "tail metrics for one random draw"
    table = pd.DataFrame(rows).T
    table.index.name = "policy"
    table.insert(0, "description", [POLICY_NAMES[i] for i in table.index])
    return {"threshold": choice, "table": table, "masks": masks, "test_frame": t}


def pcfg_sens(cfg: Config) -> list[float]:
    return list(cfg["policy"]["revenue_rate_sensitivity"])


def run_policy(cfg: Config) -> dict:
    sel_file = cfg.artifacts / "metrics" / "selection.json"
    if not sel_file.exists():
        raise FileNotFoundError("Run `creditlab evaluate` first.")
    primary = read_json(sel_file)["primary_model"]
    val, test = _cohort(cfg, "validation", primary), _cohort(cfg, "test", primary)
    econ = PolicyEconomics.from_config(cfg)
    out = cfg.artifact_dir("metrics") / "policy"
    out.mkdir(parents=True, exist_ok=True)

    res = evaluate_policies(cfg, econ, val, test)
    res["table"].to_csv(out / "policy_comparison.csv")
    write_json(out / "policy_threshold_selection.json", {
        "selected_on": "validation split (observed outcomes x assumed economics)",
        "model": primary, "economics": {
            "monthly_revenue_rate": econ.monthly_revenue_rate,
            "monthly_cost_per_account": econ.monthly_cost_per_account,
            **econ.risk.describe()},
        **res["threshold"],
    })

    # Break-even monthly revenue rate per test account (what r would make E[contribution] = 0)
    t = res["test_frame"]
    be = break_even_revenue_rate(t["pd"], test["BILL_AMT1"], t["ead"], econ.risk.lgd,
                                 econ.monthly_cost_per_account)
    finite = be[np.isfinite(be)]
    write_json(out / "break_even_revenue_rate.json", {
        "definition": "r* = (PD*LGD*EAD + c) / ((1-PD)*max(BILL_AMT1,0)); one-month horizon",
        "assumed_rate": econ.monthly_revenue_rate,
        "n_accounts": int(be.size), "n_without_drawn_balance": int((~np.isfinite(be)).sum()),
        "quantiles": {str(q): float(np.quantile(finite, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75)},
        "max_break_even_pd": float(t["break_even_pd"].max()),
        "min_model_pd": float(t["pd"].min()),
        "share_positive_expected_contribution": float((t["expected_contribution"] > 0).mean()),
    })

    # Descriptive selection rates by sensitive group (audit only; groups are not policy inputs)
    labels = group_labels(test)
    audit = []
    for attr, lab in labels.items():
        for g in sorted(lab.unique()):
            m = (lab == g).to_numpy()
            audit.append({"attribute": attr, "group": g, "n": int(m.sum()),
                          **{f"share_selected_{p}": float(res["masks"][p][m].mean())
                             for p in ("B_pd_threshold", "C_expected_contribution")}})
    pd.DataFrame(audit).to_csv(out / "policy_selection_by_group_audit.csv", index=False)

    # Sensitivity to the assumed revenue rate (B re-selected on validation each time)
    sens = []
    for r in pcfg_sens(cfg):
        e = PolicyEconomics.from_config(cfg, monthly_revenue_rate=r)
        rr = evaluate_policies(cfg, e, val, test, with_tail=False)
        for pol, row in rr["table"].iterrows():
            sens.append({"monthly_revenue_rate": r, "policy": pol,
                         "share_selected": row["share_selected"],
                         "expected_net_contribution": row["expected_net_contribution"],
                         "realized_hypothetical_contribution": row["realized_hypothetical_contribution"],
                         "b_threshold": rr["threshold"]["threshold"]})
    pd.DataFrame(sens).to_csv(out / "policy_revenue_sensitivity.csv", index=False)

    from creditlab.reporting.policy_figures import make_policy_figures

    make_policy_figures(cfg)
    update_manifest(cfg, "policy", {"model": primary, "b_threshold": res["threshold"]["threshold"]})
    tab = res["table"]
    for pol, row in tab.iterrows():
        log.info("%-24s selected %.3f | mean PD %.3f | exp. net %.0f | realized hyp. %.0f", pol,
                 row["share_selected"], row["mean_pd"], row["expected_net_contribution"],
                 row["realized_hypothetical_contribution"])
    return res



