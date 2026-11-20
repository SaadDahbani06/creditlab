"""Research extension: how does PD *estimation* uncertainty propagate into tail-risk estimates?

Status: post-hoc, exploratory (decision D21). Designed after every original test result was known.

Design (details and derivations in docs/RESEARCH_EXTENSION.md)
-------------------------------------------------------------
Outer level: model-estimation uncertainty.
    Draw B nonparametric bootstrap samples of the TRAINING split (n = 18,000, sampled with
    replacement). Refit the frozen model specification (the primary model's family, its
    hyperparameters chosen by the original training-split CV, the same preprocessing pipeline)
    on each sample. All fitted preprocessing lives inside the pipeline, so it is re-estimated on
    each bootstrap sample and never sees validation or test rows. This yields B PD vectors
    p^(1..B) for the same 6,000-account test portfolio.

Inner level: default randomness.
    For each PD vector, simulate portfolio losses with the existing one-factor engine. All PD
    vectors share the SAME factor and idiosyncratic draws (common random numbers), so
    differences between fitted models are not contaminated by between-run sampling noise.

Quantities
    * distribution across models of EL, VaR_a, ES_a  ->  "how reliable is a reported VaR?"
    * the mixture (parameter-averaged) loss distribution  F_mix(l) = (1/B) sum_b F(l | p^(b))
      and its VaR/ES  ->  "risk measured with PD uncertainty included"
    * law of total variance  Var L = E_b[Var(L | p^b)] + Var_b(E[L | p^b])
      ->  share of loss variance from estimation vs default randomness

Sensitivities: seed-only refits (algorithmic noise), re-tuning hyperparameters inside each
bootstrap sample (model-selection noise), a correlation grid, and convergence in B and in the
number of inner paths. A semi-analytical one-factor calculation gives noise-free reference values
used to measure how much of the across-model spread is inner Monte Carlo noise.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from creditlab.data.preprocess import TARGET, load_split, model_input_columns
from creditlab.models.tune import logistic_pipeline, xgboost_pipeline
from creditlab.risk.assumptions import RiskAssumptions, exposure_at_default
from creditlab.risk.simulation import one_factor_loss_quantiles, simulate_portfolio_multi
from creditlab.risk.var_es import batch_standard_error, expected_shortfall, value_at_risk
from creditlab.utils.config import Config, read_json, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)

PIPELINES = {"logistic": logistic_pipeline, "xgboost": xgboost_pipeline}


# ----------------------------------------------------------------------------- outer level
def bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) row indices drawn with replacement (nonparametric bootstrap)."""
    if n < 1 or n_boot < 1:
        raise ValueError("n and n_boot must be positive")
    return rng.integers(0, n, size=(n_boot, n))


def frozen_params(cfg: Config, family: str) -> dict:
    """Hyperparameters chosen by the ORIGINAL training-split CV (never re-chosen on test)."""
    meta = read_json(cfg.artifact_dir("models") / "training_metadata.json")
    return dict(meta["models"][f"{family}_raw"]["params"])


def _fit_predict(cfg: Config, family: str, params: dict, X: pd.DataFrame, y: pd.Series,
                 idx: np.ndarray | None, evals: dict[str, pd.DataFrame],
                 seed: int | None = None) -> dict[str, np.ndarray]:
    model = PIPELINES[family](cfg, **params)
    if seed is not None and family == "xgboost":
        model.set_params(clf__random_state=seed)
    Xf, yf = (X, y) if idx is None else (X.iloc[idx], y.iloc[idx])
    model.fit(Xf, yf)
    out = {k: model.predict_proba(v)[:, 1] for k, v in evals.items()}
    out["train_default_rate"] = np.array([float(yf.mean())])
    return out


def _retune_fit_predict(cfg: Config, X, y, ids, idx, evals, seed: int) -> dict[str, np.ndarray]:
    """Re-run the XGBoost hyperparameter search INSIDE one bootstrap sample.

    Bootstrap samples contain duplicated accounts. Ordinary K-fold CV would put copies of the same
    account in both the fitting and the scoring folds (within-sample leakage that favours
    over-fitted models), so folds are grouped by account ID.
    """
    ucfg = cfg["uncertainty"]["retune"]
    xcfg = cfg["models"]["xgboost"]
    Xb, yb, gb = X.iloc[idx], y.iloc[idx], np.asarray(ids)[idx]
    dist = {f"clf__{k}": list(v) for k, v in xcfg["param_distributions"].items()}
    ratio = float((yb == 0).sum() / (yb == 1).sum())
    dist["clf__scale_pos_weight"] = [ratio if v == "balanced" else float(v)
                                     for v in dist["clf__scale_pos_weight"]]
    search = RandomizedSearchCV(
        xgboost_pipeline(cfg), dist, n_iter=ucfg["n_iter"], scoring="neg_log_loss",
        cv=StratifiedGroupKFold(n_splits=ucfg["cv_folds"], shuffle=True, random_state=seed),
        random_state=seed, n_jobs=1, error_score="raise", refit=True,
    )
    search.fit(Xb, yb, groups=gb)
    out = {k: search.best_estimator_.predict_proba(v)[:, 1] for k, v in evals.items()}
    out["train_default_rate"] = np.array([float(yb.mean())])
    out["best_params"] = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    return out


def _stack(results: list[dict], key: str) -> np.ndarray:
    return np.vstack([r[key] for r in results])


# ----------------------------------------------------------------------------- PD summaries
def pd_uncertainty_summary(P: np.ndarray, p_hat: np.ndarray) -> dict:
    """How much do individual PDs move across fitted models?

    P: (B, n) bootstrap PDs for the same accounts; p_hat: (n,) PDs of the original fit.
    The across-model variance of each account's PD is split into a COMMON part (the shift of the
    portfolio-mean PD, shared by all accounts in a replicate) and an ACCOUNT-SPECIFIC remainder.
    Only the common part survives aggregation into portfolio expected loss.
    """
    P = np.asarray(P, float)
    sd = P.std(axis=0, ddof=1)
    lo, hi = np.quantile(P, [0.05, 0.95], axis=0)
    rel = sd / np.maximum(p_hat, 1e-12)
    mean_pd_b = P.mean(axis=1)                        # portfolio-mean PD per replicate
    common = (mean_pd_b - mean_pd_b.mean())[:, None]  # (B, 1)
    resid = P - P.mean(axis=0) - common
    total_var = float(np.mean(sd ** 2))
    common_var = float(np.var(mean_pd_b, ddof=1))
    return {
        "n_models": int(P.shape[0]), "n_accounts": int(P.shape[1]),
        "median_sd": float(np.median(sd)), "mean_sd": float(sd.mean()),
        "p90_sd": float(np.quantile(sd, 0.9)), "max_sd": float(sd.max()),
        "median_relative_sd": float(np.median(rel)),
        "p90_relative_sd": float(np.quantile(rel, 0.9)),
        "median_90pct_width": float(np.median(hi - lo)),
        "share_accounts_90pct_width_gt_0.05": float(np.mean(hi - lo > 0.05)),
        "share_accounts_90pct_width_gt_0.10": float(np.mean(hi - lo > 0.10)),
        "mean_pd_original": float(p_hat.mean()),
        "mean_pd_bootstrap_mean": float(mean_pd_b.mean()),
        "mean_pd_bootstrap_sd": float(mean_pd_b.std(ddof=1)),
        "mean_pd_bootstrap_p2_5": float(np.quantile(mean_pd_b, 0.025)),
        "mean_pd_bootstrap_p97_5": float(np.quantile(mean_pd_b, 0.975)),
        "bagging_shift_mean_abs": float(np.mean(np.abs(P.mean(axis=0) - p_hat))),
        "pd_variance_total_mean": total_var,
        "pd_variance_common_share": common_var / total_var if total_var > 0 else np.nan,
        "pd_variance_account_specific_share": float(np.mean(resid.var(axis=0, ddof=1)))
        / total_var if total_var > 0 else np.nan,
    }


def per_account_table(P: np.ndarray, p_hat: np.ndarray, ids) -> pd.DataFrame:
    lo, hi = np.quantile(P, [0.05, 0.95], axis=0)
    return pd.DataFrame({"ID": np.asarray(ids), "pd_original": p_hat, "pd_bootstrap_mean": P.mean(0),
                         "pd_bootstrap_sd": P.std(0, ddof=1), "pd_p05": lo, "pd_p95": hi})


# ----------------------------------------------------------------------------- inner level
def risk_metrics(losses: np.ndarray, levels) -> dict:
    out = {"mc_mean": float(losses.mean()), "mc_std": float(losses.std(ddof=1))}
    for a in levels:
        k = f"{round(a * 100)}"
        out[f"var_{k}"] = value_at_risk(losses, a)
        out[f"es_{k}"] = expected_shortfall(losses, a)
    return out


def risk_across_models(P: np.ndarray, w: np.ndarray, rho: float, n_sims: int, seed: int,
                       chunk: int, levels, references: dict[str, np.ndarray],
                       semi_analytic: bool = True) -> dict:
    """Simulate every PD vector (bootstrap rows + reference rows) with common random numbers.

    Returns per-model risk metrics, reference rows (original fit, bagged mean), the mixture
    distribution's risk metrics and the law-of-total-variance decomposition.
    """
    names = list(references)
    allP = np.vstack([*[references[k] for k in names], P])
    L = simulate_portfolio_multi(allP, w, n_sims=n_sims, rho=rho, seed=seed, chunk_size=chunk)
    ref_L, boot_L = L[:len(names)], L[len(names):]

    rows = []
    for b in range(P.shape[0]):
        r = {"model": b, "el": float(P[b] @ w), "mean_pd": float(P[b].mean()),
             **risk_metrics(boot_L[b], levels)}
        if semi_analytic:
            sa = one_factor_loss_quantiles(P[b], w, rho, levels)
            r.update({f"sa_{k}": v for k, v in sa.items()})
        rows.append(r)
    per_model = pd.DataFrame(rows)

    refs = {}
    for i, k in enumerate(names):
        r = {"el": float(references[k] @ w), "mean_pd": float(references[k].mean()),
             **risk_metrics(ref_L[i], levels)}
        for a in levels:
            kk = f"{round(a * 100)}"
            r[f"var_{kk}_mc_se"] = batch_standard_error(ref_L[i], lambda x, a=a: value_at_risk(x, a))
            r[f"es_{kk}_mc_se"] = batch_standard_error(ref_L[i],
                                                      lambda x, a=a: expected_shortfall(x, a))
        if semi_analytic:
            r.update({f"sa_{kk}": v for kk, v in
                      one_factor_loss_quantiles(references[k], w, rho, levels).items()})
        refs[k] = r

    pooled = boot_L.ravel()
    mixture = risk_metrics(pooled, levels)
    # Law of total variance for the equal-weight mixture. Population (ddof = 0) moments make the
    # identity exact for the empirical mixture: Var(pooled) = within + between.
    within = float(np.mean(boot_L.var(axis=1)))
    between = float(np.var(boot_L.mean(axis=1)))
    decomposition = {
        "within_model_variance": within,          # E_b[Var(L | p^b)]: default randomness
        "between_model_variance": between,         # Var_b(E[L | p^b]): PD estimation uncertainty
        "share_between": between / (within + between),
        "between_model_variance_analytic_el": float(np.var(P @ w)),
    }
    return {"per_model": per_model, "references": refs, "mixture": mixture,
            "decomposition": decomposition, "boot_losses": boot_L}


def summarize_across_models(per_model: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        v = per_model[c].to_numpy()
        rows.append({"metric": c, "mean": v.mean(), "sd": v.std(ddof=1),
                     "p2_5": np.quantile(v, 0.025), "p97_5": np.quantile(v, 0.975),
                     "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def mc_noise_diagnostic(per_model: pd.DataFrame, metric: str) -> dict:
    """How much of the across-model spread of a Monte Carlo risk estimate is inner MC noise?

    With common random numbers, MC_b = SA_b + e + d_b, where e is noise shared by every model
    (same draws) and d_b is the model-specific remainder. sd(MC - SA) across models estimates
    sd(d_b); comparing it with sd(SA) shows whether the across-model spread is real.
    """
    mc, sa = per_model[metric].to_numpy(), per_model[f"sa_{metric}"].to_numpy()
    resid = mc - sa
    return {"metric": metric, "sd_across_models_mc": float(mc.std(ddof=1)),
            "sd_across_models_semi_analytic": float(sa.std(ddof=1)),
            "common_offset_mc_minus_sa": float(resid.mean()),
            "sd_model_specific_mc_noise": float(resid.std(ddof=1)),
            "corr_mc_sa": float(np.corrcoef(mc, sa)[0, 1])}


def semi_analytic_mixture(P: np.ndarray, w: np.ndarray, rho: float, levels=(0.95, 0.99),
                          n_grid: int = 2001) -> dict:
    """Noise-free VaR/ES of the equal-weight mixture  G_mix = (1/B) sum_b G_b.

    Each G_b is the conditional-normal one-factor CDF (see one_factor_loss_quantiles); the
    mixture CDF is their average, so its quantiles carry no Monte Carlo noise. Used to decide
    whether "mixture VaR minus fixed VaR" differences are real.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    if rho == 0:
        F, wF = np.array([0.0]), np.array([1.0])
    else:
        F = np.linspace(-8.0, 8.0, n_grid)
        wF = norm.pdf(F)
        wF /= wF.sum()
    ms, ss = [], []
    for p in np.atleast_2d(P):
        pf = (p[None, :] if rho == 0 else
              norm.cdf((norm.ppf(p)[None, :] - np.sqrt(rho) * F[:, None]) / np.sqrt(1 - rho)))
        ms.append(pf @ w)
        ss.append(np.sqrt(np.maximum((pf * (1 - pf)) @ (w ** 2), 1e-12)))
    m, s = np.array(ms), np.array(ss)                       # (B, n_grid)

    def cdf(x):
        return float(np.mean(norm.cdf((x - m) / s) @ wF))

    out = {"mean": float(np.mean(m @ wF))}
    hi = m.max() + 12 * s.max()
    for a in levels:
        q = brentq(lambda x, a=a: cdf(x) - a, 0.0, hi)
        z = (q - m) / s
        excess = float(np.mean((s * norm.pdf(z) + (m - q) * norm.sf(z)) @ wF))
        k = f"{round(a * 100)}"
        out[f"var_{k}"], out[f"es_{k}"] = float(q), float(q + excess / (1 - a))
    return out


def noise_free_comparison(cfg: Config) -> dict:
    """Fixed vs bagged-mean vs mixture risk measures without Monte Carlo noise (primary family)."""
    d = cfg.artifacts / "metrics" / "uncertainty"
    ucfg = cfg["uncertainty"]
    sel = read_json(cfg.artifacts / "metrics" / "selection.json")
    fam = sel["primary_family"]
    z = np.load(d / f"pd_bootstrap_{fam}.npz")
    P, p_hat = z["test"].astype(float), z["original_test"]
    port = _portfolio(cfg, f"p_{sel['primary_model']}")
    if not np.array_equal(port["ID"].to_numpy(), z["ids"]):
        raise RuntimeError("portfolio order changed since the bootstrap run")
    a = RiskAssumptions.from_config(cfg, lgd=ucfg["lgd"])
    w = a.lgd * exposure_at_default(port["LIMIT_BAL"], port["BILL_AMT1"], a)
    out = {}
    for rho in ucfg["rho_main"]:
        fixed = one_factor_loss_quantiles(p_hat, w, rho, ucfg["levels"])
        bagged = one_factor_loss_quantiles(P.mean(axis=0), w, rho, ucfg["levels"])
        mix = semi_analytic_mixture(P, w, rho, ucfg["levels"])
        out[f"rho_{rho}"] = {
            "fixed": fixed, "bagged_mean": bagged, "mixture": mix,
            "bagging_shift_var_99": bagged["var_99"] - fixed["var_99"],
            "spread_effect_var_99": mix["var_99"] - bagged["var_99"],
            "total_change_var_99": mix["var_99"] - fixed["var_99"],
            "spread_effect_es_99": mix["es_99"] - bagged["es_99"],
        }
    write_json(d / "noise_free_comparison.json", out)
    return out


def mechanism_summary(cfg: Config) -> dict:
    """Why do refitted models disagree? Computed from saved extension artifacts only.

    (1) How much of the variation in the refitted portfolio-mean PD is explained by the default
        rate of the bootstrap training sample (a pure sampling quantity)?
    (2) How much of the across-model variation in 99% VaR is explained by expected loss alone,
        and does the remainder track the cross-sectional dispersion of the PDs?
    Uses the noise-free semi-analytic VaR so Monte Carlo noise does not enter.
    """
    d = cfg.artifacts / "metrics" / "uncertainty"
    out: dict = {}
    for fam in cfg["uncertainty"]["families"]:
        P = np.load(d / f"pd_bootstrap_{fam}.npz")["test"].astype(float)
        rate = pd.read_csv(d / f"bootstrap_model_validation_{fam}.csv").train_default_rate.to_numpy()
        mean_pd = P.mean(axis=1)
        coef = np.polyfit(rate, mean_pd, 1)
        r2 = 1 - np.var(mean_pd - np.polyval(coef, rate)) / np.var(mean_pd)
        f = {"corr_train_default_rate_vs_mean_pd": float(np.corrcoef(rate, mean_pd)[0, 1]),
             "r2_mean_pd_on_train_default_rate": float(r2), "slope": float(coef[0]),
             "sd_train_default_rate": float(rate.std(ddof=1)),
             "sd_portfolio_mean_pd": float(mean_pd.std(ddof=1)), "var99": {}}
        for rho in cfg["uncertainty"]["rho_main"]:
            pm = pd.read_csv(d / f"risk_per_model_{fam}_rho{rho}.csv")
            c = np.polyfit(pm.el, pm.sa_var_99, 1)
            res = pm.sa_var_99 - np.polyval(c, pm.el)
            f["var99"][f"rho_{rho}"] = {
                "r2_on_el": float(1 - res.var() / pm.sa_var_99.var()),
                "residual_sd": float(res.std(ddof=1)),
                "corr_residual_with_pd_dispersion": float(np.corrcoef(res, P.std(axis=1))[0, 1]),
                "sd_var99_over_sd_el": float(pm.sa_var_99.std(ddof=1) / pm.el.std(ddof=1)),
            }
        out[fam] = f
    write_json(d / "mechanism.json", out)
    return out


# ----------------------------------------------------------------------------- run
def _portfolio(cfg: Config, primary_preds: str):
    test = load_split(cfg, "test")
    preds = pd.read_parquet(cfg.artifacts / "predictions" / "test_predictions.parquet")
    port = test.merge(preds[["ID", "y", primary_preds]], on="ID", how="inner", validate="one_to_one")
    if len(port) != len(test):
        raise RuntimeError("prediction / test split mismatch")
    return port


def run_uncertainty(cfg: Config) -> dict:
    ucfg = cfg["uncertainty"]
    out = cfg.artifact_dir("metrics") / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    sel = read_json(cfg.artifacts / "metrics" / "selection.json")
    for fam in ucfg["families"]:
        if sel["families"][fam]["chosen_variant"] != "raw":
            raise NotImplementedError(
                f"{fam}: frozen variant is {sel['families'][fam]['chosen_variant']}; the extension "
                "currently propagates uncertainty for uncalibrated (raw) models only")
    primary_family = sel["primary_family"]

    train = load_split(cfg, "train")
    val = load_split(cfg, "validation")
    feats = model_input_columns(cfg)
    X, y = train[feats], train[TARGET].astype(int)
    port = _portfolio(cfg, f"p_{sel['primary_model']}")
    evals = {"test": port[feats], "validation": val[feats]}
    y_val = val[TARGET].to_numpy(int)

    a = RiskAssumptions.from_config(cfg, lgd=ucfg["lgd"])
    w = a.lgd * exposure_at_default(port["LIMIT_BAL"], port["BILL_AMT1"], a)
    levels = ucfg["levels"]
    n_boot, n_jobs = ucfg["n_bootstrap"], ucfg["n_jobs"]
    idx = bootstrap_indices(len(train), n_boot, cfg.rng("pd_bootstrap"))
    chunk = cfg["simulation"]["chunk_size"]
    sim_seed = cfg.sub_seed("simulation")      # same draws as the original scenarios

    summary: dict = {"status": "post-hoc exploratory analysis (decision D21)",
                     "portfolio": "test split (labels used only descriptively)",
                     "n_bootstrap": n_boot, "inner_sims": ucfg["inner_sims"],
                     "lgd": a.lgd, "ead": a.describe()["ead_formula"], "families": {}}
    pd_sets: dict[str, dict[str, np.ndarray]] = {}

    # ---- outer level: refit every family on bootstrap samples (fixed hyperparameters)
    for fam in ucfg["families"]:
        t0 = time.perf_counter()
        params = frozen_params(cfg, fam)
        original = _fit_predict(cfg, fam, params, X, y, None, evals)
        saved = pd.read_parquet(cfg.artifacts / "predictions" / "test_predictions.parquet")
        saved = port[["ID"]].merge(saved, on="ID")[f"p_{fam}_raw"].to_numpy()
        repro_err = float(np.max(np.abs(original["test"] - saved)))
        res = Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict)(cfg, fam, params, X, y, idx[b], evals) for b in range(n_boot))
        P, Pv = _stack(res, "test"), _stack(res, "validation")
        np.savez_compressed(out / f"pd_bootstrap_{fam}.npz", test=P.astype(np.float32),
                            validation=Pv.astype(np.float32), original_test=original["test"],
                            ids=port["ID"].to_numpy())
        perf = pd.DataFrame({
            "model": range(n_boot),
            "train_default_rate": _stack(res, "train_default_rate")[:, 0],
            "val_auc": [roc_auc_score(y_val, p) for p in Pv],
            "val_log_loss": [log_loss(y_val, p) for p in Pv],
        })
        perf.to_csv(out / f"bootstrap_model_validation_{fam}.csv", index=False)
        s = pd_uncertainty_summary(P, original["test"])
        s.update({
            "params": params,
            "refit_reproduces_saved_predictions_max_abs_diff": repro_err,
            "validation_auc_original": float(roc_auc_score(y_val, original["validation"])),
            "validation_auc_bootstrap_p2_5": float(perf.val_auc.quantile(0.025)),
            "validation_auc_bootstrap_p97_5": float(perf.val_auc.quantile(0.975)),
            "validation_log_loss_original": float(log_loss(y_val, original["validation"])),
            "validation_log_loss_bootstrap_p2_5": float(perf.val_log_loss.quantile(0.025)),
            "validation_log_loss_bootstrap_p97_5": float(perf.val_log_loss.quantile(0.975)),
            "fit_seconds": time.perf_counter() - t0,
        })
        summary["families"][fam] = {"pd": s}
        pd_sets[fam] = {"P": P, "p_hat": original["test"]}
        per_account_table(P, original["test"], port["ID"]).to_parquet(
            out / f"pd_per_account_{fam}.parquet", index=False)
        log.info("%s: %d bootstrap refits in %.0fs | median PD sd %.4f | mean-PD sd %.4f | "
                 "refit reproduces saved PDs to %.1e", fam, n_boot, time.perf_counter() - t0,
                 s["median_sd"], s["mean_pd_bootstrap_sd"], repro_err)

    # ---- algorithmic noise: XGBoost refits on the ORIGINAL data with different seeds
    if "xgboost" in ucfg["families"] and ucfg.get("n_seed_refits", 0) > 0:
        params = frozen_params(cfg, "xgboost")
        seeds = [cfg.sub_seed(f"seed_refit_{k}") for k in range(ucfg["n_seed_refits"])]
        res = Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict)(cfg, "xgboost", params, X, y, None, {"test": evals["test"]}, s)
            for s in seeds)
        S = _stack(res, "test")
        pd_sets["xgboost_seed"] = {"P": S, "p_hat": pd_sets["xgboost"]["p_hat"]}
        summary["families"]["xgboost"]["seed_only_pd"] = pd_uncertainty_summary(
            S, pd_sets["xgboost"]["p_hat"])

    # ---- model-selection noise: re-tune hyperparameters inside each bootstrap sample
    rcfg = ucfg.get("retune", {})
    if "xgboost" in ucfg["families"] and rcfg.get("n_bootstrap", 0) > 0:
        t0 = time.perf_counter()
        k = rcfg["n_bootstrap"]
        res = Parallel(n_jobs=n_jobs)(
            delayed(_retune_fit_predict)(cfg, X, y, train["ID"], idx[b], {"test": evals["test"]},
                                         cfg.sub_seed(f"retune_{b}")) for b in range(k))
        R = _stack(res, "test")
        pd_sets["xgboost_retuned"] = {"P": R, "p_hat": pd_sets["xgboost"]["p_hat"]}
        pd_sets["xgboost_fixed_first_k"] = {"P": pd_sets["xgboost"]["P"][:k],
                                           "p_hat": pd_sets["xgboost"]["p_hat"]}
        summary["families"]["xgboost"]["retuned_pd"] = {
            **pd_uncertainty_summary(R, pd_sets["xgboost"]["p_hat"]),
            "paired_fixed_hyperparameters": pd_uncertainty_summary(
                pd_sets["xgboost"]["P"][:k], pd_sets["xgboost"]["p_hat"]),
            "selected_params": [r["best_params"] for r in res],
            "seconds": time.perf_counter() - t0,
        }
        log.info("re-tuned %d bootstrap replicates in %.0fs", k, time.perf_counter() - t0)

    # ---- inner level: risk across fitted models, common random numbers
    regimes = []
    for fam in ucfg["families"]:
        P, p_hat = pd_sets[fam]["P"], pd_sets[fam]["p_hat"]
        refs = {"original_fit": p_hat, "bagged_mean": P.mean(axis=0)}
        for rho in ucfg["rho_main"]:
            t0 = time.perf_counter()
            r = risk_across_models(P, w, rho, ucfg["inner_sims"], sim_seed, chunk, levels, refs)
            tag = f"{fam}_rho{rho}"
            r["per_model"].to_csv(out / f"risk_per_model_{tag}.csv", index=False)
            cols = ["el", "mc_mean", "var_95", "var_99", "es_95", "es_99"]
            across = summarize_across_models(r["per_model"], cols)
            across.to_csv(out / f"risk_across_models_{tag}.csv", index=False)
            fam_s = summary["families"][fam].setdefault("risk", {})
            fam_s[f"rho_{rho}"] = {
                "references": r["references"], "mixture": r["mixture"],
                "decomposition": r["decomposition"],
                "across_models": across.set_index("metric").to_dict(orient="index"),
                "mc_noise": [mc_noise_diagnostic(r["per_model"], m) for m in
                             ("var_95", "var_99", "es_99")],
            }
            for regime_pd in ("fixed", "uncertain"):
                ref = r["references"]["original_fit"]
                regimes.append({
                    "family": fam, "rho": rho, "pd": regime_pd,
                    "el": ref["el"] if regime_pd == "fixed"
                    else float(across.set_index("metric").loc["el", "mean"]),
                    **{m: (ref[m] if regime_pd == "fixed" else r["mixture"][m])
                       for m in ("var_95", "var_99", "es_95", "es_99")},
                    **({f"{m}_p2_5": across.set_index("metric").loc[m, "p2_5"]
                        for m in ("el", "var_95", "var_99", "es_99")} if regime_pd == "uncertain"
                       else {}),
                    **({f"{m}_p97_5": across.set_index("metric").loc[m, "p97_5"]
                        for m in ("el", "var_95", "var_99", "es_99")} if regime_pd == "uncertain"
                       else {}),
                })
            if fam == primary_family and rho == max(ucfg["rho_main"]):
                np.save(out / f"boot_losses_{tag}.npy", r["boot_losses"].astype(np.float32))
            if fam == primary_family and rho == min(ucfg["rho_main"]):
                np.save(out / f"boot_losses_{tag}.npy", r["boot_losses"].astype(np.float32))
            log.info("%s rho=%.2f: VaR99 fixed %.2fM | across models [%.2f, %.2f]M | mixture "
                     "%.2fM | between-model share of Var(L) %.3f (%.0fs)", fam, rho,
                     r["references"]["original_fit"]["var_99"] / 1e6,
                     across.set_index("metric").loc["var_99", "p2_5"] / 1e6,
                     across.set_index("metric").loc["var_99", "p97_5"] / 1e6,
                     r["mixture"]["var_99"] / 1e6, r["decomposition"]["share_between"],
                     time.perf_counter() - t0)
    pd.DataFrame(regimes).to_csv(out / "regime_comparison.csv", index=False)

    # ---- sensitivity sets (seed-only, re-tuned vs paired fixed): risk spread at rho_main
    sens = []
    for name in ("xgboost_seed", "xgboost_retuned", "xgboost_fixed_first_k"):
        if name not in pd_sets:
            continue
        for rho in ucfg["rho_main"]:
            r = risk_across_models(pd_sets[name]["P"], w, rho, ucfg["grid_sims"], sim_seed, chunk,
                                   levels, {"original_fit": pd_sets[name]["p_hat"]},
                                   semi_analytic=True)
            pm = r["per_model"]
            sens.append({"set": name, "rho": rho, "n_models": len(pm),
                         "el_sd": pm.el.std(ddof=1), "sa_var_99_sd": pm.sa_var_99.std(ddof=1),
                         "sa_es_99_sd": pm.sa_es_99.std(ddof=1),
                         "sa_var_99_p2_5": pm.sa_var_99.quantile(.025),
                         "sa_var_99_p97_5": pm.sa_var_99.quantile(.975),
                         "mean_pd_sd": pm.mean_pd.std(ddof=1)})
    pd.DataFrame(sens).to_csv(out / "sensitivity_model_sets.csv", index=False)

    # ---- correlation grid (primary family): fixed vs across-model band vs mixture
    P, p_hat = pd_sets[primary_family]["P"], pd_sets[primary_family]["p_hat"]
    grid = []
    for rho in ucfg["rho_grid"]:
        r = risk_across_models(P, w, rho, ucfg["grid_sims"], sim_seed, chunk, levels,
                               {"original_fit": p_hat}, semi_analytic=False)
        pm = r["per_model"]
        grid.append({"rho": rho, "fixed_var_99": r["references"]["original_fit"]["var_99"],
                     "fixed_es_99": r["references"]["original_fit"]["es_99"],
                     "var_99_p2_5": pm.var_99.quantile(.025), "var_99_p97_5": pm.var_99.quantile(.975),
                     "es_99_p2_5": pm.es_99.quantile(.025), "es_99_p97_5": pm.es_99.quantile(.975),
                     "mixture_var_99": r["mixture"]["var_99"], "mixture_es_99": r["mixture"]["es_99"],
                     "share_between": r["decomposition"]["share_between"]})
    pd.DataFrame(grid).to_csv(out / "rho_grid.csv", index=False)

    # ---- convergence: in the number of bootstrap models (noise-free semi-analytic values)
    conv = []
    for rho in ucfg["rho_main"]:
        pm = pd.read_csv(out / f"risk_per_model_{primary_family}_rho{rho}.csv")
        for kk in sorted({*np.unique(np.geomspace(10, len(pm), 12).astype(int)), len(pm)}):
            v = pm.sa_var_99.to_numpy()[:kk]
            conv.append({"rho": rho, "n_models": kk, "var_99_sd": v.std(ddof=1),
                         "var_99_p2_5": np.quantile(v, .025), "var_99_p97_5": np.quantile(v, .975)})
    # ... and in the number of inner paths (mixture VaR99 from the pooled paths)
    for rho in ucfg["rho_main"]:
        L = np.load(out / f"boot_losses_{primary_family}_rho{rho}.npy")
        for m in sorted({*np.unique(np.geomspace(500, L.shape[1], 10).astype(int)), L.shape[1]}):
            conv.append({"rho": rho, "inner_paths": m,
                         "mixture_var_99": value_at_risk(L[:, :m].ravel(), 0.99),
                         "mixture_es_99": expected_shortfall(L[:, :m].ravel(), 0.99)})
    pd.DataFrame(conv).to_csv(out / "convergence.csv", index=False)

    # Observed test defaults vs model-implied expected defaults (descriptive only).
    obs = int(port["y"].sum())
    exp_def = pd_sets[primary_family]["P"].sum(axis=1)
    summary["observed_vs_expected_defaults"] = {
        "observed_test_defaults": obs, "expected_original": float(p_hat.sum()),
        "expected_bootstrap_p2_5": float(np.quantile(exp_def, .025)),
        "expected_bootstrap_p97_5": float(np.quantile(exp_def, .975)),
        "note": "descriptive; test labels are not used for any choice in this extension"}
    write_json(out / "summary.json", summary)
    mechanism_summary(cfg)
    noise_free_comparison(cfg)

    from creditlab.reporting.uncertainty_figures import make_uncertainty_figures

    make_uncertainty_figures(cfg)
    from creditlab.reporting.extension_report import run_extension_report

    run_extension_report(cfg)
    update_manifest(cfg, "uncertainty", {"n_bootstrap": n_boot, "families": ucfg["families"],
                                         "status": "post-hoc exploratory (D21)"})
    return summary
