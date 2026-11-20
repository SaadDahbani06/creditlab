# Assumptions

Each quantity in the project falls into exactly one of five categories. Results always state
which category they belong to.

| Category | Examples | Source |
|----------|----------|--------|
| **Observed historical information** | Repayment status, bills, payments, limits, default label | UCI dataset 350 |
| **Model predictions** | Raw PD from logistic regression / XGBoost | Fitted on the training split |
| **Calibrated probabilities** | Sigmoid / isotonic PDs (evaluated; not selected in the research run) | Training out-of-fold fit |
| **Assumed financial parameters** | LGD, EAD/CCF, revenue rate, cost, stress multipliers, ρ | `configs/default.yaml` (HYPOTHETICAL) |
| **Simulated financial outcomes** | Expected loss, VaR, ES, contributions | Model PDs × assumptions |

## Units and horizon

- Currency: **New Taiwan dollars (NTD)**, as recorded in the source. No conversion or inflation
  adjustment.
- Horizon: **one month**, matching the target (default on the next month's payment). Expected
  losses, simulated losses, revenues and contributions are all one-month quantities.
- Portfolio for risk and policy results: the **test split** (6,000 accounts) with out-of-sample
  PDs from the model selected on validation.

## Hypothetical financial parameters (`configs/default.yaml`)

| Parameter | Value | Why this value | Sensitivity reported |
|-----------|-------|----------------|----------------------|
| LGD (base) | 0.60 | Round mid-range value for unsecured revolving credit; not estimated | 0.30 / 0.60 / 0.90 |
| EAD method | drawn + CCF × undrawn | Uses the observed September balance and the explicit limit; never "EAD = limit" silently | `limit_fraction` alternative |
| CCF | 0.50 | Mid-range assumption for undrawn revolving lines | 0, 0.25, 0.5, 0.75, 1.0 |
| EAD cap | none | Optional | configurable |
| Stress: mild | odds × 1.5 | Illustrative | — |
| Stress: severe | odds × 2.5, LGD 0.75 | Illustrative | — |
| Correlation, base_correlated | ρ = 0.04 | Basel II/III asset-correlation value for qualifying revolving retail exposures, used as a reference point only | ρ grid 0–0.30 |
| Correlation, correlated downturn | ρ = 0.12, odds × 1.5, LGD 0.75 | Illustrative | ρ grid |
| Monthly revenue rate | 1.5% of drawn balance | Roughly an 18% annual rate; illustrative, not net of funding | 0.5%–25% sweep |
| Monthly cost per account | 0 NTD | Not modelled by default | configurable |
| Policy B risk appetite | observed bad rate ≤ 10% (validation) | Illustrative | slider in dashboard |

None of these values is estimated from the dataset, which contains no LGD, EAD, recovery,
revenue, cost or macroeconomic information. They are inputs to a calculation, not findings.

## Modelling assumptions

- Accounts are exchangeable within the cohort, and the client-level split is representative of
  the cohort. There is no time ordering.
- PDs are treated as fixed known inputs to the simulation. Their estimation uncertainty is not
  propagated into VaR/ES.
- LGD and EAD are deterministic per scenario (no LGD–PD correlation, no stochastic exposure).
- Default dependence, when used, follows a single-factor Gaussian copula with one common ρ.
- Stress scenarios move all PDs by the same odds multiplier.
