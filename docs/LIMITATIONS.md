# Limitations and Ethical Considerations

## Data

- **Selection bias.** The 30,000 rows are *existing* cardholders of one bank in Taiwan in 2005.
  Each was already approved, and their credit limits were set by past lending decisions. There
  are no rejected or new applicants, so nothing here estimates how a model would behave on an
  applicant population (the reject-inference problem). Limits, balances and default outcomes all
  depend on unobserved past decisions.
- **Single outcome month.** There is one observation window (April–September 2005) and one outcome
  month. A chronological out-of-time test is impossible. The test split measures generalisation to
  other clients *from the same period*, not to later periods or other economic conditions.
- **Undocumented codes.** Repayment-status codes -2 and 0 (66.9% of month-records), EDUCATION
  0/5/6 and MARRIAGE 0 are not documented. Code 1 is nearly absent outside September, which
  suggests the coding changed over time. The models use these values as observed categories
  without interpreting them.
- **Definition of default.** The source does not define "default payment" operationally, for
  example as days past due or as a charge-off.
- **Unusual default rate.** A 22% one-month default rate is very high for a card portfolio. The
  source does not say how the sample was drawn, so it may not represent any real portfolio's
  base rate.
- **No financial outcomes.** There is no observed LGD, EAD, recovery, write-off, revenue, cost or
  macroeconomic data. Every money figure in this project is hypothetical.

## Modelling

- Moderate discrimination: test ROC-AUC is about 0.78–0.79. Many defaulters receive low PDs and
  many non-defaulters high ones.
- Feature importance and coefficients describe *predictive association*, not causal effects.
  For example, a higher credit limit is associated with lower default here partly *because* the
  bank gave higher limits to clients it judged safer.
- Hyperparameter search is a finite random/grid search. Better configurations may exist.
- Calibration was verified on validation and test from the same period. Calibration can drift
  when the population or economy changes.

## Risk engine and simulation

- PD estimation error is not propagated into VaR/ES. The simulated tails treat PDs as exact.
- The one-factor Gaussian copula has no tail dependence. It understates joint extreme defaults
  compared with, for example, t-copulas, and a single ρ for all accounts is a strong
  simplification.
- LGD and EAD are fixed per scenario, and the empirical tendency for LGD to rise in downturns
  (PD–LGD correlation) is ignored, except crudely in the severe scenario.
- Stress multipliers are assumed, not estimated from any macroeconomic link (none is available
  for 2005). The scenarios are what-if calculations.
- Monte Carlo VaR/ES carry simulation error. Batch-means standard errors are reported; they are
  approximate.

## Decision policies

- **Not a lending or account-management system.** "Retain" means inclusion in a hypothetical
  cohort for research comparison. No output is a decision, adverse-action reason or
  recommendation about any person.
- **Not causal.** Comparing policies on the test cohort uses outcomes observed *under the bank's
  actual 2005 treatment of those accounts*. If accounts had been treated differently (limits cut,
  accounts closed), outcomes could have differed. Policy results are not proven profit
  improvements.
- The one-month economics are deliberately simple and lopsided: one month of margin against a
  one-off loss. Under the assumed values, no account has positive expected contribution.
- **Disparate impact.** Sensitive attributes are excluded from the model, but outcomes still
  differ by group because other inputs correlate with them. For example, Policy B retains
  different shares of education and age groups (`artifacts/metrics/policy/policy_selection_by_group_audit.csv`),
  and calibration-in-the-large differs slightly by group (`artifacts/metrics/audit_test_primary_model.csv`).
  Excluding attributes is not the same as fairness. Groups with fewer than 30 defaults or
  non-defaults get no AUC.
- **Legal and regulatory.** Real credit decisions are subject to fair-lending, consumer-protection,
  model-risk-management and explainability requirements in the relevant jurisdiction, none of
  which this project addresses.

## What real deployment would require

Applicant-level data with reject inference; multi-period data with out-of-time and
through-the-cycle validation; observed LGD/EAD/revenue for the economics; independent model
validation; fairness testing against legal standards; monitoring for drift; documented
adverse-action reasoning; and governance sign-off. This project does none of these and should not
be read as evidence that a model is fit for lending.
