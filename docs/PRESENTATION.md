# Presentation material

## Two-minute demo script

*Setup: `streamlit run app/streamlit_app.py`; open the Overview page.*

**0:00 – Framing (Overview).**
"CREDITLAB asks how far you can trust a credit model's probabilities, and what they imply for
portfolio losses. It uses public data on 30,000 Taiwanese cardholders from 2005. Only the default
probability is learned from data. Every money number rests on assumptions I state explicitly, and
it isn't a lending system."

**0:20 – Protocol (Overview → Model Comparison).**
"I split 60/20/20 by client. Tuning used training cross-validation only. Every choice — model,
calibration, threshold — was made on validation and frozen before the test set was touched. The
pipeline checks this with a hash of the training IDs. On test, XGBoost reaches AUC 0.790 against
0.780 for logistic regression. A paired bootstrap puts the gap at +0.010, CI 0.004 to 0.016: small
but real. The dummy model gets 78% accuracy with no skill, which is why I select on log loss
rather than accuracy."

**0:50 – Calibration.**
"I fitted Platt and isotonic calibrators on out-of-fold training predictions. On validation
neither beat the raw models by more than bootstrap noise, so I kept the raw ones. An early version
of my rule accepted a −0.00004 log-loss 'improvement'; I tightened it and documented the change."

**1:10 – Portfolio risk and stress (Portfolio Risk → Stress Testing).**
"With an assumed 60% LGD and drawn-plus-half-undrawn exposure, expected one-month loss is
NTD 70 million. With independent defaults, 99% VaR is only 7% above that. Now I drag the
correlation slider to 0.04: the mean doesn't move, but VaR jumps to about 114 million and the
median falls. Correlation, not the PD model, drives the tail. Tests check that simulated marginal
PDs match the inputs and that large-portfolio quantiles match the Vasicek formula."

**1:40 – Policy lab and limits (Policy Lab).**
"A PD cutoff chosen on validation keeps 58% of the test cohort, which defaults at 9.9%, versus 22%
for random cohorts of the same size. That is the model's ranking value. Under one-month economics
no account is profitable at any plausible margin, and these are observational comparisons on
existing accounts, so none of this is causal or a lending recommendation. The limitations doc
covers selection bias, the missing out-of-time test and group differences."

## Candidate resume bullets

Both use only verified implementation details and measured results from the research run.

1. Built an end-to-end, tested Python credit-risk pipeline (pandas, DuckDB SQL, scikit-learn,
   XGBoost) on 30,000 credit-card accounts, with leakage-checked train/validation/test selection.
   XGBoost reached test ROC-AUC 0.790 (95% bootstrap CI 0.777–0.804) against 0.780 for logistic
   regression; out-of-fold calibration showed no bootstrap-supported gain, so the raw models were
   kept.

2. Implemented a vectorised Monte Carlo engine (independent and one-factor Gaussian copula
   defaults, tie-correct VaR/ES), validated against closed-form moments and the Vasicek limit.
   It showed that default correlation of 0.04 raises 99% VaR on a hypothetical one-month
   portfolio by 52% with expected loss unchanged. Served the results in a Streamlit dashboard
   backed by 100+ pytest checks.
