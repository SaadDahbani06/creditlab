# Mathematics of CREDITLAB

This document derives every method used in the project. Each section names the code that
implements it. Notation: account $i = 1,\dots,n$; observed label $Y_i \in \{0,1\}$ (1 = default on
the next month's payment); features $X_i$ (April–September 2005 history); predicted probability
$\hat p_i$.

---

## 1. Problem formulation

The quantity of interest is the conditional default probability

$$\mathrm{PD}_i = P(Y_i = 1 \mid X_i).$$

$Y_i \mid X_i \sim \text{Bernoulli}(\mathrm{PD}_i)$, so the log-likelihood of a model $\hat p(\cdot)$ is

$$\ell = \sum_i \big[\,Y_i \log \hat p_i + (1-Y_i)\log(1-\hat p_i)\,\big].$$

Its negative mean is the **log loss**. Minimising it is maximum likelihood, and it is the
criterion used to tune and select models (`configs/default.yaml → models.selection_metric`).
PDs feed directly into the loss engine, so we want *probabilities*, not just rankings.

**Split.** The 30,000 accounts are split 60/20/20 into train/validation/test, stratified on $Y$
with a fixed seed (`data/preprocess.py::make_split`). Fitting and tuning use training data only.
Every selection (calibration variant, primary model, thresholds, policy cutoff) uses validation
data. The test split is scored once at the end. All accounts share one observation window, so this
is an *out-of-sample client* split, not an out-of-time backtest.

---

## 2. Logistic regression (`models/tune.py::logistic_pipeline`)

$$\log\frac{\hat p_i}{1-\hat p_i} = \beta_0 + \beta^\top z_i,$$

where $z_i$ is the preprocessed feature vector:

- one-hot repayment-status categories (codes $\ge 3$ grouped);
- $\operatorname{sign}(x)\log(1+|x|)$-transformed and standardised money amounts;
- standardised engineered features.

Estimation minimises the penalised negative log-likelihood

$$-\ell(\beta) + \frac{1}{2C}\lVert\beta\rVert_2^2 .$$

$C$ is the inverse regularisation strength, chosen by 5-fold CV on the training split.

**Interpretation.** $\beta_j$ is the change in log-odds for a one-standard-deviation increase in
standardised feature $j$, holding the others fixed, and $e^{\beta_j}$ is the corresponding odds ratio.
With correlated inputs (six monthly bills) individual coefficients are not separately identified
in a meaningful way. With no reference category dropped, the L2 penalty identifies the one-hot
coefficients, so compare categories within a column rather than reading each one alone.

**Why class weights distort probabilities.** Weighting positives by $w$ maximises
$\sum_i [w Y_i\log\hat p_i + (1-Y_i)\log(1-\hat p_i)]$. Its population optimum is the probability
under a re-weighted class prior, which shifts the log-odds by roughly $\log w$. Rankings (AUC)
barely change, but probabilities become inflated. The CV tables (`artifacts/models/cv_results_*.csv`)
show `class_weight="balanced"` and `scale_pos_weight` with *worse* log loss, which is why they were
not selected.

## 3. Gradient-boosted trees (XGBoost) (`models/tune.py::xgboost_pipeline`)

The model is additive in log-odds space, $F(x) = \sum_{m=1}^M f_m(x)$ with $\hat p = \sigma(F)$, and each
$f_m$ is a regression tree. At round $m$ the loss is approximated to second order around the current
prediction. With $g_i = \hat p_i - Y_i$ and $h_i = \hat p_i(1-\hat p_i)$, the objective for a tree with
leaf weights $w_j$ is

$$\sum_j \Big[ G_j w_j + \tfrac12 (H_j + \lambda) w_j^2 \Big] + \gamma T,$$

where $G_j=\sum_{i\in j} g_i$ and $H_j=\sum_{i\in j}h_i$. It is minimised by $w_j^\* = -G_j/(H_j+\lambda)$.
The quality of a split is its **gain**:

$$\tfrac12\Big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma .$$

Learning rate, depth, `min_child_weight` (a minimum $H$ per leaf), row/column subsampling and $\lambda$
were tuned by randomized search with 5-fold training CV. Trees handle the undocumented status codes
naturally: an ordered split can isolate any single code.

## 4. Evaluation metrics (`models/evaluate.py`)

- **ROC-AUC** $= P(\hat p_{\text{pos}} > \hat p_{\text{neg}})$, the Mann–Whitney probability that a
  random defaulter is ranked above a random non-defaulter. It is invariant to any monotone
  transformation of scores, so calibration cannot change it (isotonic can, slightly, through ties).
- **Average precision (PR-AUC)** $=\sum_k (R_k - R_{k-1}) P_k$ over thresholds. The no-skill
  baseline is the prevalence (0.221), not 0.5.
- **Brier score** $=\frac1n\sum(\hat p_i - Y_i)^2$, a proper scoring rule. Murphy decomposition:
  reliability − resolution + uncertainty, where uncertainty $=\bar y(1-\bar y)=0.1723$ is exactly the
  dummy model's Brier score.
- **Log loss**: see §1. Also proper, and it punishes confident errors heavily, which is why isotonic
  calibration (which can output values near 0 or 1) can worsen it.
- **Why accuracy is insufficient.** With 22.1% defaults, predicting "no default" for everyone gives
  77.9% accuracy, and the dummy model achieves exactly that. Accuracy ignores ranking and
  probability quality, and depends on an arbitrary 0.5 threshold.
- **Threshold metrics.** Precision, recall and F1 at the threshold that maximises F1 on
  *validation*, then applied unchanged to test. F1 is a convention, not a cost-optimal rule.

**Bootstrap confidence intervals.** Resample test accounts with replacement $B$ times, recompute
each metric, and report the 2.5% and 97.5% percentiles. For model comparisons the *same* resample
indices are used for both models (**paired bootstrap**), so the interval for the difference
reflects model differences rather than sample noise shared by both.

## 5. Calibration (`models/calibrate.py`)

A model is calibrated if $P(Y=1 \mid \hat p = p) = p$.

- **Reliability diagram**: sort predictions into deciles of $\hat p$ and plot mean $\hat p$ against
  the observed default rate per bin. Error bars: $\pm1.96\sqrt{\bar p(1-\bar p)/n_b}$.
- **ECE** $=\sum_b \frac{n_b}{n}\,|\bar y_b - \bar p_b|$.
- **Calibration intercept and slope**: fit $\operatorname{logit}P(Y=1)=a+b\,\operatorname{logit}(\hat p)$.
  Perfect calibration means $a=0,\ b=1$. $b<1$ means predictions are too extreme (over-confident);
  $b>1$ means too timid.
- **Platt (sigmoid) scaling**: $\tilde p = \sigma(A s + B)$ fitted by maximum likelihood on scores $s$.
- **Isotonic regression**: the non-decreasing step function minimising squared error, fitted by
  pool-adjacent-violators. It is more flexible, needs more data, and creates ties.

**Leakage-free design.** A calibrator fitted on the same rows the model was trained on learns the
model's *in-sample* over-confidence. Instead we use `CalibratedClassifierCV(cv=5, ensemble=False)`
on the training split:

1. Compute out-of-fold scores: each training account is scored by a model that never saw it.
2. Fit the calibrator on those (score, label) pairs.
3. Refit the model on all training data.

`tests/test_calibration.py` reconstructs this by hand and matches to $10^{-12}$. The choice between
raw, sigmoid and isotonic is made on validation. A calibrated variant must lower log loss, must not
raise Brier, and its paired-bootstrap 95% CI for the log-loss change must lie below zero.

## 6. Expected loss (`risk/expected_loss.py`, `risk/assumptions.py`)

Loss on account $i$ over one month: $L_i = D_i\cdot \mathrm{LGD}_i\cdot \mathrm{EAD}_i$ with
$D_i\sim\text{Bernoulli}(\mathrm{PD}_i)$. If LGD and EAD are fixed (non-random) assumptions,

$$\mathrm{EL}_i = E[L_i] = \mathrm{PD}_i\cdot\mathrm{LGD}_i\cdot\mathrm{EAD}_i,\qquad
\mathrm{EL} = E\Big[\sum_i L_i\Big]=\sum_i \mathrm{EL}_i$$

by linearity of expectation, whatever the dependence between defaults. This independence-free
identity is why correlation changes the *tail* but not the mean.

**EAD (hypothetical).**
- Default `balance_plus_ccf`: $\mathrm{EAD}=B+\mathrm{CCF}\cdot\max(\text{limit}-B,0)$ with
  $B=\max(\text{BILL\_AMT1},0)$. Drawn balance plus a share of the unused line assumed to be
  drawn before default.
- Alternative `limit_fraction`: $\mathrm{EAD}=f\cdot\text{limit}$.

**EAD-weighted PD** $=\sum_i \mathrm{PD}_i\mathrm{EAD}_i/\sum_i\mathrm{EAD}_i$, so with a common LGD,
$\mathrm{EL}/\mathrm{EAD} = \mathrm{LGD}\times$ EAD-weighted PD.

## 7. Stress transformation (`risk/stress.py::stress_pd`)

Scaling the odds by $m>0$ gives

$$\frac{p'}{1-p'} = m\frac{p}{1-p}\iff \operatorname{logit}(p')=\operatorname{logit}(p)+\log m
\iff p'=\frac{mp}{1-p+mp}.$$

Properties: $p'\in[0,1]$, $0\mapsto0$, $1\mapsto1$, the ranking is preserved, and it equals
shifting a logistic model's intercept by $\log m$. Adding a constant to $p$ would instead break the
$[0,1]$ bounds and treat a 1% and a 60% account alike. The multipliers are *assumptions*.

## 8. Monte Carlo simulation (`risk/simulation.py`)

**Estimator.** Simulate $S$ independent portfolio paths $L^{(1)},\dots,L^{(S)}$. Then
$\bar L = \frac1S\sum_s L^{(s)}$ is unbiased for $E[L]$ with standard error
$\hat\sigma_L/\sqrt S$ (CLT). Under independence,
$E[L]=\sum_i p_iw_i$ and $\operatorname{Var}[L]=\sum_i p_i(1-p_i)w_i^2$, where $w_i=\mathrm{LGD}\cdot\mathrm{EAD}_i$.
The tests compare both to simulation.

**One-factor Gaussian (Vasicek) model.**

$$Z_i=\sqrt\rho\,F+\sqrt{1-\rho}\,\varepsilon_i,\qquad F,\varepsilon_i\stackrel{iid}{\sim}N(0,1),$$

and default occurs if $Z_i<\Phi^{-1}(\mathrm{PD}_i)$.

- *Marginals preserved*: $Z_i\sim N(0,1)$, so $P(D_i=1)=\Phi(\Phi^{-1}(\mathrm{PD}_i))=\mathrm{PD}_i$ for every $\rho$.
- *Dependence*: $\operatorname{Corr}(Z_i,Z_j)=\rho$, and
  $P(D_i=D_j=1)=\Phi_2(\Phi^{-1}(p_i),\Phi^{-1}(p_j);\rho)$, the bivariate normal CDF. Default
  correlation is positive for $\rho>0$, so $\operatorname{Var}[L]$ gains covariance terms. A test checks
  the default-count variance against this formula.
- *Conditional independence*: given $F=f$, defaults are independent with
  $p_i(f)=\Phi\big((\Phi^{-1}(p_i)-\sqrt\rho f)/\sqrt{1-\rho}\big)$. Low $f$ is a bad state.
- *Vasicek limit*: for a large homogeneous portfolio the default rate converges to $p(F)$, so
  its $\alpha$-quantile is $\Phi\big((\Phi^{-1}(p)+\sqrt\rho\,\Phi^{-1}(\alpha))/\sqrt{1-\rho}\big)$.
  This is the formula behind Basel IRB capital, and it is checked in tests.
- $\rho=0$ is exactly the independent case. With the same seed the code produces identical draws.

**Why correlation lowers the median.** Correlation does not change $E[L]$ but spreads the
distribution of $p(F)$. Most paths have $F$ near or above 0 and quieter-than-average losses, while a
few bad-$F$ paths are very bad. The distribution becomes right-skewed: the median falls below the
mean while VaR/ES rise. So the effect of $\rho$ is not monotone across all quantiles.

**Implementation.** Paths are simulated in chunks, so memory is $O(\text{chunk}\times n)$. Factor
and idiosyncratic draws come from separate child seeds (`SeedSequence.spawn`), so results do not
depend on the chunk size. All scenarios use the same seed (common random numbers).

**Factor-sampling noise.** For $\rho>0$, most of the variance of $\bar L$ comes from the $S$ draws
of $F$. The research run's factor sample has mean $+0.0174$ ($z=+2.47$), which pulls the correlated
scenarios' simulated means about 2.7 SE below the analytical EL. Repeating the simulation with
independent seeds gives standardised errors centred on zero
(`artifacts/metrics/risk/mc_unbiasedness_check.json`), consistent with an unbiased estimator.
The scenario table reports `factor_sample_mean_z` for this reason.

## 9. VaR and Expected Shortfall (`risk/var_es.py`)

For loss $L$ (larger = worse):

$$\mathrm{VaR}_\alpha=\inf\{\ell: P(L\le\ell)\ge\alpha\},\qquad
\mathrm{ES}_\alpha=\frac{1}{1-\alpha}\int_\alpha^1\mathrm{VaR}_u\,du .$$

With $S$ equally likely simulated losses, VaR is the lower empirical quantile
(`np.quantile(..., method="inverted_cdf")`). Discrete distributions have atoms; for example, many
paths can share the same loss. "Average of losses beyond VaR" is then wrong, because part of the
atom at VaR belongs to the tail. The integral definition gives (Acerbi & Tasche, 2002)

$$\mathrm{ES}_\alpha=\frac{\sum_{L_s>\mathrm{VaR}}L_s+\mathrm{VaR}\cdot\big(\#\{L_s\le\mathrm{VaR}\}-S\alpha\big)}{S(1-\alpha)}
=\mathrm{VaR}_\alpha+\frac{1}{S(1-\alpha)}\sum_s\big(L_s-\mathrm{VaR}_\alpha\big)^+ .$$

The second (Rockafellar–Uryasev) form is used in code. It is algebraically identical, guarantees
$\mathrm{ES}\ge\mathrm{VaR}$, and avoids floating-point error in $S\alpha$; a test caught exactly
that error. ES is coherent (sub-additive) and VaR is not.

**Monte Carlo error of VaR/ES** is estimated by batch means: split the $S$ paths into 20 batches,
compute the statistic per batch, and use $\text{sd}/\sqrt{20}$. This is approximate, because
quantiles of small batches are slightly biased.

## 10. Hypothetical decision economics (`decisions/policies.py`)

One-month expected contribution of retaining account $i$ (all inputs except PD assumed):

$$E[C_i]=(1-\mathrm{PD}_i)\,rB_i-\mathrm{PD}_i\,\mathrm{LGD}\,\mathrm{EAD}_i-c .$$

$E[C_i]>0 \iff \mathrm{PD}_i < \dfrac{rB_i-c}{rB_i+\mathrm{LGD}\,\mathrm{EAD}_i}$, the account's
**break-even PD**. Solving for $r$ gives the break-even revenue rate
$r_i^\*=(\mathrm{PD}_i\mathrm{LGD}\,\mathrm{EAD}_i+c)/((1-\mathrm{PD}_i)B_i)$.

Over a single month a default destroys a large one-off amount ($\mathrm{LGD}\cdot\mathrm{EAD}$), while
performance earns one month of margin. With this sample's 22% one-month default rate, no account
has a positive expected contribution at plausible margins. That is a property of the horizon and
the sample, not a recommendation.

Policy B's cutoff is the largest PD threshold whose retained *validation* accounts have an observed
default rate at or below a hypothetical risk-appetite target. Policy D draws random cohorts of the
same size, which isolates the model's ranking value from the effect of simply retaining fewer
accounts.
