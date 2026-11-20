# Independent Methodological Review of CREDITLAB

*Reviewer: independent quantitative review, 2026-11-19. Scope: the full pipeline in `src/creditlab/`,
the configuration `configs/default.yaml`, the research artifacts in `artifacts/`, and the documents
`docs/RESEARCH_REPORT.md`, `docs/DECISIONS.md`, `docs/LIMITATIONS.md` and `docs/MODEL_CARD.md`.*

**Scope limits.** This review changes no code, protocol, configuration or result. Every check
below was run on a copy, or read-only against saved artifacts. The reviewer's scripts were
standalone and did not modify the repository. Where a correction would change a reported
number, the consequence is stated but the correction is **not** applied.

**About the passing tests.** All 104 automated tests pass (`pytest -q`, 12.7 s). The tests show
that the code does what its authors meant it to do: estimators match closed forms, seeds
reproduce results, and splits are disjoint. They do not show that the research design answers
its questions, that the assumptions are realistic, or that the conclusions are fully supported.
This review treats the tests as evidence of implementation correctness only. **The project should
not be described as scientifically validated on the strength of its test suite.**

---

## 0. Summary verdict

| Area | Verdict |
|---|---|
| Implementation correctness (PD models, metrics, VaR/ES, simulation) | **Sound.** Every quantity this review recomputed independently matched. |
| Data leakage into the test split | **No mechanical leakage found.** Procedural test-set hygiene is weaker than the documents state (§1.4, §2.1). |
| D9 (calibration-rule change after test metrics were printed) | **Procedurally compromised, materially harmless.** The counterfactual leaves every headline test result unchanged (§2.1). |
| D15 (Policy B rule change) | **Defensible and validation-driven**, but it makes Policy B's headline result close to true by construction (§2.2). |
| Monte Carlo VaR/ES and correlated defaults | **Correct.** Independent semi-analytical values agree within 1–2 Monte Carlo SE. Correlated tails sit ~1% low because of the factor sample (§3). |
| Reproducibility | **Bit-for-bit on the original machine and environment.** No code-version provenance (§4). |
| Policy comparisons | **The default-rate comparison is meaningful. The money comparisons are mostly algebra.** The random baseline overstates the model's value (§5). |
| Money-denominated results overall | **Not interpretable in level.** The target event is not a loss event (§6, weakness 1). |

The predictive-modelling half of the project (PD estimation, calibration assessment and
model comparison) is careful and largely well supported. The financial half (EL, VaR/ES and
policy economics) is computed correctly. However, its numbers are driven by assumptions that are
either unestimated or inconsistent with the meaning of the target variable. The documents
disclose most of these assumptions individually, but the headline sections still present the
resulting figures more prominently than their evidential weight justifies.

---

## 1. Data-leakage audit

### 1.1 Split construction — no issue

`make_split` (`src/creditlab/data/preprocess.py`) performs a stratified 60/20/20 split by client,
persisted to `data/processed/splits.csv`. `_check_no_leakage` (`src/creditlab/models/evaluate.py`)
verifies ID disjointness and that the training-ID SHA-256 equals the one stored at training time.
Re-checked: the three splits are disjoint, and each has 1,327/1,327/3,982 defaults (rate 0.2212).

### 1.2 Preprocessing and features — no issue

`engineer_features` is strictly row-wise. Its only constant is `ratio_clip = 5.0`, so it cannot
carry information across rows. All fitted transforms (scalers, one-hot encoders) are fitted
inside the scikit-learn pipelines on training folds only. Exploratory SQL runs on a
`train_accounts` view (`src/creditlab/data/sql_analysis.py:52`), and the EDA figures and notebook
state the same restriction.

### 1.3 Tuning and calibration — minor, non-material

* **Hyperparameter tuning** uses 5-fold CV on the training split only (`tune.py`). Correct.
* **Calibration** uses `CalibratedClassifierCV(cv=5, ensemble=False)` on the training split
  (`calibrate.py`). No validation or test rows touch it. Correct.
* **Mild nested-optimism issue.** The calibrator's out-of-fold scores come from a model whose
  hyperparameters were already chosen by CV on the *same* training rows. Those OOF scores are
  therefore slightly optimistic, which is a second-order leak *within the training split*. It
  cannot affect the test split, and because both calibrators were rejected, it affects no
  reported result.
* **Score-distribution mismatch.** With `ensemble=False`, the calibrator is fitted on scores from
  models trained on 80% of the training data, then applied to a model refitted on 100%. For
  boosted trees with a fixed number of estimators, the refitted model's scores are slightly more
  extreme than the OOF scores. This can make post-hoc calibration look worse than it is.
  The project's conclusion that "calibration worsened XGBoost" is correct for *this* calibration
  design but should not be generalised to "XGBoost cannot be improved by recalibration."
  **Recommended wording change:** qualify §5 of the report accordingly. This changes no result.

### 1.4 Model selection and the final test split — no mechanical leak, but the test set was scored for every candidate

`run_evaluate` correctly writes `selection.json` (computed from validation only) *before* it loads
the test split, so the code cannot feed test information back into a selection. **However,**
`predict_all` scores **all seven candidates** on the test split, and `test_metrics.csv` stores all of
them. Recomputed from `artifacts/predictions/test_predictions.parquet`:

| model | validation log loss | test log loss | test ROC-AUC |
|---|---|---|---|
| logistic_raw | 0.43325 | 0.43218 | 0.77994 |
| logistic_sigmoid | 0.43320 | 0.43217 | 0.77994 |
| logistic_isotonic | 0.43871 | 0.44274 | 0.77947 |
| **xgboost_raw** (selected) | **0.42691** | **0.42440** | **0.78991** |
| xgboost_sigmoid | 0.42977 | 0.42900 | 0.78991 |
| xgboost_isotonic | 0.42834 | **0.42437** | 0.78965 |

Two observations:

1. Selection was honest. On test, `xgboost_isotonic` happens to beat the selected `xgboost_raw`
   by 0.00003 in log loss, and the protocol correctly did not switch to it.
2. Every run exposes the full test leaderboard to the analyst. That is exactly how D9 arose (§2.1).
   A frozen-selection protocol should score only the frozen models on test.

**Recommended correction (protocol, future runs only):** restrict phase 2 of `run_evaluate` to the
models named in `selection.json` plus the dummy, or write the other candidates' test metrics to a
sealed file that is opened only after the report is final. **Consequence:** no current number
changes. The test results of non-selected variants would no longer appear in the report.

### 1.5 Downstream stages — no issue

The risk engine and policy lab use test-split PDs from `xgboost_raw`, which never saw those rows.
Policy B's cutoff is chosen on validation outcomes (`choose_pd_threshold`) and only then applied
to test. Permutation importance is computed on validation, and the fairness audit is
descriptive on test. The F1 thresholds come from validation. Their optimism on test is small:
XGBoost reaches test F1 0.537 at the validation threshold, against an oracle test F1 of 0.544.

---

## 2. Decisions D9 and D15

### 2.1 D9 — calibration selection rule tightened after test metrics had been printed

**What happened** (`docs/DECISIONS.md` D9): the first selection rule chose `logistic_sigmoid` on a
validation log-loss gain of −0.00004 (CI [−0.00014, +0.00006]). That run printed test metrics.
The rule was then tightened to require the bootstrap CI to lie below zero, so `logistic_raw` was
selected.

**Was the final evaluation compromised?** Procedurally, yes. Materially, no. The evidence:

| Quantity | Old rule (logistic_sigmoid) | New rule (logistic_raw) | Change |
|---|---|---|---|
| Primary model | xgboost_raw | xgboost_raw | none |
| Logistic test ROC-AUC | 0.77994 | 0.77994 | 0 (sigmoid is monotone, so ranking is identical) |
| Logistic test log loss | 0.43217 | 0.43218 | +0.000015 |
| Max \|Δ PD\| per account | — | — | 0.004 |
| XGB − logistic test ΔAUC, 95% paired CI | +0.0100 [+0.0039, +0.0160] | +0.0100 [+0.0039, +0.0160] | none |
| XGB − logistic test Δlog loss, 95% CI | [−0.0112, −0.0042] | [−0.0113, −0.0042] | negligible |
| EL, VaR, ES, all policy results | — | — | none (they use xgboost_raw only) |

*(Reviewer's recomputation: counterfactual rule applied to `selection.json` validation metrics, with
a paired bootstrap of 2,000 replicates on the saved test predictions.)*

1. **The change cannot have affected any downstream result.** Under both rules the primary model is
   `xgboost_raw`, which validation log loss chose by a wide margin (0.4269 vs 0.4332). Every risk
   and policy number depends only on the primary model.
2. **The change went *against* the test set.** On test, `logistic_sigmoid` is marginally *better* than
   `logistic_raw` (by 0.000015 log loss). An analyst cherry-picking on test would have kept
   sigmoid. This is reasonable circumstantial evidence that the change was driven by the
   validation CI, as D9 states.
3. **What is compromised is the claim of an "untouched" test split.** The report's abstract
   ("On the untouched test split…") and §5 ("tightened… *before* any downstream analysis") are
   true in a narrow sense, but they omit that test metrics had already been seen. D9 itself is
   candid. The report should match it.
4. **D9 cannot be audited.** The repository has no commits (`git log`: "does not have any commits
   yet"), and `run_manifest.json` records `"git_commit": null`. The order of events in D9 rests
   entirely on the authors' account. Nothing in this review contradicts that account, but nothing
   independently confirms it either.

**Recommended corrections (wording only):** replace "untouched test split" with "test split held
out from all selection; test metrics were printed once before a disclosed, non-material rule change
(D9)". **Consequence:** no number changes.

### 2.2 D15 — Policy B cutoff rule changed from "max realized contribution" to "max bad rate ≤ 10%"

**What happened:** under the assumed one-month economics, every account has negative expected
contribution, so the contribution-maximising cutoff degenerates to "retain almost nobody". The rule
was replaced by a revenue-independent risk-appetite rule.

**Assessment.**

* **The rule change was necessary and the reason is correct.** §5.3 shows the degeneracy is an
  algebraic certainty under the assumptions, not an empirical accident.
* **No test leakage.** The new rule uses validation outcomes only. D15 states that no
  *test-split policy* result existed. Note, though, that the test *model* metrics and test PDs did
  exist (the evaluate stage had already run), so "nothing on test had been seen" would overstate
  the position. The risk of influence is low because the rule does not read test data.
* **New issue introduced: the headline is close to true by construction.** The rule picks the
  cutoff whose validation cohort has a bad rate ≤ 10%. Because the model is well calibrated and
  validation and test are exchangeable random splits, test Policy B *must* land near 10%. The
  reported 9.9% therefore mostly confirms calibration transfer between two random splits. It is
  not new evidence of decision value. Across 30 re-randomised splits (hyperparameters fixed), the
  re-derived Policy B gave a test bad rate of **10.0% ± 0.5%** (range 8.7–11.0%), above the 10%
  target in 57% of splits.
* **The 10% target is itself an unexamined free parameter**, chosen after inspecting validation
  economics. Results at 8% or 12% are not reported.

**Recommended corrections:** (a) report Policy B's test bad rate with its sampling uncertainty
(9.9% ± 1.0%, 95% binomial). (b) State that ≈10% is the target the rule was built to hit. (c) Add a
small sensitivity over the target (e.g. 6–14%). **Consequence:** no headline number changes, but
the "9.9% vs 22.1%" comparison is reframed (§5.1).

---

## 3. Independent verification of Monte Carlo VaR, ES and correlated defaults

All checks re-implement the mathematics without calling `creditlab.risk.*`, and use the saved
paths in `artifacts/metrics/risk/scenario_losses.npz`.

### 3.1 Empirical VaR and ES from the saved paths — exact match

VaR was recomputed as the ⌈nα⌉-th order statistic. ES was recomputed with the Acerbi–Tasche
tie-correct formula $\mathrm{ES}_\alpha = [\sum_{i>k}L_{(i)} + L_{(k)}(k-n\alpha)]/(n(1-\alpha))$.
This is a different algebraic route from the project's Rockafellar–Uryasev form. The results
agree with `scenario_summary.csv` to every printed digit for all five scenarios at α = 0.95 and
0.99. ES ≥ VaR holds throughout.

### 3.2 Semi-analytical one-factor calculation — agrees within MC error

Conditional on the factor F, defaults are independent with
$p_i(F)=\Phi\big((\Phi^{-1}(p_i)-\sqrt\rho F)/\sqrt{1-\rho}\big)$. With n = 6,000 accounts,
$L\mid F$ is very close to normal with mean $\sum w_ip_i(F)$ and variance $\sum w_i^2p_i(F)(1-p_i(F))$.
Integrating over F on a 4,001-point grid gives the unconditional CDF, and from it VaR and ES,
without any simulation.

| scenario | ρ | MC mean | semi-analytic mean | MC VaR 99 | semi-analytic VaR 99 | MC ES 99 | semi-analytic ES 99 | VaR gap / bootstrap SE |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 69.93M | 69.91M | 74.81M | 74.78M | 75.67M | 75.49M | +0.5 |
| base_correlated | 0.04 | 69.60M | 69.91M | 114.04M | 114.84M | 121.22M | 122.78M | −1.3 |
| mild_stress | 0 | 90.37M | 90.36M | 95.83M | 95.73M | 96.68M | 96.51M | +1.7 |
| severe_stress | 0 | 151.95M | 151.93M | 159.42M | 159.41M | 160.68M | 160.47M | +0.4 |
| correlated_downturn | 0.12 | 112.09M | 112.95M | 229.44M | 231.66M | 248.45M | 252.40M | −1.7 |

Standard deviations also agree: 2.10M vs 2.09M (base), 16.48M vs 16.80M (ρ = 0.04), and
41.88M vs 42.71M (ρ = 0.12).

**Caveat on the reference values.** The normal approximation of L|F slightly understates
right-tail skew. In the independent scenarios it puts ES 99 about 0.1–0.2M *below* the MC value,
so for ρ > 0 the true gaps below are, if anything, slightly larger than shown.

**Conclusions.**

* The simulator, VaR and ES are implemented correctly.
* **The correlated tails are biased low by about 1% (VaR) to 1.6% (ES) for this seed.** The cause
  is the factor sample, whose mean is +0.0174 (z = +2.47). Its empirical 1% quantile is −2.294,
  against the exact −2.326. The report attributes the low correlated *mean* to this sample but not
  the *tail*, which is affected in the same direction. The bias is within 2 SE and immaterial
  next to the uncertainty in ρ, but it is systematic for this seed.
* **Recommended correction:** for ρ > 0, either report the semi-analytical VaR/ES alongside the MC
  values, or use stratified or quasi-random factor draws (only ~200 of 20,000 paths lie in the 1%
  factor tail). **Consequence:** base_correlated VaR 99 would rise from 114.0M to ~114.8M, and
  correlated_downturn ES 99 from 248.5M to ~252.4M. No qualitative conclusion changes.

### 3.3 Other sanity checks

| Check | Result |
|---|---|
| ρ = 0 mean/variance vs closed form $\sum p_iw_i$, $\sum p_i(1-p_i)w_i^2$ | mean within 0.81 SE; std 2.104M vs 2.093M |
| Marginal PDs preserved under correlation | max \|simulated freq − PD\| 0.012–0.017 at 20,000 paths. Binomial SD per account is 0.003–0.0035, so this is the largest of 6,000 deviations at about 4–5 SD, as expected for a maximum |
| Batch-means SE of VaR 99 vs bootstrap SE | 0.05M vs 0.06M (base); 1.56M vs 1.31M (ρ = 0.12): within ±25%, adequate |
| Stress map $p' = mp/(1-p+mp)$ | verified: mean PD 0.2215 → 0.2811 (m = 1.5) → 0.3688 (m = 2.5) |
| Implied pairwise default correlation at ρ = 0.04, p̄ = 0.22 | 0.021: plausible order of magnitude |
| Observed test defaults vs simulated distribution | 1,327 lies at the 46th percentile (base), so the model is calibrated in the large |
| Unbiasedness check (30 seeds) | z mean +0.03, SD 1.18. The SD's standard error at 30 seeds is ≈ 0.15, so this is acceptable |

**What these checks do not establish:** whether ρ = 0.04 (or any ρ) is right for this portfolio.
The value is borrowed from the Basel QRRE supervisory formula, which is calibrated to a one-year
horizon and a regulatory purpose. It is not estimated here and cannot be estimated from a single
outcome month. VaR 99 moves from 74.8M to 220.0M as ρ goes from 0 to 0.3 (`sensitivity_rho.csv`),
so **the tail figures are outputs of an assumption, not findings about this portfolio.** See
weakness 4.

---

## 4. Reproducibility and qualification of reported results

### 4.1 Reproduction

The research configuration was re-run into a separate directory
(`prepare → train → evaluate → simulate → policy`, 70 s). The `report` stage was skipped so that
`docs/` was not touched. Compared with the shipped artifacts:

| Artifact | Result |
|---|---|
| validation and test predictions (all 7 models) | max \|Δp\| = 0.0 |
| `selection.json` choices and thresholds | identical |
| `test_metrics.csv`, `test_bootstrap_ci.csv`, `validation_metrics.csv` | byte-identical |
| `scenario_summary.csv`, `sensitivity_rho.csv`, `mc_unbiasedness_check.json` | byte-identical |
| `policy_comparison.csv`, `policy_revenue_sensitivity.csv` | byte-identical |
| `audit_test_primary_model.csv`, `permutation_importance_validation.csv` | byte-identical |

Every number in the report's generated tables (`BEGIN`/`END` blocks) is therefore reproducible
**on this machine with this environment** (Python 3.11.15, macOS arm64, versions in
`run_manifest.json`).

### 4.2 Gaps

1. **No code-version provenance.** `git_commit` is `null` and the repository has no commits. A
   result cannot be tied to the code that produced it, and D9/D15 cannot be audited.
   *Correction:* commit the code and re-run, so the manifest records a hash.
2. **Cross-platform reproducibility is untested.** XGBoost `hist` and BLAS-dependent logistic fits
   can differ in the last bits across CPUs. Expect tiny numeric differences on other hardware.
   Nothing indicates that a conclusion would change.
3. **Fragile environment.** The `.venv` stores absolute paths. After the project folder was moved,
   the editable install and every console script broke (`bad interpreter`, `No module named
   creditlab.reporting`). *Correction:* document re-creating the venv after a move.
4. **Config files must sit inside the project root.** `update_manifest` calls
   `cfg.source.relative_to(PROJECT_ROOT)`, which raises an error for an external config. This is a nit.
5. **Narrative numbers are hand-written.** Values in the prose (e.g. "2.7–2.9 SE", "AUC 0.755",
   "62.7%", "6% or more") are not regenerated by `creditlab report`. All the ones checked match
   the current artifacts, but they will go stale silently if anything is re-run.

### 4.3 Single-split dependence (qualification)

All headline metrics come from one random split. Over 30 additional random 60/20/20 splits, with
hyperparameters held at the selected values:

| quantity | reported (this split) | mean ± SD over 30 splits | range |
|---|---|---|---|
| XGBoost test AUC | 0.790 | 0.786 ± 0.008 | 0.773–0.805 |
| Logistic test AUC | 0.780 | 0.776 ± 0.008 | 0.762–0.792 |
| XGB − logistic ΔAUC | +0.010 | +0.010 ± 0.003 | +0.003 to +0.017 (positive in 30/30) |
| XGB − logistic Δlog loss | −0.0079 | −0.0070 ± 0.0016 | always negative |
| Policy B test bad rate | 0.099 | 0.100 ± 0.005 | 0.087–0.110 |

The model comparison is robust. The reported split is about half an SD favourable on absolute AUC.
The bootstrap CI in the report reflects *test-sample* noise only, not split-to-split or tuning
variability. **Qualification to add:** "AUC ≈ 0.78–0.79 depending on the split."

### 4.4 Qualifications that are adequate

The documents consistently label LGD, EAD, revenue and stress multipliers as hypothetical.
They state that the split is not a chronological backtest, and they disclaim causal and
lending interpretations of the policy lab. The limitations document is unusually candid.
The issues below concern **emphasis and inference**, not missing disclaimers.

---

## 5. Are the hypothetical policy comparisons meaningful and fairly presented?

### 5.1 B vs D (equal-size random cohort): valid but a weak baseline

At equal cohort size (3,471 accounts, 57.9%), the observed test bad rate is:

| Ranking used to fill the same-size cohort | Observed default rate |
|---|---|
| Random (Policy D) | 0.221 |
| `PAY_0 ≤ 0` only (n = 4,635, larger cohort) | 0.139 |
| Transparent rule: sort by `PAY_0`, then by higher `LIMIT_BAL` | 0.129 |
| Logistic regression PD | 0.099 |
| XGBoost PD (Policy B) | 0.099 |

The report calls B vs D "the clearest evidence of the model's ranking value". Any feature with
AUC > 0.5 beats a random cohort, so the comparison mainly shows that the model is not useless.
Against a two-variable rule, the model's increment is about **3 percentage points (12.9% → 9.9%)**,
not 12. Logistic and XGBoost are indistinguishable for this purpose, so the AUC gain from XGBoost
buys nothing in this policy.

*Correction:* add a transparent-rule comparator of equal size and report B's bad rate with a CI.
**Consequence:** no number changes. The narrative of the model's value is moderated.

### 5.2 Money comparisons across A, B and D are size-driven

Under the assumed economics, **every** account has negative expected contribution (§5.3). The
cohort's total expected net contribution is then a sum of negative terms, which falls as the
cohort shrinks or its mean PD falls. So:

* "B loses less than A" (−22.3M vs −66.4M) holds for *any* cutoff below 1, with or without a model.
* "C is best" (0, retain nobody) holds by construction.
* B vs D at equal size *is* a fair money comparison (−22.3M vs −38.4M), but it repeats the
  bad-rate comparison in hypothetical currency.

Similarly, B's lower VaR 99 vs A (49.5M vs 113.0M) is mostly a size effect (446M vs 644M EAD).
Only the equal-size comparison with D (49.5M vs 65.9M) isolates selection. Even there, D's tail
figure is from **one** random draw, with no interval.

*Correction:* present money results for equal-size comparisons only, or normalise per unit of EAD.
State plainly that the A/B/C ordering is implied by the assumptions.

### 5.3 Policy C's empty cohort is an algebraic identity, not a finding

Since $\mathrm{EAD}_i \ge B_i$ (drawn balance) and $c = 0$, the account break-even PD satisfies

$$
\mathrm{PD}^*_i=\frac{rB_i}{rB_i+\mathrm{LGD}\cdot\mathrm{EAD}_i}\;\le\;\frac{r}{r+\mathrm{LGD}} .
$$

At r = 0.015 and LGD = 0.6 the bound is **0.0244**. The reported maximum break-even PD is 0.0244,
which is the bound being reached. The lowest model PD on test is 0.0353, so C must be empty
whatever the model's quality. The bound is a necessary condition only: C can retain anyone only
once r/(r+0.6) exceeds the minimum PD (r ≳ 0.022). Actual retention is lower than the bound
suggests because EAD also includes 50% of the undrawn limit. The reported shares are 0% at
r = 0.03, 5.1% at r = 0.06 and 26.9% at r = 0.12, against bound-implied ceilings of 3.9%, 25.4%
and 55.3%. This agrees with `policy_revenue_sensitivity.csv`.

The report calls C's emptiness "a consequence of the horizon … not a business finding", which is
correct. It should go further: **under these assumptions the C result carries no information
about the data or the model.**

### 5.4 Non-causal framing

The report correctly says that outcomes were observed under the bank's actual treatment and
cannot be read as counterfactual policy effects. That framing is adequate.

---

## 6. The five most serious methodological weaknesses (ranked)

### 1. The target event is not a credit-loss event, so every money figure is uninterpretable in level

`DEFAULT_NEXT_MONTH` is "default payment next month": a missed or late payment in October 2005,
with no definition by days past due or charge-off (`LIMITATIONS.md`). The risk engine applies a
60% LGD to each such event, as if it were a terminal write-off. The implied economics are
impossible for any real lender:

* Hypothetical EL is **23.8% of the drawn balance per month** (69.9M on 293.3M drawn), or 10.9% of
  EAD. Annualised, that is roughly 130% of EAD.
* A 22% monthly event rate, if i.i.d., implies a 95% annual default probability.
* Accounts already delinquent in September (`PAY_0 ≥ 1`) make up 23% of the test cohort, and
  33–77% of them "default". Many of these events are continuing delinquencies that would partly
  cure. Cure is not modelled.

The project's PD models are about the correct target. The EL, VaR, ES, stress and policy layers
interpret that target as a loss event, so their NTD outputs are best read as "loss *if* every
missed payment became a 60% charge-off". That is an upper bound with no stated relation to actual
losses, and it is the root cause of the Policy C degeneracy and D15.

*Correction:* restate all money outputs as conditional on that interpretation, or introduce an
explicit, labelled roll-to-charge-off (cure) assumption with sensitivity. *Consequence:* if the
reported tables stay as they are, no numbers change, but the abstract's "hypothetical one-month
EL of NTD 69.9M" needs an explicit qualifier. If a cure rate is added, every money number scales
down roughly in proportion and Policy C stops being degenerate.

### 2. Tail-risk results are driven by an unestimated correlation and PDs treated as exact

VaR 99 ranges from 75M to 220M across the ρ grid. The headline 114M rests on ρ = 0.04, borrowed
from a one-year regulatory formula. One outcome month gives no way to estimate ρ, so this number
is set entirely by assumption. In addition, PD estimation error is not propagated (§13 of the
report lists it as future work), the Gaussian copula has no tail dependence, and LGD is fixed
within each scenario. The report's own conclusion ("correlation, not the PD model, drives the
tail") is correct. It implies that **the tail figures should not appear in the abstract or the
README as results.**

*Correction:* present VaR/ES as a function of ρ (the existing sensitivity table) rather than as
a point value. Report the semi-analytic correlated values (§3.2). *Consequence:* no recomputation
is needed. Headline wording changes.

### 3. Policy evaluation overstates decision value

Three things combine (§2.2, §5): the 10% bad-rate result is close to true by construction, the
comparator is a random cohort rather than a simple rule, and the money comparisons are monotone in
cohort size under negative unit economics. The defensible statement is: *at equal cohort size, the
model's cohort has an observed bad rate about 3 pp below a two-variable rule and 12 pp below
random, with ±1 pp sampling uncertainty.*

*Correction:* see §5.1–5.2. *Consequence:* narrative only.

### 4. Test-set governance is weaker than stated and cannot be audited

All seven candidates are scored on test in every run. Test metrics were seen before the D9 rule
change. The test split has been regenerated several times (the milestone log records "three
research runs"). There is no version history to confirm the order of events. §2.1 shows that the
D9 change was immaterial. Still, the claim of an "untouched" test split is not accurate, and other
analyst choices made before or after test exposure cannot be verified. These include the feature
set, search grids, the 10% target and the risk bands.

*Correction:* §1.4 (score only frozen models), §4.2 (commit and record the hash), and the wording
in §2.1. *Consequence:* no current number changes.

### 5. External validity: one period, unknown sampling design, random (not temporal) split

All 30,000 accounts share one observation window and one outcome month. The 22% default rate is
far above any card portfolio's base rate, and the source does not document how the sample was
drawn. If defaults were over-sampled (a case-control design), the model's PDs are calibrated to
*the sample*, not to any population. The "well calibrated" finding and every EL number would then
be biased upward by an unknown factor, while AUC would be unaffected. Random splits from the same
period make this undetectable internally: calibration on the test split agrees with validation
because the two are exchangeable. The project discloses the single-period limitation, but it
does not connect it to the possibility that the calibration claim itself does not transfer.

*Correction:* qualify "well calibrated" as "well calibrated *to this sample's 22% base rate*". If
a population base rate were ever known, a prior-shift correction $\text{odds}' = \text{odds}\cdot
\frac{\pi/(1-\pi)}{\bar y/(1-\bar y)}$ would be required. *Consequence:* wording only for now.

### Lesser issues (not ranked)

* Calibration design (OOF from 80%-data models applied to a 100%-data refit) may understate what
  recalibration can achieve (§1.3).
* Correlated tails are ~1% low for this seed (§3.2).
* Policy D's tail metrics come from a single draw, with no interval.
* Model-comparison CIs ignore split and tuning variability (§4.3). The conclusion survives.
* The fairness audit is descriptive. Group AUC CIs are not reported (e.g. 50+ age band, n = 567),
  so differences such as 0.755 vs 0.800 may not be significant.

---

## 7. Recommended corrections, by effort

| # | Correction | Type | Changes reported numbers? |
|---|---|---|---|
| 1 | Replace "untouched test split" with an accurate description of D9 (abstract, §5) | wording | No |
| 2 | Qualify every NTD figure as conditional on "missed payment = 60% charge-off" | wording | No |
| 3 | Move VaR/ES out of the abstract/README headline, or present them as a function of ρ | wording | No |
| 4 | Report Policy B's bad rate with CI. State that ≈10% is the built-in target | wording + 1 number | No |
| 5 | Add a transparent equal-size rule comparator to the policy table | analysis | Adds rows |
| 6 | State the r/(r+LGD) bound and that C's emptiness is an identity | wording | No |
| 7 | Qualify "well calibrated" as "to this sample's base rate" | wording | No |
| 8 | Commit the code and re-run, so `git_commit` is recorded | process | No (if the code is unchanged) |
| 9 | Score only frozen models on test in phase 2 | protocol (future runs) | Removes non-selected test rows |
| 10 | Report semi-analytic or QMC tails for ρ > 0 | analysis | Correlated VaR/ES +0.7–1.6% |
| 11 | Report repeated-split variability (§4.3) alongside the bootstrap CI | analysis | Adds a table |
| 12 | Add a cure / roll-to-loss assumption with sensitivity | analysis | Yes, all money figures scale down |

Items 1–8 can be done without touching the research protocol or the frozen test results.
Items 9–12 change or extend the analysis. Before any of them is applied, it should be
recorded in `docs/DECISIONS.md` as a new, dated decision made **after** the test results were
known, so that readers can judge it as a post-hoc change.

---

## Appendix: how the checks were run

* **Re-run:** the research configuration with `processed_dir` and `artifacts_dir` redirected to a
  temporary directory, stages `prepare train evaluate simulate policy`, then a file-by-file
  comparison with `artifacts/`.
* **Risk:** standalone NumPy/SciPy script. It recomputes EAD from `accounts.parquet`, recomputes
  order-statistic VaR and tie-correct ES from `scenario_losses.npz`, computes semi-analytical
  one-factor VaR/ES by integrating over F (4,001-point grid on [−8, 8], conditional normal
  approximation of L|F), and bootstraps the SE with 300 resamples.
* **D9 counterfactual:** applies the old rule to `selection.json` validation metrics, with a paired
  bootstrap (2,000 replicates) on the saved test predictions.
* **Repeated splits:** 30 stratified 60/20/20 splits with seeds independent of the project's.
  Logistic and XGBoost pipelines are built from the project's own constructors, with the selected
  hyperparameters, no retuning, and Policy B re-derived on each validation split.
* **Policy comparators:** computed from the saved test predictions and `accounts.parquet`.
