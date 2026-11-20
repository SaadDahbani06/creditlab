# Implementation Decisions

Decisions made during implementation, with the reason for each. Numbered in the order they were made.

## D1. Project root and version control
The project lives in `~/creditlab` with its own `git init` (branch `main`), so it is independent
of any enclosing repository. Nothing is pushed anywhere.

## D2. Python 3.11 and dependency pinning
Python 3.11 satisfies the "3.11+" requirement and has wheels for every dependency.
`pyproject.toml` carries compatibility ranges; `requirements-lock.txt` pins the exact
verified versions. The code license is left for the project owner to choose; the dataset
license (CC BY 4.0) is documented in `data/README.md`.

## D3. One master seed, named sub-seeds
`seed` in the config is the only seed. Each stage derives its own seed with
`sha256(f"{seed}:{name}")`, so adding a random step in one stage cannot shift another
stage's random stream.

## D4. Two configurations
`configs/default.yaml` produces the research results. `configs/fast.yaml` inherits from it,
reduces CV folds, search iterations and simulation counts, and writes to `artifacts/_fast/`
so a smoke test can never overwrite research artifacts.

## D5. CLI stage boundaries
`train` sees only the training split (tuning by training CV, calibration by training
out-of-fold predictions). `evaluate` first makes every selection on the validation split,
saves it, and only then scores the test split. This ordering makes "the test set was not used
for selection" structural rather than a convention.

## D6. Official variable names
Raw columns `X1..X23` are renamed with the official UCI names, including the gap `PAY_0, PAY_2, …`
(there is no `PAY_1`), so results can be compared with the literature.

## D7. Undocumented codes are kept, not reinterpreted
Status codes -2 and 0, EDUCATION 0/5/6 and MARRIAGE 0 are undocumented. Secondary sources
interpret them ("no consumption", "revolving"), but the source does not, so the project treats
them as observed categories. Logistic regression uses one-hot categories (codes ≥ 3 grouped);
trees use the ordinal value and can isolate any code.

## D8. Row-wise engineered features; six-month payment ratio
Engineered features depend only on the same row, so they cannot leak across splits. They are
computed inside the model pipelines. The payment-to-bill ratio is aggregated over six months
because the source does not say which statement each payment settles.

## D9. Calibration selection rule tightened (disclosed change)
The first rule (calibrated variant must lower validation log loss and not raise Brier) selected
logistic + sigmoid on a validation log-loss change of −0.00004, whose paired-bootstrap 95% CI
[−0.00014, +0.00006] includes zero. Test metrics had been printed in that first run. The rule was
then tightened to also require the bootstrap CI of the log-loss change to lie below zero. The
change was motivated by the validation CI. The primary model (`xgboost_raw`) is the same under
both rules, and logistic raw and sigmoid have identical test AUC and log loss to four decimals.

## D10. Log loss as the selection metric; class weights evaluated and rejected
Probabilities feed the risk engine, so selection uses a proper scoring rule. Balanced class
weights left CV ROC-AUC essentially unchanged but worsened CV log loss: 0.557 vs 0.432 for
logistic, 0.523 vs 0.425 for XGBoost. Oversampling was not used.

## D11. Portfolio = test split with out-of-sample PDs
The risk engine and policy lab use the 6,000 test accounts, whose PDs come from a model that
never saw them. In-sample training PDs would understate risk dispersion.

## D12. EAD default = drawn balance + CCF × undrawn limit
This uses the observed September balance, keeps limit and EAD distinct, and is the standard
credit-conversion-factor form. `limit_fraction` is available and reported in the sensitivity grid.

## D13. Simulation design
Simulation uses a one-factor Gaussian latent variable model with ρ = 0 as the exact independent
case (same draws). Factor and idiosyncratic streams come from separate child seeds, so results do
not depend on chunk size (tested). Common random numbers are used across scenarios. Plain Monte
Carlo was kept rather than antithetic factor draws, to keep standard errors simple. Because the
factor sample of one seed can shift correlated means by a few SE, the factor z-score and a 30-seed
unbiasedness check are reported.

## D14. Expected Shortfall estimator
ES is computed as VaR + mean excess / (1 − α) (Rockafellar–Uryasev), algebraically identical to
the Acerbi–Tasche tie-correct formula. The first implementation used the count form and a test
caught a floating-point violation of ES ≥ VaR (n·α = 2000 × 0.95 ≠ 1900 exactly).

## D15. Policy B cutoff rule (disclosed change)
The first implementation chose B's cutoff by maximising realized hypothetical contribution on
validation. On inspecting the validation economics, every account had negative expected one-month
contribution (highest break-even PD 0.024 < lowest model PD 0.035), so that rule degenerates to
"retain almost nobody". The default was changed to a revenue-independent risk-appetite rule:
the largest cutoff whose retained validation accounts have an observed default rate ≤ 10%. The
original rule remains available (`policy.b_rule: max_realized_contribution`). No test-split
policy result existed before the change, because the first run stopped on the empty policy-C cohort.

## D16. Empty cohorts are valid outcomes
Retaining nobody (policy C at plausible margins) is reported as zero exposure and zero loss rather
than raised as an error.

## D17. Dashboard structure
`app/streamlit_app.py` is a router (`st.navigation`) so pages have proper titles and URLs. The
overview lives in `app/overview.py` and the other pages in `app/pages/`. The dashboard reads
artifacts and never trains. The Stress Testing and Policy Lab pages re-run the same library
functions with user-chosen hypothetical inputs.

## D18. Jupyter kernel inside the virtual environment
The notebook kernel is registered with `--sys-prefix` (inside `.venv`) so that nothing is written
to the user's global Jupyter configuration.

## D19. Figures copied into docs/
`creditlab report` copies `artifacts/figures/*.png` to `docs/figures/` so the research report
renders without the git-ignored artifacts directory.

## D20. Wording corrected: the test split is not "untouched" (2026-11-19, post-hoc)
The research report described the test split as "untouched" and said the D9 rule change happened
"before any downstream analysis". Both statements omitted that test metrics had already been
displayed (as D9 itself records). The wording in `RESEARCH_REPORT.md`, `MATHEMATICS.md` and
`MODEL_CARD.md` was corrected. No number changed. The original outputs are frozen with checksums in
`results/baseline_v1/`.

## D21. PD-estimation-uncertainty extension is post-hoc and exploratory (2026-11-19)
The extension (`docs/RESEARCH_EXTENSION.md`) was designed after every original test result was
known. It reuses the training split (bootstrap) and the test-split portfolio, and it selects
nothing using test labels. Its findings are descriptive analyses of the existing data, not an
independent validation. No new untouched data exists for this project.

## D22. Research site simplified to four read-only pages (2026-11-20)
The eight-page dashboard, with its scenario sliders, seed inputs and simulation-count controls,
was replaced by a four-page read-only site (`app/site/`): Overview, Predictive Modeling, Portfolio
Risk & Findings, and Methodology & Limitations. Every number is read from saved artifacts and the
site runs no simulation. The v1 pages, including the Stress Testing and Policy Lab tools, were moved
unchanged to `app/legacy/`. They still run on their own and are still tested, and the risk and
policy code they call is untouched. Policy Lab was dropped from the main site because its money
comparisons are mostly implied by the assumptions (`docs/INDEPENDENT_REVIEW.md` §5).

## D23. Resume material kept out of the repository (2026-11-20)
`docs/RESUME_RECOMMENDATIONS.md` contains personal career information and is listed in
`.gitignore`. The resume bullets were removed from `docs/PRESENTATION.md`; the draft revision of
the resume itself is stored outside the project folder.
