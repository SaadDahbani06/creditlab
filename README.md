# CREDITLAB

**Credit default modeling, probability calibration, Monte Carlo portfolio stress testing and
hypothetical decision research** on the UCI *Default of Credit Card Clients* dataset
(30,000 Taiwanese cardholders, 2005).

Write-up: [saaddahbani.com/work/creditlab](https://saaddahbani.com/work/creditlab) ·
Original pre-extension state: tag `v1-baseline`.

> **Research project, not an underwriting system.** It makes no decision about any real person.
> Only probability of default is estimated from data. Every loss, revenue and policy figure uses
> explicitly *hypothetical* LGD, EAD and economic assumptions.

## Results at a glance

From the research configuration (`configs/default.yaml`, seed 20051001). The full tables are in
[docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) and are regenerated from artifacts by
`creditlab report`.

| | Logistic regression | XGBoost (selected) | Base-rate dummy |
|---|---|---|---|
| Test ROC-AUC (95% bootstrap CI) | 0.780 (0.766–0.794) | 0.790 (0.777–0.804) | 0.500 |
| Test PR-AUC | 0.539 | 0.559 | 0.221 |
| Test Brier / log loss | 0.1355 / 0.4322 | 0.1335 / 0.4244 | 0.1723 / 0.5284 |

- XGBoost − logistic AUC: **+0.010**, paired bootstrap CI [+0.004, +0.016].
- Neither sigmoid nor isotonic calibration (fitted on training out-of-fold predictions) improved
  validation log loss by a bootstrap-supported margin. The raw models were kept; test calibration
  slope 1.04, ECE 0.013.
- On the 6,000-account test cohort, **hypothetical** one-month EL is NTD 69.9M (LGD 60%,
  EAD = drawn + 50% of undrawn). 99% VaR is NTD 74.8M with independent defaults and NTD 114.0M
  with one-factor correlation ρ = 0.04.
- A PD cutoff chosen on validation retains 57.9% of the test cohort with an **observed** default
  rate of 9.9%, versus 22.1% for random cohorts of the same size (a weak baseline; see
  `docs/INDEPENDENT_REVIEW.md` §5).

**Research extension: PD estimation uncertainty** ([docs/RESEARCH_EXTENSION.md](docs/RESEARCH_EXTENSION.md);
post-hoc, exploratory). 200 bootstrap refits of the model, nested Monte Carlo with common random
numbers:

- Under independent defaults the 99% VaR across refits spans NTD 72.2–77.1M (fixed-PD value
  74.8M), and PD estimation explains 22.6% of loss variance. At ρ = 0.04 the range is
  110.6–116.5M, but PD estimation explains only 0.5% of loss variance.
- Most of this uncertainty is a portfolio-wide shift in PDs, and 81% of that shift is explained by
  sampling noise in the training default rate.
- Folding the uncertainty into a single mixture VaR changes it by under 0.5% (noise-free
  calculation). The correlation assumption moves VaR roughly ten times more than PD estimation.

**Evaluation status.** All choices were frozen on validation before the test split was scored.
Test metrics were displayed once before a disclosed, non-material rule change (D9), so the test
split is not described as untouched. Original outputs are frozen with checksums in
[results/baseline_v1/](results/baseline_v1/). No independent validation data exists.

## Quick start

Requires Python 3.11+ (developed on 3.11.15, macOS arm64). No API keys, paid services or cloud.

```bash
cd creditlab                       # if you move the folder, recreate .venv (it stores absolute paths)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"            # exact verified versions: pip install -r requirements-lock.txt

creditlab all                      # fetch -> prepare -> train -> evaluate -> simulate -> policy -> uncertainty -> report
pytest                             # 133 tests, about 15 s once artifacts exist
ruff check .
streamlit run app/streamlit_app.py # research site at http://localhost:8501
```

The original pipeline takes about 71 seconds on an 8-core Apple-silicon laptop; the `uncertainty`
stage adds about 12 minutes (200 refits per model family and nested simulation). For a smoke test that writes to
`artifacts/_fast/` and never touches research artifacts:

```bash
creditlab all --skip-fetch --config configs/fast.yaml
```

If the download fails, place the official `data.csv` from
<https://archive.ics.uci.edu/static/public/350/data.csv> at `data/raw/data.csv` and run
`creditlab fetch` (or `creditlab fetch --local-file PATH`). See [data/README.md](data/README.md).

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| Fetch | `creditlab fetch` | Downloads UCI 350 via `ucimlrepo`. Stores an immutable, hash-verified raw copy and its provenance |
| Prepare | `creditlab prepare` | Validates schema and codes, maps official names, persists the stratified 60/20/20 split, builds a DuckDB database, runs 11 SQL queries, draws EDA figures |
| Train | `creditlab train` | Tunes the dummy, logistic and XGBoost models by 5-fold **training** CV; fits sigmoid/isotonic calibrators on training out-of-fold predictions |
| Evaluate | `creditlab evaluate` | Makes **all** selections on validation and freezes them, runs leakage checks, then scores the test split once with bootstrap CIs, explanations and an audit |
| Simulate | `creditlab simulate` | Expected loss; chunked Monte Carlo (20,000 paths) of independent and one-factor Gaussian defaults; stress scenarios; VaR/ES; sensitivities |
| Policy | `creditlab policy` | Hypothetical policies A–D on the test cohort; cutoff chosen on validation; break-even and revenue sensitivity |
| Uncertainty | `creditlab uncertainty` | Research extension: bootstrap refits on the training split, nested common-random-number simulation, variance decomposition, noise-free semi-analytic checks, sensitivity and convergence |
| Report | `creditlab report` | Regenerates figures and injects measured tables into `docs/RESEARCH_REPORT.md`, `docs/MODEL_CARD.md` and `docs/RESEARCH_EXTENSION.md` |

Each stage records its config fingerprint, seed, environment and package versions in
`artifacts/metrics/run_manifest.json`.

## Project structure

```
configs/            default.yaml (research), fast.yaml (smoke tests)
sql/analysis.sql    named DuckDB queries
src/creditlab/
  data/             fetch, validate, preprocess (split + features), sql_analysis
  models/           baselines/pipelines, tune, calibrate, train, evaluate, explain, audit
  risk/             assumptions (LGD/EAD), expected_loss, simulation, var_es, stress
  decisions/        policies, compare
  research/         pd_uncertainty (post-hoc research extension)
  reporting/        figures, site design system, report generation
  cli.py            `creditlab` entry point
app/site/           read-only research site (4 pages); app/legacy/ keeps the v1 interactive tools
results/baseline_v1 frozen original outputs + SHA-256 checksums
tests/              pytest suite (data, preprocessing, models, calibration, risk, simulation, policies, uncertainty, app)
notebooks/          01_exploration.ipynb (executed)
docs/               report, mathematics, model card, assumptions, limitations, decisions, status
```

## Leakage controls

- Fitted preprocessing lives inside scikit-learn pipelines fitted on the training split. A test
  checks that scaler statistics equal the training-only statistics.
- Engineered features are row-wise (tested: same values before and after splitting).
- Calibrators are fitted on training out-of-fold scores. A test reconstructs the calibrator by
  hand to $10^{-12}$.
- `evaluate` writes `selection.json` from validation data **before** loading the test split. It
  verifies a sha256 of the training IDs stored at training time, and that all splits are disjoint.
- The policy cutoff is chosen on validation. A test scrambles the test data and confirms the
  cutoff does not move.

## Documentation

| Document | Contents |
|---|---|
| [RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) | Full report with measured tables and figures |
| [RESEARCH_EXTENSION.md](docs/RESEARCH_EXTENSION.md) | PD estimation uncertainty → tail-risk uncertainty (derivations, results) |
| [INDEPENDENT_REVIEW.md](docs/INDEPENDENT_REVIEW.md) | Independent methodological review, D9/D15 audit, ranked weaknesses |
| [MATHEMATICS.md](docs/MATHEMATICS.md) | Derivations of every method used |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Model details, performance, audit, intended use |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Variables, codes, anomalies, engineered features |
| [ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Every hypothetical parameter and why |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Data, modelling, simulation, policy and ethical limits |
| [DECISIONS.md](docs/DECISIONS.md) | Implementation decisions, including two disclosed rule changes |
| [MILESTONE_STATUS.md](docs/MILESTONE_STATUS.md) | Verification commands and results per milestone |
| [PRESENTATION.md](docs/PRESENTATION.md) | Two-minute demo script |

## Data and attribution

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C55S3H>. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data are downloaded at run time
and not redistributed in this repository.
