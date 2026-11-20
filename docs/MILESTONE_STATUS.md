# Milestone Status

Each milestone is marked **VERIFIED** only after its checks have actually been run and passed.
Commands are run from the project root with the project virtual environment active.

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Foundation | VERIFIED |
| 1 | Data engineering & exploration | VERIFIED |
| 2 | Predictive modeling | VERIFIED |
| 3 | Probability calibration | VERIFIED |
| 4 | Financial risk engine | VERIFIED |
| 5 | Monte Carlo & stress testing | VERIFIED |
| 6 | Hypothetical decision policy lab | VERIFIED |
| 7 | Streamlit research dashboard | VERIFIED |
| 8 | Research report & release | VERIFIED |
| 9 | Research extension: PD estimation uncertainty (post-hoc) | VERIFIED |
| 10 | Research site redesign (4 read-only pages) | VERIFIED |

---

## Milestone 0 — Foundation

Environment: Python 3.11.15 (Homebrew), macOS arm64, `.venv` in project root.
Exact package versions: `requirements-lock.txt`.

| Check | Command | Result |
|-------|---------|--------|
| Environment installs | `pip install -e ".[dev]"` | installed, no build warnings |
| CLI help | `creditlab --help`, `creditlab fetch --help` | usage printed, exit 0 |
| Tests | `pytest -q` | 9 passed |
| Lint | `ruff check .` | All checks passed |

Delivered: `pyproject.toml` (Ruff + pytest config), `configs/default.yaml` (research) and
`configs/fast.yaml` (smoke tests; inherits default, writes to `artifacts/_fast/`),
standard-library logging, deterministic named sub-seeds derived from one master seed,
CLI entry point `creditlab` with stages `fetch, prepare, train, evaluate, simulate, policy, report, all`.

## Milestone 1 — Data engineering & exploration

| Check | Command | Result |
|-------|---------|--------|
| Download | `creditlab fetch` | ucimlrepo 0.0.7; 30,000 rows; content identical to official `data.csv` |
| Immutability | `creditlab fetch` (second run) | hash verified, file not rewritten; raw file is read-only |
| Validation + split + SQL + figures | `creditlab prepare` | validation passed; split 18,000 / 6,000 / 6,000; 11 SQL queries executed; 6 EDA figures |
| Notebook | `jupyter nbconvert --execute notebooks/01_exploration.ipynb` | 21 cells, 0 errors |
| Tests | `pytest -q` | 29 passed |
| Lint | `ruff check .` | All checks passed |

Verified facts: 30,000 rows confirmed; all 23 source variables + ID + target mapped
(`docs/DATA_DICTIONARY.md`); target in {0,1} (23,364 / 6,636; rate 0.2212); no missing values.
Undocumented codes found and documented: PAY -2 and 0 (66.9% of month-records), EDUCATION 0/5/6,
MARRIAGE 0; code 1 nearly absent outside September. Exploratory SQL is restricted to the training
split (tested). Sensitive fields excluded from model inputs (tested).

## Final verification run (2026-11-19, all milestones)

| Check | Command | Result |
|-------|---------|--------|
| Full research pipeline | `creditlab all` | exit 0, 71 s (fetch hash-verified, no re-download) |
| Fast smoke pipeline | `creditlab all --skip-fetch --config configs/fast.yaml` | exit 0, 22 s, written to `artifacts/_fast/` |
| Determinism | three research runs | identical test metrics (AUC 0.789913, Brier 0.133527) and policy outputs |
| Test suite | `pytest -q` | **104 passed**, 0 failed, 0 skipped, about 11 s |
| Lint | `ruff check .` | All checks passed |
| Imports | import every `creditlab.*` module | 34 modules OK |
| CLI smoke | `creditlab {--help, fetch, prepare, train, evaluate, simulate, policy, report, all} --help` | all exit 0 |
| Notebook | `jupyter nbconvert --execute --inplace notebooks/01_exploration.ipynb` | 21 cells, 0 errors |
| Dashboard | `streamlit run app/streamlit_app.py` + `/_stcore/health` | HTTP 200 after 2 s; all 8 pages pass `AppTest`; screenshots inspected |
| Dependencies | `pip check` | No broken requirements |

## Milestone 2 — Predictive modeling
Three models (dummy, logistic, XGBoost) are tuned by 5-fold training CV, with log loss as the
selection metric. Leakage checks: training-ID sha256 stored at train time and verified at
evaluation; split disjointness; scaler statistics equal training-only statistics (test);
`train_models` refuses non-training rows (test). Metrics recompute exactly from saved predictions,
and saved models reproduce those predictions (tests). Test split: XGBoost AUC 0.7899
[0.7765, 0.8042], logistic 0.7799 [0.7658, 0.7941]. Labelled as an out-of-sample client split,
not a chronological backtest.

## Milestone 3 — Probability calibration
Sigmoid and isotonic calibrators are fitted with `CalibratedClassifierCV(cv=5, ensemble=False)`
on training OOF predictions. A test reconstructs the isotonic calibrator by hand to 1e-12. The
choice is made on validation with a bootstrap-supported rule (disclosed revision, DECISIONS D9).
Result: raw models kept for both families. Reliability plots, before/after tables and validation
bootstrap differences are saved. Probabilities are finite and in [0, 1] (asserted at prediction
time and in tests).

## Milestone 4 — Financial risk engine
EL = PD × LGD × EAD with hypothetical LGD 0.60 and EAD = drawn + 0.5 × undrawn. Hand-calculated
unit tests cover EAD, EL, aggregation (sum of accounts) and EAD-weighted PD; LGD linearity and CCF
monotonicity are tested; invalid assumptions are rejected. Units (NTD), the one-month horizon and
the HYPOTHETICAL status are recorded in every output.

## Milestone 5 — Monte Carlo & stress testing
Chunked, vectorised simulation with 20,000 paths per scenario; measured peak-memory increase about
125 MB. Tests cover:
- losses finite and non-negative;
- fixed-seed reproducibility and chunk invariance;
- ρ = 0 identical to independent draws;
- independent mean and variance equal to closed forms;
- marginal PD match for ρ ∈ {0, 0.15, 0.4};
- bivariate-normal default-count variance;
- Vasicek quantiles;
- ES ≥ VaR with ties;
- the tail and median effects of ρ checked, not assumed.

A 30-seed unbiasedness check is saved. All five scenarios run and their results are saved.

## Milestone 6 — Hypothetical decision policy lab
Policies A–D run on the test cohort. B's cutoff is chosen on validation, and a test confirms it
does not change when the test data are scrambled. Observed outcomes, model PDs and hypothetical
money are reported in separate columns. Revenue sensitivity, break-even analysis and a group
selection-rate audit are included, along with the ethics and limitations section
(`docs/LIMITATIONS.md`). No decision automation.

## Milestone 7 — Streamlit research dashboard
Eight pages: Overview, Data Explorer, Model Comparison, Calibration, Portfolio Risk, Stress
Testing, Policy Lab, Methodology. Everything is read from artifacts or computed live by library
functions. Missing artifacts produce setup instructions (tested), and the dashboard never trains.

## Milestone 8 — Research report & release
README, DATA_DICTIONARY, MATHEMATICS, MODEL_CARD, ASSUMPTIONS, LIMITATIONS, RESEARCH_REPORT (23
generated tables), DECISIONS and PRESENTATION (demo script, resume bullets). Nothing has been
committed or published; git is initialised locally only.

## Milestone 9 — Research extension: PD estimation uncertainty (2026-11-20, post-hoc)

| Check | Command | Result |
|---|---|---|
| Baseline frozen before any change | `shasum -a 256 -c results/baseline_v1/SHA256SUMS_original_artifacts.txt` | all OK; checkpoint commit `0ac9c8d` |
| Extension run | `creditlab uncertainty` | exit 0, 722 s; 200 refits per family, 20 seed refits, 20 re-tuned refits |
| Refit reproduces frozen model | `tests/test_uncertainty.py` | max abs PD difference 0.0 (both families) |
| Fixed-PD rows reproduce original VaR/ES exactly | `tests/test_uncertainty.py` | VaR 95/99 and ES 95/99 identical to `scenario_summary.csv` |
| Multi-PD engine equals single runs (common random numbers) | `tests/test_uncertainty.py` | exact equality for ρ ∈ {0, 0.04, 0.3} and two chunk sizes |
| Law of total variance exact for the empirical mixture | `tests/test_uncertainty.py` | rel. error < 1e-9 |
| Monte Carlo noise in the across-model spread | `noise_free_comparison.json`, `summary.json` | model-specific noise < 1% of across-model variance |
| Fast smoke run | `creditlab uncertainty --config configs/fast.yaml` | exit 0, ~45 s, writes only to `artifacts/_fast/` |

## Milestone 10 — Research site redesign (2026-11-20)

| Check | Command | Result |
|---|---|---|
| Pages render, no errors | `pytest tests/test_app.py` | router + 4 site pages + 8 legacy pages render |
| No playground controls on the main site | `tests/test_app.py::test_main_site_has_no_playground_controls` | no sliders, number inputs or selectors |
| Missing artifacts show setup text | `tests/test_app.py` | passes |
| Visual check | headless Chrome screenshots at 1440 px and 390 px | layout problems found and fixed (clipped chart titles, overflowing table, stray markers) |
| Full suite and lint | `pytest && ruff check .` | 133 passed; all checks passed |
