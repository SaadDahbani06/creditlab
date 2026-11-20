# Data Dictionary

Source: UCI Machine Learning Repository, dataset 350, *Default of Credit Card Clients*
(Yeh, 2009; DOI 10.24432/C55S3H; CC BY 4.0). Official variable descriptions come from the
dataset metadata (`data/raw/uci_350_metadata.json`, field `additional_info.variable_info`).
Observed counts come from `artifacts/metrics/data_validation.json` (all 30,000 rows).

**Population:** 30,000 existing credit-card accounts of a bank in Taiwan.
**Observation window:** April–September 2005.
**Outcome:** default on the payment due the following month (October 2005).
**Currency:** New Taiwan dollars (NTD). No missing values. `ID` is unique.

## Source variables

| Raw | Project name | Official description | Role in project |
|-----|-------------|----------------------|-----------------|
| ID | `ID` | Row identifier | **Excluded** from model inputs; used only to persist the split |
| X1 | `LIMIT_BAL` | Amount of given credit (NTD), includes individual and family (supplementary) credit | Model input |
| X2 | `SEX` | 1 = male, 2 = female | **Excluded** (sensitive); aggregate audit only |
| X3 | `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = others | **Excluded** (proxy concern); aggregate audit only |
| X4 | `MARRIAGE` | 1 = married, 2 = single, 3 = others | **Excluded** (sensitive); aggregate audit only |
| X5 | `AGE` | Age in years | **Excluded** (sensitive); aggregate audit only |
| X6 | `PAY_0` | Repayment status, September 2005 | Model input |
| X7 | `PAY_2` | Repayment status, August 2005 | Model input |
| X8 | `PAY_3` | Repayment status, July 2005 | Model input |
| X9 | `PAY_4` | Repayment status, June 2005 | Model input |
| X10 | `PAY_5` | Repayment status, May 2005 | Model input |
| X11 | `PAY_6` | Repayment status, April 2005 | Model input |
| X12–X17 | `BILL_AMT1`–`BILL_AMT6` | Bill statement amount (NTD), Sep (1) … Apr (6) 2005 | Model input |
| X18–X23 | `PAY_AMT1`–`PAY_AMT6` | Amount of previous payment (NTD), Sep (1) … Apr (6) 2005 | Model input |
| Y | `DEFAULT_NEXT_MONTH` | Default payment next month (1 = yes, 0 = no) | Target |

The official names skip `PAY_1` (`PAY_0` is followed by `PAY_2`). The project keeps the official
names so that results can be checked against the literature.

## Repayment status codes (`PAY_*`)

Official scale: **-1 = pay duly; 1 = delay of one month; 2 = two months; … ; 8 = eight months;
9 = nine months and above.**

| Code | Documented? | Share of all 180,000 month-records | Project treatment |
|------|-------------|-----------------------------------|-------------------|
| -2 | **No** | 13.6% | Kept as its own category; never interpreted |
| -1 | Yes (pay duly) | — | Kept |
| 0 | **No** | 53.3% | Kept as its own category; never interpreted |
| 1 | Yes | 12.3% of accounts in Sep; 0–17 accounts per month Apr–Aug (training split) | Kept |
| 2–8 | Yes | — | Kept; 9 does not occur |

Codes -2 and 0 are frequently described in secondary sources as "no consumption" and "revolving
credit / minimum payment made". **The source does not document this**, so the project does not
rely on it. The near-absence of code 1 outside September suggests that the coding differed between
the most recent month and earlier months. The six status columns may therefore not be strictly
comparable over time. This is documented as a limitation.

## Other observed anomalies (kept as recorded)

| Finding | Count |
|---------|-------|
| `EDUCATION` codes 0, 5, 6 (undocumented) | 14, 280, 51 |
| `MARRIAGE` code 0 (undocumented) | 54 |
| Accounts with at least one negative bill (credit balance) | 1,930 |
| Accounts with at least one bill above the credit limit | 3,931 |
| Accounts with no positive bill and no payment in six months | 860 |
| Rows in groups with identical feature vectors (distinct IDs) | 108 |
| …of which groups with conflicting labels | 21 groups |
| Negative payment amounts | 0 |

## Engineered features (row-wise, deterministic)

Defined in `src/creditlab/data/preprocess.py::engineer_features`. Each depends only on the same
account's April–September 2005 record and on config constants (`ratio_clip = 5`), so no
information crosses rows or splits.

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `UTIL_1` | `BILL_AMT1 / LIMIT_BAL`, clipped to [-1, 5] | Latest utilisation |
| `UTIL_MEAN_6` | mean(`BILL_AMT1..6`) / `LIMIT_BAL`, clipped to [-1, 5] | Average utilisation |
| `BILL_TREND_6` | (`BILL_AMT1` − `BILL_AMT6`) / `LIMIT_BAL`, clipped to [-5, 5] | Balance build-up |
| `PAY_TO_BILL_6M` | Σ`PAY_AMT` / max(Σ max(`BILL_AMT`,0), 1), clipped to [0, 5] | Repayment intensity. Six-month aggregation avoids assuming which bill a payment settles |
| `NO_POSITIVE_BILL_6M` | 1 if no positive bill in six months | Separates inactive / credit-balance accounts |
| `N_MONTHS_DELAYED` | # months with status ≥ 1 | Persistence of documented delays |
| `MAX_DELAY` | max(status, 0) over six months | Worst documented delay |
| `N_MONTHS_CODE_M2` | # months with status = -2 | Count of an observed (undocumented) code |
| `N_MONTHS_CODE_M1` | # months with status = -1 | Count of "pay duly" months |

## Primary model specification

Inputs (19 source columns, configurable in `configs/default.yaml → features`):
`LIMIT_BAL, PAY_0, PAY_2–PAY_6, BILL_AMT1–6, PAY_AMT1–6`, plus the engineered features
computed inside each model pipeline.
Excluded: `ID`; `SEX, AGE, MARRIAGE` (sensitive); `EDUCATION` (proxy concern).

## Processed outputs

| File | Contents |
|------|----------|
| `data/raw/uci_350_data.csv` | Immutable raw copy (official names), read-only |
| `data/raw/uci_350_provenance.json` | URL, DOI, license, retrieval time, sha256, content hash |
| `data/raw/uci_350_metadata.json` | Official UCI metadata and variable table |
| `data/processed/accounts.parquet` | Semantic names + engineered features + `split` |
| `data/processed/splits.csv` | `ID → split` (train / validation / test) |
| `data/processed/creditlab.duckdb` | DuckDB database (`accounts` table, `train_accounts` view) |
