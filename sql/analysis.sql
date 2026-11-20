-- CREDITLAB exploratory SQL (DuckDB dialect).
--
-- Tables/views created by creditlab.data.sql_analysis:
--   accounts        all 30,000 accounts (semantic column names, engineered features, split)
--   train_accounts  development training split only (18,000 accounts)
--
-- Exploratory queries use train_accounts so that no validation/test outcome informs any
-- modelling choice. Data-quality checks and the overall default rate use all rows.
-- Each query starts with "-- name:" and "-- description:" headers; results are written to
-- artifacts/metrics/sql/<name>.csv.

-- name: overall_default_rate
-- description: Default counts and rates for the full dataset and each split (all rows).
SELECT
    COALESCE(split, 'ALL') AS split,
    COUNT(*)                               AS n_accounts,
    SUM(DEFAULT_NEXT_MONTH)                AS n_defaults,
    ROUND(AVG(DEFAULT_NEXT_MONTH), 4)      AS default_rate
FROM accounts
GROUP BY ROLLUP (split)
ORDER BY CASE COALESCE(split, 'ALL')
    WHEN 'ALL' THEN 0 WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END;

-- name: default_rate_by_pay0
-- description: Default rate by most recent (September 2005) repayment status code. Codes -2 and 0 are undocumented by the source.
SELECT
    PAY_0                                   AS status_code,
    CASE WHEN PAY_0 IN (-2, 0) THEN 'undocumented'
         WHEN PAY_0 = -1 THEN 'pay duly'
         ELSE 'delay ' || PAY_0 || ' month(s)' END AS official_meaning,
    COUNT(*)                                AS n_accounts,
    SUM(DEFAULT_NEXT_MONTH)                 AS n_defaults,
    ROUND(AVG(DEFAULT_NEXT_MONTH), 4)       AS default_rate
FROM train_accounts
GROUP BY PAY_0
ORDER BY PAY_0;

-- name: repayment_delay_patterns
-- description: Number of months (of six) with a documented payment delay (status >= 1), with default rates.
SELECT
    N_MONTHS_DELAYED                        AS months_delayed_of_6,
    COUNT(*)                                AS n_accounts,
    ROUND(COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS share_of_accounts,
    ROUND(AVG(DEFAULT_NEXT_MONTH), 4)       AS default_rate
FROM train_accounts
GROUP BY N_MONTHS_DELAYED
ORDER BY N_MONTHS_DELAYED;

-- name: status_code_by_month
-- description: Share of accounts with each repayment status code in each month (Apr-Sep 2005), via UNPIVOT.
WITH long AS (
    UNPIVOT (SELECT ID, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 FROM train_accounts)
    ON PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
    INTO NAME status_column VALUE status_code
)
SELECT
    status_column,
    CASE status_column WHEN 'PAY_0' THEN '2005-09' WHEN 'PAY_2' THEN '2005-08'
        WHEN 'PAY_3' THEN '2005-07' WHEN 'PAY_4' THEN '2005-06'
        WHEN 'PAY_5' THEN '2005-05' ELSE '2005-04' END AS month,
    ROUND(AVG((status_code = -2)::INT), 4)  AS share_code_m2,
    ROUND(AVG((status_code = -1)::INT), 4)  AS share_code_m1,
    ROUND(AVG((status_code = 0)::INT), 4)   AS share_code_0,
    ROUND(AVG((status_code = 1)::INT), 4)   AS share_code_1,
    ROUND(AVG((status_code = 2)::INT), 4)   AS share_code_2,
    ROUND(AVG((status_code >= 3)::INT), 4)  AS share_code_3plus,
    SUM((status_code = 1)::INT)             AS n_code_1
FROM long
GROUP BY status_column
ORDER BY month;

-- name: status_transition_aug_to_sep
-- description: Transition of grouped status from August (PAY_2) to September (PAY_0) with next-month default rate.
WITH g AS (
    SELECT
        CASE WHEN PAY_2 <= 0 THEN CAST(PAY_2 AS VARCHAR) ELSE 'delay>=1' END AS aug_status,
        CASE WHEN PAY_0 <= 0 THEN CAST(PAY_0 AS VARCHAR) ELSE 'delay>=1' END AS sep_status,
        DEFAULT_NEXT_MONTH
    FROM train_accounts
)
SELECT aug_status, sep_status,
       COUNT(*) AS n_accounts,
       ROUND(AVG(DEFAULT_NEXT_MONTH), 4) AS default_rate
FROM g
GROUP BY aug_status, sep_status
HAVING COUNT(*) >= 30
ORDER BY aug_status, sep_status;

-- name: bill_amount_distribution
-- description: Quantiles of monthly bill statement amounts (NTD), Apr-Sep 2005, with share of negative (credit) balances.
WITH long AS (
    UNPIVOT (SELECT ID, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6
             FROM train_accounts)
    ON BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6
    INTO NAME bill_column VALUE amount
)
SELECT
    bill_column,
    MIN(amount)                                  AS min,
    QUANTILE_CONT(amount, 0.05)                  AS p05,
    QUANTILE_CONT(amount, 0.25)                  AS p25,
    QUANTILE_CONT(amount, 0.50)                  AS median,
    QUANTILE_CONT(amount, 0.75)                  AS p75,
    QUANTILE_CONT(amount, 0.95)                  AS p95,
    MAX(amount)                                  AS max,
    ROUND(AVG(amount), 1)                        AS mean,
    ROUND(AVG((amount < 0)::INT), 4)             AS share_negative,
    ROUND(AVG((amount = 0)::INT), 4)             AS share_zero
FROM long
GROUP BY bill_column
ORDER BY bill_column;

-- name: payment_to_bill_ratio
-- description: Distribution of six-month payments / six-month positive bills (capped at 5) and default rate per band.
SELECT
    CASE
        WHEN NO_POSITIVE_BILL_6M = 1   THEN '0: no positive bill'
        WHEN PAY_TO_BILL_6M < 0.05     THEN '1: [0, 0.05)'
        WHEN PAY_TO_BILL_6M < 0.10     THEN '2: [0.05, 0.10)'
        WHEN PAY_TO_BILL_6M < 0.25     THEN '3: [0.10, 0.25)'
        WHEN PAY_TO_BILL_6M < 0.50     THEN '4: [0.25, 0.50)'
        WHEN PAY_TO_BILL_6M < 1.00     THEN '5: [0.50, 1.00)'
        ELSE                                '6: >= 1.00'
    END                                     AS pay_to_bill_band,
    COUNT(*)                                AS n_accounts,
    ROUND(AVG(DEFAULT_NEXT_MONTH), 4)       AS default_rate,
    ROUND(MEDIAN(PAY_TO_BILL_6M), 4)        AS median_ratio
FROM train_accounts
GROUP BY pay_to_bill_band
ORDER BY pay_to_bill_band;

-- name: credit_limit_deciles
-- description: Credit limit (LIMIT_BAL, NTD) deciles via NTILE window function, with default rate per decile.
WITH d AS (
    SELECT LIMIT_BAL, DEFAULT_NEXT_MONTH,
           NTILE(10) OVER (ORDER BY LIMIT_BAL, ID) AS limit_decile
    FROM train_accounts
)
SELECT limit_decile,
       MIN(LIMIT_BAL) AS min_limit,
       MAX(LIMIT_BAL) AS max_limit,
       COUNT(*)       AS n_accounts,
       ROUND(AVG(DEFAULT_NEXT_MONTH), 4) AS default_rate
FROM d
GROUP BY limit_decile
ORDER BY limit_decile;

-- name: default_by_risk_bands
-- description: Default rate across non-sensitive bands: latest utilisation x worst delay in six months.
SELECT
    CASE WHEN UTIL_1 < 0.10 THEN '1: <10%'
         WHEN UTIL_1 < 0.50 THEN '2: 10-50%'
         WHEN UTIL_1 < 0.90 THEN '3: 50-90%'
         ELSE '4: >=90%' END                AS utilisation_band,
    CASE WHEN MAX_DELAY = 0 THEN 'no delay'
         WHEN MAX_DELAY = 1 THEN '1 month'
         WHEN MAX_DELAY = 2 THEN '2 months'
         ELSE '3+ months' END               AS worst_delay_6m,
    COUNT(*)                                AS n_accounts,
    ROUND(AVG(DEFAULT_NEXT_MONTH), 4)       AS default_rate
FROM train_accounts
GROUP BY utilisation_band, worst_delay_6m
ORDER BY utilisation_band, worst_delay_6m;

-- name: data_quality_checks
-- description: Missing, undocumented and unusual values across ALL rows (no outcome information used).
SELECT 'rows' AS check_name, COUNT(*)::BIGINT AS n FROM accounts
UNION ALL SELECT 'distinct IDs', COUNT(DISTINCT ID) FROM accounts
-- NULL cells summed over every column (per-column COUNT(*) - COUNT(col), then summed).
UNION ALL SELECT 'NULL cells (all columns)', list_sum([*COLUMNS(*)])::BIGINT
    FROM (SELECT COUNT(*) - COUNT(COLUMNS(*)) FROM accounts)
UNION ALL SELECT 'target not in {0,1}', COUNT(*) FROM accounts
    WHERE DEFAULT_NEXT_MONTH NOT IN (0, 1)
UNION ALL SELECT 'PAY_0 = -2 (undocumented)', COUNT(*) FROM accounts WHERE PAY_0 = -2
UNION ALL SELECT 'PAY_0 = 0 (undocumented)', COUNT(*) FROM accounts WHERE PAY_0 = 0
UNION ALL SELECT 'any status code 1 in PAY_2..PAY_6', COUNT(*) FROM accounts
    WHERE 1 IN (PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
UNION ALL SELECT 'EDUCATION in {0,5,6} (undocumented)', COUNT(*) FROM accounts
    WHERE EDUCATION IN (0, 5, 6)
UNION ALL SELECT 'MARRIAGE = 0 (undocumented)', COUNT(*) FROM accounts WHERE MARRIAGE = 0
UNION ALL SELECT 'any negative bill amount', COUNT(*) FROM accounts
    WHERE LEAST(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6) < 0
UNION ALL SELECT 'any bill above credit limit', COUNT(*) FROM accounts
    WHERE GREATEST(BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6) > LIMIT_BAL
UNION ALL SELECT 'all bills <= 0 and no payments (inactive)', COUNT(*) FROM accounts
    WHERE NO_POSITIVE_BILL_6M = 1
      AND PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6 = 0
UNION ALL SELECT 'negative payment amount', COUNT(*) FROM accounts
    WHERE LEAST(PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6) < 0;

-- name: demographic_audit
-- description: AUDIT ONLY (training split, aggregate). Default rates by sensitive attributes with sample sizes. Never used as model inputs or policy rules.
SELECT 'SEX' AS attribute, CAST(SEX AS VARCHAR) AS code_value, COUNT(*) AS n_accounts,
       ROUND(AVG(DEFAULT_NEXT_MONTH), 4) AS default_rate
FROM train_accounts GROUP BY SEX
UNION ALL
SELECT 'EDUCATION', CAST(EDUCATION AS VARCHAR), COUNT(*), ROUND(AVG(DEFAULT_NEXT_MONTH), 4)
FROM train_accounts GROUP BY EDUCATION
UNION ALL
SELECT 'MARRIAGE', CAST(MARRIAGE AS VARCHAR), COUNT(*), ROUND(AVG(DEFAULT_NEXT_MONTH), 4)
FROM train_accounts GROUP BY MARRIAGE
UNION ALL
SELECT 'AGE_BAND',
       CASE WHEN AGE < 30 THEN '21-29' WHEN AGE < 40 THEN '30-39'
            WHEN AGE < 50 THEN '40-49' ELSE '50+' END,
       COUNT(*), ROUND(AVG(DEFAULT_NEXT_MONTH), 4)
FROM train_accounts GROUP BY 2
ORDER BY attribute, code_value;
