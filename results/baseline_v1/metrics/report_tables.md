# Generated result tables

## dataset

| split | accounts | observed defaults | default_rate |
|---|---|---|---|
| train | 18000 | 3982 | 0.2212 |
| validation | 6000 | 1327 | 0.2212 |
| test | 6000 | 1327 | 0.2212 |

Total: 30,000 accounts, 23 source features, observed default rate 0.2212 (6,636 defaults).

## test_metrics

| model | ROC-AUC | AUC 95% CI | PR-AUC (AP) | AP 95% CI | Brier | log loss | accuracy @0.5 |
|---|---|---|---|---|---|---|---|
| dummy_raw | 0.5000 | [0.5000, 0.5000] | 0.2212 | [0.2103, 0.2317] | 0.1723 | 0.5284 | 0.7788 |
| logistic_raw | 0.7799 | [0.7658, 0.7941] | 0.5389 | [0.5116, 0.5675] | 0.1355 | 0.4322 | 0.8197 |
| xgboost_raw | 0.7899 | [0.7765, 0.8042] | 0.5592 | [0.5303, 0.5887] | 0.1335 | 0.4244 | 0.8215 |

## test_difference

| metric | difference (XGBoost − logistic) | paired bootstrap 95% CI |
|---|---|---|
| roc_auc | +0.0100 | [+0.0039, +0.0161] |
| average_precision | +0.0205 | [+0.0068, +0.0338] |
| brier | -0.0020 | [-0.0032, -0.0008] |
| log_loss | -0.0079 | [-0.0114, -0.0044] |

## threshold_metrics

| model | threshold (from validation) | precision | recall | F1 | accuracy | flagged | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic_raw | 0.2635 | 0.5016 | 0.5870 | 0.5410 | 0.7797 | 0.259 | 3899 | 774 | 548 | 779 |
| xgboost_raw | 0.2866 | 0.5252 | 0.5494 | 0.5370 | 0.7905 | 0.231 | 4014 | 659 | 598 | 729 |
| dummy_raw | 0.2212 | 0.2212 | 1.0000 | 0.3622 | 0.2212 | 1.000 | 0 | 4673 | 0 | 1327 |

## validation_all

| model | roc_auc | average_precision | brier | log_loss | ece | calibration_intercept | calibration_slope |
|---|---|---|---|---|---|---|---|
| dummy_raw | 0.5000 | 0.2212 | 0.1723 | 0.5284 | 0.0001 | — | — |
| logistic_raw | 0.7718 | 0.5397 | 0.1354 | 0.4332 | 0.0198 | -0.0603 | 0.9680 |
| logistic_sigmoid | 0.7718 | 0.5397 | 0.1354 | 0.4332 | 0.0198 | -0.0494 | 0.9771 |
| logistic_isotonic | 0.7710 | 0.5263 | 0.1355 | 0.4385 | 0.0125 | -0.0801 | 0.9486 |
| xgboost_raw | 0.7844 | 0.5554 | 0.1338 | 0.4269 | 0.0137 | 0.0119 | 1.0250 |
| xgboost_sigmoid | 0.7844 | 0.5554 | 0.1346 | 0.4298 | 0.0302 | -0.0544 | 0.9756 |
| xgboost_isotonic | 0.7834 | 0.5420 | 0.1342 | 0.4283 | 0.0178 | -0.0701 | 0.9593 |

## test_all

| model | roc_auc | average_precision | brier | log_loss | ece | calibration_intercept | calibration_slope |
|---|---|---|---|---|---|---|---|
| dummy_raw | 0.5000 | 0.2212 | 0.1723 | 0.5284 | 0.0001 | — | — |
| logistic_raw | 0.7799 | 0.5389 | 0.1355 | 0.4322 | 0.0127 | -0.0326 | 0.9861 |
| logistic_sigmoid | 0.7799 | 0.5389 | 0.1355 | 0.4322 | 0.0127 | -0.0216 | 0.9954 |
| logistic_isotonic | 0.7795 | 0.5269 | 0.1354 | 0.4422 | 0.0131 | -0.0604 | 0.9602 |
| xgboost_raw | 0.7899 | 0.5592 | 0.1335 | 0.4244 | 0.0127 | 0.0435 | 1.0425 |
| xgboost_sigmoid | 0.7899 | 0.5592 | 0.1345 | 0.4290 | 0.0305 | -0.0308 | 0.9850 |
| xgboost_isotonic | 0.7897 | 0.5461 | 0.1337 | 0.4244 | 0.0147 | -0.0331 | 0.9809 |

## calibration_bootstrap

| comparison (validation) | metric | mean difference | 95% CI |
|---|---|---|---|
| logistic_sigmoid - logistic_raw | brier | -0.00000 | [-0.00004, +0.00003] |
| logistic_sigmoid - logistic_raw | log_loss | -0.00004 | [-0.00014, +0.00006] |
| logistic_isotonic - logistic_raw | brier | +0.00009 | [-0.00037, +0.00052] |
| logistic_isotonic - logistic_raw | log_loss | +0.00476 | [-0.00122, +0.01630] |
| xgboost_sigmoid - xgboost_raw | brier | +0.00073 | [+0.00015, +0.00129] |
| xgboost_sigmoid - xgboost_raw | log_loss | +0.00284 | [+0.00088, +0.00479] |
| xgboost_isotonic - xgboost_raw | brier | +0.00041 | [+0.00003, +0.00080] |
| xgboost_isotonic - xgboost_raw | log_loss | +0.00142 | [+0.00040, +0.00255] |

## selection

| family | chosen variant | qualifying calibrations | F1 threshold |
|---|---|---|---|
| logistic | raw | none | 0.2635 |
| xgboost | raw | none | 0.2866 |

Primary model (lowest validation log loss): **xgboost_raw**.

## hyperparameters

| model | selected hyperparameters | CV log loss | CV ROC-AUC | candidates |
|---|---|---|---|---|
| logistic_raw | C=0.1, class_weight=None | 0.4319 | 0.7776 | 10 |
| xgboost_raw | subsample=0.7, scale_pos_weight=1.0, reg_lambda=20.0, n_estimators=200, min_child_weight=5, max_depth=5, learning_rate=0.02, colsample_bytree=0.8 | 0.4253 | 0.7863 | 30 |

## importance

| model | feature | auc_drop_mean | auc_drop_std |
|---|---|---|---|
| logistic_raw | PAY_0 | 0.0932 | 0.0028 |
| logistic_raw | LIMIT_BAL | 0.0229 | 0.0027 |
| logistic_raw | BILL_AMT1 | 0.0135 | 0.0013 |
| logistic_raw | PAY_3 | 0.0075 | 0.0012 |
| logistic_raw | PAY_6 | 0.0058 | 0.0007 |
| logistic_raw | PAY_5 | 0.0049 | 0.0011 |
| xgboost_raw | PAY_0 | 0.0779 | 0.0027 |
| xgboost_raw | BILL_AMT1 | 0.0206 | 0.0017 |
| xgboost_raw | LIMIT_BAL | 0.0180 | 0.0031 |
| xgboost_raw | PAY_3 | 0.0084 | 0.0011 |
| xgboost_raw | PAY_2 | 0.0084 | 0.0007 |
| xgboost_raw | PAY_4 | 0.0065 | 0.0012 |

## audit

| attribute | group | n | observed_default_rate | mean_predicted_pd | predicted_minus_observed | roc_auc |
|---|---|---|---|---|---|---|
| SEX | 1 male | 2378 | 0.251 | 0.233 | -0.018 | 0.782 |
| SEX | 2 female | 3622 | 0.202 | 0.214 | +0.012 | 0.794 |
| AGE_BAND | 21-29 | 1981 | 0.230 | 0.239 | +0.010 | 0.792 |
| AGE_BAND | 30-39 | 2172 | 0.202 | 0.207 | +0.004 | 0.800 |
| AGE_BAND | 40-49 | 1280 | 0.223 | 0.215 | -0.008 | 0.784 |
| AGE_BAND | 50+ | 567 | 0.261 | 0.232 | -0.029 | 0.755 |
| EDUCATION | 1 graduate | 2138 | 0.206 | 0.194 | -0.012 | 0.810 |
| EDUCATION | 2 university | 2814 | 0.228 | 0.233 | +0.005 | 0.782 |
| EDUCATION | 3 high school | 959 | 0.248 | 0.254 | +0.006 | 0.773 |
| EDUCATION | other/undocumented (0,4,5,6) | 89 | 0.079 | 0.169 | +0.090 | — |
| MARRIAGE | 1 married | 2723 | 0.235 | 0.222 | -0.013 | 0.797 |
| MARRIAGE | 2 single | 3192 | 0.211 | 0.220 | +0.009 | 0.786 |
| MARRIAGE | other/undocumented (0,3) | 85 | 0.141 | 0.257 | +0.116 | — |

## expected_loss

| quantity | value |
|---|---|
| Accounts (test split) | 6,000 |
| Model | xgboost_raw |
| Assumed LGD | 60% |
| Assumed EAD | max(BILL_AMT1, 0) + CCF * max(LIMIT_BAL - max(BILL_AMT1, 0), 0) (CCF = 0.5) |
| Hypothetical total EAD (NTD) | 644.35M |
| Hypothetical total EL (NTD) | 69.91M |
| EL / EAD | 0.1085 |
| Mean PD | 0.2215 |
| EAD-weighted PD | 0.1808 |
| Expected defaults (Σ PD) | 1328.9 |
| Observed defaults in test split | 1,327 |
| Assumed LGD/EAD applied to observed defaults (NTD) | 72.46M |
| Share of EL in top 10% of accounts by EL | 0.391 |
| Share of EL / EAD in top PD decile | 0.268 / 0.071 |

## risk_bands

| band | n_accounts | mean_pd | observed_default_rate | share_of_accounts | share_of_ead | share_of_el |
|---|---|---|---|---|---|---|
| [0.0, 0.05) | 298 | 0.044 | 0.034 | 0.050 | 0.091 | 0.022 |
| [0.05, 0.1) | 1423 | 0.074 | 0.060 | 0.237 | 0.317 | 0.127 |
| [0.1, 0.2) | 2167 | 0.145 | 0.150 | 0.361 | 0.340 | 0.265 |
| [0.2, 0.35) | 969 | 0.255 | 0.262 | 0.162 | 0.119 | 0.168 |
| [0.35, 0.5) | 451 | 0.413 | 0.395 | 0.075 | 0.052 | 0.120 |
| [0.5, 1.0) | 692 | 0.672 | 0.685 | 0.115 | 0.082 | 0.298 |

## el_sensitivity

| ead_method | CCF or limit fraction | LGD 30% | LGD 60% | LGD 90% |
|---|---|---|---|---|
| balance_plus_ccf | 0.0 | 18.09 | 36.18 | 54.27 |
| balance_plus_ccf | 0.25 | 26.52 | 53.05 | 79.57 |
| balance_plus_ccf | 0.5 | 34.96 | 69.91 | 104.87 |
| balance_plus_ccf | 0.75 | 43.39 | 86.78 | 130.17 |
| balance_plus_ccf | 1.0 | 51.82 | 103.65 | 155.47 |
| limit_fraction | 0.25 | 12.86 | 25.73 | 38.59 |
| limit_fraction | 0.5 | 25.73 | 51.45 | 77.18 |
| limit_fraction | 0.75 | 38.59 | 77.18 | 115.77 |
| limit_fraction | 1.0 | 51.45 | 102.91 | 154.36 |

Values in NTD millions.

## scenarios

| scenario | odds × | LGD | ρ | mean PD | analytical EL | MC mean | median | std | VaR 95 | VaR 99 | ES 95 | ES 99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.0 | 0.6 | 0.0 | 0.2215 | 69.91M | 69.93M | 69.90M | 2.10M | 73.39M | 74.81M | 74.30M | 75.67M |
| base_correlated | 1.0 | 0.6 | 0.04 | 0.2215 | 69.91M | 69.60M | 68.22M | 16.48M | 98.71M | 114.04M | 107.78M | 121.22M |
| mild_stress | 1.5 | 0.6 | 0.0 | 0.2811 | 90.36M | 90.37M | 90.35M | 2.32M | 94.20M | 95.83M | 95.22M | 96.68M |
| severe_stress | 2.5 | 0.75 | 0.0 | 0.3688 | 151.93M | 151.95M | 151.93M | 3.21M | 157.24M | 159.42M | 158.61M | 160.68M |
| correlated_downturn | 1.5 | 0.75 | 0.12 | 0.2811 | 112.95M | 112.09M | 107.00M | 41.88M | 188.19M | 229.44M | 212.81M | 248.45M |

20,000 paths per scenario; NTD.

## scenario_checks

| scenario | mean_minus_el_in_se | factor_sample_mean_z | var_99_mc_se | es_99_mc_se | max_abs_marginal_pd_error | P(L > 1.1× base EL) | P(L > 1.25× base EL) | P(L > 1.5× base EL) | P(L > 2.0× base EL) |
|---|---|---|---|---|---|---|---|---|---|
| base | +0.81 | — | 0.05M | 0.09M | 0.0117 | 0.0008 | 0.0000 | 0.0000 | 0.0000 |
| base_correlated | -2.69 | +2.47 | 0.59M | 0.72M | 0.0133 | 0.3056 | 0.1439 | 0.0255 | 0.0003 |
| mild_stress | +0.73 | — | 0.06M | 0.09M | 0.0137 | 1.0000 | 0.9011 | 0.0000 | 0.0000 |
| severe_stress | +0.89 | — | 0.11M | 0.12M | 0.0119 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| correlated_downturn | -2.88 | +2.47 | 1.56M | 1.94M | 0.0165 | 0.7920 | 0.6944 | 0.5197 | 0.2361 |

## mc_unbiasedness

Unbiasedness check (correlated_downturn, 30 independent seeds × 2,000 paths): standardised error (MC mean − analytical EL)/SE has mean +0.03 and SD 1.18; share with |z| > 2: 0.03 (≈0.05 expected for an unbiased estimator).

## rho_sensitivity

| rho | mean | median | std | var_99 | es_99 |
|---|---|---|---|---|---|
| 0.0 | 69.94M | 69.91M | 2.10M | 74.77M | 75.69M |
| 0.02 | 69.75M | 69.04M | 11.75M | 100.46M | 104.79M |
| 0.04 | 69.68M | 68.21M | 16.52M | 114.19M | 121.00M |
| 0.08 | 69.55M | 66.69M | 23.41M | 136.05M | 146.64M |
| 0.12 | 69.45M | 65.10M | 28.81M | 154.09M | 168.25M |
| 0.2 | 69.27M | 61.79M | 37.57M | 185.12M | 205.09M |
| 0.3 | 69.09M | 57.65M | 46.69M | 219.98M | 244.75M |

## lgd_sensitivity

| lgd | rho | mean | var_99 | es_99 |
|---|---|---|---|---|
| 0.3 | 0.0 | 34.97M | 37.38M | 37.84M |
| 0.3 | 0.04 | 34.84M | 57.09M | 60.50M |
| 0.6 | 0.0 | 69.94M | 74.77M | 75.69M |
| 0.6 | 0.04 | 69.68M | 114.19M | 121.00M |
| 0.9 | 0.0 | 104.90M | 112.15M | 113.53M |
| 0.9 | 0.04 | 104.52M | 171.28M | 181.50M |

## observed_vs_simulated

| scenario | observed defaults | simulated mean | simulated 95% interval | share of paths below observed |
|---|---|---|---|---|
| base | 1327 | 1329.2 | [1274, 1386] | 0.464 |
| base_correlated | 1327 | 1323.7 | [822, 1931] | 0.531 |

## policy

| policy | share_selected | mean_pd | observed_default_rate | total_ead | total_expected_loss | total_expected_revenue | expected_net_contribution | realized_hypothetical_contribution | share_el_top10pct_accounts | loss_var_99_rho_0.04 |
|---|---|---|---|---|---|---|---|---|---|---|
| A_retain_all | 1.000 | 0.221 | 0.221 | 644.35M | 69.91M | 3.49M | -66.42M | -69.04M | 0.391 | 113.03M |
| B_pd_threshold | 0.579 | 0.102 | 0.099 | 446.16M | 24.95M | 2.70M | -22.25M | -20.89M | 0.268 | 49.48M |
| C_expected_contribution | 0.000 | — | — | 0.00M | 0.00M | 0.00M | 0.00M | 0.00M | — | 0.00M |
| D_random_equal_size | 0.579 | 0.221 | 0.221 | 372.84M | 40.43M | 2.02M | -38.41M | -39.92M | 0.391 | 65.92M |

Policy D: mean of 1000 random draws. 5–95% range of observed default rate: [0.214, 0.228]; of expected net contribution: [-39.30M, -37.53M].

## policy_threshold

Rule `max_bad_rate` (target observed bad rate 0.1). Selected PD cutoff **0.1748** on validation; validation observed bad rate 0.0996; validation share retained 0.579. Break-even analysis (test): share of accounts with positive expected one-month contribution at r = 0.015: 0.000; highest account break-even PD 0.0244 vs lowest model PD 0.0353; median break-even monthly revenue rate 0.298 (539 accounts have no drawn balance).

## policy_revenue

| monthly_revenue_rate | A_retain_all | B_pd_threshold | C_expected_contribution | D_random_equal_size |
|---|---|---|---|---|
| 0.005 | 1.0 | 0.578 | 0.0 | 0.578 |
| 0.015 | 1.0 | 0.578 | 0.0 | 0.578 |
| 0.03 | 1.0 | 0.578 | 0.0 | 0.578 |
| 0.06 | 1.0 | 0.578 | 0.051 | 0.578 |
| 0.12 | 1.0 | 0.578 | 0.269 | 0.578 |
| 0.25 | 1.0 | 0.578 | 0.43 | 0.578 |

## policy_audit

| attribute | group | n | share_selected_B_pd_threshold | share_selected_C_expected_contribution |
|---|---|---|---|---|
| SEX | 1 male | 2378 | 0.551 | 0.000 |
| SEX | 2 female | 3622 | 0.596 | 0.000 |
| AGE_BAND | 21-29 | 1981 | 0.537 | 0.000 |
| AGE_BAND | 30-39 | 2172 | 0.620 | 0.000 |
| AGE_BAND | 40-49 | 1280 | 0.590 | 0.000 |
| AGE_BAND | 50+ | 567 | 0.541 | 0.000 |
| EDUCATION | 1 graduate | 2138 | 0.627 | 0.000 |
| EDUCATION | 2 university | 2814 | 0.563 | 0.000 |
| EDUCATION | 3 high school | 959 | 0.508 | 0.000 |
| EDUCATION | other/undocumented (0,4,5,6) | 89 | 0.652 | 0.000 |
| MARRIAGE | 1 married | 2723 | 0.581 | 0.000 |
| MARRIAGE | 2 single | 3192 | 0.580 | 0.000 |
| MARRIAGE | other/undocumented (0,3) | 85 | 0.471 | 0.000 |
