# <br>
Computer code for "Family of linear regression mixture models stratified along the outcome" <br>

# Structure of the library

| File          | Class                 | Method(s) (public)                                             |
|---------------|-----------------------|----------------------------------------------------------------|
| two_groups.py | Joint2Regressor       | fit_ols_groups, fit_ols_jointure, fit_ols_jointure_a_b, fit_ols_jointure_smoothed, solve_ols_constrained, solve_ols_constrained_het, assemble_block_matrix, build_constraint_vector, build_constraint_matrix, variance_constrained, variance_constrained_het, check_jointure_constraint, compare_models, predict |
| k_groups.py   | JointKRegressor       | fit, check_jointure_constraint, compare_models, predict         |
| gmm.py        | Joint2GMMRegressor    | fit, solve_constrained_regression, check_jointure_constraint, predict |
| utils.py      | JointUtils            | _as_numpy, add_intercept, solve_with_fallbacks, split_by_median, find_x0_LL, find_x0 |

## Main Fit Methods

| Class               | Method                    | Description                                           |
|---------------------|--------------------------|-------------------------------------------------------|
| Joint2Regressor     | fit_ols_groups           | Fits separate OLS models to two groups of data.       |
| Joint2Regressor     | fit_ols_jointure         | Fits OLS with a continuity constraint at the join.    |
| Joint2Regressor     | fit_ols_jointure_a_b     | Fits OLS enforcing join-point continuity (two cases). |
| Joint2Regressor     | fit_ols_jointure_smoothed| Fits OLS with a soft (penalized) continuity at join.  |
| JointKRegressor     | fit                      | Fits piecewise regression across K groups.            |
| Joint2GMMRegressor  | fit                      | Fits Gaussian Mixture regression for two groups.      |

Notebook: <br>


Datasets:

|     | Name | n | p | X | y |
| --- | --- | --- | --- | --- | --- |
D1 | covid-19        | 4361  | 6   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_datasurvey.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_datasurvey.csv) |
D2 | pre-diabet      | 3059  | 4   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_prediabet.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_prediabet.csv) |
D3 | life-expectancy | 2928  | 16  | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_lifeexpectancy.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_lifeexpectancy.csv) |
D4 | pisa-2009       | 5233  | 20  | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_pisa2009.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_pisa2009.csv) |
D5 | housing         | 20640 | 8   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_california_housing.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_california_housing.csv) |


