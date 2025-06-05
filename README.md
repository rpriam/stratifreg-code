# <br>
Computer code for "Family of linear regression mixture models stratified along the outcome" <br>

## Main classes

- `Joint2Regressor`: Piecewise regression with continuity constraints for two groups.
- `JointKRegressor`: Stratified regression across K groups, with optional joint constraints and quantile support.
- `Joint2GMMRegressor`: EM algorithm for piecewise Gaussian mixture regression with constraints.
- `JointUtils`: Utilities for group splitting, median finding, etc.

## Key Features

- Joint multi-group regression with continuity or custom constraints at the join point.
- Supports quantile regression, penalized regression (lasso, ridge, elasticnet), and stratified GMM.
- Tools for piecewise or regime regression models, not directly available in scikit-learn or statsmodels.

## Notebook: <br>

## Datasets:

|     | Name | n | p | X | y |
| --- | --- | --- | --- | --- | --- |
D1 | covid-19        | 4361  | 6   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_datasurvey.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_datasurvey.csv) |
D2 | pre-diabet      | 3059  | 4   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_prediabet.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_prediabet.csv) |
D3 | life-expectancy | 2928  | 16  | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_lifeexpectancy.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_lifeexpectancy.csv) |
D4 | pisa-2009       | 5233  | 20  | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_pisa2009.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_pisa2009.csv) |
D5 | housing         | 20640 | 8   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_california_housing.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_california_housing.csv) |

Note : This package is provided “as is” for reproducing results, even if not all features are fully implemented or tested.

