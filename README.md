# computer code for communication "Family of linear regression mixture models stratified along the outcome"

Abstract: Linear regression is one of the most studied model, it assumes a clear hypothesis of linearity. Underlying issues coming from Yule-Simpson’s paradox or more generally hidden
nonlinearities lead to spurious correlations difficult to detect in practice and prone to induce a mistaken linear model.
The concern is when the model for explaining/predicting the outcome cannot be kept the same for the whole sample, it changes accordingly to the dependent variable. Hence, it
is proposed a stratification of the outcome which leads to a new family of mixture models of regressions. A break or more along the outcome changes the linear regression into 
several components instead of one. A difference with the existing mixture models of regressions is that the partioning now depends mainly on the outcome. A double check
of the change is obtained via an additional ordinal model and a discretization of the outcome. For the validation of the mixture, it is required a decrease of the bic, the aic and
a mse or mae for both the continuous and discretized outcomes. Graphically, it is also shown these indicators plus the determination coefficient for moving thresholds in order
to visualize the change between intervals of outcomes. With a threshold equal to the median, the approach is illustrated for several real datasets in the presented experiments. It is
applied with a medical dataset from the Covid-19 lockdown in spring 2020.

Datasets:

|     | Name | n | p | X | y |
| --- | --- | --- | --- | --- | --- |
D1 | covid-19        | 4361  | 6   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_datasurvey.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_datasurvey.csv) |
D2 | pre-diabet      | 3059  | 4   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_prediabet.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_prediabet.csv) |
D3 | life-expectancy | 2928  | 16  | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_lifeexpectancy.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_lifeexpectancy.csv) |
D4 | pisa-2009       | 5233  | 20  | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_pisa2009.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_pisa2009.csv) |
D5 | housing         | 20640 | 8   | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/Xf_all_california_housing.csv) | [.csv](https://github.com/rpriam/stratifreg-code/blob/main/datasets/yf_all_california_housing.csv) |


