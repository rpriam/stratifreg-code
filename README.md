# computer code for communication "Family of linear regression mixture models stratified along the outcome"

Linear regression is one of the most studied model, it as-
sumes a clear hypothesis of linearity. Underlying issues com-
ing from Yule-Simpson’s paradox or more generally hidden
nonlinearities lead to spurious correlations difficult to detect
in practice and prone to induce a mistaken linear model.
The concern is when the model for explaining/predicting the
outcome cannot be kept the same for the whole sample, it
changes accordingly to the dependent variable. Hence, it
is proposed a stratification of the outcome which leads to
a new family of mixture models of regressions. A break or
more along the outcome changes the linear regression into
several components instead of one. A difference with the
existing mixture models of regressions is that the partion-
ing now depends mainly on the outcome. A double check
of the change is obtained via an additional ordinal model
and a discretization of the outcome. For the validation of
the mixture, it is required a decrease of the bic, the aic and
a mse or mae for both the continuous and discretized out-
comes. Graphically, it is also shown these indicators plus
the determination coefficient for moving thresholds in order
to visualize the change between intervals of outcomes. With
a threshold equal to the median, the approach is illustrated
for several real datasets in the presented experiments. It is
applied with a medical dataset from the Covid-19 lockdown
in spring 2020.



