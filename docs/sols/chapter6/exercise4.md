
# Exercise 6.4

Important concepts to solve this exercise:
* <b>RSS</b>. The residual sum of squares measures the error of the predictive model. It is given by the sum of squared differences between estimated values and real values.
* <b>Variance</b>. Variance is a measure of how the model error changes when the dataset changes. In a model with high variance, we can have good performances for some datasets (e.g. training set) but poor performance for other datasets (e.g. test set). By the contrary, when the variance of the model is low, its performance does not change much with the change of the datasets.
* <b>Bias</b>. Bias is a measure of the model capacity to capture data complexity. This means that a model with high bias is a model that can't represent the relationship between predictors and response (e.g. using a linear model to represent a quadratic relationship).

# (a)

The right answer is <b>(iii)</b>. When $\lambda$ increases from 0, the flexibility of the model decreases and it gets less fitted to the training set. This means that the training RSS increases.

# (b)

The right answer is <b>(ii)</b>. The increase of $\lambda$ leads to a reduction of variance and an increase of bias. In the initial stage, the reduction of variance prevails over the bias increasement and the test RSS decreases. However, in later stages, the error due to bias tends to surpass the error due to variance. Then the test RSS start increasing.

# (c)

The right answer is <b>(iv)</b>. As the number of effective predictors reduces with the increase of $\lambda$, variance starts reducing also. In the limiting case, in which there are no predictors, the model is a constant and the prediction is the same for any dataset.

# (d)

The right answer is <b>(iii)</b>. The increase of $\lambda$ from 0 leads to a reduction of the model flexibility (less predictors considered). Thus, the model capacity to represent the relationship between predictors and response is always decreasing.

# (e)

The irreducible error is independent from the model. Consequently, changes in the model don't affect the irreducible error.
