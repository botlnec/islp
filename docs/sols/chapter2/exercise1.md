
# Exercise 2.1

To answer to this exercise, we need to understand the <b>sources of error</b> in a statistical learning method. For regression, assuming $Y = f(X) + \varepsilon$, where $E[\varepsilon]=0$ and $Var[\varepsilon]=\sigma_\varepsilon^2$, we can always obtain a decomposition of the test mean squared error, $E[(Y - \hat{f}(x_0))^2$, into the sum of the irreducible error, the squared bias and the variance [1, page 223]:

$$\mathrm{E}\big[\big(Y - \hat{f}(x)\big)^2 \big| X=x_0 \big] = \sigma_\varepsilon^2 + \mathrm{Bias}^2\big[\hat{f}(x_0)\big] + \mathrm{Var}\big[ \hat{f}(x_0) \big]$$


where $\sigma_\varepsilon^2$ is the noise or irreducible error, 

$$\mathrm{Bias}\big[\hat{f}(x_0)\big] = \mathrm{E}\big[\hat{f}(x_0) - f(x_0)\big]$$

and

$$\mathrm{Var}\big[\hat{f}(x_0)\big] = \mathrm{E}[\hat{f}(x_0)^2] - \mathrm{E}[\hat{f}(x_0)]^2.$$


Since the irreducible error corresponds to the lowest achievable error, a good test set performance of a statistical learning method requires low variance as well as low squared bias.

When we approximate a problem (possibly very complex), by a simpler model we introduce an error known as <b>bias</b>.
The simplest non-trivial example might be approximating a non-linear relationship (for example, a quadratic one) by a linear function of parameters and predictors. In this case, will have a always non-zero test error, regardless of how well we fit the model parameters, how large the training set is, or even how small the noise is (even zero). The more the true model deviates from a linear one, the larger this error will be.

On the other hand, <b>variance</b> refers to the amount by which the estimation function would change if it was estimated using a different training set.
The training set is used to estimate the model parameters, which means that we obtain different estimates from different training sets. We hope however that this difference is small, and we say that between estimates from different training sets is small, in which case we say that the learning method has low variance.
On the other hand, a method for which small changes in the training set might lead to large changes in the estimated model parameters is referred as a method with high variance.

In general, more flexible methods have less bias and have higher variance. This is referred to as the  <b>bias-variance trade-off</b> since a low test mean squared error requires both low bias and low variance.

### (a) Extremely large sample, few predictors

A flexible method is expected to be **better**. 

Since the sample size is extremely large and the number of predictors is small, a more flexible method would be able to better fit the data, while not fitting the noise due to the very large sample size. In other words a more flexible model would have the upside of a less bias, without much risk of [overfitting](https://www.youtube.com/watch?v=DQWI1kvmwRg).

### (b) Awful lot of predictors, small sample

A flexible method is expected to be  **worse**.

It is very likely that when the number of predictors is extremely large and the number of observations is small a flexible model would fit the noise, meaning that, given another random data set of the same distribuition, the fit would likely be significantly different. Therefore one would be better off using a less flexible method, which will have more bias, but will be less likely to overfit.

### (c) Highly non-linear relationship

A flexible method is expected to be  **better**.

A more flexible method will likely be necessary to model a highly non-linear relationship, otherwise the model will be too biased and not capture the non-linearities of the model. No matter how large the sample size, a less flexible model would always be limited.

### (d) Extremely high variance

A flexible method is expected to be  **worse**.

Since the variance is extremely high, a more flexible model will fit the noise more and thus very likely overfit. A less flexible model will be more likely to still capture the essential 'features' of the model without picking up extraneous ones induced by the noise.

## Further reading

The bias-variance decomposition is a fundamental aspect of machine learning which is present not only in regression. This decomposition has been generalized to more general loss functions and to classification learning methods. See, for example, [2].


* [1] Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/). Springer, 2009.
* [2] James, Gareth M. ["Variance and bias for general loss functions."](http://www-bcf.usc.edu/~gareth/research/bv.pdf)  Machine learning 51.2 (2003): 115-135. 
