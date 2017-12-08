
# Exercise 4.5

## (a)

Regardless of the Bayes decision boundary, we expect QDA to perform better than LDA on the training set. This is because QDA is more flexible which leads to a closer fit. However if the Bayes decision boundary is linear, the additional flexibility of QDA leads to overfit, and LDA is expected to perform better than QDA on the test set.

## (b)

As mentioned in the previous paragraph, due to the additional flexibility we expect QDA to perform better than LDA on the training set. If the Bayes decision boundary is non-linear we expect that QDA will also perform better on the test set, since the additional flexibility allows it to capture at least some of the non-linearity. In other words, LDA is biased leading to a worse performance on the test set (QDA could be biased as well depending on the nature of the non-linearity of the Bayes decision boundary, but it will be less so in general).

## (c)

In both cases of linear and non-linear Bayes decision boundary we expect the performance of QDA to improve relative to LDA, as $n$ increases.
In the linear boundary case, QDA will have a worse performance on the test set since its excessive flexibility will cause it to overfit, but [this overfitting](https://www.youtube.com/watch?v=DQWI1kvmwRg) will decrease as $n$ increases as the variance is reduced, and QDA will improve relative to LDA.
For a non-linear Bayes decision boundary, LDA is biased and will not improve significantly past a certain sample size. QDA on the other hand, will see its variance reduced while benefitting from a more flexible model that captures the underlying non-linearity better leading to a closer fit.

In general, as $n$ increases the more flexible model (QDA) sees its fit improve as the variance is reduced with the increasing sample size.

## (d)

False. For a linear Bayes decision boundary, QDA is too flexible compared to LDA and the noise in the data will cause it to overfit. As the sample size increases the overfitting is reduced, but in general we still expect LDA to better since it is unbiased and less prone to fit the noise.
