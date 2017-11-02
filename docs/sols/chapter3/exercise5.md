
# Exercise 3.5

This is a simple exercise of plugging the expression for $\hat{\beta}$ in the formula of the ith fitted value:

$$
\hat{y}_i = x_i \hat{\beta} = x_i \left( \sum_{i'=1}^n x_{i'} y_{i'} \right) / \left( \sum_{j=1}^n x_j^2 \right) = \sum_{i'=1}^n \left( \frac{x_i x_{i'} }{\sum_{j=1}^n x_j^2} y_{i'} \right)
$$

and comparing to the expression

$$
\hat{y}_i = \sum_{i'=1}^n a_{i'} y_{i'}
$$

to obtain:

$$
a_{i'} =  \frac{x_i x_{i'} }{\sum_{j=1}^n x_j^2}
$$
