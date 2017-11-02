
# Exercise 3.6

The least squares line is given by:

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1} \times x$$

where $\hat{\beta_0}$ and $\hat{\beta_1}$ are the least squares coefficient estimates for simple linear regression.

By definition, $\hat{\beta_0}$ is:

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1} \times \bar{x}$$

where $\bar{y}$ and $\bar{x}$ are the average values of $y$ and $x$, respectively.

Since we want to know if the least squares line always passes through the point ($\bar{x}$, $\bar{y}$), all we have to do is to substitute ($\bar{x}$, $\bar{y}$) into the first equation above and see if the condition is satisfied. We get:

$$\bar{y} = \hat{\beta_0} + \hat{\beta_1} \times \bar{x}$$

and substituting the expression above for $\hat{\beta_0}$, we obtain:

$$\bar{y} = \bar{y} - \hat{\beta_1} \times \bar{x} + \hat{\beta_1} \times \bar{x}$$

Since this is always true, we conclude that the least squares line always passes through the point ($\bar{x}$, $\bar{y}$). 

