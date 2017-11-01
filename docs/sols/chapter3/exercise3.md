
# Exercise 3.3

## (a)

$Y = \beta_0 + \beta_1 \times GPA + \beta_2 \times IQ + \beta_3 \times Gender + \beta_4 \times GPA \times IQ + \beta_5 \times GPA \times Gender$

For a fixed value of GPA and IQ, the difference between female and male is given by:

$Y_{female} - Y_{male} = \beta_3 + \beta_5 \times GPA = 35 - 10 GPA$, 

which depends on GPA. It is clear that in the normal range of the GPA (0 to 4.0), the difference in expected salary between female and male ranges linearly from 35 to -5. In particular, if GPA > 3.5, males earn on average more than females. Therefore, ** the correct answer is (iii)**.

## (b)

The **predicted salary is 137.1 (thousand dollars)**. Given the coefficients from the fit, GPA = 4.0, IQ = 110 and Gender = 1, the model predicts:

$Y = \beta_0 + \beta_1 GPA + \beta_2 IQ + \beta_3 Gender + \beta_4 (GPA \times IQ) + \beta_5 (GPA \times Gender)$

$Y = 50 + 20 \times 4.0 + 0.07 \times 110 + 35 \times 1  + 0.01 \times 4.0 \times 110 + (-10) \times 4.0 \times 1$ = 137.1

## (c)

**False**. Although the coefficient for the GPA/IQ interaction term is very small, specially when compared to the other coefficients, this does not indicate whether there is an interaction effect. First, this coefficient is multiplied by the product  of IQ and GPA which ranges from 0 to a few hundred, so that the contribution to the response would tipically add up to a value between 2 and 6, let's say. Secondly, and more importantly, evidence for the interaction effect has to be evaluated with a t-statistic or an F-statistic for a null hypothesis ($H_0: \beta_4 = 0$), yielding a certain p-value. This requires the standard error of which we have no information, and therefore cannot conclude whether there is evidence for a interaction effect.

## Additional calculations


```python
50 + 20*4 + 0.07*110 + 35*1 + 0.01*4*110 + (-10)*4*1
```




    137.1


