
# Exercise 4.6

## (a)

Multiple logistic regression model is represented by the following equation:

$$ p(x) = \frac{e^{\beta_0 + \beta_1 \times X_1 + ... + \beta_p \times X_p}}{1 + e^{\beta_0 + \beta_1 \times X_1 + ... + \beta_p \times X_p}}$$

In this case, since we have two features, the equation is as follows:

$$ p(x) = \frac{e^{\beta_0 + \beta_1 \times X_1 + \beta_2 \times X_2}}{1 + e^{\beta_0 + \beta_1 \times X_1 + \beta_2 \times X_2}}$$

Considering that $\beta_0 = -6$, $\beta_1 = 0.05$, $\beta_2 = 1$, $X_1 = 40$ and $X_2 = 3.5$, we have:

$$ p(x) = \frac{e^{-6 + 0.05 \times 40 + 1 \times 3.5}}{1 + e^{-6 + 0.05 \times 40 + 1 \times 3.5}} = 0.3775$$

Thus, the probability that a student who studies for 40 h and has an undergrad GPA of 3.5 gets an A in the class is <b>37.75%</b>.

## (b)

What we are asking is how many hours a student needs to study ($X_1$) to have p(x) = 0.5. Replacing in the equation presented before:

$$ p(x) = \frac{e^{-6 + 0.05 \times X_1 + 1 \times 3.5}}{1 + e^{-6 + 0.05 \times X_1 + 1 \times 3.5}} \Leftrightarrow \frac{1}{2} = \frac{e^{-6 + 0.05 \times X_1 + 1 \times 3.5}}{1 + e^{-6 + 0.05 \times X_1 + 1 \times 3.5}}.$$

This will be true when the numerator equals 1, that is, when

$$e^{-6 + 0.05 \times X_1 + 1 \times 3.5 }= 1 \Leftrightarrow $$ 

$$ \Leftrightarrow -6 + 0.05 \times X_1 + 1 \times 3.5 = \log(1) = 0 \Leftrightarrow $$

$$ \Leftrightarrow X_1 = 50.$$

To have a 50% chance of getting an A in the class a student needs to study <b>50 hours</b>.
