
# Exercise 4.4

## (a)

Since the set of observations X is uniformly (evenly) distributed on [0, 1] and considering that we are using a 10% range, the fraction of available information used is, on average, <b>9.75%</b>.

$$
\int_{0}^{.05} .05 + x \ dx + \int_{.05}^{.95} 1 \ dx  + \int_{.95}^{1} 1.05 -x  \ dx 
$$


## (b)

We have two sets of observations, $X_1$ and $X_2$, that are uniformly distributed on [0,1]x[0,1]. We wish to predict a test observation’s response using only observations that are within 10% of a $X_1$ range <b>and</b> within 10% of a $X_2$ range. This means that we want to evaluate an <b>intersection</b>, so we should apply the rule of multiplication. 

We have $10\% \times 10\% = 1\%$.

## (c)

This case is an extension of (b), so applying the rule of multiplication we have $10\% \times 10\% \times \cdots \times 10\%  = (10\%)^{100} = 10^{-98}\%$.

## (d)

The drawback of KNN when p is large is that there are relatively very few training observations “near” any given test observation. As we have seen above, the fraction of points near a test observation can be extremely small, and most likely 0 is p is large enough. Essentially to be "near" a point implies being "near" in every dimension, and this gets less and less likely as the number dimensions increases. In other words, there are no neighbors in high dimensions. (In practice, the underlying distributions are not uniformly random and have a certain structure so points can still lump together somewhat.)

## (e)

* $p = 1$.

In this case, the hypercube is a line segment centered around the training  observation. This line has the length needed to contain 10% of the training observations. Since our observations are uniformly distributed on [0,1], the corresponding length  of the hypercube side is <b>0.10</b>.

* $p = 2$.

Two features means that the hypercube is a square. We can imagine this square defined by two axis, both correspoding to a set of observations uniformly distributed on [0,1]x[0,1]. To have 10% of the training observations, each side $x$ of the square must have:

$$x^2 = 0.10 \times (1 \times 1) \Leftrightarrow x^2 = 0.10 \Leftrightarrow x = \sqrt{0.10}$$

Thus, the correspoding length of the hypercube side is <b>$\sqrt{0.10}\approx 0.31$</b> . So to get 10% of the observations we "need a square" whose side length is about 30% of the total width of the space.

* $p = 100$.

In the last case the hypercube is a 100-dimensional cube. Applying the same logic as above, we have:

$$x^{100} = 0.10 \times (1^{100}) \Leftrightarrow x = \sqrt[100]{0.10}$$

Accordingly, the corresponding length of the hypercube side is <b>$\sqrt[100]{0.10}\approx 0.98$</b> . Again we see that, in a way, there are no neighbors in high dimensions. It is a strange "local" method, one that uses 98% of the observations for each dimension.
