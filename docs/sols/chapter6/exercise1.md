
# Exercise 6.1

## (a)

By construction, the best subset method picks the model with exactly $k$ predictors that has the lowest training error, searching all $p!/(p-k)!$ possibilities. Forward and backward stepwise methodes only pick from (p-k) models and, if, in any earlier step it chooses any predictors not chosen by the $k$-th best subset, then it will have these predictors in the $k$-th step which lead to higher training error.

## (b)

In general this will depend on the method of validation and the training and test set, because the best subset method might overfit the training data. Forward and backward stepwise selection methods may eventually, while picking different predictors with higher training error, avoid some overfitting and fit the test set better. This will also depend on how close to the true underlying model is to one with $k$ predictors.

## (c)

### (i) and (ii)

True.

Since the $(k+1)$-th step model is chosen by picking one more predictor in addition to those of step $k$.

### (iii) and (iv)

False.

Let's explain (iii) with a counter-example, which applies in similar way to (iv). 

Say $p$ the total number of predictors is 4, and $k=2$.
Then backward selection will have chosen 2 predictors, by starting out with 4, and picking one out at each step. Let's say we end up with $x_1$ and $x_2$. Forward selection with 3 predictors is chosen by an unrelated process: it starts with 1 predictors, adding an additional 1 at each step. In general, this will not include all the predictors chosen by backward selection. We could, for example, have ended up with $x_1$, $x_3$ and $x_4$.


### (v)

False.

Model 1 is the best (less trainging error) of all models with exactly k predictors. In general, these predictors will not all be included in the best of all models with exactly k+1 predictors. See Table 6.1 of the text for a concrete example.
