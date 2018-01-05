
# Problem 5.3

## (a)

The k-fold cross-validation algorithm is a method to estimate the test error of a given statistical estimator.
We start by taking the training set (of size N) and partitioning it into k non-overlapping equal parts.
If k does not divide N evenly, we split it into k folds as "equal" as possible; for example, if N=1003, and k=10, we will have 3 folds with 101 elements and 7 with 100 elements.
We then fit the model k times, each time leaving out a different fold as validation set.
For each of these k model fits, we use the left-out fold to calculate the validation error. 
Finally, the average of the k validation errors is our estimate of the test error.

## (b)

### i.

Relative to the the k-fold cross validation, the validation set approach requires less computation (since it only fits the model once) and is simpler and easier to implement. On the other hand, the validation set approach tends to overestimate the test error since it only uses half the sample to fit the model (in general a larger sample size leads to lower test error). Additionally, fitting only half the model will make the test error estimate dependent on which half of the sample we choose. Both of these aspects are also true for the k-fold cross-validation but the effect is much smaller (the difference is evident when comparing the right-hand panels of Figures 5.2 and 5.4 of the text).

### ii.

Leave-one-out cross-validation (LOOCV) has less bias than k-fold cross validation since it uses almost all of the points of the data set, nearly unbiased.
On the other hand, this is yet another instance of the bias-variance trade-off, and LOOCV will have more variance than k-fold cross validation.
This is because when every pair of the n-1 fitted models differs only by 2 points, making the validation errors extremely correlated - they share n-2 of the n-1 points used for each fit.
With k-fold cross validation this correlation is greatly diminished if k=5 or k=10, for instance. 

Consider a k=10 example: each pair of the 10 fits shares about only 80% of the points. With k=5, each pair of the 5 model fits shares only approximately 60% of the points.
Additionally, k-fold cross validation is less computationally intensive, requiring only k model fits instead of the n fits with LOOCV. As pointed out in the text (Equation 5.2), there's an exception to this case for the least squares linear or polinomial regression - in this case LOOCV requires only the same computation as a single model fit. 
