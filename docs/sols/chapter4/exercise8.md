
# Exercise 4.8

We should prefer the method that has a lower error rate on the test data because this shows how well the model works on unseen data.
In other words, choosing the model with the lowest error rate on the test data means that we're choosing the model with better prediction capacity.

In this case, the error rate on test data for the logistic regression is explicit: 30%. For the 1-nearest neighbor we have to do some computations.

We know that when we use the 1-nearest neighbor, the average error rate is 18%. This error ($\varepsilon$) is averaged over both test and training sets, which means that:

$$ \frac{\varepsilon_{training} + \varepsilon_{test}}{2} = 0.18 \Leftrightarrow \varepsilon_{test} = 2 \times 0.18 - \varepsilon_{training}$$

For a 1-nearest neighbor model, we have $\varepsilon_{training} = 0$. This happens because for any training example, its nearest neighbor is always going to be itself.
Therefore:

$$ \varepsilon_{test} = 2 \times 0.18 - \varepsilon_{training} \Leftrightarrow \varepsilon_{test} = 0.36 - 0 \Leftrightarrow \varepsilon_{test} = 0.36$$

The method we should prefer to use for classification of new observations is the <b>logistic regression</b> method, even though its error rate on the training data is larger than the one for the 1-nearest neighbor model (18% vs. 0%).
