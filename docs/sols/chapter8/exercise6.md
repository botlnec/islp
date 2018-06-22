
# Exercise 8.6

The algorithm to fit a regression tree is as follows:

1. Use the recursive binary splitting to obtain a large tree. Consider a minimum number of observations in the terminal nodes as a criterion to stop the recursive binary splitting.
2. Apply the cost complexity pruning to get the optimum subtree. The cost complexity pruning is defined as a function of $\alpha$, which works as a tuning parameter. 
3. The tuning parameter $\alpha$ is determined using the K-fold cross-validation. A training and a test set are then defined. In each fold, the training set is splitten applying the recursive binary splitting. To the resulting tree, the cost complexity pruning is applied. Then, for a different set of specific $\alpha$s, the mean squared prediction error on the test set is evaluated. The results for the different folds and for the different $\alpha$s are averaged, being $\alpha$ the value that minimizes the average error.
4. Once we get the $\alpha$ value, we replace it in Step 2 equation to get the optimum subtree.

Some important definitions:

* <b>Subtree</b>. A tree that results from pruning a larger tree.
* <b>Recursive binary splitting.</b> Top-down greedy approach to divide the data into distinct and non-overlapping regions. It is top-down because it begins at the top of the tree (at which point all observations belong to a single region) and then successively splits the predictor space; each split is indicated via two new branches further down on the tree. It is greedy because at each step of the tree-building process, the best split is made at that particular step, rather than looking ahead and picking a split that will lead to a better tree in some future step.
* <b>Cost complexity pruning.</b> It is a way to get optimum subtrees. Mathematically, this is done adding a penalty term to the mean squared prediction error expression. This penalty term is governed by a tuning parameter, which controls the trade-off between the subtree’s complexity and its fit to the training data.
