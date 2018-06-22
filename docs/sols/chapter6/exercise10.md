
# Exercise 6.10


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # To split dataset (train + test datasets)
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS  # mlxtend package: exhaustive search for feature selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

# (a)


```python
np.random.seed(0)

# Dataframe with random numbers and the specified dimensions
n = 1000
p = 20
X = pd.DataFrame(np.random.normal(size=(n, p)))

# Epsilon
epsilon = np.random.normal(size=n)

# Coefficient b1
b1 = np.random.normal(size=p)
# Random number of b1 elements with value zero
for i in range(0, np.random.randint(0,p)):
    b1[np.random.randint(0,p)] = 0

# Final expression
# y must be a vector with 1000 rows.
y = np.dot(X, b1) + epsilon
```

# (b)


```python
# Split dataset (train + test datasets)
# We can't change the variables order. Otherwise, we save the datasets with wrong names.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .9)
```


```python
# Confirm the size of test dataset
len(X_test)
```




    900




```python
# JR: https://github.com/scipy/scipy/issues/5998
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# To solve the problem we will use exhaustive feature selection.
# This is a brute force solution but ensures the best subset selection.
# mlxtend package is used (link on References; pay attention to Example 2).
lr = LinearRegression()
score = []

for i in range(1,p):
    print(i)
    efs = EFS(lr,
              min_features = i, 
              max_features = i,
              scoring = 'neg_mean_squared_error',
              print_progress = False,
              cv = 10)

    # .fit input should be array-like.
    # X_train is a dataframe so we use as_matrix() to convert it.
    efs.fit(X_train.as_matrix(), y_train)
    score.append(efs.best_score_) 

#print(efs.best_score_)
#print(efs.best_idx_)


```

    1
    2
    3
    4



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-5-6ae6ca4077cb> in <module>()
         20     # .fit input should be array-like.
         21     # X_train is a dataframe so we use as_matrix() to convert it.
    ---> 22     efs.fit(X_train.as_matrix(), y_train)
         23     score.append(efs.best_score_)
         24 


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/mlxtend/feature_selection/exhaustive_feature_selector.py in fit(self, X, y)
        149         all_comb = len(candidates)
        150         for iteration, c in enumerate(candidates):
    --> 151             cv_scores = self._calc_score(X=X, y=y, indices=c)
        152 
        153             self.subsets_[iteration] = {'feature_idx': c,


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/mlxtend/feature_selection/exhaustive_feature_selector.py in _calc_score(self, X, y, indices)
        181                                      scoring=self.scorer,
        182                                      n_jobs=self.n_jobs,
    --> 183                                      pre_dispatch=self.pre_dispatch)
        184         else:
        185             self.est_.fit(X[:, indices], y)


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/model_selection/_validation.py in cross_val_score(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch)
        138                                               train, test, verbose, None,
        139                                               fit_params)
    --> 140                       for train, test in cv_iter)
        141     return np.array(scores)[:, 0]
        142 


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/parallel.py in __call__(self, iterable)
        756             # was dispatched. In particular this covers the edge
        757             # case of Parallel used with an exhausted iterator.
    --> 758             while self.dispatch_one_batch(iterator):
        759                 self._iterating = True
        760             else:


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/parallel.py in dispatch_one_batch(self, iterator)
        606                 return False
        607             else:
    --> 608                 self._dispatch(tasks)
        609                 return True
        610 


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/parallel.py in _dispatch(self, batch)
        569         dispatch_timestamp = time.time()
        570         cb = BatchCompletionCallBack(dispatch_timestamp, len(batch), self)
    --> 571         job = self._backend.apply_async(batch, callback=cb)
        572         self._jobs.append(job)
        573 


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/_parallel_backends.py in apply_async(self, func, callback)
        107     def apply_async(self, func, callback=None):
        108         """Schedule a func to be run"""
    --> 109         result = ImmediateResult(func)
        110         if callback:
        111             callback(result)


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/_parallel_backends.py in __init__(self, batch)
        324         # Don't delay the application, to avoid keeping the input
        325         # arguments in memory
    --> 326         self.results = batch()
        327 
        328     def get(self):


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/parallel.py in __call__(self)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 
        133     def __len__(self):


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/externals/joblib/parallel.py in <listcomp>(.0)
        129 
        130     def __call__(self):
    --> 131         return [func(*args, **kwargs) for func, args, kwargs in self.items]
        132 
        133     def __len__(self):


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/model_selection/_validation.py in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, error_score)
        236             estimator.fit(X_train, **fit_params)
        237         else:
    --> 238             estimator.fit(X_train, y_train, **fit_params)
        239 
        240     except Exception as e:


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/linear_model/base.py in fit(self, X, y, sample_weight)
        510         n_jobs_ = self.n_jobs
        511         X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
    --> 512                          y_numeric=True, multi_output=True)
        513 
        514         if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        519     X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
        520                     ensure_2d, allow_nd, ensure_min_samples,
    --> 521                     ensure_min_features, warn_on_dtype, estimator)
        522     if multi_output:
        523         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,


    /Users/disciplina/anaconda/envs/islp/lib/python3.5/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        380                                       force_all_finite)
        381     else:
    --> 382         array = np.array(array, dtype=dtype, order=order, copy=copy)
        383 
        384         if ensure_2d:


    KeyboardInterrupt: 



```python
score
```

# (d)

# (e)

# (f)

# (g)

# References
* http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html  (split dataset)
* http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/#api  (mlxtend feature selector)
