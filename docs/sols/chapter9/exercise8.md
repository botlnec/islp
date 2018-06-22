
# Exercise 9.8


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

%matplotlib inline
```


```python
df = pd.read_csv('../data/OJ.csv', index_col=0)
```


```python
# Data overview
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase</th>
      <th>WeekofPurchase</th>
      <th>StoreID</th>
      <th>PriceCH</th>
      <th>PriceMM</th>
      <th>DiscCH</th>
      <th>DiscMM</th>
      <th>SpecialCH</th>
      <th>SpecialMM</th>
      <th>LoyalCH</th>
      <th>SalePriceMM</th>
      <th>SalePriceCH</th>
      <th>PriceDiff</th>
      <th>Store7</th>
      <th>PctDiscMM</th>
      <th>PctDiscCH</th>
      <th>ListPriceDiff</th>
      <th>STORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CH</td>
      <td>237</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500000</td>
      <td>1.99</td>
      <td>1.75</td>
      <td>0.24</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CH</td>
      <td>239</td>
      <td>1</td>
      <td>1.75</td>
      <td>1.99</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
      <td>0.600000</td>
      <td>1.69</td>
      <td>1.75</td>
      <td>-0.06</td>
      <td>No</td>
      <td>0.150754</td>
      <td>0.000000</td>
      <td>0.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CH</td>
      <td>245</td>
      <td>1</td>
      <td>1.86</td>
      <td>2.09</td>
      <td>0.17</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.680000</td>
      <td>2.09</td>
      <td>1.69</td>
      <td>0.40</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.091398</td>
      <td>0.23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MM</td>
      <td>227</td>
      <td>1</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.400000</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>No</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CH</td>
      <td>228</td>
      <td>7</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.956535</td>
      <td>1.69</td>
      <td>1.69</td>
      <td>0.00</td>
      <td>Yes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define predictors and response 
X = df.drop(axis=1, labels=['Purchase'])
y = df['Purchase']
```


```python
# Dummy variables to transform qualitative into quantitative variables
X = pd.get_dummies(X)
```

# (a)


```python
# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, random_state=1)
```

# (b)


```python
# Fit SVC to data
svc = SVC(C=0.01, kernel='linear', random_state=1)
svc.fit(X_train, y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=1, shrinking=True,
      tol=0.001, verbose=False)




```python
# Number of support vectors for each class
svc.n_support_
```




    array([307, 304])



In our dataset, we have 800 observations, 2 classes, and a total of 611 support vectors. From those support vectors, 307 belong to class CH and 304 to class MM.

# (c)

Error rate (ERR) is calculated as the number of all incorrect predictions divided by the total number of the dataset. The best error rate is 0.0, whereas the worst is 1.0. The error rate is derived from the confusion matrix.

Source: https://classeval.wordpress.com/introduction/basic-evaluation-measures/


```python
# Confusion matrix
print('Train confusion matrix: ', confusion_matrix(y_train, svc.predict(X_train)))
print('Test confusion matrix: ', confusion_matrix(y_test, svc.predict(X_test)))
```

    Train confusion matrix:  [[489   7]
     [241  63]]
    Test confusion matrix:  [[150   7]
     [ 90  23]]


The count of true negatives is $C_{0,0}$, false negatives is $C_{1,0}$, true positives is $C_{1,1}$ and false positives is $C_{0,1}$.


```python
# Error rate
train_err = (7+241)/(489+7+241+63)
test_err = (7+90)/(150+7+90+23)

print('Train error rate: ', train_err)
print('Test error rate: ', test_err)
```

    Train error rate:  0.31
    Test error rate:  0.3592592592592593


# (d)

Since the selection of an optimal cost is a hypertuning parameter operation, we will use the GridSearchCV.


```python
# Hypertune cost using GridSearchCV
svc = SVC(kernel='linear', random_state=1)

parameters = {'C':np.arange(0.01, 10, 2)}

clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
```




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=1, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': array([ 0.01,  2.01,  4.01,  6.01,  8.01])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)




```python
# Best value for cost
clf.best_params_
```




    {'C': 2.0099999999999998}



# (e)


```python
# Confusion matrix
print('Train confusion matrix: ', confusion_matrix(y_train, clf.predict(X_train)))
print('Test confusion matrix: ', confusion_matrix(y_test, clf.predict(X_test)))
```

    Train confusion matrix:  [[436  60]
     [ 72 232]]
    Test confusion matrix:  [[143  14]
     [ 30  83]]



```python
# Error rate
train_err = (59+75)/(437+59+75+229)
test_err = (13+35)/(144+13+35+78)

print('Train error rate: ', train_err)
print('Test error rate: ', test_err)
```

    Train error rate:  0.1675
    Test error rate:  0.17777777777777778


# (f)


```python
# Fit SVC to data
svc = SVC(C=0.01, kernel='rbf', random_state=1)
svc.fit(X_train, y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=1, shrinking=True,
      tol=0.001, verbose=False)




```python
# Number of support vectors for each class
svc.n_support_
```




    array([321, 304])




```python
# Confusion matrix
print('Train confusion matrix: ', confusion_matrix(y_train, svc.predict(X_train)))
print('Test confusion matrix: ', confusion_matrix(y_test, svc.predict(X_test)))
```

    Train confusion matrix:  [[496   0]
     [304   0]]
    Test confusion matrix:  [[157   0]
     [113   0]]



```python
# Error rate
train_err = (0+304)/(496+0+304+0)
test_err = (0+113)/(157+0+113+0)

print('Train error rate: ', train_err)
print('Test error rate: ', test_err)
```

    Train error rate:  0.38
    Test error rate:  0.4185185185185185



```python
# Hypertune cost using GridSearchCV
svc = SVC(kernel='rbf', random_state=1)

parameters = {'C':np.arange(0.01, 10, 2)}

clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
```




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=1, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': array([ 0.01,  2.01,  4.01,  6.01,  8.01])},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)




```python
# Best value for cost
clf.best_params_
```




    {'C': 4.0099999999999998}




```python
# Confusion matrix
print('Train confusion matrix: ', confusion_matrix(y_train, clf.predict(X_train)))
print('Test confusion matrix: ', confusion_matrix(y_test, clf.predict(X_test)))
```

    Train confusion matrix:  [[450  46]
     [ 94 210]]
    Test confusion matrix:  [[145  12]
     [ 39  74]]



```python
# Error rate
train_err = (40+78)/(456+40+78+226)
test_err = (11+36)/(146+11+36+77)

print('Train error rate: ', train_err)
print('Test error rate: ', test_err)
```

    Train error rate:  0.1475
    Test error rate:  0.17407407407407408


# (g)


```python
# Fit SVC to data
svc = SVC(C=0.01, kernel='poly', degree=2, random_state=1)
svc.fit(X_train, y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=2, gamma='auto', kernel='poly',
      max_iter=-1, probability=False, random_state=1, shrinking=True,
      tol=0.001, verbose=False)




```python
# Number of support vectors for each class
svc.n_support_
```




    array([164, 164])




```python
# Confusion matrix
print('Train confusion matrix: ', confusion_matrix(y_train, svc.predict(X_train)))
print('Test confusion matrix: ', confusion_matrix(y_test, svc.predict(X_test)))
```

    Train confusion matrix:  [[435  61]
     [ 70 234]]
    Test confusion matrix:  [[140  17]
     [ 30  83]]



```python
# Error rate
train_err = (61+70)/(435+61+70+234)
test_err = (17+30)/(140+17+30+83)

print('Train error rate: ', train_err)
print('Test error rate: ', test_err)
```

    Train error rate:  0.16375
    Test error rate:  0.17407407407407408



```python
# Hypertune cost using GridSearchCV
svc = SVC(kernel='poly', degree=2, random_state=1)

parameters = {'C':np.arange(0.01, 10, 2)}

clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
```


```python
# Best value for cost
clf.best_params_
```


```python
# Confusion matrix
print('Train confusion matrix: ', confusion_matrix(y_train, clf.predict(X_train)))
print('Test confusion matrix: ', confusion_matrix(y_test, clf.predict(X_test)))
```


```python
# Error rate
train_err = (61+70)/(456+40+78+226)
test_err = (11+36)/(146+11+36+77)

print('Train error rate: ', train_err)
print('Test error rate: ', test_err)
```

# (h)

Overall, the approach that seems to give the best results on this data is <b>xxx</b>.
