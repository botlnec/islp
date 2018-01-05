
# Exercise 5.6


```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf 
import numpy as np
```


```python
df = pd.read_csv('../data/Default.csv', index_col=0)
```


```python
df.head() #just checking
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>student</th>
      <th>balance</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>No</td>
      <td>729.526495</td>
      <td>44361.625074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No</td>
      <td>Yes</td>
      <td>817.180407</td>
      <td>12106.134700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>No</td>
      <td>1073.549164</td>
      <td>31767.138947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>No</td>
      <td>529.250605</td>
      <td>35704.493935</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No</td>
      <td>No</td>
      <td>785.655883</td>
      <td>38463.495879</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.random.seed(0) #asked in the exercise
```

# (a)


```python
#using generalized linear models with statsmodel
#see the wikipedia reference to understand why family is binomial
mod1 = smf.glm(formula='default ~ income + balance', data=df, family=sm.families.Binomial()).fit() #create & fit model
print(mod1.summary()) #show results
```

                            Generalized Linear Model Regression Results                        
    ===========================================================================================
    Dep. Variable:     ['default[No]', 'default[Yes]']   No. Observations:                10000
    Model:                                         GLM   Df Residuals:                     9997
    Model Family:                             Binomial   Df Model:                            2
    Link Function:                               logit   Scale:                             1.0
    Method:                                       IRLS   Log-Likelihood:                -789.48
    Date:                             Fri, 05 Jan 2018   Deviance:                       1579.0
    Time:                                     21:08:26   Pearson chi2:                 6.95e+03
    No. Iterations:                                  9                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     11.5405      0.435     26.544      0.000      10.688      12.393
    income     -2.081e-05   4.99e-06     -4.174      0.000   -3.06e-05    -1.1e-05
    balance       -0.0056      0.000    -24.835      0.000      -0.006      -0.005
    ==============================================================================


Estimated standard errors for the coefficients associated with <b>income</b> and <b>balance</b> are <b>4.99e-06</b> and <b>0</b>, respectively.

# (b)


```python
def boot_fn(default):
    mod1 = smf.glm(formula='default ~ income + balance', data=default, family=sm.families.Binomial()).fit()
    coef_income = mod1.params[1]
    coef_balance = mod1.params[2]
    return [coef_income, coef_balance]
```


```python
boot_fn(df)
```




    [-2.0808975528986941e-05, -0.005647102950316488]



# (c)

Since there is not Python equivalent to R boot function, we will create a boot function for Python.


```python
#bootstrap function
def boot(X, bootSample_size=None):
    '''
    Sampling observations from a dataframe
    
    Parameters
    ------------
    X : pandas dataframe
        Data to be resampled
        
    bootSample_size: int, optional
        Dimension of the bootstrapped samples
    
    Returns
    ------------
    bootSample_X : pandas dataframe
        Resampled data
        
    Examples
    ----------
    To resample data from the X dataframe:
        >> boot(X)
    The resampled data will have length equal to len(X).
    
    To resample data from the X dataframe in order to have length 5:
        >> boot(X,5)
    
    References
    ------------
    http://nbviewer.jupyter.org/gist/aflaxman/6871948
    
    '''
    
    #assign default size if non-specified
    if bootSample_size == None:
        bootSample_size = len(X)
    
    #create random integers to use as indices for bootstrap sample based on original data
    bootSample_i = (np.random.rand(bootSample_size)*len(X)).astype(int)
    bootSample_i = np.array(bootSample_i)
    bootSample_X = X.iloc[bootSample_i]
    
    return bootSample_X
```

Now, we will call the *boot* function *n* times, apply *boot_fn* and compute the coefficients average value. We used *n = 100* to have convergent results. Other values could be used.


```python
#running model for bootstrapped samples
coefficients = [] #variable initialization
n = 100 #number of bootstrapped samples

for i in range(0,n):
    coef_i = boot_fn(boot(df)) #determining coefficients for specific bootstrapped sample
    coefficients.append(coef_i) #saving coefficients value

print(pd.DataFrame(coefficients).mean()) #print average of coefficients
```

    0   -0.000021
    1   -0.005716
    dtype: float64


# (d)

* <b>Results (b):</b> [-2.0808975528987073e-05, -0.005647102950316495]
* <b>Results (c):</b> [-0.000021, -0.005672]

# References
* Wikipedia, Generalized linear model, https://en.wikipedia.org/wiki/Generalized_linear_model
* http://nbviewer.jupyter.org/gist/aflaxman/6871948
