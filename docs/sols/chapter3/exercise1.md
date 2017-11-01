
# Exercise 3.1

The t-statistics computed on Table 3.4 are computed individually for each coefficient since they are independent variables. Accordingly, there are 4 null hypotheses that we are testing:

1. $H_0$ for "TV": **in the presence of** Radio and Newspaper ads (and in addition to the intercept), there is no relationship between TV and Sales;
2. $H_0$ for "Radio": **in the presence of** TV and Newspaper ads (and in addition to the intercept), there is no relationship between Radio and Sales;
3. $H_0$ for "Newspaper": **in the presence of** TV and Radio ads (and in addition to the intercept), there is no relationship between Newspaper and Sales;
4. $H_0$ for the intercept: **in the absence of** TV, Radio and Newspaper ads, Sales are zero;

versus the 4 corresponding alternative hypotheses:

$H_a$: There is some relationship between TV/Radio/Newspaper and Sales, or Sales are non-zero in the absence of the other variables.

<br><br>
Mathematically, this can be written as

$H_0:$ $\beta_i=0$, for $i = 0,1,2,3$,

versus the 4 corresponding alternative hypotheses

$H_a:$ $\beta_i\neq0$, for $i = 0,1,2,3$.

As can been seen on Table 3.4 (and below with Python), for all the variables the p-value is practically zero, except for *Newspaper* for which it is very high, namely .86, much larger than the typical confidence levels, 0.05, 0.01 and 0.001.  Given the t-statistics and the p-values we can reject the null hypothesis for the intercept, TV and Radio, but not for Newspaper.

This means that we can conclude that **there is a relationship between TV and Sales, and between Radio and Sales**. Also rejecting $\beta_0=0$, allows us to conclude that **in the absence of TV, Radio and Newspaper, Sales are non-zero**. Not being able to reject the null hypothesis $\beta_{Newspaper}=0$, suggests that there is indeed **no relationship between Newspaper and Sales, in the presence of TV and Radio**.

### Additional comment

At a 5% p-value, there would be a 19% chance of having one appear as significant out of 3 variables, even if there was no relationship for all of them. 

($1-.95^4$) 

## Auxiliary calculations


```python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('../data/Advertising.csv')

from statsmodels.formula.api import ols
model = ols("Sales ~ TV + Radio + Newspaper", df).fit()
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  Sales   R-squared:                       0.897
    Model:                            OLS   Adj. R-squared:                  0.896
    Method:                 Least Squares   F-statistic:                     570.3
    Date:                Tue, 24 Oct 2017   Prob (F-statistic):           1.58e-96
    Time:                        10:19:37   Log-Likelihood:                -386.18
    No. Observations:                 200   AIC:                             780.4
    Df Residuals:                     196   BIC:                             793.6
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      2.9389      0.312      9.422      0.000       2.324       3.554
    TV             0.0458      0.001     32.809      0.000       0.043       0.049
    Radio          0.1885      0.009     21.893      0.000       0.172       0.206
    Newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
    ==============================================================================
    Omnibus:                       60.414   Durbin-Watson:                   2.084
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
    Skew:                          -1.327   Prob(JB):                     1.44e-33
    Kurtosis:                       6.332   Cond. No.                         454.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Further reading

ISL:

* Page 67, 68
* Footnote page 68
        
$H_0$:

* [http://quant.stackexchange.com/questions/16056/null-and-alternative-hypothesis-for-multiple-linear-regression](http://quant.stackexchange.com/questions/16056/null-and-alternative-hypothesis-for-multiple-linear-regression)
    
Multiple regression:

* [https://www.datarobot.com/blog/multiple-regression-using-statsmodels/](https://www.datarobot.com/blog/multiple-regression-using-statsmodels/)
* [https://www.coursera.org/learn/regression-modeling-practice/lecture/xQRab/python-lesson-1-multiple-regression](https://www.coursera.org/learn/regression-modeling-practice/lecture/xQRab/python-lesson-1-multiple-regression)
* [http://www.scipy-lectures.org/packages/statistics/index.html#multiple-regression-including-multiple-factors](http://www.scipy-lectures.org/packages/statistics/index.html#multiple-regression-including-multiple-factors)
* [http://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python](http://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python)
    
