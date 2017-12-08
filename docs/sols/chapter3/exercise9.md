
# Exercise 3.9


```python
%matplotlib inline

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

sns.set(style="white")
plt.style.use('seaborn-white')
```


```python
df = pd.read_csv('../data/Auto.csv')
df = pd.read_csv('../data/Auto.csv', na_values='?').dropna()
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



## (a)


```python
# http://seaborn.pydata.org/generated/seaborn.PairGrid.html

g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3, legend=False);
```


![png](03_09_files/03_09_4_0.png)


## (b)


```python
# pandas' corr() function takes care of excluding non numeric data: 
# https://github.com/pandas-dev/pandas/blob/v0.19.2/pandas/core/frame.py#L4721
    
df.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>1.000000</td>
      <td>-0.777618</td>
      <td>-0.805127</td>
      <td>-0.778427</td>
      <td>-0.832244</td>
      <td>0.423329</td>
      <td>0.580541</td>
      <td>0.565209</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>-0.777618</td>
      <td>1.000000</td>
      <td>0.950823</td>
      <td>0.842983</td>
      <td>0.897527</td>
      <td>-0.504683</td>
      <td>-0.345647</td>
      <td>-0.568932</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>-0.805127</td>
      <td>0.950823</td>
      <td>1.000000</td>
      <td>0.897257</td>
      <td>0.932994</td>
      <td>-0.543800</td>
      <td>-0.369855</td>
      <td>-0.614535</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>-0.778427</td>
      <td>0.842983</td>
      <td>0.897257</td>
      <td>1.000000</td>
      <td>0.864538</td>
      <td>-0.689196</td>
      <td>-0.416361</td>
      <td>-0.455171</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>-0.832244</td>
      <td>0.897527</td>
      <td>0.932994</td>
      <td>0.864538</td>
      <td>1.000000</td>
      <td>-0.416839</td>
      <td>-0.309120</td>
      <td>-0.585005</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0.423329</td>
      <td>-0.504683</td>
      <td>-0.543800</td>
      <td>-0.689196</td>
      <td>-0.416839</td>
      <td>1.000000</td>
      <td>0.290316</td>
      <td>0.212746</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.580541</td>
      <td>-0.345647</td>
      <td>-0.369855</td>
      <td>-0.416361</td>
      <td>-0.309120</td>
      <td>0.290316</td>
      <td>1.000000</td>
      <td>0.181528</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0.565209</td>
      <td>-0.568932</td>
      <td>-0.614535</td>
      <td>-0.455171</td>
      <td>-0.585005</td>
      <td>0.212746</td>
      <td>0.181528</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Extra**

Why not a correlation heatmap as well?

[http://seaborn.pydata.org/examples/network_correlations.html](http://seaborn.pydata.org/examples/network_correlations.html)


```python
corrmat = df.corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)

f.tight_layout()
```


![png](03_09_files/03_09_8_0.png)


## (c)


```python
reg = smf.ols('mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin', df).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.821</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.818</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   252.4</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 08 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>2.04e-139</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:08</td>     <th>  Log-Likelihood:    </th> <td> -1023.5</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2063.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   384</td>      <th>  BIC:               </th> <td>   2095.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  -17.2184</td> <td>    4.644</td> <td>   -3.707</td> <td> 0.000</td> <td>  -26.350</td> <td>   -8.087</td>
</tr>
<tr>
  <th>cylinders</th>    <td>   -0.4934</td> <td>    0.323</td> <td>   -1.526</td> <td> 0.128</td> <td>   -1.129</td> <td>    0.142</td>
</tr>
<tr>
  <th>displacement</th> <td>    0.0199</td> <td>    0.008</td> <td>    2.647</td> <td> 0.008</td> <td>    0.005</td> <td>    0.035</td>
</tr>
<tr>
  <th>horsepower</th>   <td>   -0.0170</td> <td>    0.014</td> <td>   -1.230</td> <td> 0.220</td> <td>   -0.044</td> <td>    0.010</td>
</tr>
<tr>
  <th>weight</th>       <td>   -0.0065</td> <td>    0.001</td> <td>   -9.929</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.005</td>
</tr>
<tr>
  <th>acceleration</th> <td>    0.0806</td> <td>    0.099</td> <td>    0.815</td> <td> 0.415</td> <td>   -0.114</td> <td>    0.275</td>
</tr>
<tr>
  <th>year</th>         <td>    0.7508</td> <td>    0.051</td> <td>   14.729</td> <td> 0.000</td> <td>    0.651</td> <td>    0.851</td>
</tr>
<tr>
  <th>origin</th>       <td>    1.4261</td> <td>    0.278</td> <td>    5.127</td> <td> 0.000</td> <td>    0.879</td> <td>    1.973</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>31.906</td> <th>  Durbin-Watson:     </th> <td>   1.309</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  53.100</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.529</td> <th>  Prob(JB):          </th> <td>2.95e-12</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.460</td> <th>  Cond. No.          </th> <td>8.59e+04</td>
</tr>
</table>



## i

Yes, there is a relationship between the predictors and the response. 
In the table above we can see that the value of the F-statistic is 252 which is much larger than 1, so we can reject the corresponding null hypothesis:
    
$$
H_0 :\beta_{cylinders} = \beta_{displacement} = \beta_{weight} = \beta_{acceleration} = \beta_{year} = \beta_{origin}  =0.  
$$

In fact, the probability this data would be generated if $H_0$ was true is $Prob(F-Statistic) = 2 \times 10^{-139}$, a ridiculously low value.

## ii

We can see which predictors have a statistically significant relationship with the response by looking at the p-values in the table above.
The predictors that have a statistically significant relationship to the response are definitely weight, year and origin, and we could say displacement as well; while cylinders, horsepower, and acceleration do not. 

## iii

The coefficient suggests that, on average, when the other variables are held constant, an increase of one year (of production) corresponds to an increase of 0.75 of mpg (so, the more recent the more efficient).

## (d)

In R [4], by default, plot() on a fit produces 4 plots: 
 * a plot of residuals against fitted values,
 * a Scale-Location plot of sqrt(| residuals |) against fitted values,
 * a Normal Q-Q plot,
 * a plot of residuals against leverages.
 
Below, we plot each of these 4 plots. We use the code published by Emre Can [5] with a few adaptations.


```python
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

model_f = 'mpg ~ cylinders + \
                 displacement + \
                 horsepower + \
                 weight + \
                 acceleration + \
                 year + \
                 origin'

df.reset_index(drop=True, inplace=True)

model = smf.ols(formula=model_f, data=df)

model_fit = model.fit()

# fitted values (need a constant term for intercept)
model_fitted_y = model_fit.fittedvalues

# model residuals
model_residuals = model_fit.resid

# normalized residuals
model_norm_residuals = model_fit.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage, from statsmodels internals
model_leverage = model_fit.get_influence().hat_matrix_diag

# cook's distance, from statsmodels internals
model_cooks = model_fit.get_influence().cooks_distance[0]
```

### Residuals against fitted values


```python
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'mpg', data=df,
                                  lowess=True,
                                  scatter_kws={'alpha': 0.5},
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')


# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_residuals[i]));
```


![png](03_09_files/03_09_21_0.png)


### Normal Q-Q plot


```python
QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]));
```


![png](03_09_files/03_09_23_0.png)


### Scale-Location plot of sqrt(|residuals|) against fitted values


```python
plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)

plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');


for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i, 
                               xy=(model_fitted_y[i], 
                                   model_norm_residuals_abs_sqrt[i]));
```


![png](03_09_files/03_09_25_0.png)


### Residuals against leverages


```python
plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_4.axes[0].set_xlim(0, 0.20)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i, 
                               xy=(model_leverage[i], 
                                   model_norm_residuals[i]))
    
# shenanigans for cook's distance contours
def graph(formula, x_range, label=None, ls='-'):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls=ls, color='red')

p = len(model_fit.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance = .5', ls='--') # 0.5 line

graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 'Cook\'s distance = 1', ls=':') # 1 line

plt.legend(loc='upper right');
```


![png](03_09_files/03_09_27_0.png)


## Comments

No, there are no unusually large outliers, as per the the scale log location.
They are however skewedly distributed. The larger the fitted value, the larger the variance, since the spread of the residuals increases.

No, even though there is an observation (number 13) with higher leverage, it is still well within Cook's 0.5 distance.

The normal qq-plot deviates at one extreme, which could indicate that there are other explanatory predictors that we are not considering (quadratic terms, for example).
Additionally, the funnel shape of the residuals plot indicates heteroskedacity.

## References

[1] https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html

[2] https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/

## (e)

Statsmodels uses patsy which is a "mini-language" inspired by R and S to describe statistical models. The symbols ':' and '\*' have the same meaning as in R. Namely, a:b includes an interaction term between a and b, while a*b is shorthand for a + b + a:b, that is, it includes a and b as well.

References:

* http://patsy.readthedocs.io/en/latest/formulas.html
* http://stackoverflow.com/questions/33050104/difference-between-the-interaction-and-term-for-formulas-in-statsmodels-ols
* http://stackoverflow.com/questions/23672466/interaction-effects-in-patsy-with-patsy-dmatrices-giving-duplicate-columns-for

So, which pairs of variables would we expect to interact, both a priori (from our interpretation of the meaning of these variables) and from the pairs plot?

Perhaps horsepower and year? What would this mean? It would mean that, for different years, varying horsepower has different effect on mpg. It seems plausible. We could also interpret it in the reverse order: for different values of horsepower, does varying year have a different effect on mpg? For example, does the change in mpg when varying year (i.e., the derivative dmpg/dyear), differ when holding horsepower at either 130 or 160? 

Let's find out.


```python
reg = smf.ols('mpg ~ horsepower*year + displacement + weight + origin', df).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.851</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.849</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   367.0</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 08 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>7.51e-156</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:11</td>     <th>  Log-Likelihood:    </th> <td> -987.81</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1990.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   385</td>      <th>  BIC:               </th> <td>   2017.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>  -96.6688</td> <td>    9.667</td> <td>  -10.000</td> <td> 0.000</td> <td> -115.675</td> <td>  -77.663</td>
</tr>
<tr>
  <th>horsepower</th>      <td>    0.7993</td> <td>    0.092</td> <td>    8.687</td> <td> 0.000</td> <td>    0.618</td> <td>    0.980</td>
</tr>
<tr>
  <th>year</th>            <td>    1.8179</td> <td>    0.128</td> <td>   14.221</td> <td> 0.000</td> <td>    1.567</td> <td>    2.069</td>
</tr>
<tr>
  <th>horsepower:year</th> <td>   -0.0113</td> <td>    0.001</td> <td>   -8.977</td> <td> 0.000</td> <td>   -0.014</td> <td>   -0.009</td>
</tr>
<tr>
  <th>displacement</th>    <td>    0.0068</td> <td>    0.005</td> <td>    1.344</td> <td> 0.180</td> <td>   -0.003</td> <td>    0.017</td>
</tr>
<tr>
  <th>weight</th>          <td>   -0.0054</td> <td>    0.001</td> <td>  -10.170</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.004</td>
</tr>
<tr>
  <th>origin</th>          <td>    1.1866</td> <td>    0.253</td> <td>    4.684</td> <td> 0.000</td> <td>    0.688</td> <td>    1.685</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21.932</td> <th>  Durbin-Watson:     </th> <td>   1.488</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  33.066</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.411</td> <th>  Prob(JB):          </th> <td>6.60e-08</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.161</td> <th>  Cond. No.          </th> <td>5.60e+05</td>
</tr>
</table>



From the value of the p-value of the coefficient of the interaction term between horsepower and year, it does seem like there is a statistically significant relationship between the response and horsepower:year.

With 7 factors, there will be a total of 21 interaction terms.
For simplicity sake, we will exclude all the terms with cylinders and acceleration, leaving us with 10 interaction terms.
Let's try to fit a model with these terms - a total of 15 terms.


```python
model = 'mpg ~ displacement + horsepower + origin + weight + year \
               + displacement:horsepower + displacement:origin + displacement:weight + displacement:year \
               + horsepower:origin + horsepower:weight + horsepower:year + origin:weight + origin:year + weight:year'
reg = smf.ols(model, df).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.880</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.875</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   184.0</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 08 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>1.09e-162</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:11</td>     <th>  Log-Likelihood:    </th> <td> -945.49</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1923.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   376</td>      <th>  BIC:               </th> <td>   1987.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    15</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>               <td>  -51.3746</td> <td>   26.175</td> <td>   -1.963</td> <td> 0.050</td> <td> -102.843</td> <td>    0.093</td>
</tr>
<tr>
  <th>displacement</th>            <td>   -0.1818</td> <td>    0.120</td> <td>   -1.521</td> <td> 0.129</td> <td>   -0.417</td> <td>    0.053</td>
</tr>
<tr>
  <th>horsepower</th>              <td>    0.9485</td> <td>    0.232</td> <td>    4.089</td> <td> 0.000</td> <td>    0.492</td> <td>    1.405</td>
</tr>
<tr>
  <th>origin</th>                  <td>   -3.0637</td> <td>    5.496</td> <td>   -0.557</td> <td> 0.578</td> <td>  -13.871</td> <td>    7.744</td>
</tr>
<tr>
  <th>weight</th>                  <td>   -0.0174</td> <td>    0.016</td> <td>   -1.115</td> <td> 0.265</td> <td>   -0.048</td> <td>    0.013</td>
</tr>
<tr>
  <th>year</th>                    <td>    1.3975</td> <td>    0.328</td> <td>    4.267</td> <td> 0.000</td> <td>    0.754</td> <td>    2.042</td>
</tr>
<tr>
  <th>displacement:horsepower</th> <td>   -0.0001</td> <td>    0.000</td> <td>   -0.815</td> <td> 0.416</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>displacement:origin</th>     <td>    0.0282</td> <td>    0.013</td> <td>    2.172</td> <td> 0.030</td> <td>    0.003</td> <td>    0.054</td>
</tr>
<tr>
  <th>displacement:weight</th>     <td> 2.792e-05</td> <td> 5.99e-06</td> <td>    4.663</td> <td> 0.000</td> <td> 1.61e-05</td> <td> 3.97e-05</td>
</tr>
<tr>
  <th>displacement:year</th>       <td>    0.0010</td> <td>    0.001</td> <td>    0.710</td> <td> 0.478</td> <td>   -0.002</td> <td>    0.004</td>
</tr>
<tr>
  <th>horsepower:origin</th>       <td>   -0.0629</td> <td>    0.020</td> <td>   -3.104</td> <td> 0.002</td> <td>   -0.103</td> <td>   -0.023</td>
</tr>
<tr>
  <th>horsepower:weight</th>       <td>-1.175e-05</td> <td> 1.77e-05</td> <td>   -0.664</td> <td> 0.507</td> <td>-4.65e-05</td> <td>  2.3e-05</td>
</tr>
<tr>
  <th>horsepower:year</th>         <td>   -0.0114</td> <td>    0.003</td> <td>   -3.998</td> <td> 0.000</td> <td>   -0.017</td> <td>   -0.006</td>
</tr>
<tr>
  <th>origin:weight</th>           <td>    0.0014</td> <td>    0.001</td> <td>    1.200</td> <td> 0.231</td> <td>   -0.001</td> <td>    0.004</td>
</tr>
<tr>
  <th>origin:year</th>             <td>    0.0322</td> <td>    0.069</td> <td>    0.464</td> <td> 0.643</td> <td>   -0.104</td> <td>    0.169</td>
</tr>
<tr>
  <th>weight:year</th>             <td> 7.438e-05</td> <td>    0.000</td> <td>    0.394</td> <td> 0.694</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>52.658</td> <th>  Durbin-Watson:     </th> <td>   1.599</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 115.208</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.710</td> <th>  Prob(JB):          </th> <td>9.61e-26</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.244</td> <th>  Cond. No.          </th> <td>1.81e+08</td>
</tr>
</table>



These results indicate that the interactions that appear to be statistically significant are displacement:weight, horsepower:origin and horsepower:year.
Interestingly, when these are considered the only first order terms that are statistically different are horsepower and year.
By the hierarchy principle (page 89), we should nonetheless include all of the main effects (for more on this, see [these](https://stats.stackexchange.com/questions/27724/do-all-interactions-terms-need-their-individual-terms-in-regression-model) answers).

We could also have a try at [interaction plots](https://en.wikipedia.org/wiki/Interaction_%28statistics%29#Interaction_plots), which are not covered in the book, but we will leave it as a [mention](https://en.wikipedia.org/wiki/Interaction_%28statistics%29#/media/File:GSS_sealevel_interaction.png) only.



## (f)

As an example, we fit the data with a model containing the transformations indicated for the variable horsepower, starting with $X^2$. 



```python
reg = smf.ols('mpg ~ horsepower + np.power(horsepower,2) + weight + year + origin', df).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.851</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.849</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   439.5</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 08 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>7.11e-157</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:11</td>     <th>  Log-Likelihood:    </th> <td> -988.57</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1989.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   386</td>      <th>  BIC:               </th> <td>   2013.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>               <td>   -6.6457</td> <td>    3.915</td> <td>   -1.698</td> <td> 0.090</td> <td>  -14.343</td> <td>    1.052</td>
</tr>
<tr>
  <th>horsepower</th>              <td>   -0.2441</td> <td>    0.027</td> <td>   -9.099</td> <td> 0.000</td> <td>   -0.297</td> <td>   -0.191</td>
</tr>
<tr>
  <th>np.power(horsepower, 2)</th> <td>    0.0008</td> <td> 9.13e-05</td> <td>    9.170</td> <td> 0.000</td> <td>    0.001</td> <td>    0.001</td>
</tr>
<tr>
  <th>weight</th>                  <td>   -0.0044</td> <td>    0.000</td> <td>  -10.426</td> <td> 0.000</td> <td>   -0.005</td> <td>   -0.004</td>
</tr>
<tr>
  <th>year</th>                    <td>    0.7456</td> <td>    0.046</td> <td>   16.145</td> <td> 0.000</td> <td>    0.655</td> <td>    0.836</td>
</tr>
<tr>
  <th>origin</th>                  <td>    1.0465</td> <td>    0.238</td> <td>    4.405</td> <td> 0.000</td> <td>    0.579</td> <td>    1.514</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>21.819</td> <th>  Durbin-Watson:     </th> <td>   1.500</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  32.447</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.414</td> <th>  Prob(JB):          </th> <td>9.00e-08</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.140</td> <th>  Cond. No.          </th> <td>4.10e+05</td>
</tr>
</table>




```python
fig = plt.figure()
fitted = reg.fittedvalues
sns.regplot(fitted, df.mpg - fitted,  lowess=True, line_kws={'color':'r', 'lw':1})
ax = fig.axes[0]
ax.axhline(color="grey", ls="--")
ax.set_title("Residuals vs Fitted Values")
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals");
```


![png](03_09_files/03_09_40_0.png)


It is clear that this quadratic term is statistically significant. Let's try adding a logarithmic term as well.


```python
reg = smf.ols('mpg ~ horsepower + np.power(horsepower,2) + np.log(horsepower) + weight + year + origin', df).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.855</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.853</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   378.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 08 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>4.62e-158</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:11</td>     <th>  Log-Likelihood:    </th> <td> -982.62</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1979.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   385</td>      <th>  BIC:               </th> <td>   2007.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>               <td>   80.3681</td> <td>   25.568</td> <td>    3.143</td> <td> 0.002</td> <td>   30.098</td> <td>  130.638</td>
</tr>
<tr>
  <th>horsepower</th>              <td>    0.2570</td> <td>    0.148</td> <td>    1.737</td> <td> 0.083</td> <td>   -0.034</td> <td>    0.548</td>
</tr>
<tr>
  <th>np.power(horsepower, 2)</th> <td>   -0.0002</td> <td>    0.000</td> <td>   -0.571</td> <td> 0.568</td> <td>   -0.001</td> <td>    0.000</td>
</tr>
<tr>
  <th>np.log(horsepower)</th>      <td>  -27.5412</td> <td>    8.000</td> <td>   -3.443</td> <td> 0.001</td> <td>  -43.270</td> <td>  -11.812</td>
</tr>
<tr>
  <th>weight</th>                  <td>   -0.0048</td> <td>    0.000</td> <td>  -11.098</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.004</td>
</tr>
<tr>
  <th>year</th>                    <td>    0.7561</td> <td>    0.046</td> <td>   16.565</td> <td> 0.000</td> <td>    0.666</td> <td>    0.846</td>
</tr>
<tr>
  <th>origin</th>                  <td>    0.9480</td> <td>    0.236</td> <td>    4.016</td> <td> 0.000</td> <td>    0.484</td> <td>    1.412</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>17.905</td> <th>  Durbin-Watson:     </th> <td>   1.575</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  29.299</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.312</td> <th>  Prob(JB):          </th> <td>4.34e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.185</td> <th>  Cond. No.          </th> <td>2.84e+06</td>
</tr>
</table>



Now the p-value for the square term is very large.
This indicates that there is indeed a non-linearity but it seems to be better captured by the logarithm than the square. 
Let's try adding the square root term.


```python
reg = smf.ols('mpg ~ horsepower + np.power(horsepower,2) + np.log(horsepower) + np.sqrt(horsepower) + weight + year + origin', df).fit()
reg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.859</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.856</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   332.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 08 Dec 2017</td> <th>  Prob (F-statistic):</th> <td>9.16e-159</td>
</tr>
<tr>
  <th>Time:</th>                 <td>09:49:12</td>     <th>  Log-Likelihood:    </th> <td> -977.89</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   1972.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   384</td>      <th>  BIC:               </th> <td>   2004.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>               <td> -426.3991</td> <td>  167.288</td> <td>   -2.549</td> <td> 0.011</td> <td> -755.314</td> <td>  -97.485</td>
</tr>
<tr>
  <th>horsepower</th>              <td>    8.4452</td> <td>    2.676</td> <td>    3.156</td> <td> 0.002</td> <td>    3.184</td> <td>   13.706</td>
</tr>
<tr>
  <th>np.power(horsepower, 2)</th> <td>   -0.0060</td> <td>    0.002</td> <td>   -3.117</td> <td> 0.002</td> <td>   -0.010</td> <td>   -0.002</td>
</tr>
<tr>
  <th>np.log(horsepower)</th>      <td>  416.0064</td> <td>  144.951</td> <td>    2.870</td> <td> 0.004</td> <td>  131.009</td> <td>  701.004</td>
</tr>
<tr>
  <th>np.sqrt(horsepower)</th>     <td> -229.6161</td> <td>   74.927</td> <td>   -3.065</td> <td> 0.002</td> <td> -376.934</td> <td>  -82.298</td>
</tr>
<tr>
  <th>weight</th>                  <td>   -0.0048</td> <td>    0.000</td> <td>  -11.229</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.004</td>
</tr>
<tr>
  <th>year</th>                    <td>    0.7475</td> <td>    0.045</td> <td>   16.522</td> <td> 0.000</td> <td>    0.659</td> <td>    0.836</td>
</tr>
<tr>
  <th>origin</th>                  <td>    0.9088</td> <td>    0.234</td> <td>    3.886</td> <td> 0.000</td> <td>    0.449</td> <td>    1.369</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>23.395</td> <th>  Durbin-Watson:     </th> <td>   1.570</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  37.804</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.411</td> <th>  Prob(JB):          </th> <td>6.18e-09</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.281</td> <th>  Cond. No.          </th> <td>2.50e+07</td>
</tr>
</table>



So now the square term is back to a small p-value, indicating that it is statistically significant in the presence of the square root and the logarithm.
