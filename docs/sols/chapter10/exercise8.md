
# Exercise 10.8


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

%matplotlib inline
```


```python
# Import dataset
df = pd.read_csv('../data/USArrests.csv', index_col=0)
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
      <th>Murder</th>
      <th>Assault</th>
      <th>UrbanPop</th>
      <th>Rape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alabama</th>
      <td>13.2</td>
      <td>236</td>
      <td>58</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>10.0</td>
      <td>263</td>
      <td>48</td>
      <td>44.5</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>8.1</td>
      <td>294</td>
      <td>80</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>8.8</td>
      <td>190</td>
      <td>50</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>California</th>
      <td>9.0</td>
      <td>276</td>
      <td>91</td>
      <td>40.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scale data (standardize)
scl = StandardScaler()
df_scl = scl.fit_transform(df)
```

<b>Note:</b> We should standardize the data because scale is an issue in this exercise. The variance of the variable *Assault* is significantly larger than the variance of the remaining variables. Thus, if we perform PCA on the unscaled variables, then the first principal component loading vector will have a very large loading for *Assault*. This would take us to a misleading solution.

# (a)


```python
# PCA
pca = PCA()
pca.fit(df_scl)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
# Proportion of variance explained
# This is equivalent to do sdev output of the prcomp() function in R.
pca.explained_variance_ratio_
```




    array([ 0.62006039,  0.24744129,  0.0891408 ,  0.04335752])



# (b)


```python
# Loadings of the principal components
# Rows are the loading vectors (see References)
pca.components_
```




    array([[ 0.53589947,  0.58318363,  0.27819087,  0.54343209],
           [ 0.41818087,  0.1879856 , -0.87280619, -0.16731864],
           [-0.34123273, -0.26814843, -0.37801579,  0.81777791],
           [ 0.6492278 , -0.74340748,  0.13387773,  0.08902432]])




```python
# Centered and scaled variables
# In (a) we used centered ans scaled variables, so we should use the same data here.
df_scl
```




    array([[ 1.25517927,  0.79078716, -0.52619514, -0.00345116],
           [ 0.51301858,  1.11805959, -1.22406668,  2.50942392],
           [ 0.07236067,  1.49381682,  1.00912225,  1.05346626],
           [ 0.23470832,  0.23321191, -1.08449238, -0.18679398],
           [ 0.28109336,  1.2756352 ,  1.77678094,  2.08881393],
           [ 0.02597562,  0.40290872,  0.86954794,  1.88390137],
           [-1.04088037, -0.73648418,  0.79976079, -1.09272319],
           [-0.43787481,  0.81502956,  0.45082502, -0.58583422],
           [ 1.76541475,  1.99078607,  1.00912225,  1.1505301 ],
           [ 2.22926518,  0.48775713, -0.38662083,  0.49265293],
           [-0.57702994, -1.51224105,  1.21848371, -0.11129987],
           [-1.20322802, -0.61527217, -0.80534376, -0.75839217],
           [ 0.60578867,  0.94836277,  1.21848371,  0.29852525],
           [-0.13637203, -0.70012057, -0.03768506, -0.0250209 ],
           [-1.29599811, -1.39102904, -0.5959823 , -1.07115345],
           [-0.41468229, -0.67587817,  0.03210209, -0.34856705],
           [ 0.44344101, -0.74860538, -0.94491807, -0.53190987],
           [ 1.76541475,  0.94836277,  0.03210209,  0.10439756],
           [-1.31919063, -1.06375661, -1.01470522, -1.44862395],
           [ 0.81452136,  1.56654403,  0.10188925,  0.70835037],
           [-0.78576263, -0.26375734,  1.35805802, -0.53190987],
           [ 1.00006153,  1.02108998,  0.59039932,  1.49564599],
           [-1.1800355 , -1.19708982,  0.03210209, -0.68289807],
           [ 1.9277624 ,  1.06957478, -1.5032153 , -0.44563089],
           [ 0.28109336,  0.0877575 ,  0.31125071,  0.75148985],
           [-0.41468229, -0.74860538, -0.87513091, -0.521125  ],
           [-0.80895515, -0.83345379, -0.24704653, -0.51034012],
           [ 1.02325405,  0.98472638,  1.0789094 ,  2.671197  ],
           [-1.31919063, -1.37890783, -0.66576945, -1.26528114],
           [-0.08998698, -0.14254532,  1.63720664, -0.26228808],
           [ 0.83771388,  1.38472601,  0.31125071,  1.17209984],
           [ 0.76813632,  1.00896878,  1.42784517,  0.52500755],
           [ 1.20879423,  2.01502847, -1.43342815, -0.55347961],
           [-1.62069341, -1.52436225, -1.5032153 , -1.50254831],
           [-0.11317951, -0.61527217,  0.66018648,  0.01811858],
           [-0.27552716, -0.23951493,  0.1716764 , -0.13286962],
           [-0.66980002, -0.14254532,  0.10188925,  0.87012344],
           [-0.34510472, -0.78496898,  0.45082502, -0.68289807],
           [-1.01768785,  0.03927269,  1.49763233, -1.39469959],
           [ 1.53348953,  1.3119988 , -1.22406668,  0.13675217],
           [-0.92491776, -1.027393  , -1.43342815, -0.90938037],
           [ 1.25517927,  0.20896951, -0.45640799,  0.61128652],
           [ 1.13921666,  0.36654512,  1.00912225,  0.46029832],
           [-1.06407289, -0.61527217,  1.00912225,  0.17989166],
           [-1.29599811, -1.48799864, -2.34066115, -1.08193832],
           [ 0.16513075, -0.17890893, -0.17725937, -0.05737552],
           [-0.87853272, -0.31224214,  0.52061217,  0.53579242],
           [-0.48425985, -1.08799901, -1.85215107, -1.28685088],
           [-1.20322802, -1.42739264,  0.03210209, -1.1250778 ],
           [-0.22914211, -0.11830292, -0.38662083, -0.60740397]])




```python
# Application of Equation 10.8

for k in range(0,np.shape(pca.components_)[1]):
    # Numerator
    accum = 0
    num = 0
    for i in range(0, np.shape(df_scl)[0]):
        for j in range(0, np.shape(df_scl)[1]):
            accum += pca.components_[k][j] * df_scl[i][j]
        num += accum**2
        accum = 0

    # Denominator
    accum = 0
    den = 0
    for j in range(0, np.shape(df_scl)[1]):
        for i in range(0, np.shape(df_scl)[0]):
            accum += df_scl[i][j]**2
        den += accum
        accum = 0

    # Result
    print('principal component number:', k+1)
    print(num/den)
```

    principal component number: 1
    0.620060394787
    principal component number: 2
    0.247441288135
    principal component number: 3
    0.0891407951452
    principal component number: 4
    0.0433575219325


# References
* http://stackoverflow.com/questions/21217710/factor-loadings-using-sklearn (loading vectors)
