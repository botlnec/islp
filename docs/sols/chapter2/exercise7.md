
# Exercise 2.7


```python
import numpy as np
import pandas as pd

d = {'X1': pd.Series([0,2,0,0,-1,1]),
     'X2': pd.Series([3,0,1,1,0,1]),
     'X3': pd.Series([0,0,3,2,1,1]),
     'Y': pd.Series(['Red','Red','Red','Green','Green','Red'])}
df = pd.DataFrame(d)
df.index = np.arange(1, len(df) + 1)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>Red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>Red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>Red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>Green</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>Green</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Red</td>
    </tr>
  </tbody>
</table>
</div>



### (a) Euclidian distance


```python
from math import sqrt
df['distance']=np.sqrt(df['X1']**2+df['X2']**2+df['X3']**2)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>Y</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>Red</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>Red</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>Red</td>
      <td>3.162278</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>Green</td>
      <td>2.236068</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>Green</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Red</td>
      <td>1.732051</td>
    </tr>
  </tbody>
</table>
</div>



### (b) K=1


```python
df.sort_values(['distance'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>Y</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>Green</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Red</td>
      <td>1.732051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>Red</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>Green</td>
      <td>2.236068</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>Red</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>Red</td>
      <td>3.162278</td>
    </tr>
  </tbody>
</table>
</div>



As we can see by sorting the data by distance to the origin, for K=1, our prediction is **Green**, since that's the value of the nearest neighbor (point 5 at distance 1.41). 

### (c) K=3

On the other hand, for K=3 our prediction is **Red**, because that's the mode of the 3 nearest neigbours: Green, Red and Red (points 5, 6 and 2, respectively).

### (d) Highly non-linear Bayes decision boundary

A large value of K leads to a smoother decision boundary, as if the non-linearities where averaged out. This happens because KNN uses majority voting and this means less emphasis on individual points. For a large value of K, we will likely have a decision boundary which varies very little from point to point, since the result of this majority voting would have to change while for most points this will be a large majority. That is, one of its nearest neighbors changing from one class to the other would still leave the majority voting the same. By contrast, when K is very small, the decision boundary will be better able to capture local non-linearities, because it the majority of neighbors can vary significantly from point to point since the are so few neighbors. Accordingly, we would expect **the best value of K to be small**. 

Imagine this simple example: a true linear boundary that splits the plane in two semi-planes (classes A and B), but with an additional enclave in one of the regions. That is, the true model has a small region of, say, class A inside the class B semi-plane. Would we more likely capture this region for small or large K? Say we have 3 neighboring data points in the class A enclave inside the large class B region with say 50 points. If K > 4, each of the 3 points in the enclave will be classified as B, since they always lose by majority voting (unless K is so large, say 100, that many of the points from A semi-plane region enter this voting). If, on the other hand, K < 4, each of the 3 class A points inside the enclave will be classified as class A, and we will capture this non-linearity of the regions.
