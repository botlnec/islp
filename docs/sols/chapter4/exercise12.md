
# Exercise 4.12


```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
```

## (a)


```python
def Power():
    print(2**3)
```


```python
Power()
```

    8


## (b)


```python
def Power2(x,a):
    print(x**a)
```


```python
Power2(3,8)
```

    6561


## (c)


```python
Power2(10,3)
```

    1000



```python
Power2(8,17)
```

    2251799813685248



```python
Power2(131,3)
```

    2248091


## (d)


```python
def Power3(x,a):
    result = x**a
    return result
```

## (e)


```python
def Plot(log=''):
    x = np.arange(1,10)
    y = Power3(x,2)
    
    #create plot
    fig, ax = plt.subplots()
    
    #config plot
    ax.set_xlabel('x')
    ax.set_ylabel('y=x^2')
    ax.set_title('Power3()')
    
    #change scale according to axis
    if log == 'x':
        ax.set_xscale('log')
        ax.set_xlabel('log(x)')
    if log == 'y':
        ax.set_yscale('log')
        ax.set_ylabel('log(y=x^2)')
    if log == 'xy':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('log(x)')
        ax.set_ylabel('log(y=x^2)')
    
    #draw plot
    ax.plot(x, y)
```


```python
Plot(log='xy')
```


![png](04_12_files/04_12_16_0.png)


## (f)


```python
def PlotPower(start,end,power,log=''):
    x = np.arange(start,end)
    y = np.power(x,end)
    
    #create plot
    fig, ax = plt.subplots()
    
    #config plot
    ax.set_xlabel('x')
    ax.set_ylabel('y=x^2')
    ax.set_title('PlotPower()')
    
    #change scale according to axis
    if log == 'x':
        ax.set_xscale('log')
        ax.set_xlabel('log(x)')
    if log == 'y':
        ax.set_yscale('log')
        ax.set_ylabel('log(y=x^2)')
    if log == 'xy':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('log(x)')
        ax.set_ylabel('log(y=x^2)')
    
    #draw plot
    ax.plot(x, y)
```


```python
PlotPower(1,10,3)
```


![png](04_12_files/04_12_19_0.png)

