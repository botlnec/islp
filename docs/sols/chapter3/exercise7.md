
# Exercise 3.7

To show that  $R^2=r^2 \equiv cor^2(X,Y)$ for simple linear regression of $Y$ onto $X$, we will use the following formulas from the text (where we have used the simplifying assumption that $\bar{x}=\bar{y}=0$):

$$
\begin{align}
cor(X,Y) & = \frac{\sum_i(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i - \bar{x})^2}\sqrt{\sum_i(y_i - \bar{y})^2}} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2}\sqrt{\sum_i y_i^2}},\\
\hat{\beta}_0 & = \bar{y} -\hat{\beta}_1 \bar{x} = 0, \\
\hat{\beta}_1 & = \frac{\sum_i(x_i - \bar{x})(y_i - \bar{y})}{\sum_i(x_i - \bar{x})^2} = \frac{\sum_i x_i y_i}{\sum_i x_i^2}, \\
R^2 & = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_i(y_i - \hat{y})^2}{\sum_i(y_i - \bar{y})^2}= 1 - \frac{\sum_i(y_i - \hat{\beta}_1 x_i)^2}{\sum_i y_i^2}.
\end{align}
$$

Expanding the numerator in the last expression and plugging in the formula for $\hat{\beta}_1$, we obtain

$$
\begin{align}
R^2 & = 1 - \frac{\sum_i(y_i^2 - 2\hat{\beta}_1 x_i y_i + \hat{\beta}_1^2 x_i^2 )}{\sum_i y_i^2} = \frac{ 2\hat{\beta}_1 \sum_ i x_i y_i - \hat{\beta}_1^2 \sum_i x_i^2 }{\sum_i y_i^2} = \\
  & = \frac{ 2 (\sum_i x_i y_i) ( \sum_ i x_i y_i)/ (\sum_i x_i^2) -( \sum_ i x_i y_i)^2 (\sum_i x_i^2)/ (\sum_i x_i^2)^2  }{\sum_i y_i^2} = \frac{(\sum_i x_i y_i)^2}{\sum_i x_i^2 \sum_i y_i^2},
\end{align}
$$

which we can see is equal to $cor^2(X,Y)$.
