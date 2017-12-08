
# Exercise 4.1

Writing $Y=\beta_0 + \beta_1 X$ for clarity, we start with

\begin{align}
P(X) & = \frac{e^Y}{1+e^Y}, 
\end{align}

and "using a little bit of algebra", we get

\begin{align}
\frac{P(X)}{1-P(X)} & = \frac{e^Y}{1+e^Y} \left( 1 - \frac{e^Y}{1+e^Y}  \right)^{-1} = \left( \frac{(1+e^Y)-e^Y}{1+e^Y}  \right)^{-1} = \frac{e^Y}{1+e^Y} \frac{1+e^Y}{1} = e^Y = e^{\beta_0 + \beta_1 X}.
\end{align}
