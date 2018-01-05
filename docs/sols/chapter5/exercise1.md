
# Exercise 5.1

We will use the following properties of the variance:
    

\begin{align}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\Var(X + Y)    & = \Var(X)+\Var(Y) + 2\Cov(X,Y)\\
\Var(\alpha X) & = \alpha^2 \Var(X) \\
\Cov(\alpha X, \beta Y) & = \alpha \beta \Cov(X,Y)
\end{align}

With these properties, we get:

\begin{align}
\Var(\alpha X + (1-\alpha) Y) & = \Var(\alpha X)   + \Var((1-\alpha)Y)    + 2\Cov(\alpha X,(1-\alpha) Y)\\
\                             & = \alpha^2 \Var(X) + (1-\alpha)^2 \Var(Y) + 2\alpha\beta\Cov(X,Y)
\end{align}

Now, we should find the minimum in the expression above. To do so, we differentiate the expression, equal it to 0 and solve for $\alpha$:

\begin{align}
\  0 & = 2 \alpha\Var(X) - 2(1-\alpha) \Var(Y)  + 2(1-2\alpha)\Cov(X,Y) \Leftrightarrow \\
\ \Leftrightarrow   0 & = \alpha\Var(X) - (1-\alpha) \Var(Y)  + (1-2\alpha)\Cov(X,Y) \Leftrightarrow \\
\ \Leftrightarrow   \alpha\Var(X) + \alpha\Var(Y) - 2\alpha\Cov(X,Y) & =  \Var(Y)  -\Cov(X,Y) \Leftrightarrow \\
\ \Leftrightarrow   \alpha & =  \frac{\Var(Y)  -\Cov(X,Y)}{\Var(X) + \Var(Y) - 2\Cov(X,Y)} = \frac{\sigma^2_Y - \sigma_{XY}}{\sigma^2_X + \sigma^2_Y - 2\sigma_X\sigma_Y}\\
\end{align}
