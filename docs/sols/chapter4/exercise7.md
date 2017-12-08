
# Exercise 4.7

From the statement of the exercise, we know that

\begin{align}
\ Pr(X \mid Y = yes) & = f_{yes}(x)  = N(\mu = 10, \sigma^2 = 36),\\
Pr(X \mid Y = no) & = f_{no}(x) = N(\mu = 0, \sigma^2 =36),\\
Pr(Y = yes) & = \pi_{yes} = 0.8,\\
Pr(Y = no)&  = \pi_{no} = 0.2.\\
\end{align}

We want to calculate $Pr(Y = "Yes \mid X = 4)$. Using Bayes' theorem and substituting the expressions above:

\begin{align}
\ \pi_{yes}(x) & =  \frac{\pi_{yes} f_{yes}(x) }{ \pi_{yes} f_{yes}(x) + \pi_{no} f_{no}(x)},\\
\ \pi_{yes}(4) & =  \frac{\pi_{yes} f_{yes}(4) }{ \pi_{yes} f_{yes}(4) + \pi_{no} f_{no}(4)} = \\
& = \frac{0.8 e^{\left( -\frac{1}{2\times 36}(4-10)^2 \right)} }{  0.8 e^{\left( -\frac{1}{2\times 36}(4-10)^2 \right)} + 0.2 e^{\left( -\frac{1}{2\times 36}(4-0)^2 \right)}} \\
& = 0.7571.\\
\end{align}
