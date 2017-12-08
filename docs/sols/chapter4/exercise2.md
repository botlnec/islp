
# Exercise 4.2

This is equation (4.12) from the book, expressing the probability that $Y = k$, given $X = x$:

\begin{align}
p_k(x) & = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma } \exp \left( - \frac{1}{2\sigma^2 } (x-\mu_k)^2\right) } { \sum_{l=1}^K \pi_l \frac{1}{\sqrt{2\pi}\sigma } \exp \left( \frac{1}{2\sigma^2 } (x-\mu_k)^2\right)}.
\end{align}

We want to show that the class $k$ for which this probability is largest is the same as the one that is largest for equation (4.13) from the book:

\begin{align}
\delta_k(x) & = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2}+\log(\pi_k).
\end{align}

In short, this involves 3 simple steps: noticing the denominator is the same for all classes (so it can be ignored when trying to find the maximum), taking the logarithm of the whole expression (since the logarithm is monotonically increasing, the maximum will be preserved), and expanding the square of the remaining expression.

Here we go:

\begin{align}
\underset{k}{\operatorname{arg max}} p_k(x) & =  \underset{k}{\operatorname{arg max}} \pi_k \frac{1}{\sqrt{2\pi}\sigma } \exp \left( - \frac{1}{2\sigma^2 } (x-\mu_k)^2\right) = \\
& = \underset{k}{\operatorname{arg max}} \log \left(  \pi_k \frac{1}{\sqrt{2\pi}\sigma } \exp \left( - \frac{1}{2\sigma^2 } (x-\mu_k)^2\right)\right)  = \\
& = \underset{k}{\operatorname{arg max}} \log(\pi_k) - \log(\sqrt{2\pi}\sigma)  - \frac{1}{2\sigma^2 } (x-\mu_k)^2 = \\
& = \underset{k}{\operatorname{arg max}} \log(\pi_k) - \frac{1}{2\sigma^2 } (x^2 - 2x \mu_k + \mu_k^2) = \\
& = \underset{k}{\operatorname{arg max}} x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2 } +  \log(\pi_k)= \\
& = \underset{k}{\operatorname{arg max}} \delta_k(x),
\end{align}

where again we have dropped every expression that does not depend on the class $k$.
