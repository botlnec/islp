
# Exercise 4.3

This proceeds in much the same way as for the LDA case, with the additional consideration that the variance parameters are now different for each class. This means that we will not be able to dispose of these terms in the final expression which, it turns out, makes us pick up terms quadratic in x. So starting from Bayes' theorem 


$$ p_k(x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^K \pi_l f_l(x)}  ,$$

we substitute with the normal density for each class:

$$ f_k(x) \frac{1}{\sqrt{2\pi}\sigma_k } \exp \left( - \frac{1}{2\sigma_k^2} (x-\mu_k)^2\right)   .$$

We obtain the following expression

$$ p_k(x) = \frac{\pi_k \frac{1}{\sqrt{2\pi}\sigma_k } \exp \left( - \frac{1}{2\sigma_k^2 } (x-\mu_k)^2\right) } { \sum_{l=1}^K \pi_l \frac{1}{\sqrt{2\pi}\sigma_l } \exp \left( \frac{1}{2\sigma_l^2 } (x-\mu_k)^2\right)}. $$

for which we have to find class $k$ that maximizes this expression. Following the reasoning from the previous exercise, we obtain 

\begin{align}
\underset{k}{\operatorname{arg max}} p_k(x) & =  \underset{k}{\operatorname{arg max}} \pi_k \frac{1}{\sqrt{2\pi}\sigma_k } \exp \left( - \frac{1}{2\sigma_k^2 } (x-\mu_k)^2\right) = \\
& = \underset{k}{\operatorname{arg max}} \log \left(  \pi_k \frac{1}{\sqrt{2\pi}\sigma_k } \exp \left( - \frac{1}{2\sigma_k^2 } (x-\mu_k)^2\right)\right)  = \\
& = \underset{k}{\operatorname{arg max}} \log(\pi_k) - \log(\sqrt{2\pi}\sigma_k)  - \frac{1}{2\sigma_k^2 } (x-\mu_k)^2 = \\
& = \underset{k}{\operatorname{arg max}} \log(\pi_k) - \frac{1}{2\sigma_k^2 } (x^2 - 2x \mu_k + \mu_k^2) = \\
& = \underset{k}{\operatorname{arg max}}  - \frac{\mu_k^2}{2\sigma_k^2 } +  \log(\pi_k)  + x \cdot \frac{\mu_k}{\sigma_k^2} - x^2 \cdot \frac{1}{2\sigma_k^2} = \\
& = \underset{k}{\operatorname{arg max}} \delta_k(x),
\end{align}

It is clear that this simplified expression in quadratic in x (hence the name - quadratic discriminant analysis), since the variance is in general different for each class, and we cannot reduce it further.
