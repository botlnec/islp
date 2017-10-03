
# Exercise 2.6

That this exercise does not ask for a precise definition of parametric and non-parametric methods and that the main text does also not provide one (section 2.1.2) is a sign that this definition is not consensual, and in fact it depends on the context in which it is used in statistics [1].
For our purposes we follow both the main text and Murphy [2] (and a few others [3-6]), for a definition of a parametric and non-parametric model (and by model, this we mean a probability distribution or density, or a regression function).
So, for our purposes, a parametric model means two thin
gs: (1) strong, explicit assumptions about the form of the model, (2) which is parametrized by a finite, fixed number of parameters that does not change with the amount of data.
(Yes, the first part of this definition is vague.)
Conversely, a non-parametric model means that, in general, the number of parameters depends on the amount of data and that we try to keep assumptions as weak as possible.
'Non-parametric' is thus an unfortunate name since it does not mean 'no parameters'.
On the contrary, they do contain parameters (often many more) but these tend to determine the model complexity rather than the form.
The typical examples of a parametric and non-parametric models are linear regression and KNN, respectively.
Another good example, is the one from the text and figures 2.4, 2.5 and 2.6.

The advantages of a parametric approach are, in general, that it requires less observations, is faster and more computationally tractable, and more robust (less misguided by noise).
The disadvantages are the stronger assumptions that make the model less flexible, perhaps unable to capture the underlying adequately regardless of the amount of training data.

There are also other types of models such as semi-parametric models [6], which have two components: a parametric and a non-parametric one.
One can also have a non-parametric model by using a parametric model but aditionally adapting the number of parameters as needed (say the degree of the polynomial in a linear regression).

### References

* [1] https://en.wikipedia.org/wiki/Nonparametric_statistics#Definitions
* [1] Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.
* [2] Domingos, Pedro. "A few useful things to know about machine learning." Communications of the ACM 55.10 (2012): 78-87.
* [3] Wasserman, Larry. All of statistics: a concise course in statistical inference. Springer Science & Business Media, 2013.
* [4] Bishop, C.M.: Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag New York, Inc., Secaucus, NJ, USA (2006)
* [5] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
* [6] Wellner, Jon A., Chris AJ Klaassen, and Ya'acov Ritov. "Semiparametric models: a review of progress since BKRW (1993)." (2006): 25-44.
