
# Exercise 2.4

### (a) Classification

**Spam filters**. 
Any self-respecting email service has a good spam filter: ideally it keeps any undesired email away from your inbox, and let's through every email you're interested in.
The response has just two categories: 'spam' or 'not spam', and the predictors can include several different variables: whether you have corresponded with the sender, whether you have their email address in some of your non-spam emails, the words and sequences of words of the emails (for example, if the email includes any reference to a 'Nigerian prince'), whether similar emails have already been tagged as spam by other users, etc.
The main goal is **prediction** because the most important task is to accurately be able to predict whether a future email message is spam or not-spam. 
An important aspect in this case is the rate of false positives and false negatives.
In the case of email it is usually much more acceptable to have false negatives (spam that got through to your inbox) than false positives (important email that was classified as spam and you have never noticed). 

**Digit recognition. (or text or speech or facial recognition)** 
An important task required everyday around the world is the recognition of handwritten digits and addresses for postal service or document scanning and OCR (Optical Character Recognition).
Depending on how we define the learning task, the predictors can be the original digital photographs or scans of the documents, or just a cropped, grayscale image of a single digit (that is, a N by M matrix of real numbers, each describing the gray scale value of a single pixel).
The response might be one of ten digits (0 to 9) or it might include commas and the full numbers (not just the digits), depending on how we decide to define the task.
Again here the main goal is **prediction**, since the most important thing is to recognize a correct address or bank account number from a letter or document. 

**Social analysis**. Classify people into supporters of [Sporting](https://en.wikipedia.org/wiki/Sporting_Clube_de_Portugal), supporters of [Benfica](https://en.wikipedia.org/wiki/S.L._Benfica), neither or both (so the response is one of these four categories).
As predictors, include factors such as age, nationality, address, income, education, gender, whether they appreciate classical music, their criminal record, etc.
Here the main goal is **inference**, since more than predicting whether someone supports a specific club we are interested in understanding and studying the different factors and discovering interesting relationships between them.

**Other examples**: fraud detection, medical diagnosis, stock prediction (buy, sell, hold), astronomical objects classification, choosing [strawberries](http://www.geekwire.com/2016/jeff-bezos-sees-future-amazon-echo-alexa-healthcare/) or [cucumbers](https://cloud.google.com/blog/big-data/2016/08/how-a-japanese-cucumber-farmer-is-using-deep-learning-and-tensorflow).

### (b) Regression

**Galton's original 'regression' to the mean**. 
Study the height of children and its relationship to the height of their parents.
Here the response is the height of the child and the predictor is the height of both parents (or the mean of both).
The main goal would be **inference**, since we are trying to better understand, from a scientific perspective, how much genetics influences an individual's height.
Unless, of course, you are an overly zealous future parent trying to predict the chances of your child playing in the NBA given your own height and your partner's - in which case your main goal might be prediction.
This problem is [where the name 'regression' comes from](https://en.wikipedia.org/wiki/Regression_toward_the_mean) and, perhaps surprisingly, [Galton's statistical method](https://www.wired.com/2009/03/predicting-height-the-victorian-approach-beats-modern-genomics/) has stood the test of time quite well.

**House prices**. Buying a house is probably the biggest investment that many people do during their life time.
Accordingly, it is natural that people want to know the value of a house in order to do the best deal possible.
The main goal in this cases is **prediction**, since we want to predict house's price considering a set of predictors. 
There are several factors used to predict house's price. 
For example, a [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/kernels) in which we participated, suggests a total of 79 predictors to predict the final price of each house in Boston.

**Weather**.
The response can be temperature, wind velocity or precipitation amount in different locations, and the predictors can be the same variables for previous times.
The goal is both **prediction** (will it rain too much this month and ruin the plantations; will there be enough snow for the snowtrip?) and **inference** (what causes hurricanes, or the current climate change?). 

**Other examples**:
returns of investment in stocks, predicting school grades from amount and type of study, predicting the popularity of a book, film, article or tweet.

### (c) Cluster analysis

**Outlier detection**.
If you can define a distance between the data points or a density function in the predictors space, you can use cluster analysis to identify candidate outliers such as the data points farther from the majority of the points, or those in sparser regions.

**Market research**. 
Using cluster analysis, market researchers can group consumers into market segments according to their similarities.
This can provide a better understanding on group characteristics, as well as it can provide useful insights about how different groups of consumers/potential customer behave and relate with different groups.


**Recommender systems**.
Will you think watching 'Pulp Fiction' is time well spent?
If enough is known about you, including some of your film preferences (and perhaps how many times you have watched 'Stalker' or 'Pulp Fiction') and some similar data about other people, we can have an indication of whether will you think rewatching 'Pulp Fiction' is worth your time...
Are you 'closer' to the people who think the answer is 'No' or to the ones that think 'Yes'?

**Other examples**:
gene sequencing, data compression, social network analysis, representation learning, topography classification.
