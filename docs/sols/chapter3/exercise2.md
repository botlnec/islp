
# Exercise 3.2

Both models share a similar principle: the output values are computed based on their K closest points (nearest neighbours) value.

The KNN classifier starts by identifying the K nearest neighbours. Then, the output of each K observation is considered and, by majority vote, we determine the label of our observation. Example: if we are trying to classify an observation as 'blue' or 'red', and in the K nearest neighbours we have two of them classified as 'blue' and one as 'red', our observation will be classified as 'blue'. Notice that if we have a tie, a common solution is to increase or decrease K.

Regarding the KNN regression method, it also starts by identifying the K nearest neighbours. However, in this situation, we compute our output averaging the output of the K nearest neighbours. Example: if our K nearest neighbours have as output the values 3,4 and 5, our output will be (3+4+5)/3 = 4. 
