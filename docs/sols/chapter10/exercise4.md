
# Exercise 10.4

To solve this exercise, there are some definitions that we should take into account:

* <b>Linkage</b> - measures the dissimilarity between two groups of observations (cluster).
* <b>Single linkage</b> - linkage is given by the smallest pairwise distance between the observations in cluster A and the observations in cluster B.  
* <b>Complete linkage</b> - linkage is given by the largest pairwise distance between the observations in cluster A and the observations in cluster B.

# (a)

<b>We don't have enough information to tell</b>. If all observations in cluster {1,2,3} and cluster {4,5} have the same pairwise distance, the fusion between these clusters occur at the same height on the tree. An example would be *d(1,4)=d(1,5)=d(2,4)=d(2,5)=d(3,4)=d(2,5)=1*, where *d(x,y)* denotes the distance between observation *x* and observation *y*.

In contrast, if observations between clusters have different distances, the fusion will occur higher on the complete linkage. For example, if *d(1,4)=1*, *d(1,5)=2*, *d(2,4)=3*, *d(2,5)=4*, *d(3,4)=5*, and *d(3,5)=6*, single linkage would fuse at height 1 and complete linkage at height 6.

# (b)

<b>They fuse at the same height</b>. The distance between two observations is unique. Thus, the smallest and largest pairwise distance are the same and the will fuse at the same height.
