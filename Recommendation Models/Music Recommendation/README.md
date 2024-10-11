## Mock Spotify Music Recommendation System ##
**Project Overview:**
In this project, we will build a song recommendation system based on your personal Spotify data that divides the songs that you liked into 'k' number of playlists using the K-Means Clustering algorithm based on similarity in audio features such as energy, tempo, danceability, etc. Taking a look at the clustering, you will get an idea about what each playlist represents, e.g. you may notice that playlist #1 contains slow and melancholic songs, etc. Further, you get to test the recommendation ability of the system by getting new songs by a particular artist/any other way to get a bunch of unseen random songs to test whether it makes sense for the new songs to be classified under the category they have been assigned to.
K-Means is a popular unsupervised machine learning algorithm used for clustering data into groups based on feature similarity. The goal is to partition a dataset into 'k' distinct clusters, where each data point belongs to the cluster with the nearest mean.
**Steps of the algorithm:**
Initialization: Choose 'k' initial centroids randomly from the data points.
Assignment Step: Assign each data point to the nearest centroid, forming kk clusters.
Update Step: Recalculate the centroids as the mean of the points in each cluster.
Convergence Check: Repeat the assignment and update steps until centroids no longer change significantly or a maximum number of iterations is reached.
We use the library function from sklearn to achieve our purpose here.
**Pros and Cons of the algorithm:**
Pros:

* Simple to implement and understand.
* Efficient for large datasets.
* Works well with spherical clusters.

Cons:

* Requires the number of clusters 'k' to be specified in advance.
* Sensitive to initial centroid placement.
* Can converge to local minima.

