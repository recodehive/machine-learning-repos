from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
X = iris.data

# Choose the number of clusters (k)
k = 3

# Initialize and fit the KMeans algorithm
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Visualize the clustering results
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5)
plt.title('Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Print the cluster centers
print(kmeans.cluster_centers_)

# Print the number of data points in each cluster
for i in range(k):
    print(f'Number of data points in cluster {i}: {sum(labels == i)}')
