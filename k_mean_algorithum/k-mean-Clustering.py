#pip install scikit-learn
# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#Load the dataset
# Replace this with your own dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the dataset
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (k)
# In this example, we use the elbow method to determine the optimal k value
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow method, choose the optimal k value
# In this example, we choose k = 3
k = 3

# Initialize the KMeans algorithm
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the algorithm to the dataset
kmeans.fit(X_scaled)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Visualize the clustering results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5)
plt.title('Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Print the cluster centers
print(kmeans.cluster_centers_)

# Print the cluster labels for each data point  
print(labels)

# Print the number of data points in each cluster
for i in range(k):
    print(f'Number of data points in cluster {i}: {sum(labels == i)}')

# Print the total number of data points
print(f'Total number of data points: {len(labels)}')

# Print the total number of clusters
print(f'Total number of clusters: {k}')


