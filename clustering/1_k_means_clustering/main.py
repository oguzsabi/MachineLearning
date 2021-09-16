import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.scatter(np.arange(1, 11), wcss, color='red')
plt.plot(np.arange(1, 11), wcss, color='blue')
plt.xlabel('WCSS')
plt.ylabel('Number of Clusters')
plt.title('The Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
colors = ['red', 'green', 'blue', 'orange', 'yellow']

for i in range(0, 5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], color=colors[i], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, color='black', label='Cluster Centers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters of Customers')
plt.legend()
plt.show()
