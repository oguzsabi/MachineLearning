import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
wcss = []

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.title('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
colors = ['red', 'green', 'blue', 'orange', 'yellow']

for i in range(0, 5):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], color=colors[i], label=f'Cluster {i}')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters of Customers')
plt.legend()
plt.show()
