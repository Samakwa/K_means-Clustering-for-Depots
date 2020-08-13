

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import pandas as pd
list1 =[]
#Data
#df = pd.read_csv('National_data2.csv', encoding='latin1')
df = pd.read_csv('National_Demand_Points.csv', encoding='latin1')
#centers = np.load('National_data2.csv', encoding='latin1')
for index, row in df.iterrows():
    # print(row['longitude'], row['latitude'])
    a = []
    p = list(a)
    k = []
    cord = []
    #demand1 =[]
    k.append(row['longitude'])
    k.append(row['latitude'])
    #popn.append(row['population'])
    #k.append(row['id'])
    #k.append(row['address'])
    #k.append(row['city'])
    #k.append(str(row['zip']))

    for x in k:
        p.append(x)

    list1.append((p))
#centers = [[1, 1], [-1, -1], [1, -1]]
#centers = [[1, 1], [-1, -1], [1, -1]]

#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
X, _ = (list1[0], list1[1])

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
