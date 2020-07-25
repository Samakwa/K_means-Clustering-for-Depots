import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

f = open('National_data2.csv', encoding='latin1')
EOCs = pd.read_csv(f)
kmeans = cluster.KMeans(37)
kmeans.fit(EOCs[['longitude', 'latitude']])
labels = kmeans.labels_
plt.scatter(EOCs['longitude'], EOCs['latitude'], c=labels)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.title (" Initial Clusters ")
plt.show()

kmeans = cluster.KMeans(37)
kmeans.fit(EOCs[['longitude', 'latitude']])
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(EOCs['longitude'], EOCs['latitude'], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', c='black')
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.title (" Optimised Clusters ")
plt.show()
plt.show()

# K-means clustering based on population
kmeans = cluster.KMeans(37)
kmeans.fit(EOCs[['Popn']])
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(EOCs['longitude'], EOCs['latitude'], c=labels)
plt.show()
# print(centroids)