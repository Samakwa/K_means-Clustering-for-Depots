import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv



df = pd.read_csv('Enugu_PODs_Intial.csv', encoding='latin1')
df.head(10)

#print (df.head(10))

K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = df[['Latitude']]
X_axis = df[['Longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
#plt.show()


# Variable with the Longitude and Latitude
X=df.loc[:,['I.D','Latitude','Longitude']]
X.head(10)
#print (X.head(10))

kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(X[X.columns[1:3]]) # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:5]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(X[X.columns[1:3]]) # Labels of each point
print(X.head(15))
df = X.head(15)

X.plot.scatter(x = 'Latitude', y = 'Longitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#plt.show()

df.to_csv (r'C:\Users\Filepath\_result.csv', index = False, header=True)

print (df)

#clustered_data.to_csv ('clustered_data.csv', index=None, header = True)
centers = kmeans.cluster_centers_

print ("Cluster Centroids:")
print(centers)