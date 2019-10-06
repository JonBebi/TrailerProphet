from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score,recall_score,f1_score,precision_score,confusion_matrix
from sklearn.linear_model import LogisticRegression



###### train_test_split #####
y = df['freshness']
X = df.drop(columns=['rt_score','freshness','movie','total_likes','total_replies','total_retweets'])
X_scaled = StandardScaler().fit_transform(merged_df.drop(columns=['movie']))
# y = df['rt_scores']
# X = df.drop(columns=['tweet','rt_scores','movie','reply_to','retweet_date'])
xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.5,random_state=3)

### Encoding for movies but saving for genres in the future
# data = merged_df['movie']
# values = np.array(data)
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)

########################################
test = merged_df[['avg_sentiment','freshness']]
merged_df.columns
kmeans = KMeans(n_clusters=5).fit(test)
centroids = kmeans.cluster_centers_
print(centroids)

from sklearn.metrics import silhouette_score
X = test
silhouette_plot = []
for k in tqdm(range(2, 10)):
    clusters = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusters.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_plot.append(silhouette_avg)

import numpy as np
plt.figure(figsize=(15,8))
plt.subplot(121, title='Silhouette coefficients over k')
plt.xlabel('k')
plt.ylabel('silhouette coefficient')
plt.plot(range(2, 10), silhouette_plot)
plt.axhline(y=np.mean(silhouette_plot), color="red", linestyle="--")
plt.grid(True)

X = test
distorsions = []

# Calculate SSE for different K
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state = 301)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

# Plot values of SSE
plt.figure(figsize=(15,8))
plt.subplot(121, title='Elbow curve')
plt.xlabel('k')
plt.plot(range(2, 10), distorsions)
plt.grid(True)


###################################################################
####################### Hierarchical Clustering ###################
###################################################################


plt.scatter(X_scaled[:,0], X_scaled[:,1])

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_scaled[:5000], 'single')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X_scaled[:5000]))

# calculate and construct the dendrogram
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# trimming and truncating the dendrogram
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

plt.scatter(X_scaled[:,0],X_scaled[:,2])

cluster = AgglomerativeClustering(n_clusters=2)
cluster
pred_cluster = cluster.fit_predict(X_scaled)

from sklearn.metrics import silhouette_score
silhouette_score(X_scaled, pred_cluster)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c = pred_cluster)
