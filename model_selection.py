from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score,recall_score,f1_score,precision_score,confusion_matrix
from sklearn.linear_model import LogisticRegression



###### train_test_split #####
y = merged_df['freshness']
X = merged_df.drop(columns=['freshness','movie','total_likes','total_replies','total_retweets'])
X_scaled = StandardScaler().fit_transform(X)
y.shape
X_scaled.shape
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

#########################
## Logistic Regression ##
#########################
lr = LogisticRegression(class_weight='balanced',random_state=5,C=2)
lr.fit(xTrain,yTrain)
y_pred = lr.predict(xTest)


confusion_matrix(y_oct,y_pred)
accuracy_score(yTest,y_pred)
f1_score(yTest,y_pred)
recall_score(yTest,y_pred)
precision_score(yTest,y_pred)

oct_df = oct_df[oct_df['freshness']!= "NA"]
oct_df.rename(columns={'total tweets':'total_tweets'},inplace=True)
y_oct
y_oct = oct_df['freshness']
x_oct = oct_df.drop(columns=['freshness','movie','total_likes','total_replies','total_retweets'])

# pip install imblearn
from imblearn.over_sampling import SMOTE
smt = SMOTE()
xTrain, yTrain = smt.fit_sample(xTrain,yTrain)

################################
########## ~ S V M ~ ###########
################################
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

clf = SVC(kernel='linear',C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()
clf.fit(xTrain,yTrain)
pred = clf.predict(xTest)
confusion_matrix(yTest,pred)
accuracy_score(yTest,pred)
f1_score(yTest,pred)
recall_score(yTest,pred)
precision_score(yTest,pred)

def plot_feature_importances(model,X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances(lr,xTrain)

from sklearn.model_selection import GridSearchCV

# Grid search function for Random Forest model
def grid_search(xTrain,xTest,yTrain,yTest):
    gs = GridSearchCV(estimator=SVC(),
                     param_grid={'max_depth': [3,8],
                                 'n_estimators': (25,50,100),
                                 'max_features': (4,5)},
                     cv=4,n_jobs=-1,scoring='balanced_accuracy')
    model = gs.fit(xTrain,yTrain)
    print(f'Best score: {model.best_score_}')
    print(f'Best parms: {model.best_params_}')

grid_search(xTrain,xTest,yTrain,yTest)


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
