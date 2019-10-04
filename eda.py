from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score,recall_score,f1_score,precision_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(X)
df_pca = pd.DataFrame(data=pca_data,columns=['PC1','PC2','PC3'])
result_df = pd.concat([df_pca, y],axis=1)

import numpy as np
plt.scatter(pca_data[:,0],pca_data[:,1])

sns.barplot(pca.explained_variance_ratio_,df_movies.columns)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

score_rt = []
for m in tqdm(df['movie']):
    if m in list(rt_dict.keys()):
        score_rt.append(rt_dict[m])
    else:
        score_rt.append(1)
df['rt_score'] = score_rt
df_movies.corr()



df_movies['like_percent'] = df_movies['total_likes'] / df_movies['total tweets']
df_movies['reply_percent'] = df_movies['total_replies'] / df_movies['total tweets']
df_movies['retweet_percet'] = df_movies['total_retweets'] / df_movies['total tweets']

df_movies[df_movies['avg_sentiment']!=0].drop(columns=['total_likes','total_replies','total_retweets']).sort_values(by='total tweets')

df_movies.corr()

df.columns
df_movies[df_movies['like_percent']<20].plot.scatter(x='avg_sentiment',y='freshness')
############################
### Train Test Split #######
############################
y = df_movies['freshness']
X = df_movies.drop(columns=['freshness','movie','total_likes','total_replies','total_retweets'])
# X_scaled = StandardScaler().fit_transform(df.drop(columns=['movie','tweet','reply_to','retweet_date']))
# y = df['rt_scores']
# X = df.drop(columns=['tweet','rt_scores','movie','reply_to','retweet_date'])
xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.5,random_state=3)


#########################
## Logistic Regression ##
#########################
lr = LogisticRegression(class_weight='balanced',random_state=5)
lr.fit(xTrain,yTrain)
y_pred = lr.predict(xTest)

confusion_matrix(yTest,y_pred)
accuracy_score(yTest,y_pred)
f1_score(yTest,y_pred)
recall_score(yTest,y_pred)
precision_score(yTest,y_pred)

################################
########## ~ S V M ~ ###########
################################
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

clf = svm.SVC(kernel='linear',C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()
clf.fit(xTrain,yTrain)
pred = clf.predict(xTest)
confusion_matrix(yTest,pred)
accuracy_score(yTest,pred)
f1_score(yTest,pred)
recall_score(yTest,pred)
precision_score(yTest,pred)
