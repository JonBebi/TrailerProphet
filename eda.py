from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

merged_df.head()

## Create merged_df columns for ratios
merged_df['like_percent'] = merged_df['total_likes'] / merged_df['total_tweets']
merged_df['reply_percent'] = merged_df['total_replies'] / merged_df['total_tweets']
merged_df['retweet_percet'] = merged_df['total_retweets'] / merged_df['total_tweets']

oct_df['like_percent'] = oct_df['total_likes'] / oct_df['total tweets']
oct_df['reply_percent'] = oct_df['total_replies'] / oct_df['total tweets']
oct_df['retweet_percet'] = oct_df['total_retweets'] / oct_df['total tweets']

oct_df.shape
oct_df.head()
oct_df.describe()

oct_df = oct_df[oct_df['total tweets']> 7]

merged_df.shape

merged_df.head()

merged_df.describe()
merged_df['total_tweets'].sum()
## Investigate the ones with no likes
merged_df[merged_df['total_likes'] == 0]
## Dropping those movies
merged_df = merged_df[merged_df['total_likes'] != 0]
## Dropping movies with less than 30 tweets
merged_df = merged_df[merged_df['total_tweets'] > 30]
merged_df.corr()


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(X)
df_pca = pd.DataFrame(data=pca_data,columns=['PC1','PC2','PC3'])
result_df = pd.concat([df_pca, y],axis=1)

import numpy as np
plt.scatter(pca_data[:,0],pca_data[:,1])
import numpy as np
sns.barplot(pca.explained_variance_ratio_,merged_df.columns)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

score_rt = []
for m in tqdm(merged_df['movie']):
    if m in list(rt_dict.keys()):
        score_rt.append(rt_dict[m])
    else:
        score_rt.append(1)
merged_df['freshness'] = score_rt
merged_df.corr()


merged_df[merged_df['avg_sentiment']!=0].drop(columns=['total_likes','total_replies','total_retweets']).sort_values(by='total_tweets')

merged_df.corr()
merged_df.corr()
df.columns
merged_df[merged_df['like_percent']<20].plot.scatter(x='avg_sentiment',y='freshness')


merged_df.head()
#########################
## Logistic Regression ##
#########################
lr = LogisticRegression(class_weight='balanced',random_state=5)
lr.fit(xTrain,yTrain)
y_pred = lr.predict(x_oct)


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
