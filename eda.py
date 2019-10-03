from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score,recall_score,f1_score,precision_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

df_movies.head(20)

df_movies[df_movies['freshness']==0].corr()

df_movies['like_percent'] = df_movies['total_likes'] / df_movies['total tweets']
df_movies['reply_percent'] = df_movies['total_replies'] / df_movies['total tweets']
df_movies['retweet_percet'] = df_movies['total_retweets'] / df_movies['total tweets']

df_movies[df_movies['avg_sentiment']!=0].drop(columns=['total_likes','total_replies','total_retweets']).sort_values(by='like_percent')
############################
### Train Test Split #######
############################
y = df_movies['freshness']
X = df_movies.drop(columns=['freshness','movie','total_likes','total_replies','total_retweets'])
X_scaled = StandardScaler().fit_transform(X)
xTrain,xTest,yTrain,yTest = train_test_split(X,y,test_size=.3,random_state=10)

#########################
## Logistic Regression ##
#########################
lr = LogisticRegression(class_weight='balanced',random_state=10)
lr.fit(xTrain,yTrain)
y_pred = lr.predict(xTest)

confusion_matrix(yTest,y_pred)
accuracy_score(yTest,y_pred)
f1_score(yTest,y_pred)
recall_score(yTest,y_pred)
precision_score(yTest,y_pred)


### From Learn.co for reference
###################
# Now let's compare a few different regularization performances on the dataset:
weights = [None, 'balanced', {1:2, 0:1}, {1:10, 0:1}, {1:100, 0:1}, {1:1000, 0:1}]
names = ['None', 'Balanced', '2 to 1', '10 to 1', '100 to 1', '1000 to 1']
colors = sns.color_palette("Set2")

plt.figure(figsize=(10,8))

for n, weight in enumerate(weights):
    #Fit a model
    logreg = LogisticRegression(fit_intercept = False, C = 1e12, class_weight=weight,solver='lbfgs') #Starter code
    model_log = logreg.fit(X_train, y_train)
    print(model_log) #Preview model params

    #Predict
    y_hat_test = logreg.predict(X_test)

    y_score = logreg.fit(X_train, y_train).decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
    lw = 2
    plt.plot(fpr, tpr, color=colors[n],
             lw=lw, label='ROC curve {}'.format(names[n]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
