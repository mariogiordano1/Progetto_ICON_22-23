import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import metrics, linear_model, model_selection, svm
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import f1_score, explained_variance_score, r2_score, max_error, zero_one_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cdist
import statsmodels.api as sm
from collections import Counter
from matplotlib import pyplot
from scipy.stats import zscore
from sklearn.inspection import permutation_importance


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Database/CSVS/merged_files.csv", index_col=False)
df_finale = df.drop(columns=["Country", "City", "CO_AQI_Category", "Ozone_AQI_Category",
                         "NO2_AQI_Category", "PM2_5_AQI_Category", "Unnamed: 0", "PM10_Category", "AQI_Value"])
                         

df_finale.dropna(axis=0, how="any",inplace=True)

X = df_finale.drop(columns="AQI_Category")
y = df_finale["AQI_Category"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
numeric_cols = df_finale.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X.apply(zscore)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42, shuffle=False)
train_scores_dec, test_scores_dec = list(), list()
train_scores_forest, test_scores_forest = list(), list()
train_scores_knn, test_scores_knn = list(), list()

knn_k = [i for i in range(1,35, 2)]
for i in knn_k:
    knn = KNeighborsClassifier(n_neighbors=i)

    X_train_KNN = X_train.drop(columns=["CO_AQI_Value", "Ozone_AQI_Value",
                            "NO2_AQI_Value", "PM2_5_AQI_Value", "PM10"])
    knn.fit(X_train, y_train)
    y_hat = knn.predict(X_test)
    y_hat_train =  knn.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, y_hat_train)
    train_scores_knn.append(train_acc)

    test_acc = metrics.accuracy_score(y_test, y_hat)
    test_scores_knn.append(test_acc)
    with open("Database/TXTS/KNN_Metrics.txt", 'a') as f:
        f.write(str(i) + "---" + "\n")
        f.write(str(metrics.accuracy_score(y_test, y_hat)))
        f.write("\n")
        f.write(str(metrics.f1_score(y_test, y_hat, average='weighted')))
        f.write("\n")
        f.write(str(metrics.precision_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.recall_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.zero_one_loss(y_test, y_hat, normalize=True)))
        f.write("\n")
    
pyplot.plot(knn_k, train_scores_knn, "-o", label="train")
pyplot.plot(knn_k, test_scores_knn, "-o", label="test")
pyplot.legend()
fig = pyplot.gcf()
fig.savefig("Database/PNGS/knn_acc.png")
pyplot.show()
fig.savefig("Database/PNGS/knn_acc.png")


X[numeric_cols] = X.apply(zscore)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42, shuffle=False)


values = [i for i in range(1,25)]
for i in values:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    y_hat_train =  clf.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, y_hat_train)
    train_scores_dec.append(train_acc)

    test_acc = metrics.accuracy_score(y_test, y_hat)
    test_scores_dec.append(test_acc)
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    with open("Database/TXTS/DEC_Metrics.txt", 'a') as f:
        f.write(str(i) + "---" + "\n")
        f.write(str(metrics.accuracy_score(y_test, y_hat)))
        f.write("\n")
        f.write(str(metrics.f1_score(y_test, y_hat, average='weighted')))
        f.write("\n")
        f.write(str(metrics.precision_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.recall_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.zero_one_loss(y_test, y_hat, normalize=True)))
        f.write("\n")
    

with open("Database/TXTS/Importances_Dec_tree.txt", 'w') as f2:
    for name, importance in zip(X_train.columns, clf.feature_importances_):
        f2.write(name + ":" + str(importance) + "\n")

pyplot.plot(values, train_scores_dec, "-o", label="train")
pyplot.plot(values, test_scores_dec, "-o", label="test")
pyplot.legend()
fig = pyplot.gcf()
fig.savefig("Database/PNGS/Dec_acc_tree.png")
pyplot.show()
fig.savefig("Database/PNGS/Dec_acc_tree.png")

    
for i in values:
    clf = RandomForestClassifier(criterion='entropy', max_depth=i)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    y_hat_train =  clf.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, y_hat_train)
    train_scores_forest.append(train_acc)
    test_acc = metrics.accuracy_score(y_test, y_hat)
    test_scores_forest.append(test_acc)
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    
    with open("Database/TXTS/Forest_Metrics.txt", 'a') as f:
        f.write(str(i) + "---" + "\n")
        f.write(str(metrics.accuracy_score(y_test, y_hat)))
        f.write("\n")
        f.write(str(metrics.f1_score(y_test, y_hat, average='weighted')))
        f.write("\n")
        f.write(str(metrics.precision_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.recall_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.zero_one_loss(y_test, y_hat, normalize=True)))
        f.write("\n")
    
    
pyplot.plot(values, train_scores_forest, "-o", label="train")
pyplot.plot(values, test_scores_forest, "-o", label="test")
pyplot.legend()
fig = pyplot.gcf()
fig.savefig("Database/PNGS/Forest_acc.png")
pyplot.show()
fig.savefig("Database/PNGS/Forest_acc.png")
with open("Database/TXTS/Importances_Forest.txt", 'w')as f2:
    for name, importance in zip(X_train.columns, clf.feature_importances_):
        f2.write(name + ":" + str(importance) + "\n")

train_scores2, test_scores2 = list(), list()

sv = svm.SVC(C=0.9, kernel='rbf', gamma='scale', class_weight='balanced')
sv.fit(X_train, y_train)
y_hat = sv.predict(X_test)
y_hat_train =  sv.predict(X_train)
train_acc = metrics.f1_score(y_train, y_hat_train, average='weighted')
train_scores2.append(train_acc)

perm_importance = permutation_importance(sv, X_test, y_test)

feature_names = X_train.columns
features = np.array(feature_names)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
fig = plt.gcf()
fig.savefig("Database/PNGS/SVM_acc.png")
plt.show()
fig.savefig("Database/PNGS/SVM_acc.png",dpi=500)
print(sorted_idx)
for x,y in zip(features, perm_importance.importances_mean.argsort()):
    print(x,":",y)
with open("Database/TXTS/SVM_Metrics.txt", 'w') as f:
        f.write(str(i) + "---" + "\n")
        f.write(str(metrics.accuracy_score(y_test, y_hat)))
        f.write("\n")
        f.write(str(metrics.f1_score(y_test, y_hat, average='weighted')))
        f.write("\n")
        f.write(str(metrics.precision_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.recall_score(y_test, y_hat, average='macro')))
        f.write("\n")
        f.write(str(metrics.zero_one_loss(y_test, y_hat, normalize=True)))
        f.write("\n")


    
