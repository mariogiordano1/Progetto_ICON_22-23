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
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("merged_files.csv", index_col=False)
df_finale = df.drop(columns=["Country", "City", "CO_AQI_Category", "Ozone_AQI_Category",
                         "NO2_AQI_Category", "PM2_5_AQI_Category", "PM10_Category", "AQI_Value"])
df_finale.rename(columns={"AQI_Value" :"AQI", "CO_AQI_Value": "CO", "NO2_AQI_Value": "NO2", "PM2_5_AQI_Value" : "PM25", "Ozone_AQI_Value" : "Ozone"}, inplace=True)

df_finale.dropna(inplace=True)
print(df_finale)
X = df_finale.drop(columns="AQI_Category")
y = df_finale["AQI_Category"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
numeric_cols = df_finale.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X.apply(zscore)
kfoldcv = KFold()

def KNN_Est(X, y):
    train_scores_knn, test_scores_knn = list(), list()
    best_f1 = 0
    best_acc = 0
    best_recall = 0
    best_prec = 0
    knn_k = [i for i in range(1,35, 2)]
    for i in knn_k:
        fold = 1
        for train, test in kfoldcv.split(X, y):
            knn = KNeighborsClassifier(n_neighbors=i)

            X_train_KNN = X.iloc[train].drop(columns=["CO", "Ozone",
                                    "NO2", "PM25", "PM10"])
            X_test_KNN = X.iloc[test].drop(columns=["CO", "Ozone",
                                    "NO2", "PM25", "PM10"])
            knn.fit(X_train_KNN, y[train])
            y_hat = knn.predict(X_test_KNN)
            y_hat_train =  knn.predict(X_train_KNN)
            train_acc = metrics.accuracy_score(y[train], y_hat_train)
            train_scores_knn.append(train_acc)

            test_acc = metrics.accuracy_score(y[test], y_hat)
            test_scores_knn.append(test_acc)
            
            if best_f1 <= metrics.f1_score(y[test], y_hat, average='weighted'):
                best_f1 = metrics.f1_score(y[test], y_hat, average='weighted')
                best_f1_k = str(i)
                best_fold_f1 = fold
            if best_acc <= metrics.accuracy_score(y[test], y_hat):
                best_acc = metrics.accuracy_score(y[test], y_hat)
                best_acc_k = str(i)
                best_fold_acc = fold
            if best_prec <= metrics.precision_score(y[test], y_hat, average='macro'):
                best_prec =metrics.precision_score(y[test], y_hat, average='macro')
                best_prec_k = str(i)
                best_fold_prec = fold
            if best_recall <= metrics.recall_score(y[test], y_hat, average='macro'):
                best_recall =metrics.recall_score(y[test], y_hat, average='macro')
                best_recall_k = str(i)
                best_fold_rec = fold
        
            with open("KNN_Metrics.txt", 'a') as f:
                f.write(str(i) + "---" + "\n")
                f.write(str(metrics.accuracy_score(y[test], y_hat)))
                f.write("\n")
                f.write(str(metrics.f1_score(y[test], y_hat, average='weighted')))
                f.write("\n")
                f.write(str(metrics.precision_score(y[test], y_hat, average='macro')))
                f.write("\n")
                f.write(str(metrics.recall_score(y[test], y_hat, average='macro')))
                f.write("\n")
                f.write(str(metrics.zero_one_loss(y[test], y_hat, normalize=True)))
                f.write("\n")
            fold = fold +1
            
    with open("best_score.txt", 'a') as best:
        best.write(best_f1_k + "--- " + str(best_fold_f1) + " f1" +"\n")
        best.write(str(best_f1))
        best.write("\n")
        best.write(best_acc_k + "--- " + str(best_fold_acc) + " acc" +"\n")
        best.write(str(best_acc))
        best.write("\n")
        best.write(best_recall_k + "--- " + str(best_fold_rec)+ " recall"  + "\n")
        best.write(str(best_recall))
        best.write("\n")
        best.write(best_prec_k + "--- " + str(best_fold_prec) + " prec" "\n")
        best.write(str(best_prec))
        best.write("\n")

def Dec_tree_PM25(X, y):
    train_scores_dec, test_scores_dec = list(), list()
    values = [i for i in range(1,25)]
    best_f1 = 0
    best_acc = 0
    best_recall = 0
    best_prec = 0
    for i in values:
        fold = 1
        for train, test in kfoldcv.split(X, y):
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
            clf.fit(X.iloc[train], y[train])
            y_hat = clf.predict(X.iloc[test])
            y_hat_train =  clf.predict(X.iloc[train])
            train_acc = metrics.accuracy_score(y[train], y_hat_train)
            train_scores_dec.append(train_acc)

            test_acc = metrics.accuracy_score(y[test], y_hat)
            test_scores_dec.append(test_acc)
            print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
            if best_f1 <= metrics.f1_score(y[test], y_hat, average='weighted'):
                    best_f1 = metrics.f1_score(y[test], y_hat, average='weighted')
                    best_f1_k = str(i)
                    best_fold_f1 = fold
            if best_acc <= metrics.accuracy_score(y[test], y_hat):
                    best_acc = metrics.accuracy_score(y[test], y_hat)
                    best_acc_k = str(i)
                    best_fold_acc = fold
            if best_prec <= metrics.precision_score(y[test], y_hat, average='macro'):
                    best_prec =metrics.precision_score(y[test], y_hat, average='macro')
                    best_prec_k = str(i)
                    best_fold_prec = fold
            if best_recall <= metrics.recall_score(y[test], y_hat, average='macro'):
                    best_recall =metrics.recall_score(y[test], y_hat, average='macro')
                    best_recall_k = str(i)
                    best_fold_rec = fold
            with open("DEC_Metrics.txt", 'a') as f:
                f.write(str(i) + "---" + "\n")
                f.write(str(metrics.accuracy_score(y[test], y_hat)))
                f.write("\n")
                f.write(str(metrics.f1_score(y[test], y_hat, average='weighted')))
                f.write("\n")
                f.write(str(metrics.precision_score(y[test], y_hat, average='macro')))
                f.write("\n")
                f.write(str(metrics.recall_score(y[test], y_hat, average='macro')))
                f.write("\n")
                f.write(str(metrics.zero_one_loss(y[test], y_hat, normalize=True)))
                f.write("\n")
            fold = fold +1
    with open("best_score_dec.txt", 'a') as best:
            best.write(best_f1_k + "--- " + str(best_fold_f1) + " f1" +"\n")
            best.write(str(best_f1))
            best.write("\n")
            best.write(best_acc_k + "--- " + str(best_fold_acc) + " acc" +"\n")
            best.write(str(best_acc))
            best.write("\n")
            best.write(best_recall_k + "--- " + str(best_fold_rec)+ " recall"  + "\n")
            best.write(str(best_recall))
            best.write("\n")
            best.write(best_prec_k + "--- " + str(best_fold_prec) + " prec" "\n")
            best.write(str(best_prec))
            best.write("\n")
def Dec_tree_noPM(X, y):
    X_noPM = X.drop(columns=["PM25"])
    train_scores_dec, test_scores_dec = list(), list()
    values = [i for i in range(1,25)]
    best_f1 = 0
    best_acc = 0
    best_recall = 0
    best_prec = 0
    for i in values:
        fold = 1
        for train, test in kfoldcv.split(X_noPM, y):
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
            clf.fit(X_noPM.iloc[train], y[train])
            y_hat = clf.predict(X_noPM.iloc[test])
            y_hat_train =  clf.predict(X_noPM.iloc[train])
            train_acc = metrics.accuracy_score(y[train], y_hat_train)
            train_scores_dec.append(train_acc)

            test_acc = metrics.accuracy_score(y[test], y_hat)
            test_scores_dec.append(test_acc)
            print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
            if best_f1 <= metrics.f1_score(y[test], y_hat, average='weighted'):
                    best_f1 = metrics.f1_score(y[test], y_hat, average='weighted')
                    best_f1_k = str(i)
                    best_fold_f1 = fold
            if best_acc <= metrics.accuracy_score(y[test], y_hat):
                    best_acc = metrics.accuracy_score(y[test], y_hat)
                    best_acc_k = str(i)
                    best_fold_acc = fold
            if best_prec <= metrics.precision_score(y[test], y_hat, average='macro'):
                    best_prec =metrics.precision_score(y[test], y_hat, average='macro')
                    best_prec_k = str(i)
                    best_fold_prec = fold
            if best_recall <= metrics.recall_score(y[test], y_hat, average='macro'):
                    best_recall =metrics.recall_score(y[test], y_hat, average='macro')
                    best_recall_k = str(i)
                    best_fold_rec = fold
            with open("DEC_Metrics_noPM.txt", 'a') as f:
                f.write(str(i) + "---" + "\n")
                f.write(str(metrics.accuracy_score(y[test], y_hat)))
                f.write("\n")
                f.write(str(metrics.f1_score(y[test], y_hat, average='weighted')))
                f.write("\n")
                f.write(str(metrics.precision_score(y[test], y_hat, average='macro')))
                f.write("\n")
                f.write(str(metrics.recall_score(y[test], y_hat, average='macro')))
                f.write("\n")
                f.write(str(metrics.zero_one_loss(y[test], y_hat, normalize=True)))
                f.write("\n")
            fold = fold +1
    with open("best_score_dec_noPM.txt", 'a') as best:
            best.write(best_f1_k + "--- " + str(best_fold_f1) + " f1" +"\n")
            best.write(str(best_f1))
            best.write("\n")
            best.write(best_acc_k + "--- " + str(best_fold_acc) + " acc" +"\n")
            best.write(str(best_acc))
            best.write("\n")
            best.write(best_recall_k + "--- " + str(best_fold_rec)+ " recall"  + "\n")
            best.write(str(best_recall))
            best.write("\n")
            best.write(best_prec_k + "--- " + str(best_fold_prec) + " prec" "\n")
            best.write(str(best_prec))
            best.write("\n")

values = [i for i in range(1,25)]
train_scores_forest, test_scores_forest = list(), list()
train_scores_dec, test_scores_dec = list(), list()
best_f1 = 0
best_acc = 0
best_recall = 0
best_prec = 0
for i in values:
    fold = 1
    for train, test in kfoldcv.split(X, y):
        clf = RandomForestClassifier(criterion='entropy', max_depth=i)
        clf.fit(X.iloc[train], y[train])
        y_hat = clf.predict(X.iloc[test])
        y_hat_train =  clf.predict(X.iloc[train])
        train_acc = metrics.accuracy_score(y[train], y_hat_train)
        train_scores_forest.append(train_acc)
        test_acc = metrics.accuracy_score(y[test], y_hat)
        test_scores_forest.append(test_acc)
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
        if best_f1 <= metrics.f1_score(y[test], y_hat, average='weighted'):
                    best_f1 = metrics.f1_score(y[test], y_hat, average='weighted')
                    best_f1_k = str(i)
                    best_fold_f1 = fold
        if best_acc <= metrics.accuracy_score(y[test], y_hat):
                best_acc = metrics.accuracy_score(y[test], y_hat)
                best_acc_k = str(i)
                best_fold_acc = fold
        if best_prec <= metrics.precision_score(y[test], y_hat, average='macro'):
                best_prec =metrics.precision_score(y[test], y_hat, average='macro')
                best_prec_k = str(i)
                best_fold_prec = fold
        if best_recall <= metrics.recall_score(y[test], y_hat, average='macro'):
                best_recall =metrics.recall_score(y[test], y_hat, average='macro')
                best_recall_k = str(i)
                best_fold_rec = fold
        with open("Forest_Metrics.txt", 'a') as f:
            f.write(str(i) + "---" + "\n")
            f.write(str(metrics.accuracy_score(y[test], y_hat)))
            f.write("\n")
            f.write(str(metrics.f1_score(y[test], y_hat, average='weighted')))
            f.write("\n")
            f.write(str(metrics.precision_score(y[test], y_hat, average='macro')))
            f.write("\n")
            f.write(str(metrics.recall_score(y[test], y_hat, average='macro')))
            f.write("\n")
            f.write(str(metrics.zero_one_loss(y[test], y_hat, normalize=True)))
            f.write("\n")
        fold = fold +1
    with open("best_score_forest.txt", 'a') as best:
            best.write(best_f1_k + "--- " + str(best_fold_f1) + " f1" +"\n")
            best.write(str(best_f1))
            best.write("\n")
            best.write(best_acc_k + "--- " + str(best_fold_acc) + " acc" +"\n")
            best.write(str(best_acc))
            best.write("\n")
            best.write(best_recall_k + "--- " + str(best_fold_rec)+ " recall"  + "\n")
            best.write(str(best_recall))
            best.write("\n")
            best.write(best_prec_k + "--- " + str(best_fold_prec) + " prec" "\n")
            best.write(str(best_prec))
            best.write("\n")


values = [i for i in range(1,25)]
train_scores_forest, test_scores_forest = list(), list()
train_scores_dec, test_scores_dec = list(), list()
best_f1 = 0
best_acc = 0
best_recall = 0
best_prec = 0
X_noPM = X.drop(columns=["PM25"])
for i in values:
    fold = 1
    for train, test in kfoldcv.split(X, y):
        clf = RandomForestClassifier(criterion='entropy', max_depth=i)
        clf.fit(X_noPM.iloc[train], y[train])
        y_hat = clf.predict(X_noPM.iloc[test])
        y_hat_train =  clf.predict(X_noPM.iloc[train])
        train_acc = metrics.accuracy_score(y[train], y_hat_train)
        train_scores_forest.append(train_acc)
        test_acc = metrics.accuracy_score(y[test], y_hat)
        test_scores_forest.append(test_acc)
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
        if best_f1 <= metrics.f1_score(y[test], y_hat, average='weighted'):
                    best_f1 = metrics.f1_score(y[test], y_hat, average='weighted')
                    best_f1_k = str(i)
                    best_fold_f1 = fold
        if best_acc <= metrics.accuracy_score(y[test], y_hat):
                best_acc = metrics.accuracy_score(y[test], y_hat)
                best_acc_k = str(i)
                best_fold_acc = fold
        if best_prec <= metrics.precision_score(y[test], y_hat, average='macro'):
                best_prec =metrics.precision_score(y[test], y_hat, average='macro')
                best_prec_k = str(i)
                best_fold_prec = fold
        if best_recall <= metrics.recall_score(y[test], y_hat, average='macro'):
                best_recall =metrics.recall_score(y[test], y_hat, average='macro')
                best_recall_k = str(i)
                best_fold_rec = fold
        with open("Forest_Metrics_noPM.txt", 'a') as f:
            f.write(str(i) + "---" + "\n")
            f.write(str(metrics.accuracy_score(y[test], y_hat)))
            f.write("\n")
            f.write(str(metrics.f1_score(y[test], y_hat, average='weighted')))
            f.write("\n")
            f.write(str(metrics.precision_score(y[test], y_hat, average='macro')))
            f.write("\n")
            f.write(str(metrics.recall_score(y[test], y_hat, average='macro')))
            f.write("\n")
            f.write(str(metrics.zero_one_loss(y[test], y_hat, normalize=True)))
            f.write("\n")
        fold = fold +1
    with open("best_score_forest_noPM.txt", 'a') as best:
            best.write(best_f1_k + "--- " + str(best_fold_f1) + " f1" +"\n")
            best.write(str(best_f1))
            best.write("\n")
            best.write(best_acc_k + "--- " + str(best_fold_acc) + " acc" +"\n")
            best.write(str(best_acc))
            best.write("\n")
            best.write(best_recall_k + "--- " + str(best_fold_rec)+ " recall"  + "\n")
            best.write(str(best_recall))
            best.write("\n")
            best.write(best_prec_k + "--- " + str(best_fold_prec) + " prec" "\n")
            best.write(str(best_prec))
            best.write("\n")