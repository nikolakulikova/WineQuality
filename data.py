import math
from numpy import where
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score


def loadData(filename):
    data = pd.read_csv('datasets/' + filename)
    data.drop('Id', inplace=True, axis=1)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y, data


def find_attr(X, y, data):
    # Information gain: alcohol, sulphates, volatile acidity, citric acid, chlorides, density
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, data.columns[0:len(data.columns) - 1])
    feat_importances.plot(kind='barh', color="blue")
    plt.show()

    # by Lasso: fixed acidity,volatile acidity,free sulfur dioxide,total sulfur dioxide,sulphates,alcohol
    last_score = math.inf
    best_alpha = 1
    a = 1

    #find best alpha for lasso
    while a != 0:
        clf = Lasso(alpha=a)
        score = -np.mean(cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error'))

        if score < last_score:
            last_score = score
            best_alpha = a

        a -= 0.01
        a = round(a, 7)

    clf = Lasso(alpha=best_alpha)
    clf.fit(X, y)
    coef = clf.coef_
    print(coef)
    indexes = [i for i in range(len(coef)) if coef[i] != 0]
    return indexes


def plot_data(X, y, attr1, attr2):
    for quality in range(10):
        row_ix = where(y == quality)
        plt.scatter(X[row_ix, attr1.get("index")], X[row_ix, attr2.get("index")],
                    label=str(quality))

    plt.xlabel(attr1.get("name"))
    plt.ylabel(attr2.get("name"))
    plt.legend()
    plt.show()
