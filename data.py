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
    # alcohol, sulphates, volatile acidity, citric acid, chlorides, density
    """Information Gain

        Information gain calculates the reduction in entropy from the transformation of a dataset.
        It can be used for feature selection by evaluating the Information gain of each variable in the context of the
        target variable.
        https://machinelearningmastery.com/information-gain-and-mutual-information/
        """
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, data.columns[0:len(data.columns) - 1])
    feat_importances.plot(kind='barh', color="blue")
    plt.show()

    # by Lasso: fixed acidity,volatile acidity,free sulfur dioxide,total sulfur dioxide,sulphates,alcohol
    last_score = math.inf
    best_alpha = 1
    a = 1

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


def plot_data(X, y):
    for quality in range(10):
        row_ix = where(y == quality)
        plt.scatter(X[row_ix, 10], X[row_ix, 1],
                    label=str(quality))

    plt.xlabel("alcohol")
    plt.ylabel("volatile acidity")
    plt.legend()
    plt.show()

    # for quality in range(10):
    #     row_ix = where(y == quality)
    #     plt.scatter(X[row_ix, 10], X[row_ix, 8],
    #                 label=str(quality))
    #
    #
    # plt.xlabel("pH")
    # plt.ylabel("quality")
    # plt.legend()
    # plt.show()