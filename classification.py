import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def predictRandomForest(X_train, y_train, X_val, y_val, X_test, y_test):
    rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0)
    rfc.fit(X_train, y_train)
    score = cross_val_score(rfc, X_val, y_val)
    print("Validation score for RFC: {}".format(np.mean(score)))
    print("Traning accuracy for RFC: {}".format(rfc.score(X_train, y_train)))
    print("Test accuracy for RFC: {}".format(rfc.score(X_test, y_test)))


def predictSVM(X_train, y_train, X_val, y_val, X_test, y_test):
    svc = svm.SVC(kernel='rbf', random_state = 0, C=2,gamma=2)
    svc.fit(X_train, y_train)
    score = cross_val_score(svc, X_val, y_val, cv=5)
    print("Validation score form SVM: {}".format(np.mean(score)))
    print("Traning accuracy for SVM: {}".format(svc.score(X_train, y_train)))
    print("Test accuracy for SVM: {}".format(svc.score(X_test, y_test)))


def predictKNeighbors(X_train, y_train, X_val, y_val, X_test, y_test):
    svc = KNeighborsClassifier(n_neighbors=4)
    svc.fit(X_train, y_train)
    score = cross_val_score(svc, X_val, y_val)
    print("Validation score for KNeighbors: {}".format(np.mean(score)))
    print("Traning accuracy for KNeighbors: {}".format(svc.score(X_train, y_train)))
    print("Test accuracy for KNeighbors: {}".format(svc.score(X_test, y_test)))

