import numpy as np
from classification import predictRandomForest, predictSVM, predictKNeighbors
from data import loadData, find_attr, plot_data


def main():
    # load data
    X, y, data = loadData('WineQT.csv')
    # make train, validation, test data
    X_train = X[:int(len(X) * 0.6), :]
    y_train = y[:int(len(X) * 0.6)]
    X_val = X[int(len(X) * 0.6):int(len(X) * 0.8), :]
    y_val = y[int(len(X) * 0.6):int(len(X) * 0.8)]
    X_test = X[int(len(X) * 0.8):, :]
    y_test = y[int(len(X) * 0.8):]

    # normalize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    predictSVM(X_train, y_train, X_val, y_val, X_test, y_test)
    predictRandomForest(X_train, y_train, X_val, y_val, X_test, y_test)
    predictKNeighbors(X_train, y_train, X_val, y_val, X_test, y_test)

    find_attr(X, y, data)
    plot_data(X, y)

if __name__ == '__main__':
    main()
