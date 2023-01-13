import numpy as np
from classification import final_predict, mlp, get_hyper_param_KNC, get_hyper_param_SVC, get_hyper_param_RFT
from data import loadData, plot_data, find_attr


def main():
    # load data
    X, y, data = loadData('WineQT.csv')
    # split data to train, test
    X_train, y_train, X_test, y_test = split_data(X, y, 0.8)

    # normalize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    '''
    RFC hyper (64, 7)
    SVC hyper (0.2, 0.16)
    KNC hyper 67 
    '''

    # print("RFC hyper " + str(get_hyper_param_RFT(X_train, y_train, X_test, y_test)))
    # print("SVC hyper " + str(get_hyper_param_SVC(X_train, y_train, X_test, y_test)))
    # print("KNC hyper " + str(get_hyper_param_KNC(X_train, y_train, X_test, y_test)))
    # find_attr(X, y, data)
    final_predict(X_train, y_train, X_test, y_test)
    # mlp(X, y)

    # find_attr(X, y, data)
    # plot_data(X, y, {"index": 10, "name": "alcohol"}, {"index": 1, "name": "volatile acidity"})
    # plot_data(X, y, {"index": 9, "name": "sulphates"}, {"index": 1, "name": "volatile acidity"})
    # plot_data(X, y, {"index": 9, "name": "sulphates"}, {"index": 10, "name": "alcohol"})


def split_data(X, y, train_size):
    X_train = X[:int(len(X) * train_size), :]
    y_train = y[:int(len(X) * train_size)]
    X_test = X[int(len(X) * train_size):, :]
    y_test = y[int(len(X) * train_size):]
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    main()
