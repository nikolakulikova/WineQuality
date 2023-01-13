from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


def final_predict(X_train, y_train, X_test, y_test):
    models = [{"model": RandomForestClassifier(n_estimators=64, max_depth=7), "name": "RFC"},
              {"model": svm.SVC(kernel='rbf', random_state=0, C=0.2, gamma=0.16), "name": "SVC"},
              {"model": KNeighborsClassifier(n_neighbors=9), "name": "KNC"}]

    for model in models:
        model.get("model").fit(X_train, y_train)
        print(model.get("name"))
        print("Traning accuracy: {}".format(model.get("model").score(X_train, y_train)))
        print("Test accuracy: {}".format(model.get("model").score(X_test, y_test)))

        common_params = {
            "X": X_train,
            "y": y_train,
            "score_type": "both",
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
            "score_name": "Accuracy",
        }
        LearningCurveDisplay.from_estimator(model.get("model"), **common_params)
        plt.show()

def mlp(X, y):
    mlp = Sequential()
    mlp.add(Dense(16, activation='sigmoid', input_dim=X.shape[1]))
    mlp.add(Dense(64, activation='sigmoid'))
    mlp.add(Dense(9, activation='softmax'))

    mlp.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])

    history = mlp.fit(X, keras.utils.to_categorical(y), epochs=100, validation_split=0.1, verbose=False)

    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.legend(loc='best')
    plt.show()
    return mlp


def get_hyper_param_RFT(X_train, y_train, X_test, y_test):
    best_e, best_d, best_test = 0, 0, 0
    for e in range(1, 100):
        for d in range(1, 100):
            model = RandomForestClassifier(n_estimators=e, max_depth=d)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_test:
                best_d = d
                best_e = e
                best_test = score
    return best_e, best_d


def get_hyper_param_SVC(X_train, y_train, X_test, y_test):
    best_c, best_g, best_test = 0, 0, 0
    for c in range(1, 100):
        for g in range(1, 100):
            c = c / 10
            g = g / 10
            model = svm.SVC(kernel='rbf', random_state=0, C=c, gamma=g)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_test:
                best_c = c
                best_g = g
                best_test = score
    return best_g, best_c


def get_hyper_param_KNC(X_train, y_train, X_test, y_test):
    best_n, best_test = 0, 0
    for n in range(1, 11):
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_test:
            best_n = n
            best_test = score
    return best_n
