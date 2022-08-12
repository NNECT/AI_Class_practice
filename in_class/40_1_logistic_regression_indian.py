import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from tensorflow import keras
from keras.api._v2 import keras


def pid():
    df = pd.read_csv('data/pima-indians-diabetes.csv', skiprows=9, header=None)
    x = df.values[:, :-1]
    y = df.values[:, -1:]

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    x_train = preprocessing.scale(x_train)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.001),
                  metrics='acc')
    model.fit(x_train, y_train, epochs=1000, verbose=2, validation_data=(x_test, y_test))

    bools = model.predict(x_test, verbose=0) > 0.5
    print((bools == y_test).mean())


if __name__ == "__main__":
    pid()
