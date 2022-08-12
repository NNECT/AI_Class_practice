import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from tensorflow import keras
from keras.api._v2 import keras


def load_leaf_xy_1():
    df = pd.read_csv('data/leaf_train.csv', index_col=0)
    x = df.values[:, 1:]
    y = df.values[:, :1]

    x = preprocessing.scale(x)
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)

    return model_selection.train_test_split(x, y, train_size=0.7)


def sr_leaf_1():
    x_train, x_test, y_train, y_test = load_leaf_xy_1()

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(x_train.shape[1], )))
    model.add(keras.layers.Dense(y_train.shape[1], activation=keras.activations.softmax))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.1),
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))


def load_leaf_xy_2():
    df = pd.read_csv('data/leaf_train.csv', index_col=0)

    x = preprocessing.scale(df.values[:, 1:])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df.values[:, :1])

    return model_selection.train_test_split(x, y, train_size=0.7)


def sr_leaf_2():
    x_train, x_test, y_train, y_test = load_leaf_xy_2()

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(x_train.shape[1], )))
    model.add(keras.layers.Dense(np.max(np.concatenate(y_train, y_test)), activation=keras.activations.softmax))
    model.summary()

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.1),
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_test, y_test))


if __name__ == "__main__":
    sr_leaf_2()
