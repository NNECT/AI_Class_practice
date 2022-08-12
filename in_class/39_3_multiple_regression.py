import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.api._v2 import keras


if __name__ == "__main__":
    x = np.array([[1, 2],
                  [2, 1],
                  [4, 5],
                  [5, 4],
                  [8, 9],
                  [9, 8]])
    y = np.array([[3],
                  [3],
                  [9],
                  [9],
                  [17],
                  [17]])

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape))
    model.add(keras.layers.Dense(1))
    model.summary()
    exit()

    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01))
    model.fit(x, y, epochs=100, verbose=2)
    t = np.array([[3, 8],
                  [6, 1]])
    tp = model.predict(t, verbose=0)
    print(tp)

    plt.scatter(x[:, 0], x[:, 1], s=(y*2)**2, c='r', alpha=0.5)
    plt.scatter(t[:, 0], t[:, 1], s=(tp*2)**2, c='b', alpha=0.5)
    plt.show()
