import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.api._v2 import keras


def lr():
    # 2명 탈락, 4명 통과
    x = [[1, 2],
         [2, 1],
         [4, 5],
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.015),
                  metrics='acc')
    model.fit(x, y, epochs=1000, verbose=2)

    bools = model.predict(x, verbose=0) > 0.5
    print(bools)

    print('acc :', np.mean(bools == y))


if __name__ == "__main__":
    lr()

    # -log(x), -log(1-x) 두 그래프를 y값이 1, 0일때 각각 사용
    #
    # if y == 1:
    #   functionA()
    # else:
    #   functionB()
    #
    # y * A() + (1 - y) * B()

