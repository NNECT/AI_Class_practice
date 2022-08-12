import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from tensorflow import keras
from keras.api._v2 import keras


def sr():
    # 2명 탈락, 4명 통과
    x = [[1, 2],    # C
         [2, 1],
         [4, 5],    # B
         [5, 4],
         [8, 9],    # A
         [9, 8]]
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(2, )))
    #                       . class(결과값 y의 차원)의 개수
    # (6, 3) = (6, 2) * (2, 3)
    #                    ^ feature(x의 column)의 개수
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.1),
                  metrics='acc')

    model.fit(x, y, epochs=1000, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x, verbose=0)
    print(pd.DataFrame(p))

    y_arg = np.argmax(y, axis=1)
    p_arg = np.argmax(p, axis=1)
    print(y_arg)
    print(p_arg)

    print('acc :', (p_arg == y_arg).mean())


if __name__ == "__main__":
    sr()
