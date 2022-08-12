import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.api._v2 import keras


if __name__ == "__main__":
    df = pd.read_csv('data/trees.csv', index_col=0)
    x = df.values[:, :-1]
    y = df.values[:, -1:]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.SGD(learning_rate=0.0001))
    model.fit(x, y, epochs=3000, verbose=2)

    t = [[10, 70],
         [15, 80]]
    tp = model.predict(t, verbose=0)
    print(tp)
