import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras.api._v2 import keras


if __name__ == "__main__":
    df = pd.read_csv('data/cars.csv', index_col=0)
    # print(df)

    x = df['speed']
    y = df['dist']
    print(x, '\n', y)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.SGD(learning_rate=0.001))

    k = model.fit(x, y, epochs=1000, verbose=0)  # 0-없음, 1-전체, 2-일부
    # print(model.evaluate(x, y, verbose=0))

    a = list(range(4, 51))
    ap = model.predict(a, verbose=0)

    t = [[0],
         [30],
         [50]]
    tp = model.predict(t)
    plt.plot(x, y, 'ro')
    plt.plot(a, ap, 'r')
    plt.plot(t, tp, 'bo')
    plt.plot([0, 50], [0, tp[2]], 'g')
    plt.plot([0, 50], [tp[0], tp[2]], 'b')
    plt.show()
