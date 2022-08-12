import numpy as np
from tensorflow import keras
from keras.api._v2 import keras

x = [[1], [2], [3]]
y = [[1], [2], [3]]

model = keras.Sequential()
# model.add(keras.layers.InputLayer(input_shape=(1, )))
model.add(keras.layers.Dense(1))
# model.summary()
# exit()

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.SGD(learning_rate=0.1))

model.fit(x, y, epochs=10, verbose=2)   # 0-없음, 1-전체, 2-일부
print(model.evaluate(x, y, verbose=0))

p = model.predict(x)
print(p)

print(p - y)
print((p - y) ** 2)
print(np.mean((p - y) ** 2))

print(model.predict([[5],
                     [7]]))
