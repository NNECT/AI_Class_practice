import numpy as np
from sklearn import preprocessing
from sklearn import impute
import pandas as pd


def make_xy_from_iris():
    data = pd.read_csv('iris.csv')

    x = data.drop(['species'], axis=1)

    # lb = preprocessing.LabelBinarizer()
    lb = preprocessing.LabelEncoder()
    y = lb.fit_transform(data.loc[:, 'species'])

    return x.values, y


def make_xy_from_iris_2():
    data = pd.read_csv('iris.csv')

    x = data.values[:, :-1]
    x = np.float32(x)

    # lb = preprocessing.LabelBinarizer()
    lb = preprocessing.LabelEncoder()
    y = lb.fit_transform(data.values[:, -1])

    return x, y


if __name__ == '__main__':
    # make_xy_from_iris()
    x, y = make_xy_from_iris()
    print(x)
    print(y)
    x, y = make_xy_from_iris_2()
    print(x)
    print(x.dtype)
    print(y)
