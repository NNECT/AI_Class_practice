import numpy as np
import pandas as pd
from sklearn import datasets, svm, model_selection, preprocessing
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def sklearn_basic_1():
    iris = datasets.load_iris()
    # print(iris)
    # print(iris.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

    # print(iris['DESCR'])

    print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
    print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(iris.frame)

    print(iris.data.shape)  # (150, 4)
    print(iris.target)


def sklearn_basic_2():
    # iris = datasets.load_iris(as_frame=True)
    # print(iris.keys())
    # # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    #
    # print(iris.frame)

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(df)

    # scatter_matrix(df)
    scatter_matrix(df, c=iris.target, cmap='jet')
    plt.show()


def sklearn_basic_3():
    x, y = datasets.load_iris(return_X_y=True)
    # print(x.shape, y.shape)  # (150, 4) (150,)

    clf1 = svm.SVC()
    clf1.fit(x, y)
    p = clf1.predict(x)

    # np.set_printoptions(linewidth=500)
    # print(p)
    # print(y)
    print(np.sum(p == y) / p.size)
    print(np.mean(p == y))
    print(clf1.score(x, y))


def sklearn_basic_4():
    x, y = datasets.load_digits(return_X_y=True)
    # sep = int(len(x) * 0.8)
    # clf2 = svm.SVC()
    # clf2.fit(x[:sep], y[:sep])
    # p = clf2.predict(x[sep:])
    # print(np.mean(p == y[sep:]))

    # data = model_selection.train_test_split(x, y)
    # data = model_selection.train_test_split(x, y, train_size=0.8)
    data = model_selection.train_test_split(x, y, train_size=1200)
    x_train, x_test, y_train, y_test = data
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    p = clf.predict(x_test)
    print(np.mean(p == y_test))


def leaf_model():
    base = pd.read_csv('data/leaf_train.csv', index_col=0)
    x = base[base.columns[1:]]
    y = base['species']

    x = preprocessing.scale(x)
    
    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    p = clf.predict(x_test)
    print(np.mean(p == y_test))


if __name__ == "__main__":
    # sklearn_basic_1()
    # sklearn_basic_2()
    # sklearn_basic_3()
    # sklearn_basic_4()
    leaf_model()
