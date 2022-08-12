import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn import datasets, linear_model, model_selection, utils, preprocessing


def cv_1():
    x, y = datasets.make_blobs(random_state=23, centers=7, n_samples=100, n_features=5)
    # print(x, y)
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.show()

    # x = preprocessing.scale(x)
    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data
    reg = linear_model.LogisticRegression(solver='liblinear')
    reg.fit(x_train, y_train)
    print('acc:', reg.score(x_test, y_test))


def cv_2():
    num = 10
    x, y = datasets.make_blobs(random_state=23, centers=31, n_samples=1000, n_features=5)
    reg = linear_model.LogisticRegression(solver='liblinear')
    result = np.zeros(num, dtype=np.float32)
    for n in range(num):
        data = model_selection.train_test_split(x, y, train_size=0.8)
        x_train, x_test, y_train, y_test = data
        reg.fit(x_train, y_train)
        result[n] = reg.score(x_test, y_test)
    print(result.mean())


def cv_3():
    num = 10
    x, y = datasets.make_blobs(random_state=23, centers=31, n_samples=10000, n_features=5)
    reg = linear_model.LogisticRegression(solver='liblinear')
    acc = np.zeros(num, dtype=np.float32)
    for i in range(num):
        s = i * 100
        e = s + 100
        x_train = np.vstack([x[:s], x[e:]])
        y_train = np.concatenate([y[:s], y[e:]])
        x_test, y_test = x[s:e], y[s:e]

        # data = model_selection.train_test_split(x, y, train_size=0.9)
        # k = x[n::num]
        # print(k, len(k))
        #
        # x_train, x_test, y_train, y_test = data
        # reg.fit(x_train, y_train)
        # acc[n] = reg.score(x_test, y_test)
    # print(acc.mean())


def cv_4():
    num = 10
    result = []

    x, y = datasets.make_blobs(random_state=23, centers=31, n_samples=1000, n_features=5)
    reg = linear_model.LogisticRegression(solver='liblinear')
    sss = model_selection.StratifiedShuffleSplit(num, test_size=0.2, random_state=23)
    for train_idx, test_idx in sss.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        reg.fit(x_train, y_train)
        result.append(reg.score(x_test, y_test))
    print(np.mean(result))


def cv_5():
    x, y = datasets.make_blobs(random_state=23, centers=31, n_samples=10000, n_features=5)
    reg = linear_model.LogisticRegression(solver='liblinear')
    score = model_selection.cross_val_score(reg, x, y, cv=model_selection.StratifiedShuffleSplit(n_splits=10))
    print(score, score.mean())


if __name__ == "__main__":
    # cv_1()
    # cv_2()
    # cv_3()
    # cv_4()
    cv_5()
