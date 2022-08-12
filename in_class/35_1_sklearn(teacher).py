# 35_1_sklearn.py
import numpy as np
from sklearn import datasets, svm, model_selection, preprocessing, neighbors
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import colors


np.set_printoptions(linewidth=1000)


def sklearn_basic_1():
    iris = datasets.load_iris()
    print(type(iris))
    print(iris.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR',
    # 'feature_names', 'filename', 'data_module'])

    print(iris['DESCR'])
    print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
    print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(iris.frame)

    print(iris.data.shape)      # (150, 4)
    print(iris.target)


def sklearn_basic_2():
    # iris = datasets.load_iris(as_frame=True)
    # print(iris.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

    # print(iris.frame)

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # scatter_matrix(df)
    # scatter_matrix(df, c=iris.target)
    scatter_matrix(df, c=iris.target, cmap='jet')
    plt.show()


def sklearn_basic_3():
    x, y = datasets.load_iris(return_X_y=True)
    print(x.shape, y.shape)         # (150, 4) (150,)

    # 퀴즈
    # 예측 결과에 대해 정확도를 구하세요
    clf = svm.SVC()
    clf.fit(x, y)
    p = clf.predict(x)
    print(p)
    print(y)
    print(sum(p == y) / len(p))
    print(np.mean(p == y))

    print(clf.score(x, y))


# 퀴즈
# digits 데이터에 대해 80%의 데이터로 학습하고 20%의 데이터에 대해 정확도를 구하세요
def sklearn_basic_4():
    x, y = datasets.load_digits(return_X_y=True)
    # print(x.shape, y.shape)       # (1797, 64) (1797,)

    # train_size = int(len(x) * 0.8)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # data = model_selection.train_test_split(x, y)     # 75:25
    # data = model_selection.train_test_split(x, y, train_size=0.8)
    data = model_selection.train_test_split(x, y, train_size=1200)
    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))


# 퀴즈
# leaf_train.csv에 대해 80%의 데이터로 학습하고 20%의 데이터에 대해 정확도를 구하세요
def leaf_model():
    # leaf = pd.read_csv('data/leaf_train.csv', index_col=0)
    # print(leaf)

    leaf = pd.read_csv('data/leaf_train.csv')

    x = leaf.values[:, 2:]
    # y = leaf.values[:, 1]
    y = leaf.species
    print(y)
    print(x.dtype)

    x = preprocessing.scale(x)
    # scaler = preprocessing.StandardScaler()
    # x = scaler.fit_transform(x)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))


# 퀴즈
# leaf_train.csv에 대해 80%의 데이터로 학습하고 20%의 데이터에 대해 정확도를 구하세요
# 하이퍼 파라미터에 해당하는 이웃의 갯수를 2명에서 10명까지로 테스트하고 결과를 막대 그래프로 보여주세요
def leaf_model_knn():
    leaf = pd.read_csv('data/leaf_train.csv', index_col=0)
    # print(leaf)

    x = leaf.values[:, 1:]
    y = leaf.species.values

    # x = preprocessing.scale(x)
    x = preprocessing.minmax_scale(x)

    data = model_selection.train_test_split(x, y, train_size=0.8)
    x_train, x_test, y_train, y_test = data

    scores = []
    for n in range(2, 11):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)

        print(n, score)
        scores.append(score)

    plt.bar(range(2, 11), scores, color=colors.TABLEAU_COLORS)
    plt.title('knn')
    plt.show()


# sklearn_basic_1()
# sklearn_basic_2()
# sklearn_basic_3()
# sklearn_basic_4()

# leaf_model()
leaf_model_knn()




