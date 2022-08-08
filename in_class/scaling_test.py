import numpy as np
from sklearn import preprocessing


def z_score(x: list[list[int | float]] | np.ndarray):  # == preprocessing.scale(x) 표준화
    if type(x) == list:
        x = np.array(x)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def minmax_scale(x: list[list[int | float]] | np.ndarray):
    if type(x) == list:
        x = np.array(x)

    mx = np.max(x, axis=0)
    mn = np.min(x, axis=0)
    return (x - mn) / (mx - mn)


if __name__ == '__main__':
    # column == feature라는 것을 잊지 말자
    x = [[1, -1, -3],
         [2, 0, 1],
         [0, 1, 7]]

    # scaler = preprocessing.StandardScaler()
    # print(scaler.fit_transform(x))
    # print(preprocessing.scale(x))
    # print(z_score(x))

    # scaler = preprocessing.MinMaxScaler()
    # print(scaler.fit_transform(x))
    # print(preprocessing.minmax_scale(x))
    print(minmax_scale(x))

