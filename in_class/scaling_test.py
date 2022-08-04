from sklearn import preprocessing


if __name__ == '__main__':
    # column == feature라는 것을 잊지 말자
    x = [[1, -1, -3],
         [2, 0, 1],
         [0, 1, 7]]

    scaler = preprocessing.StandardScaler()
    print(scaler.fit_transform(x))
    print(preprocessing.scale(x))
