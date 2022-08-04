# 33_1_preprocessing.py
import numpy as np
from sklearn import preprocessing
from sklearn import impute


def add_dummy_feature():
    x = [[0, 1],
         [2, 3]]

    print(preprocessing.add_dummy_feature(x))
    print(type(preprocessing.add_dummy_feature(x)))
    print(preprocessing.add_dummy_feature(x).transpose())
    print(preprocessing.add_dummy_feature(x).T)

    # 퀴즈
    # 첫 번째 행에 전체가 1로 채워진 행을 넣으세요
    print(preprocessing.add_dummy_feature(np.transpose(x)).T)


def binarizer():
    x = [[1, -1, 0.2],
         [2, 0, 0],
         [0, 1, -1]]

    # bin = preprocessing.Binarizer(threshold=0.5)
    bin = preprocessing.Binarizer()
    bin.fit(x)
    print(bin.transform(x))

    bin1 = preprocessing.Binarizer()
    bin2 = bin1.fit(x)
    print(bin1)
    print(bin2)

    bin = preprocessing.Binarizer().fit(x)
    print(bin.transform(x))


def imputer():
    # 3 : (1 + 5) / 2
    # 5 : (2 + 4 + 9) / 3
    x = [[1, 2],
         [np.nan, 4],
         [5, 9]]

    # allowed_strategies = ["mean", "median", "most_frequent", "constant"]
    imp = impute.SimpleImputer()
    print(imp.fit_transform(x))

    z = [[np.nan, np.nan]]
    print(imp.transform(z))

    print(imp.missing_values)
    print(imp.statistics_)
    print(imp.strategy)


def label_encoder():
    x = [4, 7, 1, 2, 4]

    enc = preprocessing.LabelEncoder()
    print(enc.fit_transform(x))

    x = ['yes', 'no', 'cancel', 'no']

    enc = preprocessing.LabelEncoder()
    z = enc.fit_transform(x)
    print(z)

    print(enc.classes_)

    # 퀴즈
    # x를 변환한 결과를 원래대로 복구하세요
    print(enc.inverse_transform(z))
    print(enc.classes_[z])

    lb = preprocessing.LabelBinarizer()
    z2 = lb.fit_transform(z)
    print(z2)

    # classes = np.sort(np.unique(z))
    size = enc.classes_.size
    # size = classes.shape[0]
    # binaries = np.zeros((size, size), dtype=np.int32)
    # np.fill_diagonal(binaries, 1)
    # binaries = np.eye(size, dtype=np.int32)
    binaries = np.identity(size, dtype=np.int32)
    # print(z)
    print(binaries[z])



def label_binarizer():
    x = [4, 7, 1, 2, 4]

    lb = preprocessing.LabelBinarizer()
    print(lb.fit_transform(x))

    x = ['yes', 'no', 'cancel', 'no']

    lb = preprocessing.LabelBinarizer()
    z = lb.fit_transform(x)
    print(z)

    print(lb.classes_)

    # 퀴즈
    # x를 변환한 결과를 원래대로 복구하세요
    print(lb.inverse_transform(z))

    p = np.argmax(z, axis=1)
    print(p)
    print(lb.classes_[p])


# add_dummy_feature()
# binarizer()
# imputer()

label_encoder()
# label_binarizer()

#              black, white, yellow, zion.t
# 단순 인코딩 :    0      1      2       3
# 이진수:         00     01     10      11
# 원핫 인코딩 :  1 0 0  0 1 0  0 0 1
#             1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1






