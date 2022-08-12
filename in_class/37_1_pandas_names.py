import os
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors


def read_yob(year):
    filename = 'pickle_temp/yob' + str(year) + '.pickle'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    df = pd.read_csv('data/yob' + str(year) + '.txt', header=None, names=['Name', 'Gender', 'Count'])
    df.to_pickle(filename)
    return df


if __name__ == "__main__":
    start_time = time.time()
    names = read_yob(1880)
    print(time.time() - start_time)
    print(names)

    # 남녀 이름 개수
    print('남자:', (names['Gender'] == 'M').sum())
    print('여자:', (names['Gender'] == 'F').sum())
    print(names.groupby(by='Gender').size())
    print(names.Gender.value_counts())

    # 남녀 출생수
    print('여자:', names[names.Gender == 'F'].Count.sum())
    print('남자:', names[names.Gender == 'M'].Count.sum())
    print(names.groupby(by='Gender').sum('Count'))
    print(names.pivot_table(values='Count', index='Gender', aggfunc=np.sum))

    # for by in names.groupby(by='Gender'):
    #     print(by)

    # plt.pie(male5.Count, labels=male5.Name)
    male5 = names[names.Gender == 'M'].sort_values('Count', ascending=False).head()
    male5.index = male5.Name
    del male5['Name']
    male5.plot(y='Count', kind='pie', legend=False, autopct='%1.1f%%', title='Male Top5 Names')
    plt.show()
