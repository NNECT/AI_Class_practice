import re
import pandas as pd
import numpy as np


if __name__ == "__main__":
    s = pd.Series([1, 3, 7, 4])
    print(s)
    print()

    print(s.values)
    print(type(s.values))
    print()

    print(s.index)
    # print(type(s.index))
    print(s.index.values)
    print(type(s.index.values))
    print()

    s.index = list('abcd')
    print(s)
    print()

    print(s[1:3])
    print(s['b':'c'])
    print()

    filename = "2016_GDP.txt"
    f = open(filename, encoding='utf-8')
    data = np.array([re.sub(',', '', line.strip()).split(':') for line in f.readlines()])
    f.close()

    df = pd.DataFrame(data[1:, 1:])
    df.columns = data[0, 1:]
    # print(df)

    print(df.head())
    print()

    print(df.head(3))
    print()

    print(df.tail(3))
    print()

    print(df['국가'])
    print(df.국가)
    print()

    print(df.iloc[0])
    print(df.loc[0])    # index명

    print(df.iloc[-3:])
    # print(df.loc[list('def')])
    print()

    print(df.iloc[4, 0])
    print(df.loc[4, '국가'])
    print(df.국가[4])
    print(df.values[4, 0])
    print()

