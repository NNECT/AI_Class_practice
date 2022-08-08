import os

import numpy as np
import pandas as pd
import pickle


def get_movie_lens():
    if os.path.exists('pickle_temp/ml_1m_movie_data.pickle'):
        with open('pickle_temp/ml_1m_movie_data.pickle', 'br') as f:
            return pickle.load(f)

    movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='ISO-8859-1',
                         names='MovieID::Title::Genres'.split('::'))
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='ISO-8859-1',
                         names='UserID::MovieID::Rating::Timestamp'.split('::'))
    users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='ISO-8859-1',
                         names='UserID::Gender::Age::Occupation::Zip-code'.split('::'))

    data = users.merge(ratings).merge(movies)
    with open('pickle_temp/ml_1m_movie_data.pickle', 'bw') as f:
        pickle.dump(data, f)

    return data


if __name__ == "__main__":
    df = get_movie_lens()
    by_title = df.groupby(by='Title').size()
    over500 = by_title[by_title.values >= 500]
    # over500.index.values

    # 영화별 각 성별 레이팅
    ratings = df.pivot_table(values='Rating', index='Title', columns='Gender')
    ratings500 = ratings.loc[over500.index]
    print(ratings500)

    # 여성 레이팅으로 정렬
    female500 = ratings500.sort_values(by='F', ascending=False)
    print(female500.head())

    # 각 성별 레이팅의 차가 작은 순으로 정렬
    diff_ratings = (ratings.M - ratings.F).abs().sort_values()
    print(diff_ratings.head())
