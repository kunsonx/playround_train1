import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import collections

print(tf.__version__)


# 加载数据
def load_data():
    df_user = pd.read_csv("./datas/ml-1m/users.dat",
                          sep="::", header=None, engine="python",
                          names="UserID::Gender::Age::Occupation::Zip-code".split("::"))

    df_movie = pd.read_csv("./datas/ml-1m/movies.dat",
                           sep="::", header=None, engine="python",
                           names="MovieID::Title::Genres".split("::"))

    df_rating = pd.read_csv("./datas/ml-1m/ratings.dat",
                            sep="::", header=None, engine="python",
                            names="UserID::MovieID::Rating::Timestamp".split("::"))

    return df_user, df_movie, df_rating


# 计算电影中每个题材的次数
def top_labels(df_movie):
    genre_count = collections.defaultdict(int)
    for genres in df_movie["Genres"].str.split("|"):
        for genre in genres:
            genre_count[genre] += 1
    return genre_count


# 只保留最有代表性的题材
def get_highrate_genre(x, genre_count):
    sub_values = {}
    for genre in x.split("|"):
        sub_values[genre] = genre_count[genre]
    return sorted(sub_values.items(), key=lambda x: x[1], reverse=True)[0][0]


# 只保留一个 label 到数据
def set_highrate_genre(df_movie, genre_count):
    df_movie["Genres"] = df_movie["Genres"].map(lambda j: get_highrate_genre(j, genre_count))


# 加载数据并且预处理
def load_that_data():
    users, movies, ratings = load_data()
    genre_count = top_labels(movies)
    set_highrate_genre(movies, genre_count)
    return users, movies, ratings
