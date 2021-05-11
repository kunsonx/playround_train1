import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import collections

print(tf.__version__)


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
