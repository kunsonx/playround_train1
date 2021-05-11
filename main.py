import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

print(tf.__version__)

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)


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


# 对数据进行下表化处理
def add_index_column(param_df, column_name):
    values = list(param_df[column_name].unique())
    value_index_dict = {value: idx for idx, value in enumerate(values)}
    param_df[f"{column_name}_idx"] = param_df[column_name].map(value_index_dict)


# 加载数据并且预处理
def load_that_data():
    users, movies, ratings = load_data()
    genre_count = top_labels(movies)
    set_highrate_genre(movies, genre_count)
    add_index_column(users, "UserID")
    add_index_column(users, "Gender")
    add_index_column(users, "Age")
    add_index_column(users, "Occupation")
    add_index_column(movies, "MovieID")
    add_index_column(movies, "Genres")
    # 合并成一个df
    df = pd.merge(pd.merge(ratings, users), movies)
    df.drop(columns=["Timestamp", "Zip-code", "Title"], inplace=True)
    return df


# 准备开始训练
def train(df):
    num_users = df["UserID_idx"].max() + 1
    num_movies = df["MovieID_idx"].max() + 1
    num_genders = df["Gender_idx"].max() + 1
    num_ages = df["Age_idx"].max() + 1
    num_occupations = df["Occupation_idx"].max() + 1
    num_genres = df["Genres_idx"].max() + 1

    min_rating = df["Rating"].min()
    max_rating = df["Rating"].max()
    df["Rating"] = df["Rating"].map(lambda x: (x - min_rating) / (max_rating - min_rating))

    print(df.sample(frac=1).head(3))

    x = df[["UserID_idx", "Gender_idx", "Age_idx", "Occupation_idx", "MovieID_idx", "Genres_idx"]]
    y = df.pop("Rating")

    model = get_model(num_users, num_genders, num_ages, num_occupations, num_movies, num_genres)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.RMSprop())

    fit_x_train = [
        x["UserID_idx"],
        x["Gender_idx"],
        x["Age_idx"],
        x["Occupation_idx"],
        x["MovieID_idx"],
        x["Genres_idx"]
    ]

    t = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/logs_" + t)

    history = model.fit(
        x=fit_x_train,
        y=y,
        batch_size=32,
        epochs=5,
        verbose=1,
        callbacks=[tensorboard_callback]
    )

    print(history)

    inputs = df.sample(frac=1.0)[
        ["UserID_idx", "Gender_idx", "Age_idx", "Occupation_idx", "MovieID_idx",
         "Genres_idx"]].head(10)

    # 对于（用户ID，召回的电影ID列表），计算分数
    out = model.predict([
        inputs["UserID_idx"],
        inputs["Gender_idx"],
        inputs["Age_idx"],
        inputs["Occupation_idx"],
        inputs["MovieID_idx"],
        inputs["Genres_idx"]
    ])

    print(out)

    model.save("./datas/ml-latest-small/tensorflow_two_tower.h5")


def get_model(num_users, num_genders, num_ages, num_occupations, num_movies, num_genres):
    """函数式API搭建双塔DNN模型"""

    # 输入
    user_id = keras.layers.Input(shape=(1,), name="user_id")
    gender = keras.layers.Input(shape=(1,), name="gender")
    age = keras.layers.Input(shape=(1,), name="age")
    occupation = keras.layers.Input(shape=(1,), name="occupation")
    movie_id = keras.layers.Input(shape=(1,), name="movie_id")
    genre = keras.layers.Input(shape=(1,), name="genre")

    # user 塔
    user_vector = tf.keras.layers.concatenate([
        layers.Embedding(num_users, 100)(user_id),
        layers.Embedding(num_genders, 2)(gender),
        layers.Embedding(num_ages, 2)(age),
        layers.Embedding(num_occupations, 2)(occupation)
    ])
    user_vector = layers.Dense(32, activation='relu')(user_vector)
    user_vector = layers.Dense(8, activation='relu',
                               name="user_embedding", kernel_regularizer='l2')(user_vector)

    # movie塔
    movie_vector = tf.keras.layers.concatenate([
        layers.Embedding(num_movies, 100)(movie_id),
        layers.Embedding(num_genres, 2)(genre)
    ])
    movie_vector = layers.Dense(32, activation='relu')(movie_vector)
    movie_vector = layers.Dense(8, activation='relu',
                                name="movie_embedding", kernel_regularizer='l2')(movie_vector)

    # 每个用户的embedding和item的embedding作点积
    dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1)
    dot_user_movie = tf.expand_dims(dot_user_movie, 1)

    output = layers.Dense(1, activation='sigmoid')(dot_user_movie)

    return keras.models.Model(inputs=[user_id, gender, age, occupation, movie_id, genre],
                              outputs=[output])
