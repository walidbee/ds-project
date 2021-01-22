import pandas as pd

def ratingsMatrix():
    df = pd.read_csv("./data/ratings.csv")
    df = df.drop(columns=['timestamp'])
    df = df.pivot(index='userId', columns='movieId', values='rating')
    df = df.fillna(0.0)
    rec_matrix = df.to_numpy()

    return rec_matrix