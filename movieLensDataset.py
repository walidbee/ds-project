import pandas as pd
import numpy as np

def ratingsMatrix():
    df = pd.read_csv("./data/ratings.csv")
    df = df.drop(columns=['timestamp'])
    df = df.pivot(index='userId', columns='movieId', values='rating')
    df = df.fillna(0.0)
    rec_matrix = df.to_numpy()

    return rec_matrix

def getUtilityMatrix():
    ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

    movies_cols = ['movieId', 'title','release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('./data/ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')
    nb_users = ratings.userId.unique().shape[0]
    nb_movies = ratings.movieId.unique().shape[0]
    utility_matrix = np.zeros((nb_users, nb_movies))
    for row in ratings.itertuples():
        utility_matrix[row[1]-1, row[2]-1] = row[3]

    return utility_matrix