from movieLensDataset import *
import numpy as np
from numpy.linalg import lstsq

class NMF():

    def __init__(self, M, K, steps, beta):
        """
        Non-negative matrix factorization using ALS 

        Arguments
        - M (matrix)   : rating matrix
        - K (int)       : number of latent features
        - steps (int) : number of times we run the algorithm
        - beta (float) : added number so we don't divide by 0 (used only for Multiplication Update algorithm)
        """

        self.M = M
        self.X, self.Y = M.shape
        self.K = K
        self.steps = steps
        self.beta = beta

    def computeALS(self):
        # Alternating Least Squares
        # M = PQ
        #initialize P and Q matrices
        P = np.random.rand(self.X, self.K)
        Q = np.random.rand(self.K, self.Y)
    
        for _ in range(self.steps):
            #resolve P @ Q = M
            Q = lstsq(P, self.M, rcond = -1)[0]
            #negative terms = 0
            Q[Q < 0] = 0

            #resolve Q.T @ P = M.T
            P = lstsq(Q.T, self.M.T, rcond = -1)[0].T
            #negative terms = 0
            P[P < 0] = 0

        return (P @ Q).round(decimals=1)

    def computeMU(self):
        # Multiplication Update
        # M = PQ
        #initialize P and Q matrices
        P = np.random.rand(self.X, self.K)
        Q = np.random.rand(self.K, self.Y)

        for _ in range(self.steps):

            Q = Q * (P.T.dot(self.M) / P.T.dot(P).dot(Q) + self.beta)

            P = P * (self.M.dot(Q.T) / P.dot(Q).dot(Q.T) + self.beta)

        return (P @ Q).round(decimals=1)

rec_matrix = ratingsMatrix()

NMF = NMF(rec_matrix, K=2, steps=20, beta= 0.002)

result = NMF.computeMU()

print(result)

