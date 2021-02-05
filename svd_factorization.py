import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

!wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

csv_file=pd.read_csv("ml-latest-small/ratings.csv")
utility_matrix=csv_file.pivot(index='movieId', columns='userId', values='rating').fillna(0.0)
utility_matrix

R = utility_matrix.to_numpy()


class SVD():
    def __init__(self, M):
        self.M = M
        
    def computeSVD(self,M):
        U, S, VT = np.linalg.svd(self.M, full_matrices=False)
        s=np.diag(S[0:20])
        return U, S, VT, s
        
    def plot_singular_values(self,s):
        plt.semilogy(np.diagonal(s))
        plt.title('plot Singular Values')
        plt.show()
        
    def rank_S(self,S):
        print("rank:")
        return len(S)
  

#----------------------------------------------------------------------------

svd=SVD(R)
U, S, VT, s=svd.computeSVD(R)

svd.plot_singular_values(s)


# calculate initial P and Q 
# we fill missing rating of the original matrix with zeros
# we calculate the svd of this matrix
# we do the svd of this matriw , we will not use the svd to predict missing values but only to factorize the 
# initial estimated matrix model
# R = U S VT
# lets make : Q=U, and pT=S * VT

# lets start : 

print("Initializing Q and pT \n")
Q=U[:,:20]
print("Q matrix:",Q.shape,"\n",Q)
pT=s@VT[:20,:]
print("\npT matrix:",pT.shape,"\n",pT)

# -----------------------------------------------------------------------------------------

# Perform SGD 

grad_Q=0 #np.zeros((len_items,K))

n=0.1
lamda=0.1

Q_=Q
pT_=pT

for _ in range(10): 
    
    for i_ in range(R.shape[0]):
        for k in range(2):
            for x in range(R.shape[1]):
                grad_Q+=-2*(R[i_,x]-(Q[i_,:]@pT[:,x]))*pT[k,x] + 2*lamda*Q[i_,k]
            Q_[i_,k]-=n*grad_Q


print(Q_.shape)
print(Q_)
        
    
grad_pT=0 #np.zeros((K,len_users))

for _ in range(10): 
    
    for k_ in range(2):
        for x in range(R.shape[1]):
            for i in range(R.shape[0]):
                grad_pT+=-2*(R[i,x]-(Q[i,:]@pT[:,x]))*Q[i,k_] + 2*lamda*pT[k_,x]
            pT_[k_,x]-=n*grad_pT

           
   
