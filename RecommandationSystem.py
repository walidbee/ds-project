from movieLensDataset import *
import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from sklearn.model_selection import train_test_split

class RecommandationSystem():

    def __init__(self):
        #load the MovieLens Dataset and convert it to a Numpy matrix
        self.M = ratingsMatrix()
        self.train_set, self.test_set = train_test_split(
            ratingsMatrix(), test_size=0.3
        )
        self.X, self.Y = self.M.shape
    
    #Non-negative matrix factorization implementations
    def NMF_MU(self, K, steps, beta):
        """
        Non-negative matrix factorization using Multiplicative Update
        https://eric.univ-lyon2.fr/~mselosse/teaching/NMF.pdf

        Arguments
        - K (int) : number of latent features
        - steps (int) : number of times we run the algorithm
        - beta (float) : added number so we don't divide by 0 
        """

        #initialize P and Q matrices
        P = np.random.rand(self.X, K)
        Q = np.random.rand(K, self.Y)

        for _ in range(steps):
            Q = Q * (P.T.dot(self.M) / P.T.dot(P).dot(Q) + beta)

            P = P * (self.M.dot(Q.T) / P.dot(Q).dot(Q.T) + beta)

        return (P @ Q).round(decimals=1)
    
    def NMF_ALS(self, K, steps):
        """
        Non-negative matrix factorization using Alternating Least Squares
        https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-explo-nmf.pdf

        Arguments
        - K (int) : number of latent features
        - steps (int) : number of times we run the algorithm
        """
        #initialize P and Q matrices
        P = np.random.rand(self.X, K)
        Q = np.random.rand(K, self.Y)
    
        for _ in range(steps):
            #resolve P @ Q = M
            Q = lstsq(P, self.M, rcond = -1)[0]
            #negative terms = 0
            Q[Q < 0] = 0

            #resolve Q.T @ P = M.T
            P = lstsq(Q.T, self.M.T, rcond = -1)[0].T
            #negative terms = 0
            P[P < 0] = 0

        return (P @ Q).round(decimals=1)

    def computeCUR(self, r, N=10, dataset = "train"):
      """
      CUR decomposition

      Arguments
      - r (int) : random number of selected rows and columns
      """
      if (dataset == "train"): M = self.train_set
      else : M = self.test_set
      self.r = r
      self.R, self.C, self.selectedRowsInd, self.selectedColsInd = self.computeCR(dataset)
      self.U = self.computeU()
      self.CUR = self.C.dot(self.U.dot(self.R))
      self.error = self.RMSE(self.CUR, M)
      #print("Initial error: %f"%self.error)
      
      for _ in range(N):
        R_temp, C_temp, selectedRowsInd_temp, selectedColsInd_temp = self.computeCR(dataset)
        U_temp = self.computeU(dataset)
        CUR_temp = C_temp.dot(U_temp.dot(R_temp))
        error_temp = self.RMSE(CUR_temp, M)

        if error_temp <= self.error:
          #print("error_Temp at %d = %f"%(i, error_temp))
          
          self.selectedRowsInd = selectedRowsInd_temp
          self.selectedColsInd = selectedColsInd_temp
          self.C = C_temp
          self.U = U_temp
          self.R = R_temp
          self.CUR = CUR_temp
          self.error = error_temp
      

      return self.CUR, self.error

    def computeCR(self, dataset = "train"):
      """
      Computation of C and R matrices

      """
      if (dataset == "train"): M = self.train_set
      else : M = self.test_set
      
      nbRows = M.shape[0]
      nbCols = M.shape[1]

      # compute the square of the Frobenius norm of M
      squareFrobenious = (M**2).sum()

      # columns squared Frobenius norm
      sumSquaresCols = (M**2).sum(axis=0)

      # columns probabilities
      probCols = sumSquaresCols / squareFrobenious

      # selected columns' indices
      selectedColsInd = np.random.choice(np.arange(0, nbCols),size=self.r,replace=True,p=probCols)

      # selected columns' values
      selectedCols = M[:,selectedColsInd]

      # dividing columns' elements by the square root of the expected number of times this column would be picked
      selectedCols = np.divide(selectedCols,(self.r*probCols[selectedColsInd])**0.5)

      # rows squared Frobenius norm
      sumSquaresRows = (M**2).sum(axis=1)

      # rows probabilities
      probRows = sumSquaresRows / squareFrobenious

      # selected rows' indices
      selectedRowsInd = np.random.choice(np.arange(0, nbRows),size=self.r,replace=True,p=probRows)
      # selected rows' values
      selectedRows = M[selectedRowsInd,:]

      # dividing rows' elements by the square root of the expected number of times this column would be picked
      tmp = np.array([(self.r*probRows[selectedRowsInd])**0.5])
      selectedRows = np.divide(selectedRows,tmp.transpose())

      return selectedRows, selectedCols, selectedRowsInd, selectedColsInd

    def computeU(self, dataset = "train"):
      """
      Computation of middle matrix U
      
      """
      if (dataset == "train"): M = self.train_set
      else : M = self.test_set

      tmp = M[self.selectedRowsInd,:]
      W = tmp[:,self.selectedColsInd]

      U, s, Vh = np.linalg.svd(W, full_matrices=False)
      """
      s = np.diag(s)

      for i in range(min(s.shape[0], s.shape[1])):
        if s[i][i] != 0:
          s[i][i] = 1 / s[i][i]

      u = Vh.transpose().dot(np.square(s).dot(U.transpose()))
      """
      #"""
      for i in range(len(s)):
        if s[i] == 0:
            continue
        else:
            s[i] = 1 / s[i]
      s = np.diag(s)
      u = np.dot(np.dot(Vh, s), U.T)
      #u = Vh.dot(s.dot(U.transpose()))
      #"""

      return u
      
    def calculateOptimalValueR(self, data_set = "train"):
      """
      Calculates optimal value of number of columns and rows for CUR decomposition
      """
      min_error = 1e50
      min_error_index = 0

      if (data_set == "train"):
        r_range = [1,2,3,4,5,6,7,8,9,10,20,50,100,200,300,400]
      else:
        r_range = [1,2,3,4,5,6,7,8,9,10,20,50,100,150]
      print("Calculating optimal value of r...")
      for i in r_range:
        m, error = self.computeCUR(i, dataset = data_set)
        #error = self.RMSE(m)
        if error < min_error:
          min_error = error
          min_error_index = i
          best_m = m
        #print(min_error,i)
      #print('min error at ',min_error,min_error_index)
      #print("Error at %d is %f"%(i, error))
      return best_m, min_error, min_error_index

    #Error computing
    def RMSE(self, PM, OM = None):
        """
        Computes Root Mean Square Error 

        Arguments
        - PM (matrix) : predicted matrix to compare with original
        """
        if OM is None: OM = self.M
        return np.sqrt(np.mean((PM-OM)**2))

    def recPreF1(self, PM):
        """
        Computes Recall, Precision and F1-Score for each rating

        Arguments
        - PM (matrix) : predicted matrix to compare with original
        """

        results = []
        for score in np.arange(0, 5.5, 0.5):
          tp=0
          fn=0
          fp=0
          tn=0
          vals = []

          for i in range(self.X):
            for j in range(self.Y):
              predicted = PM[i,j]
              value = self.M[i,j]

              if value>=score :
                if predicted>=score :
                  tp = tp+1
                else:
                  fn = fn+1
              else:
                if predicted>=score :
                  fp = fp+1
                else:
                  tn = tn+1

              if tp == 0:
                precision = 0
                recall = 0
                f1 = 0
              else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)   

          vals = [score, tp, fp, tn ,fn, precision, recall, f1]
          results.append(vals)

        results = pd.DataFrame(results)
        results.rename(columns={0:'score', 1:'tp', 2: 'fp', 3: 'tn', 4:'fn', 5: 'Precision', 6:'Recall', 7:'F1'}, inplace=True)

        return results

    def calculatePrecisionOnTopK(self, k):
      """
      Calculates the Precision on Top K
      Arguments
        - k (int) : The k in Precision on Top k
      
      Output
        - Precision on Top K
        - Recall on Top K
      """
      test_set = self.test_set
      recall = 0
      precision = 0 
      #pred_mat_test = []
      #temp = test_set.dot(self.R.T)
      for i in range(test_set.shape[0]):
        query = test_set[i,:]
        query = np.reshape(query,(1, query.shape[0]))

        temp = query.dot(self.R.T)

        # needed to rescale predicted ratings
        #prediction = np.divide(temp.dot(self.R), 100000)
        prediction = temp.dot(self.R)
        #pred_mat_test.append(prediction)

        idx = prediction.argsort()[0, ::-1]
        prediction = prediction[0,idx]
        query = query[0, idx]

        prediction[prediction < 3] = 0
        prediction[prediction > 3] = 1

        query[query == 0] = -1
        query[(query < 3) & (query > 0)] = 0
        query[query >= 3] = 1

        #idx = prediction.argsort()[0, ::-1][:k]
        #prediction = prediction[0,idx]
        
        #query = query[0, idx]]
        relevant_items = 0
        recommended_items = 0
        rec_relevant_items = 0
        recall_item = 0
        precision_item = 0 

        #Recall on top k
        for i in range(0,query.shape[0]):
          if (relevant_items == k): 
            break
          if (query[i] != -1):
            #print("predicted: %d - True: %d"%(prediction[i], query[i]))
            if (query[i] == 1):
              relevant_items += 1 
              if(prediction[i] == 1):
                rec_relevant_items +=1
        if (relevant_items != 0):
          recall_item = rec_relevant_items / relevant_items
        else:
          recall_item = 0.0
        recall += recall_item

        #precision on top k
        rec_relevant_items = 0
        for i in range(0,query.shape[0]):
          if (recommended_items == k): 
            break
          if (query[i] != -1):
            #print("predicted: %d - True: %d"%(prediction[i], query[i]))
            if (prediction[i] == 1):
              recommended_items += 1
              if(query[i] == 1):
                rec_relevant_items +=1
        
        if (recommended_items != 0):
          precision_item = rec_relevant_items / recommended_items
        else:
          precision_item = 0.0
        precision += precision_item

      CUR_on_test, RMSE_on_test, r_value = self.calculateOptimalValueR(data_set = "test")
      return RMSE_on_test, r_value, precision / test_set.shape[0], recall / test_set.shape[0]

r = RecommandationSystem()
#print(r.recPreF1(r.NMF_MU(2, 10, 0.002)))

#CUR and evaluation
print("#########################################")
print("Computing CUR and RMSE on training set.")
CUR, RMSE, r_value = r.calculateOptimalValueR("train")
topK = 50
print("Computing RMSE on testing set and evaluation.")
rmse_test, r_val_test, precision, recall = r.calculatePrecisionOnTopK(topK)
print("Train set: RMSE = %f. Optimal value of r = %d\nTest set: RMSE = %f. Optimal value of r = %d. Precision on TOP %d = %f. Recall on TOP %d = %f"%(RMSE, r_value, rmse_test, r_val_test, topK, precision, topK, recall))
print("#########################################")
"""
#----SVD factorization --------------

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
"""
           
   
