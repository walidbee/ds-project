from movieLensDataset import *
import numpy as np
from numpy.linalg import lstsq
import pandas as pd

class RecommandationSystem():

    def __init__(self):
        #load the MovieLens Dataset and convert it to a Numpy matrix
        self.M = ratingsMatrix()
        #self.M = np.array([[1, 1, 1, 0, 0], [3, 3, 3, 0, 0], [4, 4, 4, 0, 0], [5, 5, 5, 0, 0], [0, 0, 0, 4, 4], [0, 0, 0, 5, 5], [0, 0, 0, 2, 2]])
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

    def computeCUR(self, r):
      """
      CUR decomposition

      Arguments
      - r (int) : random number of selected rows and columns
      """

      self.r = r
      self.R, self.C, self.selectedRowsInd, self.selectedColsInd = self.computeCR()
      self.U = self.computeU()

      return self.C.dot(self.U.dot(self.R))

    def computeCR(self):
      """
      Computation of C and R matrices

      """
      nbRows = self.M.shape[0]
      nbCols = self.M.shape[1]

      # compute the square of the Frobenius norm of M
      squareFrobenious = (self.M**2).sum()

      # columns squared Frobenius norm
      sumSquaresCols = (self.M**2).sum(axis=0)

      # columns probabilities
      probCols = sumSquaresCols / squareFrobenious

      # selected columns' indices
      selectedColsInd = np.random.choice(np.arange(0, nbCols),size=self.r,replace=True,p=probCols)

      # selected columns' values
      selectedCols = self.M[:,selectedColsInd]

      # dividing columns' elements by the square root of the expected number of times this column would be picked
      selectedCols = np.divide(selectedCols,(self.r*probCols[selectedColsInd])**0.5)

      # rows squared Frobenius norm
      sumSquaresRows = (self.M**2).sum(axis=1)

      # rows probabilities
      probRows = sumSquaresRows / squareFrobenious

      # selected rows' indices
      selectedRowsInd = np.random.choice(np.arange(0, nbRows),size=self.r,replace=True,p=probRows)
      # selected rows' values
      selectedRows = self.M[selectedRowsInd,:]

      # dividing rows' elements by the square root of the expected number of times this column would be picked
      tmp = np.array([(self.r*probRows[selectedRowsInd])**0.5])
      selectedRows = np.divide(selectedRows,tmp.transpose())

      return selectedRows, selectedCols, selectedRowsInd, selectedColsInd

    def computeU(self):
      """
      Computation of middle matrix U
      
      """
      tmp = self.M[self.selectedRowsInd,:]
      W = tmp[:,self.selectedColsInd]

      U, s, Vh = np.linalg.svd(W, full_matrices=False)
      s = np.diag(s)

      for i in range(min(s.shape[0], s.shape[1])):
        if s[i][i] != 0:
          s[i][i] = 1 / s[i][i]

      u = Vh.transpose().dot(np.square(s).dot(U.transpose()))

      return u

    def calculateOptimalValueR(self):
      """
      Calculates optimal value of number of columns and rows for CUR decomposition
      """
      min_error = 1e50
      min_error_index = 0
      for i in range(1,self.X):
        m = self.computeCUR(i)
        error = self.RMSE(m)
        if error < min_error:
          min_error = error
          min_error_index = i
        #print(min_error,i)
      print('min error at ',min_error,min_error_index)

    #Error computing
    def RMSE(self, PM):
        """
        Computes Root Mean Square Error 

        Arguments
        - PM (matrix) : predicted matrix to compare with original
        """
        return np.sqrt(np.mean((PM-self.M)**2))

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
      
      recall = 0
      precision = 0 
      for i in range(self.X):
        query = self.M[i,:]
        query = np.reshape(query,(1, query.shape[0]))

        temp = query.dot(self.R.T)

        # needed to rescale predicted ratings
        #prediction = np.divide(temp.dot(self.R), 100000)
        prediction = temp.dot(self.R)

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
      return precision / self.X, recall / self.X

r = RecommandationSystem()
print(r.recPreF1(r.NMF_MU(2, 10, 0.002)))

#CUR and evaluation
CUR = r.computeCUR(r = 4)
RMSE = r.RMSE(CUR)
precision, recall = r.calculatePrecisionOnTopK(10)
print("RMSE = %f. Precision on TOP 10 = %f. Recall on TOP 10 = %f"%(RMSE, precision, recall))