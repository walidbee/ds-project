import pandas as pd
import numpy as np
from scipy import sparse as sp
from movieLensDataset import *

class CUR():
	def __init__(self, M):
		self.M = M

	def computeCR(self, r):
		nbRows = self.M.shape[0]
		nbCols = self.M.shape[1]
		# compute the square of the Frobenius norm of M
		squareFrobenious = (self.M**2).sum()

		# columns squared Frobenius norm
		sumSquaresCols = (self.M**2).sum(axis=0)

		# rows squared Frobenius norm
		sumSquaresRows = (self.M**2).sum(axis=1)

		# columns probabilities
		probCols = sumSquaresCols / squareFrobenious
		#print(probCols)

		# rows probabilities
		probRows = sumSquaresRows / squareFrobenious
		#print(probRows)

		selectedColsInd = np.random.choice(np.arange(0, nbCols),size=r,replace=False,p=probCols)
		#print(selectedColsInd)

		selectedCols = self.M[:,selectedColsInd]
		#print(selectedCols)

		selectedCols = np.divide(selectedCols,(r*probCols[selectedColsInd])**0.5)
		#print(selectedCols)

		selectedRowsInd = np.random.choice(np.arange(0, nbRows),size=r,replace=False,p=probRows)
		#print(selectedRowsInd)

		selectedRows = self.M[selectedRowsInd,:]
		#print(selectedRows)

		tmp = np.array([(r*probRows[selectedRowsInd]**0.5)])
		selectedRows = np.divide(selectedRows,tmp.transpose())
		#print(selectedRows)
		return selectedCols, selectedColsInd, selectedRows, selectedRowsInd

	def computeU(self, selectedColsInd, selectedRowsInd):
		tmp = self.M[selectedRowsInd,:]
		W = tmp[:,selectedColsInd]

		u, s, Vh = np.linalg.svd(W, full_matrices=False)

		s = np.diag(s)

		for i in range(min(s.shape[0], s.shape[1])):
			s[i][i] = 1 / s[i][i]

		u = Vh.transpose().dot(np.square(s).dot(u.transpose()))

		return u

utility_matrix = getUtilityMatrix()

cur = CUR(utility_matrix)

r = 100

C, selectedColsIndices, R, selectedRowsIndices = cur.computeCR(r)

U = cur.computeU(selectedColsIndices, selectedRowsIndices)

CUR = C.dot(U.dot(R))

print(CUR)