import numpy as np 
import pandas as pd 
from sklearn.metrics import pairwise_distances
import sys

NUM_USERS = 943
NUM_ITEMS = 1682
USER_ID_COL = "User Id"
ITEM_ID_COL = "Item Id"
RATING_COL = "Rating"
TIMESTAMP_COL = "Timestamp"


def convertFileToDataframe(filename):
	df = pd.read_csv("../../ml-100k/"+filename , sep = "\t" , names = ["User Id" , "Item Id" , "Rating" , "Timestamp"] , header = None)
	return df  

def convertDataframeToMatrix(df):
	mat = np.zeros(shape=(NUM_USERS , NUM_ITEMS)) 
	for i in range(len(df)):
		rating = df.loc[i , RATING_COL]
		user_id = df.loc[i , USER_ID_COL]
		item_id = df.loc[i , ITEM_ID_COL]
		mat[user_id - 1][item_id - 1] = rating
	return mat 

def get_mask(filename):
	df = pd.read_csv("../../ml-100k/"+filename , sep = "\t" , names = ["User Id" , "Item Id" , "Rating" , "Timestamp"] , header = None)
	mask = np.zeros((NUM_USERS , NUM_ITEMS))
	for i in range(len(df)):
		rating = df.loc[i , RATING_COL]
		user_id = df.loc[i , USER_ID_COL]
		item_id = df.loc[i , ITEM_ID_COL]
		mask[user_id - 1][item_id - 1] = 1 

	return mask 

def get_NMAE(total_matrix , X , filename):
	df = pd.read_csv("../../ml-100k/"+filename , sep = "\t" , names = ["User Id" , "Item Id" , "Rating" , "Timestamp"] , header = None)
	error = 0
	total_pred = 0
	for i in range(len(df)):
		total_pred += 1
		rating = df.loc[i , RATING_COL]
		user_id = df.loc[i , USER_ID_COL]
		item_id = df.loc[i , ITEM_ID_COL]
		error += (abs(rating - X[user_id - 1][item_id - 1]) / 4)

	return error / total_pred





def train_(total_matrix , fold_num , num_iters , lmbda):
	mask = get_mask("u"+str(fold_num)+".base")
	X = np.random.rand(NUM_USERS , NUM_ITEMS)
	# u, s, v = np.linalg.svd(X)
	# print(s)
	for i in range(num_iters):
		if i % 100 == 0:
			print(i)
		
		T_ = X + total_matrix - np.multiply(mask , X)
		u, s, v = np.linalg.svd(T_) 
		# print("u shape" , u.shape)
		# print("s" , s.shape)
		# print("v shape" , v.shape)
		# print(s.shape)
		sigma = np.zeros((NUM_USERS , NUM_ITEMS))
		for j in range(s.shape[0]):
			sigma[j][j] = max(0 , s[j] - (lmbda / 2)) 
		# print("u shape" , u.shape)
		# print("Sigma Shape" , sigma.shape)
		# print("v shape" , v.shape)
		X = np.matmul(u , np.matmul(sigma , v))

	error = get_NMAE(total_matrix , X , "u"+fold_num+".test")
	print("For fold {} the error is {}".format(fold_num , error))
	return error










if __name__ == "__main__":
	total_matrix = convertDataframeToMatrix(convertFileToDataframe("u.data"))
	TOTAL_FOLDS = 5
	per_fold_error = []
	for i in range(1 , TOTAL_FOLDS + 1):
		e = train_(total_matrix , i + 1 , 1000 , 0.2)
		per_fold_error.append(e)
	print(per_fold_error)

	
