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
		pred_rating = X[user_id - 1][item_id - 1]
		# print("prediction {}".format(pred_rating))
		error += abs(rating - X[user_id - 1][item_id - 1]) / 4

	return error / total_pred


def populate_matrix(filename):
	df_train = convertFileToDataframe(filename+".base")
	# df_test = convertFileToDataframe(filename+".test") 
	Y = np.zeros((NUM_USERS , NUM_ITEMS))
	for i in range(len(df_train)):
		rating = df_train.loc[i , RATING_COL]
		user_id = df_train.loc[i , USER_ID_COL]
		item_id = df_train.loc[i , ITEM_ID_COL]
		Y[user_id - 1][item_id - 1] = rating
	# for i in range(len(df_test)):
	# 	rating = df_test.loc[i , RATING_COL]
	# 	user_id = df_test.loc[i , USER_ID_COL]
	# 	item_id = df_test.loc[i , ITEM_ID_COL]
	# 	Y[user_id - 1][item_id - 1] = rating

	return Y 


def train_(fold_num , num_iters , lmbda):

	mask = get_mask("u"+str(fold_num)+".base")
	total_matrix = populate_matrix("u"+str(fold_num))
	# print(total_matrix)
	X = np.random.rand(NUM_USERS , NUM_ITEMS)
	for i in range(num_iters):
		if i % 20 == 0:
			print(i)	
		T_ = X + total_matrix - np.multiply(mask , X)
		u, s, v = np.linalg.svd(T_) 
		sigma = np.zeros((NUM_USERS , NUM_ITEMS))
		for j in range(s.shape[0]):
			sigma[j][j] = max(0 , s[j] - (lmbda / 2)) 
		X = np.matmul(u , np.matmul(sigma , v))

	error = get_NMAE(total_matrix , X , "u"+str(fold_num)+".test")
	print("For fold {} the error is {}".format(fold_num , error))
	return error

def per_regulariser(lmbda):
	TOTAL_FOLDS = 5
	per_fold_error = []
	for i in range(1 , TOTAL_FOLDS + 1):
		e = train_(i , 100 , lmbda)
		per_fold_error.append(e)
	print("for lambda {}".format(lmbda))
	print(per_fold_error)
	print("Avg. Error {}".format(sum(per_fold_error) / 5)) 

if __name__ == "__main__":
	lmbdas = [2 , 0.2 , 0.5, 1 , 3]
	for l in lmbdas:
		per_regulariser(l) 
	

	
