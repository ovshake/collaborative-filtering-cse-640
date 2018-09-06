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
USER_AXIS = 0
ITEM_AXIS = 1

def convertFileToDataframe(filename):
	df = pd.read_csv("../ml-100k/"+filename , sep = "\t" , names = ["User Id" , "Item Id" , "Rating" , "Timestamp"] , header = None)
	return df  

def convertDataframeToMatrix(df):
	mat = np.zeros(shape=(NUM_USERS , NUM_ITEMS)) 
	for i in range(len(df)):
		rating = df.loc[i , RATING_COL]
		user_id = df.loc[i , USER_ID_COL]
		item_id = df.loc[i , ITEM_ID_COL]
		mat[user_id - 1][item_id - 1] = rating
	return mat 
   


def get_k_most_similiar(k , user ,cosine_similarity):
	USER = 0
	RATING_GIVEN = 1
	user_tups = []
	for user_id in range(cosine_similarity.shape[0]):
		if user_id != user:
			tup = (user_id , cosine_similarity[user_id][user])
			user_tups.append(tup) 
	user_tups = sorted(user_tups , key = lambda x : x[RATING_GIVEN] , reverse = True)
	user_tups = user_tups[:k] 
	return user_tups


def getPredictedRating(item , user , rating_matrix_base , cosine_similarity, k = -1):
	
	total_cosine_similarity = 0
	predicted_rating = 0
	if k == -1:
		for user_id in range(rating_matrix_base.shape[USER_AXIS]):
			if user_id != user:
				total_cosine_similarity += cosine_similarity[user][user_id]
		for user_id in range(rating_matrix_base.shape[USER_AXIS]):
			if user_id != user:
				predicted_rating += (cosine_similarity[user][user_id] / total_cosine_similarity) * rating_matrix_base[user_id][item]
		return predicted_rating
	else:
		user_tups = get_k_most_similiar(k , user , cosine_similarity)
		total_cosine_similarity = 0
		for user_id , cos_sim in user_tups:
			total_cosine_similarity += cos_sim 
		for user_id , cos_sim in user_tups:
			predicted_rating += (cosine_similarity[user][user_id] / total_cosine_similarity) * rating_matrix_base[user_id][item]
		return predicted_rating 


def getMeanAbsoluteError(filename , k):
	train_file_name = filename+".base"
	test_file_name = filename+".test"
	rating_matrix_base = convertDataframeToMatrix(convertFileToDataframe(train_file_name))
	rating_matrix_test = convertDataframeToMatrix(convertFileToDataframe(test_file_name))
	cosine_similarity = 1 - pairwise_distances(rating_matrix_base , metric='cosine')
	mean_absolute_error = 0
	total_prediction = 0
	for user_id in range(rating_matrix_test.shape[USER_AXIS]):
		for item_id in range(rating_matrix_test.shape[ITEM_AXIS]):
			if rating_matrix_test[user_id][item_id] != 0:
				mean_absolute_error += abs(getPredictedRating(item_id , user_id , rating_matrix_base , cosine_similarity , k) - rating_matrix_test[user_id][item_id])
				total_prediction += 1
	
	return mean_absolute_error / total_prediction

if __name__ == "__main__":
	k_vals = [-1, 10 , 20 , 30 , 40 , 50]
	for k in k_vals:
		mean_error = 0
		for i in range(1, 6):
			split_error = getMeanAbsoluteError("u"+str(i) , k = k)
			print("for u{} the mean abs error is {} k = {}".format(i , split_error , k))
			mean_error += split_error
		print(mean_error / 5)





