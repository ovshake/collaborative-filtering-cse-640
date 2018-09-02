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
    
def getPredictedRating(item , user , rating_matrix_base , cosine_similarity):
    
    total_cosine_similarity = 0
    predicted_rating = 0
    for user_id in range(rating_matrix_base.shape[USER_AXIS]):
        if user_id != user:
            total_cosine_similarity += cosine_similarity[user][user_id]

    # print("total cosine similarity" , total_cosine_similarity)
    for user_id in range(rating_matrix_base.shape[USER_AXIS]):
        if user_id != user:
            predicted_rating += (cosine_similarity[user][user_id] / total_cosine_similarity) * rating_matrix_base[user_id][item]
    return predicted_rating


def getMeanAbsoluteError(filename):
    train_file_name = filename+".base"
    test_file_name = filename+".test"
    rating_matrix_base = convertDataframeToMatrix(convertFileToDataframe(train_file_name))
    rating_matrix_test = convertDataframeToMatrix(convertFileToDataframe(test_file_name))
    # print(rating_matrix_test[333][3])
    cosine_similarity = 1 - pairwise_distances(rating_matrix_base , metric='cosine')
    mean_absolute_error = 0
    # p = getPredictedRating(3 , 333 , rating_matrix_base , cosine_similarity)
    # print("jnbdsjdf" , p)
    # sys.exit()
    for user_id in range(rating_matrix_test.shape[USER_AXIS]):
        for item_id in range(rating_matrix_base.shape[ITEM_AXIS]):
            if rating_matrix_test[user_id][item_id] != 0:
                mean_absolute_error += abs(getPredictedRating(item_id , user_id , rating_matrix_base , cosine_similarity) - rating_matrix_test[user_id][item_id])
    
    return mean_absolute_error
mean_error = 0
for i in range(1,6):
    split_error = getMeanAbsoluteError("u"+str(i))
    print("for u{} the mean abs error is {}".format(i , split_error))
    mean_error += split_error

print("Mean Error is {}".format(mean_error / 5))





