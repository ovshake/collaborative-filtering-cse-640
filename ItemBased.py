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
USER_AXIS = 1
ITEM_AXIS = 0

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
    for item_id in range(rating_matrix_base.shape[ITEM_AXIS]):
        if item_id != item:
            total_cosine_similarity += cosine_similarity[item_id][item]
    if total_cosine_similarity == 0:
        return 0 
    for item_id in range(rating_matrix_base.shape[ITEM_AXIS]):
        if item_id != item: 
            predicted_rating += (cosine_similarity[item_id][item] / total_cosine_similarity) * rating_matrix_base[item_id][user]
    return predicted_rating


def getMeanAbsoluteError(filename):
    train_file_name = filename+".base"
    test_file_name = filename+".test"
    rating_matrix_base = convertDataframeToMatrix(convertFileToDataframe(train_file_name))
    rating_matrix_base = np.transpose(rating_matrix_base)
    rating_matrix_test = convertDataframeToMatrix(convertFileToDataframe(test_file_name))
    rating_matrix_test = np.transpose(rating_matrix_test)
    cosine_similarity = 1 - pairwise_distances(rating_matrix_base , metric='cosine')
    mean_absolute_error = 0
    total_prediction = 0
    for item_id in range(rating_matrix_test.shape[ITEM_AXIS]):
        for user_id in range(rating_matrix_base.shape[USER_AXIS]):
            if rating_matrix_test[item_id][user_id] != 0:
                mean_absolute_error += abs(getPredictedRating(item_id , user_id , rating_matrix_base , cosine_similarity) - rating_matrix_test[item_id][user_id])
                total_prediction += 1
    return mean_absolute_error / total_prediction

if __name__ == "__main__":
    mean_error = 0
    fold_error = []
    for i in range(1,6):
        split_error = getMeanAbsoluteError("u"+str(i))
        print("for u{} the mean abs error is {}".format(i , split_error))
        mean_error += split_error
        fold_error.append(split_error)
    print(fold_error)

    print("Mean Error is {}".format(mean_error / 5))





