from Utils import * 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import numpy as np 

data_path = "../../ml-100k/"
def train_(filename):
    matrix = convertDataframeToMatrix(convertFileToDataframe(data_path+filename))
    X_train = np.array([])
    Y_train = np.array([]) 
    USER_FEATURES = get_user_info() 
    ITEM_FEATURES = get_item_feature() 
    for user_id in range(matrix.shape[0]):
        for item_id in range(matrix.shape[1]):
            # input_vec = np.array([])
            # input_vec.extend(USER_FEATURES[user_id+1])
            # input_vec.extend(ITEM_FEATURES[item_id+1])
            input_vec = np.append(USER_FEATURES[user_id+1], ITEM_FEATURES[item_id + 1])
            # X_train.append(input_vec)
            X_train = np.append(X_train, input_vec)
            # Y_train.append(matrix[user_id][item_id]) 
            Y_train = np.append(Y_train, matrix[user_id][item_id])
    
    clf = SVC(verbose=True)
    clf.fit(X_train , Y_train) 
    return clf 

def test_(filename, clf):
    matrix = convertDataframeToMatrix(
        convertFileToDataframe(data_path+filename))
    X_test = np.array([])
    Y_test = np.array([])
    USER_FEATURES = get_user_info()
    ITEM_FEATURES = get_item_feature()
    for user_id in range(matrix.shape[0]):
        for item_id in range(matrix.shape[1]):
            # input_vec = np.array([])
            # input_vec.extend(USER_FEATURES[user_id+1])
            # input_vec.extend(ITEM_FEATURES[item_id+1])
            input_vec = np.append(
                USER_FEATURES[user_id+1], ITEM_FEATURES[item_id + 1])
            X_test = np.append(X_test, input_vec)
            # X_test.append(input_vec)
            Y_test = np.append(Y_test , matrix[user_id][item_id])
            # Y_test.append(matrix[user_id][item_id])
    
    Y_pred = clf.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)
    print(score) 


def per_split(num):
    train_filename = "u"+str(num)+".base" 
    test_filename = "u"+str(num)+".test" 
    clf = train_(train_filename)
    test_(test_filename, clf) 


if __name__ == "__main__":
    for i in range(1,6):
        per_split(i) 
