import pandas as pd 
import numpy as np 
NUM_USERS = 943
NUM_ITEMS = 1682
USER_ID_COL = "User Id"
ITEM_ID_COL = "Item Id"
RATING_COL = "Rating"
TIMESTAMP_COL = "Timestamp"

def convertFileToDataframe(filename):
	df = pd.read_csv("../ml-100k/"+filename, sep="\t",
	                 names=["User Id", "Item Id", "Rating", "Timestamp"], header=None)
	return df


def convertDataframeToMatrix(df):
	mat = np.zeros(shape=(NUM_USERS, NUM_ITEMS))
	for i in range(len(df)):
		rating = df.loc[i, RATING_COL]
		user_id = df.loc[i, USER_ID_COL]
		item_id = df.loc[i, ITEM_ID_COL]
		mat[user_id - 1][item_id - 1] = rating
	return mat

def get_age_one_hot(age):
	index = -1 
	if age <= 14:
		index = 0
	elif age <= 21:
		index = 1
	elif age <= 28:
		index = 3
	elif age <= 36:
		index = 4
	elif age <= 48:
		index = 5
	elif age <= 55:
		index = 6 
	elif age <= 65:
		index = 7 
	else:
		index = 8 
	vec = np.zeros((8, ))
	vec[index - 1] = 1
	return vec 

def get_one_hot(index, total):
	vec = np.zeros((total,))
	vec[index-1] = 1
	return vec 

def occu_key_value_dict():
	data_path = "../../ml-100k/"
	occupation = pd.read_csv(data_path + "u.occupation", header = None)
	occu_dict = {}
	for i in range(occupation.shape[0]):
		occu_dict[i] = occupation.loc[i][0]
		# print(occu_dict[i])
	occu_dict = {value: key for (key, value) in occu_dict.items()}
	# print(occu_dict)
	return occu_dict

def genre_jey_value_dict():
	data_path = "../../ml-100k/"
	genre = pd.read_csv(data_path + "u.genre", names = ["Genre", "Id"],sep='|')
	key_val = {} 
	for i in range(genre.shape[0]):
		id = int(genre.loc[i, "Id"])
		g = genre.loc[i,"Genre"]
		key_val[id] = g 
	return key_val


def get_user_info():
	data_path = "../../ml-100k/"
	df = pd.read_csv(data_path+"u.user", sep="|", names=["Id", "Age", "Sex", "Occupation", "Pincode"])
	occu_dict = occu_key_value_dict() 
	# print(occu_dict)
	USER_FEATURES = {} 
	for i in range(df.shape[0]):
		id = int(df.loc[i,"Id"])
		features = []
		age = df.loc[i, "Age"]
		sex = df.loc[i, "Sex"]
		occupation = df.loc[i, "Occupation"]
		features = np.array([])
		age_v = get_age_one_hot(int(age))
		sex_v = -1
		occupation_v = get_one_hot(occu_dict[occupation] + 1, len(occu_dict))	
		if sex == 'M':
			sex_v = get_one_hot(1, 2)
		else:
			sex_v = get_one_hot(2,2)
		features = np.append(np.append(age_v , sex_v), occupation_v)
		# features.extend(age_v)
		# features.extend(sex_v)
		# features.extend(occupation_v)
		USER_FEATURES[id] = features 
	return USER_FEATURES 


def get_item_feature():
	data_path = "../../ml-100k/"
	header_ = ["Id", "Name", "Date", "Something"]
	for i in range(5,24):
		s = "feature"+str(i) 
		header_.append(s) 
	# print(header_) 
	items = pd.read_csv(data_path + 'u.item', header=None,
	            encoding="ISO-8859-1", sep="|", engine='python')
	items = items.values 
	ITEM_FEATURES = {} 
	for item in items:
		ITEM_FEATURES[int(item[0])] = item[5:] 
	
	return ITEM_FEATURES 

	


if __name__ == "__main__":
	USER = get_user_info() 
	print(USER)
