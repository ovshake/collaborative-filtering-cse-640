import numpy as np
import json
from pprint import pprint
import re
import pandas as pd
# import tensorflow as tf
import argparse
import sys
import os
import heapq
import math

DATA_PATH = "data/"
USER_FILE = DATA_PATH+"users.csv"
MOVIES_FILE = DATA_PATH+"movies.csv"
RATINGS_FILE = DATA_PATH+"ratings_correctFormat.json"

for i in TRAIN_FOLDS:
	TRAIN_FOLDS_FILES.append(DATA_PATH+i)

for i in TEST_FOLDS:
	TEST_FOLDS_FILES.append(DATA_PATH+i)

class Model:
	def __init__(self, args, rating_matrix,movie_ids, ratings):
		self.dataName = 'data'
		# self.dataSet = DataSet(self.dataName)
		self.shape = np.asarray(rating_matrix).shape
		self.maxRate = 5.0

		# self.train = self.dataSet.train
		# self.test = self.dataSet.test

		self.rating_matrix=rating_matrix
		# self.users=users
		# self.movies=movies
		self.movie_ids=movie_ids
		self.ratings=ratings

		self.train, self.test=self.train_test_split(test_size=0.2)

		print(self.train.shape, self.test.shape)

		self.trainDict=self.getTrainDict()

		self.negNum = 7
		self.testNeg = self.getTestNeg(self.test, 99)
		self.add_embedding_matrix()

		self.add_placeholders()

		self.userLayer = args.userLayer
		self.itemLayer = args.itemLayer
		self.add_model()

		self.add_loss()

		self.lr = args.lr
		self.add_train_step()

		self.checkPoint = args.checkPoint
		self.init_sess()

		self.maxEpochs = args.maxEpochs
		self.batchSize = args.batchSize

		self.topK = args.topK
		self.earlyStop = args.earlyStop

	def train_test_split(self, test_size):
		train=[]
		test=[]

		test_len=test_size*len(self.ratings)

		for i in range(int(test_len)):
			test.append(self.ratings[i])

		for i in range(int(test_len), len(self.ratings)):
			train.append(self.ratings[i])

		return np.asarray(train), np.asarray(test)

	def getTestNeg(self, testData, negNum):
		user = []
		item = []
		for s in testData:
			tmp_user = []
			tmp_item = []
			u = s[0]
			i = s[1]
			tmp_user.append(u)
			tmp_item.append(i)
			neglist = set()
			neglist.add(i)
			for t in range(negNum):
				j = np.random.randint(self.shape[1])
				while (u, j) in self.trainDict or j in neglist:
					j = np.random.randint(self.shape[1])
				neglist.add(j)
				tmp_user.append(u)
				tmp_item.append(j)
			user.append(tmp_user)
			item.append(tmp_item)
		return [np.array(user), np.array(item)]

	def getTrainDict(self):
			dataDict = {}
			for i in self.train:
				dataDict[(i[0], i[1])] = i[2]
			return dataDict

	def add_placeholders(self):
		self.user = tf.placeholder(tf.int64)
		self.item = tf.placeholder(tf.int64)
		self.rate = tf.placeholder(tf.float64)
		self.drop = tf.placeholder(tf.float64)

	def add_embedding_matrix(self):
		# print(type(self.rating_matrix[0][2]))
		self.user_item_embedding = tf.convert_to_tensor(self.rating_matrix)
		self.item_user_embedding = tf.transpose(self.user_item_embedding)

	def add_model(self):
		user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
		item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

		def init_variable(shape, name):
			return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float64, stddev=0.01), name=name)

		with tf.name_scope("User_Layer"):
			user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
			user_out = tf.matmul(user_input, user_W1)
			for i in range(0, len(self.userLayer)-1):
				W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
				b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
				user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

		with tf.name_scope("Item_Layer"):
			item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
			item_out = tf.matmul(item_input, item_W1)
			for i in range(0, len(self.itemLayer)-1):
				W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
				b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
				item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

		norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
		norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
		self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keepdims=False) / (norm_item_output* norm_user_output)
		self.y_ = tf.maximum(np.float64(1e-6), self.y_)

	def add_loss(self):
		regRate = self.rate / self.maxRate
		losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
		loss = -tf.reduce_sum(losses)
		# regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
		# self.loss = loss + self.reg * regLoss
		self.loss = loss

	def add_train_step(self):
		'''
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.lr = tf.train.exponential_decay(self.lr, global_step,
											 self.decay_steps, self.decay_rate, staircase=True)
		'''
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_step = optimizer.minimize(self.loss)

	def init_sess(self):
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.config.allow_soft_placement = True
		self.sess = tf.Session(config=self.config)
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		if os.path.exists(self.checkPoint):
			[os.remove(f) for f in os.listdir(self.checkPoint)]
		else:
			os.mkdir(self.checkPoint)

	def getInstances(self, data, negNum):
		user = []
		item = []
		rate = []
		for i in data:
			user.append(i[0])
			item.append(i[1])
			rate.append(i[2])
			for t in range(negNum):
				j = np.random.randint(self.shape[1])
				while (i[0], j) in self.trainDict:
					j = np.random.randint(self.shape[1])
				user.append(i[0])
				item.append(j)
				rate.append(0.0)
		return np.array(user), np.array(item), np.array(rate)

	def run(self):
		best_hr = -1
		best_NDCG = -1
		best_epoch = -1
		print("Start Training!")
		for epoch in range(self.maxEpochs):
			print("="*20+"Epoch ", epoch, "="*20)
			self.run_epoch(self.sess)
			print('='*50)
			print("Start Evaluation!")
			hr, NDCG = self.evaluate(self.sess, self.topK)
			print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
			if hr > best_hr or NDCG > best_NDCG:
				best_hr = hr
				best_NDCG = NDCG
				best_epoch = epoch
				self.saver.save(self.sess, self.checkPoint)
			if epoch - best_epoch > self.earlyStop:
				print("Normal Early stop!")
				break
			print("="*20+"Epoch ", epoch, "End"+"="*20)
		print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
		print("Training complete!")

	def run_epoch(self, sess, verbose=10):
		train_u, train_i, train_r = self.getInstances(self.train, self.negNum)
		train_len = len(train_u)
		shuffled_idx = np.random.permutation(np.arange(train_len))
		train_u = train_u[shuffled_idx]
		train_i = train_i[shuffled_idx]
		train_r = train_r[shuffled_idx]

		num_batches = len(train_u) // self.batchSize + 1

		losses = []
		for i in range(num_batches):
			min_idx = i * self.batchSize
			max_idx = np.min([train_len, (i+1)*self.batchSize])
			train_u_batch = train_u[min_idx: max_idx]
			train_i_batch = train_i[min_idx: max_idx]
			train_r_batch = train_r[min_idx: max_idx]

			feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
			_, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
			losses.append(tmp_loss)
			if verbose and i % verbose == 0:
				sys.stdout.write('\r{} / {} : loss = {}'.format(
					i, num_batches, np.mean(losses[-verbose:])
				))
				sys.stdout.flush()
		loss = np.mean(losses)
		print("\nMean loss in this epoch is: {}".format(loss))
		return loss

	def create_feed_dict(self, u, i, r=None, drop=None):
		return {self.user: u,
				self.item: i,
				self.rate: r,
				self.drop: drop}

	def evaluate(self, sess, topK):
		def getHitRatio(ranklist, targetItem):
			for item in ranklist:
				if item == targetItem:
					return 1
			return 0
		def getNDCG(ranklist, targetItem):
			for i in range(len(ranklist)):
				item = ranklist[i]
				if item == targetItem:
					return math.log(2) / math.log(i+2)
			return 0


		hr =[]
		NDCG = []
		testUser = self.testNeg[0]
		testItem = self.testNeg[1]
		for i in range(len(testUser)):
			target = testItem[i][0]
			feed_dict = self.create_feed_dict(testUser[i], testItem[i])
			predict = sess.run(self.y_, feed_dict=feed_dict)

			item_score_dict = {}

			for j in range(len(testItem[i])):
				item = testItem[i][j]
				item_score_dict[item] = predict[j]

			ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

			tmp_hr = getHitRatio(ranklist, target)
			tmp_NDCG = getNDCG(ranklist, target)
			hr.append(tmp_hr)
			NDCG.append(tmp_NDCG)
		return np.mean(hr), np.mean(NDCG)

def retrieve_users():
	user_list=[]
	old_to_new_user_id_dict = {}
	new_to_old_user_id_dict = {}  
	df = pd.read_csv(USER_FILE) 
	for i in range(df.shape[0]):
		name = df.loc[i ,'_id'] 
		occupation = df.loc[i , 'job']
		address = df.loc[i , 'state']
		dob = df.loc[i , 'dob']
		gender = df.loc[i , 'gender'] 
		all_langs = df.loc[i, 'languages']
		langs_known = [] 
		if type(all_langs) == float:
			user_list.append((name , langs_known , occupation, address , dob , gender))
			continue
		for u in all_langs[1:-1].split(','):
			u = u.replace('"', '')
			u = u.replace(" ", '')
			langs_known.append(u) 
		user_list.append((name , langs_known , occupation, address , dob , gender))
		old_to_new_user_id_dict[name] = i
		new_to_old_user_id_dict[i] = name  
	return user_list , old_to_new_user_id_dict , new_to_old_user_id_dict

def retrieve_movies():
	movies_list=[]
	data_frame=pd.read_csv(MOVIES_FILE)
	movies_list=data_frame.values
	return movies_list

def define_movie_id(movies):
	old_to_new_movie_ids={x[0] : i for i,x in enumerate(movies)}
	new_to_old_movie_ids = {i : x[0] for i,x in enumerate(movies)}
	return old_to_new_movie_ids , new_to_old_movie_ids

def retrieve_ratings_matrix(movie_ids, old_to_new_user_id_dict): 
	num_users =  924 #len(users)
	num_movies = 2850 #len(movies)
	user_item_matrix = np.zeros((num_users, num_movies))
	data_frame=pd.read_json(RATINGS_FILE)
	ratings = [] 
	for i in range(data_frame.shape[0]):
		old_user_id = data_frame.loc[i , '_id']
		all_ratings = data_frame.loc[i , 'rated']
		new_user_id = old_to_new_user_id_dict[old_user_id] 
		print(new_user_id , "new user if")
		for k in all_ratings:
			if 's' not in k:
				user_item_matrix[new_user_id][movie_ids[k]] = int(all_ratings[k][0] )
				tup = (new_user_id , movie_ids[k] , int(all_ratings[k][0])) 
				ratings.append(tup) 
	ratings = np.asarray(ratings)
	return user_item_matrix , ratings

# def run_dmf():



USERS , old_to_new_user_id_dict , new_to_old_user_id_dict = retrieve_users()
print(old_to_new_user_id_dict)
MOVIES = retrieve_movies()
old_to_new_movie_ids , new_to_old_movie_ids = define_movie_id(MOVIES)
RATINGS_MATRIX , RATINGS = retrieve_ratings_matrix(old_to_new_movie_ids, old_to_new_user_id_dict)


parser=argparse.ArgumentParser(description="Options")

# parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
# parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
parser.add_argument('-topK', action='store', dest='topK', default=10)

args = parser.parse_args()
classifier = Model(args, RATINGS_MATRIX, old_to_new_movie_ids, RATINGS)

# print("Arguments:", args)
classifier.run()


