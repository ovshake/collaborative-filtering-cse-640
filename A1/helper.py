import numpy as np 
import matplotlib.pyplot as plt 

def get_cosine_similarity(a , b):
	norm_a = 0
	norm_b = 0  
	sim = 0
	for i in range(len(a)):
		if a[i] * b[i] != 0:
			sim += a[i] * b[i] 
			norm_b += b[i]**2 
			norm_a += a[i]**2 
	if norm_a == 0 or norm_b == 0:
		return 0 
	return sim / ((norm_a**0.5) * (norm_b**0.5))



def my_pairwise_distance(matrix):
	l = len(matrix)
	cosine_sim = np.zeros((l , l)) 
	for i in range(l):
		for j in range(l):
			if i != j:
				sim = get_cosine_similarity(matrix[i] , matrix[j])
				cosine_sim[i][j] = sim 
				cosine_sim[j][i] = sim 


