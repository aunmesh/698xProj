import numpy as np 
from copy import copy as cp

'''
alpha is Dirichlet prior , size : Batch_size * T 
Returns: sample from Dirichlet alpha, Batch_size * T
'''
def sample_mixing_prop(alpha):
	b_size, T = alpha.shape
	result = np.ones((b_size,T) , dtype=float)

	for i in range(b_size):
		result[i] = np.random.dirichlet(alpha[i])

	return result



'''
Args:
	minibatch - data and metadata
	lambda - T * F array. T - size of topics. F is size of meta-data features
	theta - T * V array. V is vocabulary size

'''




def Gibbs_sampler(minibatch, lamda, theta , sample_size = 100, burn_in = 100, num_sim = 200):
	

	metadata = minibatch['meta-data'] # Batch_size * F
	data = minibatch['data'] # List of size Batch-size. Each list element carries a vector of size nd for that document

	assert(lamda.shape[0] == theta.shape[0], "Lambda and Theta array number of rows should be same")
	assert(lamda.shape[1] == metadata.shape[1], "Lambda and Metadata array number of cols should be same")

	#Calculating alpha or Dirichlet prior for mixing proportions for each document
	log_alpha = np.transpose( lamda.dot(np.transpose(metadata)) )  # Batch_size * T
	alpha = np.exp(log_alpha)

	T = lamda.shape[0]
	F = lamda.shape[1]
	b_size = metadata.shape[0]

	






	#Initialise z
	z_initial = []
	for doc in data:
		temp = np.ones(doc.shape[0][0], dtype = int) * -1
		z_initial.append(temp)





