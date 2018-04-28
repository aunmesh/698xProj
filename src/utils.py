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
args:
z - 

'''
def calc_ndk_dot(z,T):
	b_size = len(z)
	ndk_dot = np.ones((b_size,T), dtype=int)  #Size : Batch_size * T

	for ind,doc in enumerate(z):
		nd = doc.shape[0] # nd stores number of words in that document

		temp = z[ind]

		temp2 = np.zeros(T)
		for l in range(T):
			temp2[l] = np.sum(temp==l)

		ndk_dot[ind] = temp2
	return ndk_dot  #Size: b_size * T


'''
alpha : batch_size * T
theta : T * V
ndk_dot : Size: b_size * T
z : size : list of Batch_size. Each element of list is an array of size : nd, nd is number of words in that document

Returns a list of size batch_size, each element is an array of shape nd * T
'''
def calc_z_posterior(alpha, theta, ndk_dot, data, z):
	result = []
	T = alpha.shape[1]
	
	for ind,doc in enumerate(data):
		nd  = len(doc)
		temp = np.zeros((nd,T), dtype=float)
		
		for pos,word in enumerate(doc):
			mul1 = alpha[ind] + ndk_dot[ind] #Size: T
			
			for i in range(T):
				if(z[ind][pos] == i):
					mul1[i]-=1
					break  # As there is only one topic corresponding to a word
				else:
					continue
			wdi = data[ind][pos] # wdi stores the index in vocabulary of the current word
			theta_wdi = theta[:,wdi]
			temp[pos] = ( mul1 * theta_wdi ) / np.sum(mul1 * theta_wdi)

		result.append(temp)

	return result


#Input : list of size batch_size, each element is an array of shape nd * T
#Output: Each element in the list is an array of size nd. where nd is number of words in that document.
def gen_sample_z(z_posterior):
	result = []
	
	for doc_post in z_posterior: #posterior vector for each document
		nd, T = doc_post.shape

		temp = np.zeros(nd)
		for pos in range(nd):
			probab = doc_post[pos,:]
			temp[pos] = np.random.choice( T, size = 1, p = probab )[0]
		
		result.append(temp)

	return result

'''
Args:
	minibatch - data and metadata
	lambda - T * F array. T - size of topics. F is size of meta-data features
	theta - T * V array. V is vocabulary size

'''

def Gibbs_sampler(minibatch, lamda, theta , sample_size = 2, burn_in = 2):

	num_sim = sample_size + burn_in

	metadata = minibatch['metadata'] # Batch_size * F
	data = minibatch['data'] # List of size Batch-size. Each list element carries a vector of size nd for that document

	assert lamda.shape[0] == theta.shape[0], "Lambda and Theta array number of rows should be same"
	assert lamda.shape[1] == metadata.shape[1], "Lambda and Metadata array number of cols should be same " + str(lamda.shape) + " " + str(metadata.shape)

	#Calculating alpha or Dirichlet prior for mixing proportions for each document
	log_alpha = np.transpose( lamda.dot(np.transpose(metadata)) )  # Batch_size * T
	alpha = np.exp(log_alpha)

	T = lamda.shape[0]
	F = lamda.shape[1]
	b_size = metadata.shape[0]

	eta_initial = sample_mixing_prop(alpha)


	#Initialise z , size : list of Batch_size. Each element of list is an array of size : nd, nd is number of words in that document
	z_initial = []

	for ind,doc in enumerate(data):
		nd = len(doc)  # nd stores number of words in that document

		temp = np.random.choice( T, size = nd, p = eta_initial[ind] )
		z_initial.append(temp)



	final_sample = [[]] * b_size # list of lists. Each sublist has z_samples for that document
	
	for step in range(num_sim):
		ndk_dot = calc_ndk_dot(z_initial,T)

		z_posterior = calc_z_posterior(alpha, theta, ndk_dot, data, z_initial) #list of size batch_size, each element is an array of shape nd * T

		if(step > burn_in):
			temp_sample = gen_sample_z(z_posterior)
			for b in range(b_size):
				final_sample[b].append( temp_sample[b] )
		z_initial = gen_sample_z(z_posterior)
	return final_sample

