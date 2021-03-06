import numpy as np
from utils import *
import scipy.special as SP
from copy import copy as cp

class MetaLDA:

	def __init__(self, mu, sigma_sq, beta, num_topics, vocab, dataset_size , feature_size, a, b, gamma=0.55):

		'''
		Constructor for MetaLDA:
		mu = Mean of Topic-Metadata correlation matrix
		sigma_sq = Variance of the matrix
		lamda = Topic-Metadata correlation matrix , size : T * F
		beta = (hyperparameter) prior on pi
		num_topics = number of topics in the model
		pi = multinoulli over vocabulary size: T * V
		vocab = Dictionary containing vocabulary words
		
		key:
		T - description of T
		F - description of F
		V - description of V
		'''
		
		#Store state variables in a dict.
		self.a = a*1.0
		self.b = b*1.0
		self.gamma = gamma*-1.0
		self.lr = self.a * ( self.b ** self.gamma )

		self.mu = mu
		self.sigma_sq = sigma_sq

		self.beta = beta
		self.num_topics = num_topics
		self.vocab = vocab
		self.vocab_size = len(vocab)

		self.dataset_size = dataset_size
		self.inference_step = 0
		self.num_features = feature_size

		self.lamda =  np.random.normal(self.mu,1,(self.num_topics,self.num_features)) * (self.sigma_sq ** 0.5) 

		temp = np.ones(self.vocab_size,dtype=float) * self.beta
		self.theta =  np.random.gamma(temp, size = (self.num_topics, self.vocab_size))

		self.pi = np.zeros((self.num_topics,self.vocab_size), dtype=float)

		for t in range(num_topics):
			self.pi[t,:] = self.theta[t,:] / np.sum(self.theta[t,:])
		
		self.current_iteration = 0

		self.current_state = dict(lamda = self.lamda, pi = self.pi, theta=self.pi)

		self.state = []
		self.state.append(self.current_state)

	'''
	args:
	z is a numpy aray of size nd
	output- scalar ndk.
	'''
	def calc_ndk_dot(self,z,topic_id):
		res = np.sum(z == topic_id)
		return res

	'''
	res outputs ndk vector , of size V(vocab_size). Num_of times each word in vocab has appeared in the doc from topic_id
	'''
	def calc_ndk(self, z, doc, topic_id):
		res = np.zeros(self.vocab_size)
		
		for pos,word in enumerate(doc):
			if(z[pos] == topic_id):
				res[int(word)] = res[int(word)] + 1

		return res


	'''
	z_doc : All samples of z for a document. Size: nd
	Output Size: V dimensional vector
	'''
	def calc_expectation_theta_doc( self, doc, z_doc , topic_id):
		res = np.zeros(self.vocab_size, dtype=float)
		size = 0.0
		for z in z_doc:
			ndk = self.calc_ndk(z, doc, topic_id)
			ndk_dot = self.calc_ndk_dot(z , topic_id)
			temp = ndk - ndk_dot * self.current_state['pi'][topic_id , :]
			res = res + temp
			size = size + 1.0

		return res/size

	'''
	Output size: V dimensional vector 
	'''
	def calc_expectation_sum_theta(self,z_sample, minibatch, topic_id):
		data = minibatch['data']
		res = np.zeros(self.vocab_size, dtype=float)

		for ind,doc in enumerate(data):
			res = res + self.calc_expectation_theta_doc( data[ind], z_sample[ind] , topic_id)

		return res


	'''
	minibatch - 
	Output: - Updated Theta . Dimension - T * V
	z-sample - size : list of Batch_size. Each element of list is an array of size : nd, nd is number of words in that document
	'''
	def update_theta(self, minibatch, z_sample):
		#Calculate expectation term first
		b_size, _  = minibatch['metadata'].shape
		multiplier = (self.dataset_size * 1.0) / b_size

		#expectation is a matrix
		expectation = np.zeros((self.num_topics, self.vocab_size), dtype=float)
		
		for t in range(self.num_topics):
			expectation[t,:] = self.calc_expectation_sum_theta(z_sample, minibatch,t)


		grad_mat = ( self.beta * 1.0) - self.current_state['theta'] + expectation* multiplier

		noise_term = np.random.normal(0,1,size=(self.num_topics, self.vocab_size)) * ( (self.lr * 1.0 ) ** 0.5)

		noise_term = noise_term * self.current_state['theta']
		theta_new = self.current_state['theta'] + self.lr * 0.5 * grad_mat + noise_term

		return np.abs(theta_new)

	def convert_theta2pi(self , theta):
		res = np.zeros((self.num_topics, self.vocab_size), dtype=float)
		for t in range(self.num_topics):
			res[t,:] = theta[t,:] / np.sum(theta[t,:])

		return res

	'''
	Calculates alpha for 1 document. 
	Input : meta_doc , array of size F
	'''
	def calc_alpha(self, meta_doc, state=None):
		#Calculating alpha or Dirichlet prior for mixing proportions for each document
		if(state==None):
			log_alpha = self.current_state['lamda'].dot(meta_doc)  #T
		else:
			log_alpha = state['lamda'].dot(meta_doc)  #T

		alpha = np.exp(log_alpha)
		return alpha



	'''Calculates expected number of words of topic t for each topic. Used in update of lamda

	'''

	def calc_n_topic(self, z_doc):
		sample_size = len(z_doc)
		T = self.num_topics

		res = np.ones((T), dtype=float)  #Size : T
		nd = z_doc[0].shape[0] # nd stores number of words in that document

		for ind,sample in enumerate(z_doc):
			temp2 = np.zeros(T)

			for l in range(T):
				temp2[l] = np.sum(sample==l)

			res = res + temp2

		return res*1.0/ sample_size  #Size: T



	'''
		output size: T * F
	'''
	def calc_grad_lamda(self, z_sample, minibatch):
		b_size, _  = minibatch['metadata'].shape
		multiplier = (self.dataset_size * 1.0) / b_size
		
		prior_grad = (self.current_state['lamda']*(-1.0)) / self.sigma_sq
		res = np.zeros((self.num_topics, self.num_features))


		for index in range(b_size):
			
			doc = minibatch['data'][index]
			meta_doc = minibatch['metadata'][index,:]
			z_doc = z_sample[index]  #z_doc is a list containing arrays of size n_doc
			n_doc = len(doc)

			alpha_doc = self.calc_alpha(meta_doc)
			alpha_sum = np.sum(alpha_doc)

			n_topic = self.calc_n_topic(z_doc)

			phi_t = SP.digamma(alpha_sum) - SP.digamma(alpha_sum + n_doc) + SP.digamma(alpha_doc + n_topic) - SP.digamma(alpha_doc)

			ki_t = alpha_doc * phi_t
			ki_t = ki_t.reshape((self.num_topics, 1))
			meta_doc_reshaped = meta_doc.reshape((1,self.num_features))
			res = res + ki_t.dot(meta_doc_reshaped)
		res = res * multiplier
		res = res + prior_grad

		return res

	def update_lamda(self, minibatch, z_sample):
		lamda_grad = self.calc_grad_lamda( z_sample, minibatch)
		noise = np.random.normal(0,1,(self.num_topics, self.num_features)) * (self.lr)**0.5
		res = self.current_state['lamda'] + 0.5 * self.lr * lamda_grad + noise
		return res


	def update_lr(self):
		self.inference_step = self.inference_step + 1
		self.lr = self.a * ( (self.inference_step + self.b) ** self.gamma)

	'''
	performs inference over an minibatch and updates posterior sample and state for self.lamda and self.pi.
	self.lamda is updated using SGLD
	self.pi is updated using SGRLD
	Args:
		minibatch : minibatch of documents and metadata [Dictionary of arrays]

	'''
	def inference(self, minibatch):

		#Sample z using Gibbs Sampling - needs Alpha, current z allocations for each word(to calculate ndk.), Current Theta
		#Infer lamda using SGLD - To infer lamda need current Learning Rate , grad of log prior of lamda, grad of log likelihood of data, and a gaussian noise
		#Infer theta using SGRLD - Needs A sample of large enough size to calculate expectation wrt to z. Needs beta , previous value of theta.
		#Store the inferences in class variable

		# As a minibatch comes in we need to calculate Alpha and initialize z for that mini-batch. A good initialisation will be important.
		#For that sampling from Alpha will be required using mixing proportion.

		#Once we have an initial z, we can calculate ndk./i and sample from posterior z.

		#Update Learning Rate for SGLD and SGRLD optimizers
		
		z_sample = Gibbs_sampler(minibatch, self.current_state["lamda"], self.current_state["theta"])

		# theta_new size: T * V
		theta_new = self.update_theta(minibatch , z_sample)
		pi_new = self.convert_theta2pi(theta_new)
		lamda_new = self.update_lamda(minibatch, z_sample)
		
		#Updates Learning Rate as well as inference_step number
		self.update_lr()

		self.current_state["lamda"] = lamda_new
		self.current_state["pi"] = pi_new
		self.current_state["theta"] = theta_new
		self.state.append( cp(self.current_state) )


	'''
	Averages the updates of theta, pi and lamda, using the n latest updates
	'''
	def calc_average_state(self, n):
		assert n < len(self.state), " n should be less than total number of update steps taken {}".format(len(self.state))
		temp_theta = self.state[-n]["theta"]
		temp_pi = self.state[-n]["pi"]
		temp_lamda = self.state[-n]["lamda"]
		for i in range(1,n):
			temp_theta = temp_theta + self.state[-n + i]["theta"]
			temp_pi = temp_pi + self.state[-n + i]["pi"]
			temp_lamda = temp_lamda +self.state[-n + i]["lamda"]
		avg_state = dict(lamda = temp_lamda/(n*1.0) , theta = temp_theta/(n*1.0) , pi = temp_pi/(n*1.0))
		return avg_state


	def calc_nd_dot(self,z):
		res = np.zeros( (self.num_topics), dtype = float)
		
		for t in range(self.num_topics):
			res[t] = self.calc_ndk_dot(z,t)
		return res


	'''
		calculates perplexity on test set
		test-set is a dict with a train set and test set

	'''
	def eval_perplexity(self, test_set , n = 1):
		
		train_words = test_set['train']  #List containing words(word indices in vocabulary)
		test_words = test_set['test']  #List containing words(word indices in vocabulary)
		test_meta = test_set['metadata']  #numpy array containing meta data for the single document

		avg_state = self.calc_average_state(n)
		minibatch_train = dict(data = train_words, metadata = test_meta)
		
		z_sample = Gibbs_sampler(minibatch_train, avg_state['lamda'], avg_state['theta'])[0]

		for i,z in enumerate(z_sample):
			if(i==0):
				res = self.calc_nd_dot(z)
			else:
				res = res + self.calc_nd_dot(z)

		res = (res*1.0)/len(z_sample)

		#ndk_dot_train is evaluated on train words size T
		ndk_dot_train = res

		alpha = self.calc_alpha(test_meta.reshape((-1,1)) , avg_state)

		eta = alpha + ndk_dot_train.reshape((-1,1))
		eta = eta*1.0 /np.sum(eta)

		#pwd gives array of size V.
		pwd = eta.reshape((1,-1)).dot(avg_state['pi'])[0]

		neg_log_pwd = -1 * np.log(pwd)
		log_perplexity = 0
		
		for word in test_words[0]:
			log_perplexity = log_perplexity + neg_log_pwd[int(word)]

		log_perplexity=log_perplexity / (1.0*len(test_words))

		perplexity = np.exp(log_perplexity)
		
		return perplexity



